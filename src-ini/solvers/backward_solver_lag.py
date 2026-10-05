"""LRP backward pass: route Level Set -> fixed-A refresh -> node Level Set."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import math
import os
from numbers import Integral
import time

from cuts.benders_cuts import clean_pi, add_unique_cut
from core.backend_telemetry import backend_scope
from core.stage2_tolerance import effective_abs_gap
from cuts.lagrangian_cuts import (LagrangianCutManager, _reaudit_fixed_trial_witness,
                                  outer_gap_tight_exit_abs)
from models.stage_builder import StageModelBuilder, _instance, _state_keys
from models.subproblem_builder import SubproblemBuilder
from solvers.forward_solver import ForwardSolver, solve_stage2_node, solve_stage3_node
from s2forward.refresh_handoff import build_refresh_handoff, _node_key
from s2forward.refinement_budget import RefinementBudget
from solvers.backward_solver_sbc import BackwardSolverSBC
from solvers.forward_policy_certification import (
    InvalidForwardPolicy, certify_lrp_forward_policy, certify_stage3_forward_policy,
)


class BackwardSolverLagrangian:
    """Retain the original three-layer backward flow and Level Set manager.

    All physical node solves use the current LRP builders. No enumeration or
    extensive model is used here. Independent cuts may run in worker pools;
    whole-context refreshes can also run in workers; commits remain in the parent.
    """
    def __init__(self, prob_data, stage_builder=None, cut_manager=None,
                 lambda_level=0.3, mu_level=0.5, norm_option=1, tol=1e-3, iter_limit=100,
                 adaptive_alpha=False, sub_time_limit=60.0,
                 inner_s3_rounds=0, inner_s3_rel_tol=1e-6,
                 inner_s3_time_limit=300.0, s2_piece_cache=None,
                 s2_cut_time_limit=0.0):
        self.prob_data = prob_data
        self.stage_builder = stage_builder or StageModelBuilder(prob_data)
        self.cut_manager = cut_manager or LagrangianCutManager({})
        self.cut_manager.stage2_lazy_threshold = self.stage_builder.lazy_threshold
        self.cut_manager._model_env = self.stage_builder.env
        self.lambda_level, self.mu_level = lambda_level, mu_level
        self.norm_option, self.tol, self.iter_limit = norm_option, tol, iter_limit
        self.adaptive_alpha = bool(adaptive_alpha)
        self.sub_time_limit = float(sub_time_limit)
        self.inner_s3_rounds = max(0, int(inner_s3_rounds))
        self.inner_s3_rel_tol = max(0., float(inner_s3_rel_tol))
        self.inner_s3_time_limit = float(inner_s3_time_limit)
        self.s2_cut_time_limit = float(s2_cut_time_limit)
        if math.isnan(self.sub_time_limit) or self.sub_time_limit <= 0:
            raise ValueError('sub_time_limit must be positive; +inf means no per-solve cap')
        if not math.isfinite(self.inner_s3_time_limit) or self.inner_s3_time_limit <= 0:
            raise ValueError('LRP refinement limit must be positive and finite')
        if not math.isfinite(self.s2_cut_time_limit) or self.s2_cut_time_limit < 0:
            raise ValueError('S2 pass budget must be finite and nonnegative; zero disables it')
        self.last_inner_policy = None
        self.last_stage2_handoff = None
        self.last_refinement_limited = False
        self.last_refinement_diagnostic = None
        self.last_cut_diagnostics = []
        self.last_s2_cut_diagnostics = []
        self.solve_counts = dict(backward_passes=0, stage2_refresh_solves=0,
                                 stage3_route_solves=0, s2_cuts=0, s3_cuts=0)
        self._refinement_requested = False
        self._refinement_budget_multiplier = 1
        self._next_second = 0
        self._next_s2_cut = 0
        self._s2_tight_exit_abs = None
        self._next_third = {}
        self.s2_piece_cache = None
        self.sbc_prepass_enabled = os.environ.get('LRP_S3_SBC_PREPASS', '0') == '1'
        self.worker_pids = set()
        self._route_sbc = BackwardSolverSBC(prob_data, self.stage_builder,
            subproblem_builder=SubproblemBuilder(prob_data, lazy_threshold=self.stage_builder.lazy_threshold, env=self.stage_builder.env),
            strengthen_s2=True, strengthen=False, sub_time_limit=min(1., self.sub_time_limit))
        self._route_witness_buffer = None
        self.cut_manager._route_witness_sink = None

    def configure_route_witness_collection(self, enabled=True, *, max_routes=512,
                                           max_routes_per_scope=32):
        """Opt-in primal evidence only; no extra oracle or optimizer invocation."""
        if type(enabled) is not bool:
            raise TypeError('route witness enabled flag must be bool')
        if not enabled:
            self._route_witness_buffer = None
            self.cut_manager._route_witness_sink = None
            return
        if any(type(v) is not int or v < 1 for v in (max_routes, max_routes_per_scope)):
            raise ValueError('route witness limits must be positive integers')
        from solvers.lrp_backward_route_witnesses import BackwardRouteWitnessBuffer
        limits = dict(max_routes=max_routes, max_routes_per_scope=max_routes_per_scope)
        current = self._route_witness_buffer
        if current is None or current.limits != limits:
            current = BackwardRouteWitnessBuffer(self.prob_data, **limits)
            self._route_witness_buffer = current
        self.cut_manager._route_witness_sink = current

    @property
    def route_witness_count(self):
        return 0 if self._route_witness_buffer is None else len(self._route_witness_buffer)

    def peek_route_witnesses(self, max_routes=None, *, deadline=None):
        """Reaudited snapshot; callers may stop on their deadline without losing rows."""
        return (() if self._route_witness_buffer is None else
                self._route_witness_buffer.peek(max_routes, deadline=deadline))

    def discard_route_witnesses(self, witnesses):
        return (0 if self._route_witness_buffer is None else
                self._route_witness_buffer.discard(witnesses))

    def drain_route_witnesses(self, max_routes=None, *, deadline=None):
        return (() if self._route_witness_buffer is None else
                self._route_witness_buffer.drain(max_routes, deadline=deadline))

    @staticmethod
    def _resolve_s3_num_processes(num_processes, s3_process_multiplier):
        if num_processes <= 1:
            return 1
        requested = int(num_processes) * max(1, int(s3_process_multiplier))
        budget = max(1, int(os.environ.get('LRP_CPU_BUDGET',
                                          os.environ.get('VRP_CPU_BUDGET', '12'))))
        return max(1, min(requested, os.cpu_count() or budget, budget))

    def _parallel_cut_batch(self, stage, entries, cut_lag, cut_Dict, options,
                            deadline, workers, pool, *, seed_only=False):
        """Run independent physical node jobs; commit in scheduler order."""
        from solvers.lrp_parallel import map_worker_jobs
        from solvers.lrp_parallel_levelset import (make_lagrangian_job,
            lagrangian_worker, merge_lagrangian_result)
        if not entries:
            return []
        deadline = deadline if deadline is not None and math.isfinite(deadline) else None
        remaining = math.inf if deadline is None else max(0., deadline-time.monotonic())
        if remaining <= 0:
            return []
        slots = min(max(1, int(workers)), len(entries))
        seconds = remaining / math.ceil(len(entries) / slots) if math.isfinite(remaining) else None
        if seed_only:
            seconds = min(2., seconds) if seconds is not None else 2.
            if seconds < .5:
                return []
        jobs = [make_lagrangian_job(self, stage, node, index, parent, target,
                cut_lag, cut_Dict, options, deadline, node_seconds=seconds,
                seed_only=seed_only, skip_seed=skip_seed)
                for node, index, parent, target, skip_seed in entries]
        results = map_worker_jobs(lagrangian_worker, jobs, workers, pool=pool)
        visited = []
        for job, result in zip(jobs, results):
            merge_lagrangian_result(self, job, result, cut_lag, cut_Dict)
            if result[0]['started']:
                visited.append(job['index'])
                self.worker_pids.add(result[0]['worker_pid'])
        return visited

    def _route_plan(self, scen_tree, second_ind, x_star, cost_star, cut_lag):
        """Keep one physical parent's current route trials in cursor order."""
        routes = list(scen_tree[2][second_ind].successor)
        offset = self._next_third.get(second_ind, 0) % max(1, len(routes))
        parent = x_star[2][second_ind]
        pending = [r for r in routes[offset:] + routes[:offset] if
            cost_star[3][r] - ForwardSolver._envelope(
                cut_lag.get(3, {}).get(r, ()), parent) > max(1e-7,
                self.inner_s3_rel_tol * max(1., abs(cost_star[3][r])))]
        return second_ind, routes, pending, parent

    def _separate_route_wave(self, scen_tree, plans, cost_star, cut_lag,
                             cut_Dict, options, route_deadline, workers, pool):
        """Batch independent contexts; finish all merges before any refresh."""
        trials = [(q, routes, r, parent) for q, routes, pending, parent in plans
                  for r in pending]
        seeded_routes = set()
        if self.sbc_prepass_enabled and self.prob_data.shape[1] > 8 and trials:
            seed_started = time.monotonic()
            seed_deadline = min(route_deadline, seed_started + 2. * len(trials))
            if workers > 1 and len(trials) > 1:
                entries = [(scen_tree[3][r], r, parent, cost_star[3][r], False)
                           for _, _, r, parent in trials]
                seeded_routes.update(self._parallel_cut_batch(3, entries, cut_lag,
                    cut_Dict, options, seed_deadline, workers, pool, seed_only=True))
            else:
                for position, (_, _, r, parent) in enumerate(trials):
                    now = time.monotonic()
                    share = (seed_deadline-now) / (len(trials)-position)
                    if share < .5:
                        break
                    self._generate_cut(3, scen_tree[3][r], r, parent,
                        cost_star[3][r], cut_lag, cut_Dict, options,
                        min(seed_deadline, now+min(2., share)), seed_only=True)
                    seeded_routes.add(r)
            trials = [(q, routes, r, parent) for q, routes, r, parent in trials if
                cost_star[3][r] - ForwardSolver._envelope(
                    cut_lag.get(3, {}).get(r, ()), parent) > max(1e-7,
                    self.inner_s3_rel_tol * max(1., abs(cost_star[3][r])))]
        visited = []
        if workers > 1 and len(trials) > 1:
            entries = [(scen_tree[3][r], r, parent, cost_star[3][r], r in seeded_routes)
                       for _, _, r, parent in trials]
            visited = self._parallel_cut_batch(3, entries, cut_lag, cut_Dict,
                options, route_deadline, workers, pool)
            scopes = {r: (q, routes) for q, routes, r, _ in trials}
            for r in visited:
                q, routes = scopes[r]
                self._next_third[q] = (routes.index(r)+1) % len(routes)
        else:
            for position, (q, routes, r, parent) in enumerate(trials):
                now = time.monotonic()
                if now >= route_deadline:
                    break
                self._next_third[q] = (routes.index(r)+1) % len(routes)
                target = cost_star[3][r]
                envelope = ForwardSolver._envelope(cut_lag.get(3, {}).get(r, ()), parent)
                if target-envelope <= max(1e-7, self.inner_s3_rel_tol*max(1., abs(target))):
                    continue
                local_deadline = now + (route_deadline-now) / (len(trials)-position)
                self._generate_cut(3, scen_tree[3][r], r, parent, target,
                    cut_lag, cut_Dict, options, local_deadline, skip_seed=r in seeded_routes)
                visited.append(r)
        return visited

    def request_refinement(self):
        """Restore Final's explicit effort escalation within the outer deadline."""
        self._refinement_requested = True
        self._refinement_budget_multiplier *= 2

    def scheduler_checkpoint(self):
        """Save traversal and requested effort, scoped to the physical data."""
        return dict(version='lrp_backward_scheduler_v1',
                    instance_sha256=_instance(self.prob_data).logical_hash(),
                    next_second=int(self._next_second),
                    next_s2_cut=int(self._next_s2_cut),
                    next_third=deepcopy(self._next_third),
                    refinement_effort=dict(version='lrp_refinement_effort_v1',
                        budget_multiplier=self._refinement_budget_multiplier,
                        pending_force=self._refinement_requested))

    def restore_scheduler_state(self, state):
        """Restore a compatible cursor without restoring models or targets."""
        data = _instance(self.prob_data)
        if (not isinstance(state, Mapping) or
                state.get('version') != 'lrp_backward_scheduler_v1' or
                state.get('instance_sha256') != data.logical_hash()):
            raise ValueError('Backward scheduler checkpoint belongs to different LRP data')
        m, _, H, _, S = data.shape

        def offset(value, size):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
                raise ValueError('Invalid backward scheduler cursor')
            return int(value)

        second = offset(state.get('next_second'), H*S)
        s2_cut = offset(state.get('next_s2_cut'), H*S)
        saved_third = state.get('next_third')
        if not isinstance(saved_third, Mapping):
            raise ValueError('Invalid backward route scheduler cursors')
        third = {}
        for key, value in saved_third.items():
            # JSON object keys are strings; pickle preserves their integers.
            key = int(key) if isinstance(key, str) and key.isdecimal() else key
            node = offset(key, H*S)
            if node in third:
                raise ValueError('Duplicate backward route scheduler cursor')
            third[node] = offset(value, m)
        effort = state.get('refinement_effort')
        multiplier, pending_force = 1, False
        if effort is not None:
            if (not isinstance(effort, Mapping) or
                    effort.get('version') != 'lrp_refinement_effort_v1'):
                raise ValueError('Invalid refinement effort checkpoint')
            multiplier = effort.get('budget_multiplier')
            pending_force = effort.get('pending_force')
            if (isinstance(multiplier, bool) or not isinstance(multiplier, Integral)
                    or multiplier < 1 or type(pending_force) is not bool):
                raise ValueError('Invalid refinement effort state')
            try:
                finite_budget = math.isfinite(self.inner_s3_time_limit * multiplier)
            except OverflowError:
                finite_budget = False
            if not finite_budget:
                raise ValueError('Nonfinite refinement effort budget')
        self._next_second, self._next_s2_cut, self._next_third = second, s2_cut, third
        self._refinement_budget_multiplier = int(multiplier)
        self._refinement_requested = pending_force

    def _record_manager(self):
        self.solve_counts.update(getattr(self.cut_manager, 'solve_counts', {}))
        diagnostic = deepcopy(getattr(self.cut_manager, 'last_cut_diagnostic', None))
        if diagnostic is not None:
            self.last_cut_diagnostics.append(diagnostic)
            if diagnostic['stage'] == 2:
                self.last_s2_cut_diagnostics.append(diagnostic)

    def _generate_cut(self, stage, node, node_index, parent, target,
                      cut_lag, cut_Dict, options, deadline=None, *, seed_only=False, skip_seed=False):
        seeded = False
        if self.prob_data.shape[1] > 8 and not skip_seed:
            remaining = math.inf if deadline is None else deadline-time.monotonic()
            if remaining > .1:
                self._route_sbc.sub_time_limit = min(1., self.sub_time_limit, remaining/3.)
                self._route_sbc.last_cut_diagnostics = []
                generate = (self._route_sbc._generate_third_stage_cut if stage == 3 else
                            self._route_sbc._generate_second_stage_cut)
                seed = generate(node, parent, cut_lag, node_index,
                    **({'deadline': deadline} if deadline is not None else {}))
                count_key = f's{stage}_sbc_seed_calls'
                self.solve_counts[count_key] = self.solve_counts.get(count_key, 0)+1
                self.solve_counts[f's{stage}_seed_lp_solves'] = self._route_sbc.solve_counts[f'stage{stage}_lp']
                self.solve_counts[f's{stage}_seed_mip_solves'] = self._route_sbc.solve_counts[f'stage{stage}_mip']
                if self._route_sbc.last_cut_diagnostics:
                    diagnostic = deepcopy(self._route_sbc.last_cut_diagnostics[-1])
                    diagnostic['source'] = ('phase2_route_dfj_sbc_seed' if stage == 3 else
                                            'phase2_assignment_sbc_seed')
                    self.last_cut_diagnostics.append(diagnostic)
                if seed is not None:
                    seeded = add_unique_cut(cut_lag.setdefault(stage, {}).setdefault(node_index, []), *seed)
                    self.solve_counts[f's{stage}_cuts'] += int(seeded)
                    # A tight SBC row can still have a large multiplier norm
                    # and be weak on other assignments. It seeds the archive,
                    # but does not replace Phase-2's minimum-norm Level Set.
                    # S2's seed also propagates the refreshed route envelope
                    # before its separate multiplier-search budget expires.
        if seed_only:
            return seeded
        archive = cut_Dict.setdefault(stage, {}).setdefault(node_index, [])
        with backend_scope(phase=2, path='backward', stage=stage):
            pi, _, intercept, _, converged = self.cut_manager.solve_lagrangian_dual(
                self.prob_data, stage, node, cut_lag, archive,
                options['lambda_level'], options['mu_level'], options['norm_option'],
                options['tol'], options['iter_limit'], float(target),
                node_ind=node_index, x_prev=parent, adaptive_alpha=self.adaptive_alpha,
                sub_time_limit=self.sub_time_limit, deadline=deadline,
                tight_exit_abs=getattr(self, '_s2_tight_exit_abs', None) if stage == 2 else None,
                s2_abs_tol=getattr(self, '_s2_abs_tol', None),
                s2_rel_cap=getattr(self, '_s2_rel_cap', None))
        self._record_manager()
        if intercept is None:
            return seeded
        pi, intercept, _, _ = clean_pi(pi, intercept)
        changed = add_unique_cut(cut_lag.setdefault(stage, {}).setdefault(node_index, []),
                                 pi, intercept)
        self.solve_counts[f's{stage}_cuts'] += int(changed)
        return changed or seeded

    def _remember_policy(self, scen_tree, x_star, cost_star):
        certificate = certify_lrp_forward_policy(self.prob_data, x_star, scen_tree)
        candidate = dict(ub=certificate['feasible_upper_bound'],
                         x_star=deepcopy(x_star), cost_star=deepcopy(cost_star),
                         certificate=certificate, certified=True)
        if self.last_inner_policy is None or candidate['ub'] < self.last_inner_policy['ub']:
            self.last_inner_policy = candidate
        return certificate

    def _fixed_route_witness(self, node, assignment):
        """A cached tour is a primal incumbent only at its complete physical state."""
        cache = getattr(self.cut_manager, '_fixed_target_cache', {})
        if not cache:
            return None
        names = _state_keys(node.context, int(node.info))
        if any(name not in assignment for name in names):
            return None
        trial = {name: float(assignment[name]) for name in names}
        key = (3, node.context.key, int(node.info), tuple(trial.items()), ())
        cached = cache.get(key, {})
        witness = cached.get('fixed_trial_witness')
        if witness is None:
            return None
        try:
            return _reaudit_fixed_trial_witness(self.prob_data, node, trial, witness)
        except (ValueError, KeyError):
            # Old checkpoints may lack a usable route; their scalar target
            # never substitutes for independently checked primal decisions.
            return None

    def _apply_fixed_route_witnesses(self, tree, x_star, cost_star, second_indices):
        """Keep a coherent improved policy before any assignment is refreshed."""
        changed = False
        for second_ind in second_indices:
            assignment = x_star[2][second_ind]
            for third_ind in tree[2][second_ind].successor:
                node = tree[3][third_ind]
                witness = self._fixed_route_witness(node, assignment)
                if witness is None:
                    continue
                _, previous = certify_stage3_forward_policy(
                    self.prob_data, node, assignment, x_star[3][third_ind])
                if witness['physical_upper'] < previous:
                    values = dict(witness['route'], stage_cost=witness['physical_upper'])
                    x_star[3][third_ind] = values
                    cost_star[3][third_ind] = witness['physical_upper']
                    self.solve_counts['stage3_witness_improvements'] = self.solve_counts.get(
                        'stage3_witness_improvements', 0) + 1
                    changed = True
        if changed:
            self._remember_policy(tree, x_star, cost_star)
        return changed

    @staticmethod
    def _assignment_signature(second_ids, x_star):
        """Physical discrete trials, excluding changing epigraphs/cost fields."""
        return tuple((q, tuple(sorted((key, float(value))
            for key, value in x_star[2][q].items()
            if key.startswith(('alpha[', 'u[', 'e[', 'z['))))) for q in second_ids)

    def _scheduled_inner_search_sufficient(self, scores, complete):
        """Effort stopping only: actual route and capped S2 intervals suffice."""
        if not complete or not scores:
            return False
        for score in scores.values():
            upper = score.get('envelope_upper')
            lower = score.get('envelope_lower')
            residual = score.get('route_residual')
            physical = score.get('physical_upper')
            if any(value is None or not math.isfinite(float(value))
                   for value in (upper, lower, residual, physical)):
                return False
            tolerance = effective_abs_gap(self._s2_abs_tol, upper, self._s2_rel_cap)
            if (tolerance is None or not 0. <= upper-lower <= tolerance or
                    not 0. <= residual <= max(1e-7,
                        self.inner_s3_rel_tol * max(1., abs(physical)))):
                return False
        return True

    def _refresh(self, scen_tree, x_star, cost_star, cut_lag, deadline, second_indices=None):
        """Solve assignment envelopes and their actual routes at this same A."""
        scores = {}
        handoff_solver = ForwardSolver(self.prob_data, self.stage_builder,
                                       sub_time_limit=self.sub_time_limit)
        handoff_solver.set_stage2_tolerance(getattr(self, '_s2_abs_tol', None),
                                           getattr(self, '_s2_rel_cap', None))
        model_options = handoff_solver._handoff_model_options()
        solve_policy = handoff_solver._period_solve_policy()
        customers = _instance(self.prob_data).shape[1]
        for second_ind in (scen_tree[1][0].successor if second_indices is None else second_indices):
            started = time.monotonic()
            remaining = deadline - started
            if remaining <= 0:
                return scores, False
            node = scen_tree[2][second_ind]
            # The old inner loop evaluated physical routes separately from
            # S2. Under one shared deadline, reserve time for that work.
            assignment_deadline = started + .6 * remaining
            result = solve_stage2_node(self.prob_data, node, cut_lag, x_star[1][0],
                sub_time_limit=min(self.sub_time_limit, assignment_deadline-started), stage_builder=self.stage_builder,
                s2_abs_tol=getattr(self, '_s2_abs_tol', None), s2_rel_cap=getattr(self, '_s2_rel_cap', None),
                deadline=assignment_deadline,
                previous_policy=(dict(x_star[2][second_ind])
                                 if second_ind in x_star[2] else None))
            self.solve_counts['stage2_refresh_solves'] += 1
            assignment_seconds = time.monotonic()-started
            assignment = result['x']
            routes, costs, route_diagnostics = {}, {}, []
            witnesses = {q: self._fixed_route_witness(scen_tree[3][q], assignment)
                         for q in node.successor}
            unpaid_routes = sum(witnesses[q] is None and any(
                assignment[f'alpha[{int(scen_tree[3][q].info)},{j}]']
                for j in range(customers)) for q in node.successor)
            for third_ind in node.successor:
                remaining = deadline - time.monotonic()
                third = scen_tree[3][third_ind]
                i = int(third.info)
                chosen = [j+1 for j in range(customers) if assignment[f'alpha[{i},{j}]']]
                record = dict(node=third_ind, remaining_seconds=max(0., remaining),
                              solve_called=False, deadline_exhausted=remaining <= 0,
                              previous_cost=None, solver_cost=None)
                if not chosen:
                    values, cost = certify_stage3_forward_policy(self.prob_data, third, assignment, {})
                    values['stage_cost'] = cost
                    routes[third_ind], costs[third_ind] = values, cost
                    self.solve_counts['stage3_empty_routes'] = self.solve_counts.get('stage3_empty_routes', 0)+1
                    route_diagnostics.append(dict(record, source='empty', physical_cost=cost))
                    continue
                # An earlier physical tour remains a feasible incumbent only
                # if a fresh audit accepts this exact new assignment/context.
                # It is never a route bound or an oracle/Level-Set target.
                best = None
                previous = x_star.get(3, {}).get(third_ind)
                if isinstance(previous, Mapping):
                    try:
                        values, cost = certify_stage3_forward_policy(
                            self.prob_data, third, assignment, previous)
                    except InvalidForwardPolicy:
                        pass  # This refresh changed the assigned customers.
                    else:
                        best = (values, cost, 'previous_audited')
                        record['previous_cost'] = cost
                remaining = deadline-time.monotonic()
                record['remaining_seconds'] = max(0., remaining)
                record['deadline_exhausted'] = remaining <= 0
                route = None
                witness = witnesses[third_ind]
                if witness is not None:
                    # This exact fixed-state route was already paid for in
                    # Level Set. Reuse its feasible policy, not its target LB.
                    cost = witness['physical_upper']
                    record['witness_cost'] = cost
                    if best is None or cost < best[1]:
                        best = (dict(witness['route']), cost, 'fixed_target_witness')
                    self.solve_counts['stage3_fixed_witness_reuses'] = self.solve_counts.get(
                        'stage3_fixed_witness_reuses', 0) + 1
                elif remaining > 0:
                    route_seconds = min(self.sub_time_limit, remaining / unpaid_routes)
                    route_deadline = min(deadline, time.monotonic() + route_seconds)
                    record.update(allocated_seconds=route_seconds, solve_deadline=route_deadline)
                    route = solve_stage3_node(self.prob_data, third, cut_lag,
                        assignment, sub_time_limit=route_seconds,
                        stage_builder=self.stage_builder, deadline=route_deadline)
                    self.solve_counts['stage3_route_solves'] += 1
                    record['solve_called'] = True
                else:
                    self.solve_counts['stage3_route_deadlines'] = self.solve_counts.get('stage3_route_deadlines', 0)+1
                if witness is None:
                    unpaid_routes -= 1
                if route is not None and route['x'] is not None:
                    values, cost = certify_stage3_forward_policy(
                        self.prob_data, third, assignment, route['x'])
                    record['solver_cost'] = cost
                    if best is None or cost < best[1]:
                        best = (values, cost, 'solver')
                elif route is not None:
                    self.solve_counts['stage3_no_incumbent'] = self.solve_counts.get('stage3_no_incumbent', 0)+1
                # A route solve timing out does not invalidate the assignment.
                # A complete directed tour in customer-index order is always
                # feasible in this TSP domain, including nonmetric costs and
                # zero-demand customers. Preserve the refreshed assignment so
                # its actual route state can still be separated next round.
                if best is None:
                    path = [0]+chosen+[0]
                    candidate = {f'r[{i},{v},{w}]':1. for v,w in zip(path[:-1],path[1:])}
                    values, cost = certify_stage3_forward_policy(
                        self.prob_data, third, assignment, candidate)
                    best = (values, cost, 'index_fallback')
                    self.solve_counts['stage3_feasible_tour_fallbacks'] = self.solve_counts.get('stage3_feasible_tour_fallbacks', 0)+1
                route_values, cost, source = best
                if source == 'previous_audited':
                    self.solve_counts['stage3_reused_tours'] = self.solve_counts.get('stage3_reused_tours', 0)+1
                    if remaining <= 0:
                        self.solve_counts['stage3_reused_tour_deadlines'] = self.solve_counts.get('stage3_reused_tour_deadlines', 0)+1
                route_values['stage_cost'] = cost
                routes[third_ind], costs[third_ind] = route_values, cost
                route_diagnostics.append(dict(record, source=source, physical_cost=cost))
            x_star[2][second_ind] = assignment
            cost_star[2][second_ind] = result['objective']
            x_star[3].update(routes); cost_star[3].update(costs)
            physical = result['stage_cost'] + math.fsum(costs.values())
            bound = result['lower_bound']
            route_residual = math.fsum(max(0., costs[r] - assignment[f'theta[{r}]'])
                                       for r in node.successor)
            scores[second_ind] = dict(physical_upper=physical, envelope_lower=bound,
                envelope_upper=result['objective'],
                fixed_A_gap=None if bound is None else physical-bound,
                route_residual=route_residual,
                envelope_solver_gap=None if bound is None else result['objective']-bound,
                diagnostic=result['diagnostic'], assignment_seconds=assignment_seconds,
                assignment_deadline=assignment_deadline, refresh_deadline=deadline,
                refresh_seconds=time.monotonic()-started, routes=route_diagnostics,
                fresh_solve=result.get('fresh_solve', False),
                exact_optimal=result.get('exact_optimal', False),
                solve_time_limit=result.get('solve_time_limit'),
                refresh_key=_node_key(self.prob_data, scen_tree, node, x_star[1][0],
                                      cut_lag, model_options, dict(solve_policy,
                                          sub_time_limit=result.get('solve_time_limit'))))
        return scores, True

    def _save_stage2_handoff(self, tree, x_star, cuts, scores):
        forward = ForwardSolver(self.prob_data, self.stage_builder,
                                sub_time_limit=self.sub_time_limit)
        forward.set_stage2_tolerance(self._s2_abs_tol, self._s2_rel_cap)
        self.last_stage2_handoff = build_refresh_handoff(self.prob_data, tree,
            x_star[1][0], x_star[2], scores, cuts,
            model_options=forward._handoff_model_options(),
            solve_policy=forward._period_solve_policy())

    def backward_pass(self, scen_tree, x_star, cost_star, cut_lag, cut_Dict,
                      lambda_level=None, mu_level=None, norm_option=None,
                      tol=None, iter_limit=None, num_processes=1, pool=None,
                      s3_pool=None, s3_process_multiplier=4, outer_gap=None,
                      outer_gap_abs=None, s2_abs_tol=None, s2_rel_cap=None,
                      stage1_eta=None, policy_checkpoint=None, deadline=None,
                      allow_shared_pool=False):
        self._s2_abs_tol, self._s2_rel_cap = s2_abs_tol, s2_rel_cap
        self._s2_tight_exit_abs = outer_gap_tight_exit_abs(outer_gap_abs)
        self.worker_pids = set()
        self._refresh_worker_pids = set()
        s2_workers = int(getattr(pool, '_processes', num_processes))
        s3_workers = int(getattr(s3_pool, '_processes',
            self._resolve_s3_num_processes(num_processes, s3_process_multiplier)))
        if s2_workers > 1 and s3_pool is not None and s3_pool is pool and not allow_shared_pool:
            raise ValueError('Stage-2 and Stage-3 pools must be independent')
        self.last_inner_policy = None
        self.last_stage2_handoff = None
        self.last_cut_diagnostics = []
        self.last_s2_cut_diagnostics = []
        self.last_refinement_limited = False
        self.solve_counts['backward_passes'] += 1
        options = dict(lambda_level=self.lambda_level if lambda_level is None else lambda_level,
                       mu_level=self.mu_level if mu_level is None else mu_level,
                       norm_option=self.norm_option if norm_option is None else norm_option,
                       tol=self.tol if tol is None else tol,
                       iter_limit=self.iter_limit if iter_limit is None else iter_limit)
        # Original search precision schedule. This changes effort only; the
        # emitted cut and global convergence still require valid certificates.
        if self.adaptive_alpha and outer_gap is not None and 0 < outer_gap < math.inf:
            if outer_gap > 5.0:
                options['tol'] *= 5
            elif outer_gap > 1.0:
                options['tol'] *= 2
        self._remember_policy(scen_tree, x_star, cost_star)
        # Additional rounds keep A fixed. Route epigraph residual alone is
        # insufficient: a time-limited S2 incumbent can still miss assignments.
        rounds = max(1, self.inner_s3_rounds)
        if self._refinement_requested:
            rounds = max(rounds, 2)
        self._refinement_requested = False
        outer_deadline = deadline
        now = time.monotonic()
        deadline = now + min(self.inner_s3_time_limit * self._refinement_budget_multiplier,
                            math.inf if outer_deadline is None else .7*max(0., outer_deadline-now))
        # Expose the already allocated shared refinement budget to UB-only polish.
        self._refinement_budget = RefinementBudget(deadline, deadline-now)
        scores, closed, complete = {}, False, False
        global_closed = False
        completed_rounds = 0
        scheduled_search_sufficient = False
        trial_unchanged = False
        refinement_stop_reason = None
        visited_routes, refreshed_nodes = [], []
        second_ids = list(scen_tree[1][0].successor)
        for round_index in range(rounds):
            initial_trial = self._assignment_signature(second_ids, x_star)
            initial_route_cuts = self.solve_counts['s3_cuts']
            # Fill the S3 pool across independent physical contexts, then
            # refresh that bounded wave before selecting any new trial state.
            sweep_start = self._next_second
            order = second_ids[sweep_start:] + second_ids[:sweep_start]
            scores = {}
            position = 0
            while position < len(order):
                now = time.monotonic()
                if now >= deadline:
                    break
                plans = []
                for second_ind in order[position:]:
                    plan = self._route_plan(scen_tree, second_ind, x_star, cost_star, cut_lag)
                    plans.append(plan)
                    # The original parallel algorithm queues the complete
                    # sweep of independent, frozen route trials. Workers can
                    # take another job before the slowest previous one ends.
                    # Preserve the existing per-context serial path.
                    if s3_workers <= 1:
                        break
                wave_deadline = (deadline if s3_workers > 1 else
                    now + (deadline-now) * len(plans) / (len(order)-position))
                route_deadline = now + .6 * (wave_deadline-now)
                visited_routes.extend(self._separate_route_wave(scen_tree, plans,
                    cost_star, cut_lag, cut_Dict, options, route_deadline, s3_workers, s3_pool))
                self._apply_fixed_route_witnesses(scen_tree, x_star, cost_star,
                    [plan[0] for plan in plans])
                refreshed = 0
                committed = []
                if s2_workers > 1 and len(plans) > 1:
                    from solvers.lrp_parallel_refresh import parallel_refresh
                    local_scores, committed = parallel_refresh(self, scen_tree,
                        x_star, cost_star, cut_lag, wave_deadline,
                        [plan[0] for plan in plans], s2_workers, pool=pool)
                    scores.update(local_scores)
                    refreshed_nodes.extend(committed)
                    refreshed = len(committed)
                    for second_ind in committed:
                        self._next_second = (second_ids.index(second_ind)+1) % len(second_ids)
                else:
                    for refresh_position, (second_ind, _, _, _) in enumerate(plans):
                        now = time.monotonic()
                        if now >= wave_deadline:
                            break
                        node_deadline = now + (wave_deadline-now) / (len(plans)-refresh_position)
                        local_scores, local_complete = self._refresh(
                            scen_tree, x_star, cost_star, cut_lag, node_deadline, [second_ind])
                        scores.update(local_scores)
                        if not local_complete:
                            break
                        refreshed_nodes.append(second_ind)
                        committed.append(second_ind)
                        refreshed += 1
                        self._next_second = (second_ids.index(second_ind)+1) % len(second_ids)
                position += refreshed
                if refreshed < len(plans):
                    # A later queued context can finish before an earlier one.
                    # Keep the first uncommitted context, not a count-based cursor.
                    missing = next(plan[0] for plan in plans if plan[0] not in committed)
                    self._next_second = second_ids.index(missing)
                    break
            else:
                # Rotate the first context after a complete sweep, as before.
                self._next_second = (sweep_start+1) % len(second_ids)
            complete = len(scores) == len(second_ids)
            completed_rounds += 1
            self._remember_policy(scen_tree, x_star, cost_star)
            closed = complete and all(
                score['fixed_A_gap'] is not None and
                -2e-6 <= score['fixed_A_gap'] <= max(1e-7,
                    self.inner_s3_rel_tol * max(1., abs(score['physical_upper']))) and
                score['route_residual'] <= max(1e-7,
                    self.inner_s3_rel_tol * max(1., abs(score['physical_upper'])))
                for score in scores.values())
            if policy_checkpoint is not None:
                global_certificate = policy_checkpoint(self.last_inner_policy)
                if global_certificate is not None:
                    self.last_inner_policy.update(global_certificate)
                    global_closed = True
                    refinement_stop_reason = 'global_gap'
                    break
            scheduled_search_sufficient = self._scheduled_inner_search_sufficient(scores, complete)
            trial_unchanged = (complete and
                initial_trial == self._assignment_signature(second_ids, x_star) and
                self.solve_counts['s3_cuts'] == initial_route_cuts)
            if closed:
                refinement_stop_reason = 'fixed_A_closed'
            elif scheduled_search_sufficient:
                refinement_stop_reason = 'route_and_s2_tolerance'
            elif trial_unchanged:
                refinement_stop_reason = 'trial_unchanged'
            elif not complete or time.monotonic() >= deadline:
                refinement_stop_reason = 'deadline'
            elif round_index+1 == rounds:
                refinement_stop_reason = 'round_limit'
            if refinement_stop_reason is not None:
                break
        self.last_refinement_limited = not (closed or global_closed)
        self.last_refinement_diagnostic = dict(rounds=completed_rounds,
            fixed_A_closed=closed, global_gap_closed=global_closed, all_nodes_refreshed=complete,
            stop_reason=refinement_stop_reason,
            scheduled_search_sufficient=scheduled_search_sufficient,
            trial_unchanged=trial_unchanged,
            time_limited=time.monotonic() >= deadline,
            nodes=scores, visited_routes=visited_routes, refreshed_nodes=refreshed_nodes,
            requested_processes=int(num_processes), worker_pids=sorted(self.worker_pids),
            execution=('parallel cut and context refresh nodes' if self._refresh_worker_pids else
                'parallel independent cut nodes; serial refresh' if self.worker_pids else 'serial nodes'))
        if global_closed:
            self._record_counts_only()
            return cut_lag, cut_Dict
        # The S2 value function here is the CURRENT route envelope. Cuts from
        # its free-A oracle lower-bound the true recourse even if inner rounds
        # have not closed yet. Subsequent passes strengthen both cut layers.
        # Bound this pass's S2 multiplier search separately. Dividing the whole
        # remaining run budget here can postpone every subsequent S3 pass
        # until the outer deadline, even when S3 is the remaining bottleneck.
        s2_started = time.monotonic()
        s2_deadline = math.inf if outer_deadline is None else outer_deadline
        if self.s2_cut_time_limit > 0:
            s2_deadline = min(s2_started + self.s2_cut_time_limit, s2_deadline)
        s2_start = self._next_s2_cut
        s2_order = second_ids[s2_start:] + second_ids[:s2_start]
        visited_s2 = []
        if s2_workers > 1 and len(s2_order) > 1:
            entries = [(scen_tree[2][q], q, x_star[1][0], cost_star[2][q], False) for q in s2_order]
            visited_s2 = self._parallel_cut_batch(2, entries, cut_lag, cut_Dict,
                options, s2_deadline, s2_workers, pool)
            for second_ind in visited_s2:
                self._next_s2_cut = (second_ids.index(second_ind)+1) % len(second_ids)
            if len(visited_s2) == len(s2_order):
                self._next_s2_cut = (s2_start+1) % len(second_ids)
        else:
            for position, second_ind in enumerate(s2_order):
                now = time.monotonic()
                if now >= s2_deadline:
                    break
                node = scen_tree[2][second_ind]
                local_deadline = None if not math.isfinite(s2_deadline) else now + (s2_deadline-now)/(len(s2_order)-position)
                self._generate_cut(2, node, second_ind, x_star[1][0],
                    cost_star[2][second_ind], cut_lag, cut_Dict, options, local_deadline)
                visited_s2.append(second_ind)
                self._next_s2_cut = (second_ids.index(second_ind)+1) % len(second_ids)
            else:
                self._next_s2_cut = (s2_start+1) % len(second_ids)
        self.last_refinement_diagnostic.update(
            worker_pids=sorted(self.worker_pids),
            execution=('parallel cut and context refresh nodes' if self._refresh_worker_pids else
                'parallel independent cut nodes; serial refresh' if self.worker_pids else 'serial nodes'),
            s2_cut_budget=self.s2_cut_time_limit,
            s2_cut_seconds=time.monotonic()-s2_started,
            s2_cut_nodes=visited_s2,
            s2_cut_time_limited=time.monotonic() >= s2_deadline)
        self._save_stage2_handoff(scen_tree, x_star, cut_lag, scores)
        self._record_counts_only()
        return cut_lag, cut_Dict

    def _record_counts_only(self):
        self.solve_counts.update(getattr(self.cut_manager, 'solve_counts', {}))
