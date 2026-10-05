"""Production LRP forward pass: facility plan -> assignment -> actual routes."""
from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace
from copy import deepcopy
from functools import wraps

import core.customized_subprob  # preserve the original refresh handoff module path
from s2forward.refresh_handoff import RefreshHandoff, RefreshHandoffPass
from core.solver_settings import (configured_gurobi_threads,
                                 configured_forward_s2_backend, forward_s2_bpc_policy,
                                 forward_s2_requested_mode)
from core.backend_telemetry import backend_call
from core.stage2_tolerance import effective_abs_gap
from core.solve_deadline import SolveDeadlineReached, bounded_solve_time
from models.stage_builder import StageModelBuilder, _instance, _node_context
from models.stage_model_core import InvalidSolverPrimal, evaluate_model
from solvers.forward_period_dedup import (
    stage2_period_key, forward_stage3_route_key, remap_stage2_forward_values,
)
from solvers.forward_policy_certification import (
    InvalidForwardPolicy, certify_lrp_forward_policy,
    certify_stage1_forward_policy, certify_stage2_forward_policy,
    certify_stage3_forward_policy,
)


def canonical_stage1_fleet_signature(prob_data, x_star_1):
    """Compatibility name for a shared LRP facility-plan signature."""
    m, _, _, L, _ = _instance(prob_data).shape
    return tuple(tuple(int(round(x_star_1[f'A[{i},{k}]'])) for k in range(L)) for i in range(m))


def stage1_s2_cut_activity(root_node, cut_lag, x_star_1, eta_by_node, *, tolerance=1e-6):
    records = []
    for node_ind in root_node.successor:
        eta = float(eta_by_node[node_ind])
        local = []
        for index, (pi, intercept) in enumerate(cut_lag.get(2, {}).get(node_ind, ())):
            value = float(intercept) + math.fsum(float(c) * x_star_1[k] for k, c in pi.items())
            local.append(dict(node=int(node_ind), cut_index=index, value=value, slack=eta-value))
        envelope = max((record['value'] for record in local), default=-math.inf)
        for record in local:
            scale = max(1., abs(eta), abs(record['value']))
            record.update(active=envelope-record['value'] <= tolerance*scale,
                          binding=abs(record['slack']) <= tolerance*scale)
        records.extend(local)
    return records


def _within_period_cache(method):
    """Cache only this forward pass, including exception/deadline exits."""
    @wraps(method)
    def run(self, scen_tree, *args, **kwargs):
        self._period_cache = {2: {}, 3: {}} if self.period_dedup else None
        self._period_tree = scen_tree
        handoff, self._pending_stage2_handoff = self._pending_stage2_handoff, None
        self._stage2_handoff_pass = (handoff.begin_pass(self.prob_data, scen_tree)
                                    if handoff is not None else RefreshHandoffPass())
        self.period_dedup_stats = dict(stage2_hits=0, stage3_hits=0)
        try:
            return method(self, scen_tree, *args, **kwargs)
        finally:
            self._period_cache = None
            self._period_tree = None
            self.stage2_handoff_stats = dict(hits=self._stage2_handoff_pass.hits,
                                            rejected=self._stage2_handoff_pass.rejected)
            self._stage2_handoff_pass = RefreshHandoffPass()
    return run


def _phase1_large_auto_probe_plan(phase, context):
    """Final's guaranteed memory-rejected range, without importing fleet DP.

    Final's pure count ceiling is at most 22 under every memory/worker gate.
    Only active>22 is restored here; memory-dependent smaller buckets and the
    physical subset-DP adapter are deliberately outside this limited change.
    """
    if phase != 1 or forward_s2_requested_mode()['mode'] != 'auto':
        return None
    active = sum(float(v) > .5 for v in context.active)
    if active <= 22:
        return None
    raw = None
    for prefix in ('LRP_', 'VRP_'):
        value = os.environ.get(prefix + 'S2_FORWARD_AUTO_GRB_PROBE_S')
        if value is not None and value.strip():
            raw = value
            break
    seconds = float(.1 if raw is None else raw)
    if not math.isfinite(seconds) or seconds <= 0.:
        raise ValueError('S2_FORWARD_AUTO_GRB_PROBE_S must be finite and positive')
    return dict(seconds=seconds, active_customers=active,
                reason='Final_guaranteed_memory_reject_active_gt_22',
                contract='phase1_certified_feasible_trial_v1')


def _fixed_route_native_time_limit():
    """Preserve Investment's configurable fixed-TSP time allowance.

    Native calls are still capped by the caller's subproblem/global deadline.
    A short process-start delay must not force a much slower Gurobi fallback.
    """
    raw = None
    for prefix in ('LRP_', 'VRP_'):
        value = os.environ.get(prefix + 'CONCORDE_TIMEOUT_S')
        if value is not None and value.strip():
            raw = value
            break
    seconds = float(30. if raw is None else raw)
    if not math.isfinite(seconds) or seconds <= 0.:
        raise ValueError('CONCORDE_TIMEOUT_S must be finite and positive')
    return seconds


class ForwardSolver:
    """Keep the original forward API while using only LRP state and costs.

    S1 and S2 objectives contain recourse epigraphs; the policy upper bound is
    independently rebuilt only after every S3 physical route is available.
    Independent scenario-period nodes use process-isolated solver environments
    when workers are requested; policy and bound certification stays in the parent.
    Repeated periods reuse only exactly identical physical LRP models; each
    target assignment and route is independently re-audited before use.
    """
    phase = 2

    def __init__(self, prob_data, stage_builder=None, total_stage=3,
                 sub_time_limit=60.0, period_dedup=True, *, stage2_sub_time_limit=None,
                 stage2_mip_gap=None):
        if total_stage != 3:
            raise ValueError('LRP uses three computational layers')
        if math.isnan(float(sub_time_limit)) or sub_time_limit <= 0:
            raise ValueError('sub_time_limit must be positive; +inf means no per-solve cap')
        self.prob_data = prob_data
        self.instance = _instance(prob_data)
        self.stage_builder = stage_builder or StageModelBuilder(prob_data)
        self.stage2_mip_gap = (self.stage_builder.mip_gap if stage2_mip_gap is None
                              else float(stage2_mip_gap))
        if not math.isfinite(self.stage2_mip_gap) or self.stage2_mip_gap < 0:
            raise ValueError('stage2_mip_gap must be finite and nonnegative')
        self.total_stage = total_stage
        self.sub_time_limit = float(sub_time_limit)
        self.stage2_sub_time_limit = (self.sub_time_limit if stage2_sub_time_limit is None
                                     else float(stage2_sub_time_limit))
        if math.isnan(self.stage2_sub_time_limit) or self.stage2_sub_time_limit <= 0:
            raise ValueError('stage2_sub_time_limit must be positive; +inf means no per-solve cap')
        self.period_dedup = bool(period_dedup)
        self._period_cache = None
        self._period_tree = None
        self.period_dedup_stats = dict(stage2_hits=0, stage3_hits=0)
        self.s2_piece_cache = None
        self._s2_abs_tol = None
        self._s2_rel_cap = None
        self._pending_stage2_handoff = None
        self._stage2_handoff_pass = RefreshHandoffPass()
        self.stage2_handoff_stats = dict(hits=0, rejected=0)
        self._last_eta_per_omega = {}
        self.last_forward_diagnostics = []
        self.solve_counts = {"stage1": 0, "stage2": 0, "stage3": 0}
        self.last_policy_certificate = None
        self._native_routes = {}
        self._refresh_previous_hint = None

    def set_stage2_handoff(self, handoff):
        """Accept one refresh pass; matching and policy audit happen at use."""
        if handoff is not None and not isinstance(handoff, RefreshHandoff):
            raise TypeError("Stage-2 refresh handoff has an invalid payload type")
        self._pending_stage2_handoff = handoff

    def _take_stage2_handoff(self, tree, node, parent, cuts, deadline=None):
        if tree is None or not self._stage2_handoff_pass.entries:
            return None
        # An expired handoff is consumed by this pass but never starts work.
        if deadline is not None and time.monotonic() >= deadline:
            self._stage2_handoff_pass.entries.pop(node.index, None)
            return None
        record = self._stage2_handoff_pass.take(self.prob_data, tree, node.index,
            parent, cuts, model_options=self._handoff_model_options(),
            solve_policy=self._period_solve_policy())
        if record is None:
            return None
        self.last_forward_diagnostics.append(dict(stage=2, node=node.index,
            status='refresh_handoff', optimization_completed=False, gurobi_executed=False,
            policy_reaudited=True, objective=record.cost_star_value,
            certified_lower_bound=record.objective_lower_bound,
            source_time_limit=record.source_time_limit,
            time_limit_compatible_by_certificate=(record.source_time_limit != self.stage2_sub_time_limit
                                                   and record.exact_optimal)))
        values = dict(record.x_dict)
        values['stage_cost'] = record.stage_cost_value
        return values, SimpleNamespace(report={'objective': record.cost_star_value},
            certified_lower_bound=(record.objective_lower_bound
                                   if math.isfinite(record.objective_lower_bound) else None),
            optimal=record.exact_optimal)

    def _handoff_model_options(self):
        # Plain model settings; no live Gurobi Env is serialized.
        return dict(mip_gap=self.stage2_mip_gap,
                    lazy_threshold=self.stage_builder.lazy_threshold,
                    connectivity=self.stage_builder.connectivity)

    def set_stage2_tolerance(self, abs_tol, rel_cap=None):
        self._s2_abs_tol, self._s2_rel_cap = abs_tol, rel_cap

    def _deadline_result(self, stage, node, native_result=None):
        record = dict(stage=stage, node=getattr(node, 'index', None),
                      status='global_deadline', optimization_completed=False,
                      gurobi_executed=False, certified_lower_bound=None)
        self.last_forward_diagnostics.append(record)
        self.solve_counts['deadline_skips'] = self.solve_counts.get('deadline_skips', 0) + 1
        if native_result is not None:
            # An audited native incumbent/bound remain valid after the clock
            # expires. Only the unstarted Gurobi follow-up is skipped.
            record['native_attempt'] = native_result['diagnostic']
            record['certified_lower_bound'] = native_result['lower_bound']
            self.solve_counts[f'stage{stage}'] += 1
            return native_result['x'], SimpleNamespace(
                report={'objective': native_result['objective']},
                certified_lower_bound=native_result['lower_bound'], optimal=native_result['exact'])
        return None, SimpleNamespace(report={'objective': None}, certified_lower_bound=None, optimal=False)

    def _period_solve_policy(self):
        s2_backend = configured_forward_s2_backend(self.instance.shape[1])
        return dict(mip_gap=self.stage2_mip_gap,
                    sub_time_limit=self.stage2_sub_time_limit,
                    s2_abs_tol=self._s2_abs_tol, s2_rel_cap=self._s2_rel_cap,
                    threads=configured_gurobi_threads(),
                    route_backend=os.environ.get('LRP_S3_BACKEND', 'native'),
                    s2_backend=s2_backend,
                    s2_bpc_policy=forward_s2_bpc_policy(self.phase) if s2_backend == 'bpc' else None,
                    connectivity=self.stage_builder.connectivity,
                    lazy_threshold=self.stage_builder.lazy_threshold)

    def _solve_model(self, stage, node, cut_lag, parent, deadline=None):
        if stage == 2:
            handed = self._take_stage2_handoff(self._period_tree, node, parent, cut_lag, deadline)
            if handed is not None:
                return handed
        cache = self._period_cache
        if cache is None or stage not in (2, 3):
            return self._solve_uncached(stage, node, cut_lag, parent, deadline=deadline)
        # Preserve the original deadline behavior: do not start work, including
        # a cache replay, after expiry. The normal policy fallbacks stay intact.
        try:
            bounded_solve_time(self.sub_time_limit, deadline)
        except SolveDeadlineReached:
            return self._deadline_result(stage, node)
        policy = self._period_solve_policy()
        key = (stage2_period_key(node, parent, cut_lag, self.prob_data,
                                self._period_tree, solve_policy=policy) if stage == 2 else
               forward_stage3_route_key(node.info, node, parent, self.prob_data,
                                        solve_policy=policy))
        saved = cache[stage].get(key)
        if saved is not None:
            values = (remap_stage2_forward_values(saved['values'], saved['node'], node,
                                                  self._period_tree) if stage == 2 else
                      deepcopy(saved['values']))
            if stage == 2:
                normalized, physical_cost = certify_stage2_forward_policy(self.prob_data, node, parent, values)
            else:
                normalized, physical_cost = certify_stage3_forward_policy(self.prob_data, node, parent, values)
            values.update(normalized)
            values['stage_cost'] = physical_cost
            self.period_dedup_stats[f'stage{stage}_hits'] += 1
            self.last_forward_diagnostics.append(dict(stage=stage, node=node.index,
                source_node=saved['node'].index, status='period_dedup',
                optimization_completed=False, gurobi_executed=False,
                cached_certificate_reused=True, policy_reaudited=True,
                objective=saved['objective'], certified_lower_bound=saved['lower_bound']))
            return values, SimpleNamespace(report={'objective': saved['objective']},
                certified_lower_bound=saved['lower_bound'], optimal=saved['optimal'])
        values, result = self._solve_uncached(stage, node, cut_lag, parent, deadline=deadline)
        if values is not None:
            try:
                if stage == 2:
                    certify_stage2_forward_policy(self.prob_data, node, parent, values)
                else:
                    certify_stage3_forward_policy(self.prob_data, node, parent, values)
            except InvalidForwardPolicy:
                # Preserve existing caller fallback behavior for an unauditable
                # solver incumbent; it must never populate an equivalence cache.
                pass
            else:
                cache[stage][key] = dict(values=deepcopy(values), node=node,
                    objective=result.report['objective'],
                    lower_bound=result.certified_lower_bound, optimal=result.optimal)
        return values, result

    def _solve_phase1_feasible_probe(self, node, cut_lag, parent, plan, deadline=None):
        """Accept only an audited/rescored policy, never claim probe optimality."""
        from cuts.lrp_static_bounds import basic_route_cuts
        from models.stage_builder import _as_cut, _route_pools
        from solvers.forward_stage2_bpc import _score
        ctx = _node_context(self.instance, node, stage=2)
        pools, successors = _route_pools(ctx, node, cut_lag)
        cuts = {i: basic_route_cuts(ctx, i) + [_as_cut(row, ctx, i) for row in pools[i]]
                for i in range(ctx.m)}
        try:
            limit = bounded_solve_time(min(self.stage2_sub_time_limit, plan['seconds']), deadline)
        except SolveDeadlineReached:
            return self._deadline_result(2, node)
        model = self.stage_builder.build_stage_problem(2, node, cut_lag, parent)
        result = None
        diagnostic = dict(stage=2, node=node.index, backend='gurobi_probe_feasible',
            probe_plan=dict(plan), native_executed=False, optimality_claimed=False,
            requested_mip_gap=0., requested_mip_abs_gap=0., requested_time_limit=limit)
        try:
            result = evaluate_model(model, time_limit=limit, deadline=deadline,
                mip_gap=0., mip_abs_gap=0., threads=configured_gurobi_threads(), output=False,
                optimize_context=lambda: backend_call('gurobi', 'optimize', model=model,
                    phase=1, path='forward', stage=2, attempt_kind='feasible_probe'))
            self.solve_counts['stage2'] += 1
            diagnostic.update(result.summary())
        except SolveDeadlineReached:
            diagnostic.update(status='global_deadline', optimization_completed=False,
                              gurobi_executed=False)
        finally:
            model.dispose()
        values = None
        if result is not None and result.x is not None:
            spec = result.problem.linear
            values = {name: (float(round(value)) if spec.integer[col] else float(value))
                      for col, (name, value) in enumerate(zip(spec.names, result.x))}
        if values is not None:
            try:
                scored = _score(self.prob_data, node, ctx, parent, values,
                                cuts, successors, diagnostic)
            except InvalidForwardPolicy:
                values = None
                diagnostic['fallback'] = 'all_outsourcing_after_exact_capacity_audit'
        if values is None:
            values, _ = self._outsourcing_fallback(node, parent, cut_lag)
            diagnostic.setdefault('fallback', 'all_outsourcing')
            scored = _score(self.prob_data, node, ctx, parent, values,
                            cuts, successors, diagnostic)
        # Retain only the actual fixed-A compact-model lower certificate, and
        # reject an inverted pair rather than clipping it to the policy score.
        bound = None if result is None else result.certified_lower_bound
        if bound is not None and (not math.isfinite(bound) or bound > scored['objective']):
            bound = None
        diagnostic.update(objective=scored['objective'], certified_lower_bound=bound,
            policy_certified=True, complete_archive_rescored=True,
            optimality_claimed=False, closed_numerical_optimality_certificate=False,
            exact_optimal=False)
        self.last_forward_diagnostics.append(diagnostic)
        return scored['x'], SimpleNamespace(report={'objective': scored['objective']},
            certified_lower_bound=bound, optimal=False)

    def _solve_uncached(self, stage, node, cut_lag, parent, deadline=None):
        started = time.monotonic()
        sub_time_limit = self.stage2_sub_time_limit if stage == 2 else self.sub_time_limit
        native_result = None
        s2_native_attempt = None
        try:
            bounded_solve_time(sub_time_limit, deadline)
        except SolveDeadlineReached:
            return self._deadline_result(stage, node)
        if stage == 2 and self.phase == 1:
            ctx = _node_context(self.instance, node, stage=2)
            plan = _phase1_large_auto_probe_plan(self.phase, ctx)
            if plan is not None:
                return self._solve_phase1_feasible_probe(node, cut_lag, parent, plan, deadline)
        if stage == 2 and configured_forward_s2_backend(self.instance.shape[1]) == 'bpc':
            from solvers.forward_stage2_bpc import (solve_s2_forward_with_bpc,
                _independent_reference_lower_bound)
            policy = forward_s2_bpc_policy(self.phase)
            try:
                limit = bounded_solve_time(min(sub_time_limit, policy['time_limit_s']), deadline)
            except SolveDeadlineReached:
                return self._deadline_result(stage, node)
            result = solve_s2_forward_with_bpc(self.prob_data, node, cut_lag, parent,
                time_limit_s=limit, phase=self.phase, deadline=deadline,
                forward_gap=policy['forward_gap'], stage_builder=self.stage_builder)
            s2_native_attempt = dict(result['diagnostic'], requested_time_limit=limit,
                                    requested_forward_gap=policy['forward_gap'])
            if s2_native_attempt.get('native_executed'):
                self.solve_counts['stage2_native'] = self.solve_counts.get('stage2_native', 0) + 1
            if result['ok']:
                reference = _independent_reference_lower_bound(result.get('lower_bound'),
                    s2_native_attempt.get('reference_lp', {}), result['objective'])
                self.solve_counts['stage2'] += 1
                self.last_forward_diagnostics.append(dict(s2_native_attempt,
                    stage=2, node=node.index, status='audited_native_primal',
                    optimization_completed=bool(s2_native_attempt.get('native_executed')),
                    gurobi_executed=False, objective=result['objective'],
                    certified_lower_bound=reference))
                # This is the independently verified fixed-A full-LP reference,
                # never a native bound or an integer optimality claim.
                return result['x'], SimpleNamespace(report={'objective': result['objective']},
                    certified_lower_bound=reference, optimal=False)
            s2_native_attempt['fallback_reason'] = result['reason']
            if time.monotonic() - started >= sub_time_limit:
                stopped = self._deadline_result(stage, node)
                self.last_forward_diagnostics[-1].update(status='s2_budget_exhausted',
                                                        native_attempt=s2_native_attempt)
                return stopped
        if stage == 3 and os.environ.get('LRP_S3_BACKEND', 'native') == 'native':
            from solvers.lrp_native_oracle import LRPNativeRouteOracle, NativeUnavailable
            try:
                key = getattr(node, 'index', id(node))
                oracle = self._native_routes.get(key)
                if oracle is None:
                    oracle = LRPNativeRouteOracle(self.prob_data, node)
                    self._native_routes[key] = oracle
                i, ctx = oracle.facility, oracle.context
                result = oracle.solve_fixed([parent[f'alpha[{i},{j}]'] for j in range(ctx.n)],
                                            parent[f'u[{i}]'], time_limit=bounded_solve_time(min(_fixed_route_native_time_limit(), sub_time_limit), deadline),
                                            deadline=deadline)
            except SolveDeadlineReached:
                return self._deadline_result(stage, node)
            except NativeUnavailable:
                # An absent/stale binary never silently enters the certified
                # chain; the verified Gurobi model remains available.
                pass
            else:
                self.solve_counts['stage3_native'] = self.solve_counts.get('stage3_native', 0)+1
                record = dict(stage=3, node=getattr(node, 'index', None),
                    objective=result['objective'], certified_lower_bound=result['lower_bound'],
                    **result['diagnostic'])
                narrow = (result['diagnostic'].get('integer_optimality_proven', False)
                          and result['objective']-result['lower_bound'] <= 1e-7*max(1., abs(result['objective'])))
                if result['exact'] or narrow:
                    self.solve_counts['stage3'] += 1
                    self.last_forward_diagnostics.append(record)
                    return result['x'], SimpleNamespace(
                        report={'objective':result['objective']},
                        certified_lower_bound=result['lower_bound'], optimal=result['exact'])
                native_result = result
        try:
            bounded_solve_time(sub_time_limit, deadline)
        except SolveDeadlineReached:
            stopped = self._deadline_result(stage, node, native_result)
            if s2_native_attempt is not None:
                self.last_forward_diagnostics[-1]['native_attempt'] = s2_native_attempt
            return stopped
        model = self.stage_builder.build_stage_problem(stage, node, cut_lag, parent)
        try:
            limit = (sub_time_limit if native_result is None and s2_native_attempt is None
                     else max(1e-6, sub_time_limit-(time.monotonic()-started)))
            limit = bounded_solve_time(limit, deadline)
            abs_gap = 1e-8
            refresh_hint = (self._refresh_previous_hint
                            if stage == 2 and self.phase == 2 else None)
            warm_start_columns = 0
            if refresh_hint is not None:
                # Only the audited current-A/current-archive vector is seeded.
                # No old theta, objective, route_base, or lower bound survives.
                start = refresh_hint['x']
                for variable in model.getVars():
                    if variable.VarName in start:
                        variable.Start = float(start[variable.VarName])
                        warm_start_columns += 1
            if stage == 2:
                ctx = _node_context(self.instance, node, stage=2)
                reference = math.fsum(float(ctx.outsourcing[j]) * int(ctx.active[j]) for j in range(ctx.n))
                if refresh_hint is not None:
                    reference = refresh_hint['objective']
                scheduled = effective_abs_gap(self._s2_abs_tol, reference, self._s2_rel_cap)
                abs_gap = max(abs_gap, scheduled or 0.)
            solve_options = dict(time_limit=limit, deadline=deadline,
                mip_gap=(min(self.stage_builder.mip_gap, 1e-4) if stage == 1 else
                         self.stage2_mip_gap if stage == 2 else self.stage_builder.mip_gap),
                mip_abs_gap=abs_gap, threads=configured_gurobi_threads(), output=False,
                optimize_context=lambda: backend_call("gurobi", "optimize", model=model,
                    phase=self.phase, path="forward", stage=stage))
            refresh_retry = None
            try:
                result = evaluate_model(model, **solve_options)
            except InvalidSolverPrimal as original_rejection:
                if stage != 2 or self.phase != 2:
                    raise
                # The same S2 matrix rejection can occur in ordinary forward
                # as in refresh. Retry once without reading its failed bound;
                # old-policy recovery still requires a separately audited hint.
                refresh_retry = dict(original_rejection_type=type(original_rejection).__name__,
                    original_rejection_reason=str(original_rejection), retry_count=0,
                    retry_reset_performed=False, failed_attempts_counted=True)
                def count_rejection(exc):
                    self.solve_counts['stage2'] += 1
                    key = 'stage2_primal_rejections'
                    self.solve_counts[key] = self.solve_counts.get(key, 0) + 1
                    exc._lrp_refresh_retry = refresh_retry
                count_rejection(original_rejection)
                # In particular, deadline=None must not grant a second full
                # sub_time_limit. Include build/audit time already spent here.
                retry_deadline = started + sub_time_limit
                if deadline is not None:
                    retry_deadline = min(retry_deadline, deadline)
                try:
                    remaining = bounded_solve_time(sub_time_limit, retry_deadline)
                    model.reset()
                    model.Params.Presolve = 1
                    refresh_retry['retry_reset_performed'] = True
                    remaining = bounded_solve_time(remaining, retry_deadline)
                    refresh_retry.update(retry_requested_seconds=remaining,
                                         retry_deadline=retry_deadline)
                    result = evaluate_model(model, **dict(solve_options,
                        time_limit=remaining, deadline=retry_deadline))
                except SolveDeadlineReached:
                    refresh_retry['retry_status'] = 'deadline_before_retry_optimize'
                    raise original_rejection
                except InvalidSolverPrimal as retry_rejection:
                    refresh_retry.update(retry_count=1, retry_status='invalid_solver_primal',
                                         retry_rejection_reason=str(retry_rejection))
                    self.solve_counts['stage2_primal_retries'] = self.solve_counts.get('stage2_primal_retries', 0) + 1
                    count_rejection(retry_rejection)
                    raise
                else:
                    refresh_retry.update(retry_count=1, retry_status=result.report.get('status'))
                    self.solve_counts['stage2_primal_retries'] = self.solve_counts.get('stage2_primal_retries', 0) + 1
            self.solve_counts[f"stage{stage}"] += 1
            record = dict(stage=stage, node=getattr(node, 'index', getattr(node, 'ind', None)),
                          **result.summary(), requested_time_limit=float(model.Params.TimeLimit),
                          requested_mip_gap=float(model.Params.MIPGap))
            self.last_forward_diagnostics.append(record)
            if refresh_retry is not None:
                record['refresh_primal_retry'] = dict(refresh_retry)
            if refresh_hint is not None:
                record.update(refresh_previous_warm_start_columns=warm_start_columns,
                              scheduled_gap_reference=reference,
                              scheduled_gap_reference_source='audited_current_archive_policy')
            if s2_native_attempt is not None:
                record['native_attempt'] = s2_native_attempt
            if native_result is not None:
                record['native_attempt'] = native_result['diagnostic']
                if result.x is None or native_result['objective'] < result.report['objective']:
                    bounds = [native_result['lower_bound']]
                    if result.certified_lower_bound is not None:
                        bounds.append(result.certified_lower_bound)
                    return native_result['x'], SimpleNamespace(
                        report={'objective':native_result['objective']},
                        certified_lower_bound=max(bounds), optimal=native_result['exact'])
            if result.x is None:
                return None, result
            spec = result.problem.linear
            values = {name: (float(round(value)) if spec.integer[col] else float(value))
                      for col, (name, value) in enumerate(zip(spec.names, result.x))}
            return values, result
        except SolveDeadlineReached:
            stopped = self._deadline_result(stage, node, native_result)
            if s2_native_attempt is not None:
                self.last_forward_diagnostics[-1]['native_attempt'] = s2_native_attempt
            return stopped
        finally:
            model.dispose()

    @staticmethod
    def _envelope(cuts, state):
        return max([0.] + [float(v) + math.fsum(float(c)*state[k] for k,c in pi.items())
                          for pi,v in cuts])

    def _root_fallback(self, root_node, cut_lag):
        m, _, _, L, _ = self.instance.shape
        raw = {f'A[{i},{k}]': 1. for i in range(m) for k in range(L)}
        values, cost = certify_stage1_forward_policy(self.prob_data, raw)
        values['stage_cost'] = cost
        for second_ind in root_node.successor:
            values[f'eta[{second_ind}]'] = self._envelope(cut_lag.get(2, {}).get(second_ind, ()), values)
        objective = cost + math.fsum(float(self.instance.arrays['scenario_prob'][second_ind // self.instance.shape[2]])
                        * values[f'eta[{second_ind}]'] for second_ind in root_node.successor)
        return values, objective

    def _outsourcing_fallback(self, node, root, cut_lag):
        ctx = _node_context(self.instance, node, stage=2)
        values = {f'alpha[{i},{j}]': 0. for i in range(ctx.m) for j in range(ctx.n)}
        values.update({f'u[{i}]': 0. for i in range(ctx.m)})
        values.update({f'e[{j}]': float(ctx.active[j]) for j in range(ctx.n)})
        values, cost = certify_stage2_forward_policy(self.prob_data, node, root, values)
        values['stage_cost'] = cost
        values.update({f'z[{i}]': root[f'A[{i},{ctx.interval}]'] for i in range(ctx.m)})
        for third_ind in node.successor:
            values[f'theta[{third_ind}]'] = self._envelope(cut_lag.get(3, {}).get(third_ind, ()), values)
        objective = cost + math.fsum(values[f'theta[{third_ind}]'] for third_ind in node.successor)
        return values, objective

    def _forward_recourse_node(self, node, third_nodes, cut_lag, root_values, deadline=None, *, stage2_result=None):
        """One independent scenario-period, including its whole-node fallback."""
        second_ind = node.index
        x_star, cost_star = {1: {0: root_values}, 2: {}, 3: {}}, {2: {}, 3: {}}
        values, result = (stage2_result if stage2_result is not None else
                          self._solve_model(2, node, cut_lag, x_star[1][0], deadline=deadline))
        if values is None:
            values, objective = self._outsourcing_fallback(node, x_star[1][0], cut_lag)
            self.last_forward_diagnostics[-1]['fallback'] = 'all_outsourcing'
        else:
            try:
                normalized, cost = certify_stage2_forward_policy(self.prob_data, node, x_star[1][0], values)
            except InvalidForwardPolicy:
                # FeasibilityTol may allow a tiny actual capacity overload;
                # such an incumbent is unusable as a policy upper bound.
                values, objective = self._outsourcing_fallback(node, x_star[1][0], cut_lag)
                self.last_forward_diagnostics[-1]['fallback'] = 'all_outsourcing_after_exact_capacity_audit'
            else:
                values.update(normalized)
                values['stage_cost'] = cost
                objective = float(result.report['objective'])
        x_star[2][second_ind], cost_star[2][second_ind] = values, objective
        failed_route = False
        for third_ind in node.successor:
            third = third_nodes[third_ind]
            route_values, route_result = self._solve_model(3, third, cut_lag, values, deadline=deadline)
            if route_values is None:
                self.last_forward_diagnostics[-1]['fallback'] = 'whole_node_all_outsourcing'
                failed_route = True
                break
            route, cost = certify_stage3_forward_policy(self.prob_data, third, values, route_values)
            route_values.update(route)
            route_values['stage_cost'] = cost
            x_star[3][third_ind], cost_star[3][third_ind] = route_values, cost
        if failed_route:
            # Replacing the whole node also changes the assignment passed
            # backward, so the recorded policy and the cut anchor agree.
            values, objective = self._outsourcing_fallback(node, x_star[1][0], cut_lag)
            x_star[2][second_ind], cost_star[2][second_ind] = values, objective
            for third_ind in node.successor:
                third = third_nodes[third_ind]
                route, cost = certify_stage3_forward_policy(self.prob_data, third, values, {})
                route['stage_cost'] = cost
                x_star[3][third_ind], cost_star[3][third_ind] = route, cost

        return {'node': second_ind, 'x2': x_star[2][second_ind],
                'cost2': cost_star[2][second_ind], 'x3': x_star[3], 'cost3': cost_star[3]}


    @_within_period_cache
    def forward_pass(self, scen_tree, cut_lag, num_processes=1, pool=None, deadline=None):
        self.last_forward_diagnostics = []
        self.last_policy_certificate = None
        x_star, cost_star = {1: {}, 2: {}, 3: {}}, {1: {}, 2: {}, 3: {}}
        root = scen_tree[1][0]
        values, result = self._solve_model(1, root, cut_lag, {}, deadline=deadline)
        bound = result.certified_lower_bound
        lb = max(0., bound) if bound is not None else 0.
        if values is None:
            values, objective = self._root_fallback(root, cut_lag)
            self.last_forward_diagnostics[-1]['fallback'] = 'all_facilities_available'
        else:
            normalized, cost = certify_stage1_forward_policy(self.prob_data, values)
            values.update(normalized)
            values['stage_cost'] = cost
            objective = float(result.report['objective'])
        x_star[1][0], cost_star[1][0] = values, objective
        self._last_eta_per_omega = {q: values[f'eta[{q}]'] for q in root.successor}
        root_callback = getattr(self, 'record_root_certificate', None)
        if root_callback is not None:
            # Publish only the actual solver certificate, never lb's fallback 0.
            # The root plan has already passed the original policy audit.
            root_callback(bound, root=values)

        worker_pids = []
        if int(num_processes) > 1 and len(root.successor) > 1:
            from solvers.lrp_parallel import solve_forward_nodes
            packets = solve_forward_nodes(self, scen_tree, cut_lag, x_star[1][0],
                                          num_processes, pool=pool, deadline=deadline)
        else:
            packets = (self._forward_recourse_node(scen_tree[2][q],
                        {r: scen_tree[3][r] for r in scen_tree[2][q].successor},
                        cut_lag, x_star[1][0], deadline=deadline) for q in root.successor)
        for packet in packets:
            second_ind = packet['node']
            x_star[2][second_ind], cost_star[2][second_ind] = packet['x2'], packet['cost2']
            x_star[3].update(packet['x3'])
            cost_star[3].update(packet['cost3'])
            if packet.get('worker_pid') is not None:
                worker_pids.append(packet['worker_pid'])

        certificate = certify_lrp_forward_policy(self.prob_data, x_star, scen_tree)
        certificate['period_dedup'] = dict(enabled=self.period_dedup, **self.period_dedup_stats)
        certificate['stage2_handoff'] = dict(hits=self._stage2_handoff_pass.hits,
                                            rejected=self._stage2_handoff_pass.rejected)
        certificate['requested_processes'] = int(num_processes)
        certificate['worker_pids'] = sorted(set(worker_pids))
        certificate['effective_processes'] = len(set(worker_pids)) if worker_pids else 1
        certificate['execution'] = ('parallel scenario-period nodes' if worker_pids else 'serial nodes')
        certificate['stage2_solve_tolerance'] = {'mip_gap': self.stage2_mip_gap,
            'abs_tol': self._s2_abs_tol, 'rel_cap': self._s2_rel_cap}
        certificate['fallbacks'] = [record for record in self.last_forward_diagnostics if 'fallback' in record]
        self.last_policy_certificate = certificate
        ub = certificate['feasible_upper_bound']
        if lb > ub + 2e-6 + 1e-10 * max(1., abs(ub)):
            raise RuntimeError(f'LRP forward certified lower bound {lb} exceeds audited policy cost {ub}')
        return lb, x_star, ub, cost_star


def solve_stage2_node(prob_data, node, cut_lag, x_prev, sub_time_limit=60.0, stage_builder=None, *,
                      s2_abs_tol=None, s2_rel_cap=None, deadline=None, previous_policy=None):
    """Refresh this assignment envelope under an unchanged shared facility plan."""
    solver = ForwardSolver(prob_data, stage_builder, sub_time_limit=sub_time_limit)
    solver.set_stage2_tolerance(s2_abs_tol, s2_rel_cap)
    previous, hint_rejection = None, None
    cuts = successors = ctx = None
    if previous_policy is not None:
        from cuts.lrp_static_bounds import basic_route_cuts
        from models.stage_builder import _as_cut, _route_pools
        from solvers.forward_stage2_bpc import _score
        ctx = _node_context(solver.instance, node, stage=2)
        # Validate physical reuse before interpreting stale epigraph fields.
        # Domain/malformed-cut errors still propagate instead of being hidden.
        try:
            certify_stage2_forward_policy(prob_data, node, x_prev, previous_policy)
        except InvalidForwardPolicy as exc:
            hint_rejection = str(exc)
        else:
            pools, successors = _route_pools(ctx, node, cut_lag)
            cuts = {i: basic_route_cuts(ctx, i) + [_as_cut(row, ctx, i) for row in pools[i]]
                    for i in range(ctx.m)}
            previous = _score(prob_data, node, ctx, x_prev, previous_policy,
                              cuts, successors, {'source': 'previous_policy'})
            solver._refresh_previous_hint = previous
    try:
        values, result = solver._solve_model(2, node, cut_lag, x_prev, deadline=deadline)
    except InvalidSolverPrimal as exc:
        if previous is None:
            raise
        # Only a separately audited current-A policy permits this recovery.
        # Its theta/objective were rebuilt above from the complete archive.
        # The failed model has already been disposed by _solve_uncached;
        # none of its incumbent, objective, bound or optimality is retained.
        retry = getattr(exc, '_lrp_refresh_retry', None)
        if retry is None:
            solver.solve_counts['stage2'] += 1
            key = 'stage2_primal_rejections'
            solver.solve_counts[key] = solver.solve_counts.get(key, 0) + 1
        solver.last_forward_diagnostics.append(dict(
            stage=2, node=node.index, status='invalid_solver_primal',
            optimization_completed=True, gurobi_executed=True,
            solver_primal_rejected=True, rejection_type=type(exc).__name__,
            rejection_reason=str(exc), solver_bound_used=False,
            objective=None, certified_lower_bound=None,
            optimality_claimed=False, closed_numerical_optimality_certificate=False,
            recovery='audited_previous_policy'))
        if retry is not None:
            solver.last_forward_diagnostics[-1]['refresh_primal_retry'] = dict(retry)
        values = None
        result = SimpleNamespace(report={'objective': None},
                                 certified_lower_bound=None, optimal=False)
    if previous is not None:
        # Even after deadline, an audited old physical policy can be rescored
        # and returned without a model or optimizer; it is never a new LB.
        fresh, fresh_rejection = None, None
        if values is not None:
            try:
                fresh = _score(prob_data, node, ctx, x_prev, values,
                               cuts, successors, {'source': 'fresh_policy'})
            except InvalidForwardPolicy as exc:
                fresh_rejection = str(exc)
        choose_previous = fresh is None or previous['objective'] < fresh['objective']
        selected = previous if choose_previous else fresh
        diagnostic = solver.last_forward_diagnostics[-1]
        diagnostic.update(previous_policy_audited=True,
            previous_policy_current_archive_objective=previous['objective'],
            selected_policy_source='previous_audited_policy' if choose_previous else 'fresh_audited_policy',
            selected_policy_objective=selected['objective'],
            complete_archive_rescored=True, policy_certified=True)
        if fresh_rejection is not None:
            diagnostic['fresh_policy_rejection'] = fresh_rejection
        if choose_previous:
            diagnostic['fallback'] = ('previous_policy_after_no_valid_incumbent'
                                      if fresh is None else 'previous_policy_better_than_fresh')
        bound = result.certified_lower_bound
        if bound is not None and (not math.isfinite(bound) or bound > selected['objective']):
            diagnostic['rejected_lower_bound_above_selected_policy'] = bound
            bound = None
        # Leave the raw solver report/matrix audit intact. Its incumbent may
        # differ from the independently selected returned physical policy.
        diagnostic['selected_policy_lower_bound'] = bound
        from s2forward.refresh_handoff import _lrp_closed_interval
        exact = bool(not choose_previous and result.optimal and bound is not None
                     and _lrp_closed_interval(selected['objective'], bound))
        return {'x': selected['x'], 'objective': selected['objective'],
                'stage_cost': selected['stage_cost'], 'lower_bound': bound,
                'diagnostic': diagnostic,
                'fresh_solve': bool(not choose_previous and diagnostic.get('optimization_completed')),
                'exact_optimal': exact,
                'solve_time_limit': diagnostic.get('requested_time_limit')}
    if hint_rejection is not None:
        solver.last_forward_diagnostics[-1]['previous_policy_rejection'] = hint_rejection
    fallback = values is None
    if not fallback:
        try:
            normalized, stage_cost = certify_stage2_forward_policy(prob_data, node, x_prev, values)
        except InvalidForwardPolicy:
            fallback = True
    if fallback:
        values, objective = solver._outsourcing_fallback(node, x_prev, cut_lag)
        stage_cost = values['stage_cost']
        solver.last_forward_diagnostics[-1]['fallback'] = 'all_outsourcing'
    else:
        values.update(normalized)
        values['stage_cost'] = stage_cost
        objective = float(result.report['objective'])
    return {'x': values, 'objective': objective, 'stage_cost': stage_cost,
            'lower_bound': result.certified_lower_bound,
            'diagnostic': solver.last_forward_diagnostics[-1],
            'fresh_solve': not fallback and bool(solver.last_forward_diagnostics[-1].get('optimization_completed')),
            'exact_optimal': bool(result.optimal),
            'solve_time_limit': solver.last_forward_diagnostics[-1].get('requested_time_limit')}


def solve_stage3_node(prob_data, node, cut_lag, x_prev, sub_time_limit=60.0, stage_builder=None, *, deadline=None):
    """Solve a real facility route for a fixed assignment, never a route envelope."""
    solver = ForwardSolver(prob_data, stage_builder, sub_time_limit=sub_time_limit)
    values, result = solver._solve_model(3, node, cut_lag, x_prev, deadline=deadline)
    cost = None
    if values is not None:
        normalized, cost = certify_stage3_forward_policy(prob_data, node, x_prev, values)
        values.update(normalized)
        values['stage_cost'] = cost
    return {'x': values, 'objective': cost, 'stage_cost': cost,
            'lower_bound': result.certified_lower_bound,
            'diagnostic': solver.last_forward_diagnostics[-1]}
