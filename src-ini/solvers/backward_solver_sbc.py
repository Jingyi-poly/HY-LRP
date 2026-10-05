"""LRP strengthened Benders pass: fixed-state LP dual -> free-state MIP."""
from __future__ import annotations

import math
import os
import time

import gurobipy as gp

from core.solver_settings import (configured_gurobi_threads, configured_native_probe_seconds,
                                 configured_backward_s3_backend, phase1_s2_oracle_policy,
                                 configured_s3_native_options)
from core.backend_telemetry import backend_call
from core.solve_deadline import SolveDeadlineReached, bounded_solve_time
from cuts.benders_cuts import add_unique_cut, clean_pi
from core.solver_bounds import certified_gurobi_minimization_lower_bound, extract_verified_fixed_rhs_dual_cut
from models.stage_builder import StageModelBuilder
from models.subproblem_builder import SubproblemBuilder
from models.stage_model_core import InvalidSolverPrimal, cut_from_backward, evaluate_model
from solvers.route_lp_separation import separate_route_lp_dfj


def _grb_certified_lb(model):
    return certified_gurobi_minimization_lower_bound(model)


class BackwardSolverSBC:
    """Preserve SBC's LP multipliers and optional integer-oracle strengthening.

    Cuts stay in unweighted node cost units. Only the root weights S2 values by
    scenario probability. No cut is copied between different node contexts.
    """
    phase = 2

    def __init__(self, prob_data, stage_builder=None, subproblem_builder=None,
                 strengthen_s2=True, strengthen=False, sub_time_limit=60.0,
                 route_lp_separation_time_limit=None):
        if math.isnan(float(sub_time_limit)) or sub_time_limit <= 0:
            raise ValueError('sub_time_limit must be positive; +inf means no per-solve cap')
        self.prob_data = prob_data
        self.stage_builder = stage_builder or StageModelBuilder(prob_data)
        self.subproblem_builder = subproblem_builder or SubproblemBuilder(
            prob_data, lazy_threshold=self.stage_builder.lazy_threshold,
            env=self.stage_builder.env)
        self.strengthen_s2 = bool(strengthen_s2)
        self.strengthen = bool(strengthen)
        self.sub_time_limit = float(sub_time_limit)
        self.route_lp_separation_time_limit = (0.0
            if route_lp_separation_time_limit is None else float(route_lp_separation_time_limit))
        if not math.isfinite(self.route_lp_separation_time_limit) or self.route_lp_separation_time_limit < 0:
            raise ValueError('route_lp_separation_time_limit must be finite and nonnegative')
        self.last_cut_diagnostics = []
        self.solve_counts = {"stage2_lp": 0, "stage2_mip": 0, "stage3_lp": 0, "stage3_mip": 0,
                             "zero_route_skips": 0}
        self.s2_piece_tables = {}
        self._native_routes = {}

    def _solve_s2_gurobi(self, node, cut_lag, pi, limit, deadline, *, probe=False):
        """Only certified ObjBound endpoints strengthen the fixed-LP cut."""
        bounded_solve_time(limit, deadline)
        free = self.subproblem_builder.build_subproblem(2, node, cut_lag, pi)
        try:
            result = evaluate_model(free, time_limit=limit, deadline=deadline,
                mip_gap=0.0 if probe else self.stage_builder.mip_gap,
                mip_abs_gap=0.0 if probe else 1e-8,
                threads=configured_gurobi_threads(), output=False,
                optimize_context=lambda: backend_call('gurobi', 'optimize', model=free,
                    phase=self.phase, path='backward', stage=2,
                    kind='mip_probe' if probe else 'free_state_mip'))
            self.solve_counts['stage2_mip'] += 1
            certificate = (cut_from_backward(result).to_dict()
                           if result.certified_lower_bound is not None else None)
            return result, certificate
        finally:
            free.dispose()

    def _strengthen_s2_phase1(self, node, cut_lag, pi, intercept, record, deadline):
        """LP + short Gurobi probe + optional native root, as in Investment.

        A full integer solve is reserved for an explicit Gurobi selection.
        All strengthening candidates share one remaining node allowance;
        a failed/timed-out optional oracle cannot discard the valid LP cut.
        """
        policy = phase1_s2_oracle_policy(self.prob_data, node)
        backend = policy['backend']
        record['configured_s2_backend'] = backend
        record['s2_oracle_policy'] = policy
        record['oracle_candidates'] = {'fixed_rhs_lp': intercept}
        oracle_deadline = time.monotonic() + self.sub_time_limit
        if deadline is not None:
            oracle_deadline = min(oracle_deadline, deadline)
        try:
            if backend == 'gurobi' or (backend == 'auto' and policy['gurobi_probe_seconds'] > 0):
                probe = backend == 'auto'
                limit = policy['gurobi_probe_seconds'] if probe else self.sub_time_limit
                bounded_solve_time(limit, oracle_deadline)
                result, certificate = self._solve_s2_gurobi(
                    node, cut_lag, pi, limit, oracle_deadline, probe=probe)
                record['oracle'] = result.summary()
                record['gurobi_oracle_time_limit'] = limit
                record['oracle_candidates']['gurobi_probe' if probe else 'gurobi'] = result.certified_lower_bound
                if certificate is not None:
                    intercept = max(intercept, result.certified_lower_bound)
                    record['oracle_certificate'] = certificate
                if backend == 'gurobi' or result.optimal:
                    record['oracle_stop_reason'] = ('explicit_gurobi' if backend == 'gurobi'
                                                    else 'probe_numerically_closed')
                    return intercept
            reason = policy['bpc_skip_reason'] if backend == 'auto' else (
                None if policy['bpc_enabled'] else 'phase1_s2_bpc_disabled')
            if reason is not None:
                record['native_skip_reason'] = reason
                return intercept
            limit = policy['bpc_seconds'] if backend == 'auto' else self.sub_time_limit
            if not math.isfinite(limit):
                # The native API is finite-budget only. Preserve its original
                # backend cap when an explicit Python node budget is unlimited.
                limit = float(os.environ.get('LRP_S2_BP_TIME_LIMIT_S',
                              os.environ.get('VRP_S2_BP_TIME_LIMIT_S', '1800')))
                if not math.isfinite(limit) or limit <= 0:
                    raise ValueError('S2_BP_TIME_LIMIT_S must be positive and finite')
            if limit <= 0:
                record['native_skip_reason'] = 'bpc_budget_zero'
                return intercept
            limit = bounded_solve_time(limit, oracle_deadline)
            from solvers.lrp_backward_s2_bpc import solve_s2_backward_with_bpc
            native = solve_s2_backward_with_bpc(self.prob_data, node, cut_lag, pi,
                time_limit_s=limit, deadline=oracle_deadline if math.isfinite(oracle_deadline) else None,
                root_bound_only=True, phase=1)
            self.solve_counts['stage2_native'] = self.solve_counts.get('stage2_native', 0) + 1
            record['native_oracle'] = native
            record['native_probe_time_limit'] = limit
            lower = native.get('outer_lb')
            valid = (native.get('lb_certified') is True and
                     isinstance(lower, (int, float)) and not isinstance(lower, bool) and
                     math.isfinite(lower) and abs(lower) < 1e100)
            record['oracle_candidates']['bpc'] = lower if valid else None
            if valid:
                intercept = max(intercept, float(lower))
            record['oracle_stop_reason'] = 'optional_native_root_completed'
        except SolveDeadlineReached:
            record['oracle_stop_reason'] = 'shared_deadline_exhausted_keep_lp'
        return intercept

    def _strengthen_route_cut(self, node, cut_lag, pi, intercept, record, deadline):
        """Keep the LP slope; merge original-domain certified oracle bounds.

        Native and Gurobi share one integer-strengthening allowance. The fixed
        LP has its original separate allowance. Expiry never discards an
        already certified LP or native lower bound.
        """
        oracle_deadline = time.monotonic() + self.sub_time_limit
        if deadline is not None:
            oracle_deadline = min(oracle_deadline, deadline)
        route_backend = configured_backward_s3_backend(self.phase)
        record['configured_route_backend'] = route_backend
        if self.phase == 1 and -1e-6 <= intercept <= 1e-6:
            # The free domain contains the empty route, with value zero.
            # Keep the original signed LP intercept; never round it to zero.
            record['oracle_stop_reason'] = 'lp_zero_band'
            return intercept
        if route_backend == 'native':
            from solvers.lrp_native_oracle import LRPNativeRouteOracle, NativeUnavailable
            probe_seconds = self.sub_time_limit if self.phase == 1 else configured_native_probe_seconds()
            native_options = configured_s3_native_options()
            try:
                bounded_solve_time(probe_seconds, oracle_deadline)
                key = (record['context'], int(node.info))
                oracle = self._native_routes.get(key)
                if oracle is None:
                    oracle = LRPNativeRouteOracle(self.prob_data, node)
                    self._native_routes[key] = oracle
                limit = bounded_solve_time(probe_seconds, oracle_deadline)
                before = oracle.solve_count
                try:
                    native = oracle.solve(pi, time_limit=limit, deadline=oracle_deadline,
                                          **native_options)
                finally:
                    calls = oracle.solve_count - before
                    if calls:
                        self.solve_counts['stage3_native'] = self.solve_counts.get('stage3_native', 0) + calls
                record['native_oracle'] = native
                record['native_probe_time_limit'] = limit
                lower = native.get('outer_lb')
                upper = native.get('inner_value')
                valid_lower = (isinstance(lower, (int, float)) and not isinstance(lower, bool)
                    and math.isfinite(lower) and abs(lower) < 1e100 and lower <= 0.)
                valid_upper = (isinstance(upper, (int, float)) and not isinstance(upper, bool)
                    and math.isfinite(upper) and abs(upper) < 1e100)
                # Empty is feasible. A native incumbent is an upper support,
                # never an intercept; reject any inverted lower endpoint.
                if valid_lower and (not valid_upper or lower <= upper):
                    intercept = max(intercept, float(lower))
                    if (native.get('exact') is True
                            and native.get('incumbent_policy_certified') is True
                            and valid_upper
                            and upper - lower <= 1e-7 + 1e-12 * max(1., abs(upper), abs(lower))):
                        record['oracle_stop_reason'] = 'native_exact'
                        return intercept
            except (NativeUnavailable, ValueError) as exc:
                record['native_fallback_reason'] = str(exc)
            except SolveDeadlineReached:
                record['oracle_stop_reason'] = 'shared_deadline_exhausted'
                return intercept
            if self.phase == 1:
                record['oracle_stop_reason'] = 'native_bound_or_lp_retained'
                return intercept
        free = None
        try:
            bounded_solve_time(self.sub_time_limit, oracle_deadline)
            free = self.subproblem_builder.build_subproblem(3, node, cut_lag, pi)
            limit = bounded_solve_time(self.sub_time_limit, oracle_deadline)
            result = evaluate_model(free, time_limit=limit, deadline=oracle_deadline,
                mip_gap=0.0, threads=configured_gurobi_threads(), output=False,
                optimize_context=lambda: backend_call('gurobi', 'optimize', model=free,
                    phase=self.phase, path='backward', stage=3, kind='free_state_mip'))
            self.solve_counts['stage3_mip'] += 1
            record['oracle'] = result.summary()
            record['gurobi_oracle_time_limit'] = limit
            if result.certified_lower_bound is not None:
                certificate = cut_from_backward(result)
                intercept = max(intercept, certificate.intercept)
                record['oracle_certificate'] = certificate.to_dict()
        except SolveDeadlineReached:
            record['oracle_stop_reason'] = 'shared_deadline_exhausted'
        finally:
            if free is not None:
                free.dispose()
        return intercept

    def _generate_cut(self, stage, node, x_prev, cut_lag, node_ind, deadline=None):
        # No model build or optimizer starts after the shared deadline.
        if deadline is not None and time.monotonic() >= deadline:
            return None
        options = {'learned_cut_purpose': 'dual_lp'} if stage == 2 else {'reduce_nodes': False}
        if stage == 3 and hasattr(self.stage_builder, 'build_stage3_dual_problem'):
            build_dual = self.stage_builder.build_stage3_dual_problem
            capability = getattr(getattr(build_dual, '__func__', build_dual),
                                 '_lrp_accepts_route_dfj_reuse', False)
            reuse = {'reuse_route_dfj': True} if self.phase == 2 and capability is True else {}
            fixed = build_dual(node, cut_lag, x_prev, **reuse)
        else:
            fixed = self.stage_builder.build_stage_problem(stage, node, cut_lag, x_prev, **options)
        relaxation = None
        record = {'stage': stage, 'node': int(node_ind), 'phase': self.phase,
                  'context': fixed._lrp_spec.context.key,
                  'method': 'fixed_RHS_LP_then_optional_free_state_oracle',
                  'parent_state': {name: float(x_prev[name]) for name, _ in fixed._lrp_dual_bindings}}
        reuse_report = getattr(fixed._lrp_spec.linear, 'route_dfj_reuse', None)
        if reuse_report is not None:
            record['route_dfj_reuse'] = dict(reuse_report)
        try:
            bounded_solve_time(self.sub_time_limit, deadline)
            bindings = tuple(fixed._lrp_dual_bindings)
            fixed.update()
            relaxation = fixed.relax()
            relaxation.Params.OutputFlag = 0
            relaxation.Params.Threads = configured_gurobi_threads()
            relaxation.Params.TimeLimit = bounded_solve_time(self.sub_time_limit, deadline)
            relaxation.Params.FeasibilityTol = 1e-9
            relaxation.Params.OptimalityTol = 1e-9
            with backend_call("gurobi", "optimize", model=relaxation, phase=self.phase, path="backward", stage=stage, kind="fixed_rhs_lp"):
                relaxation.optimize()
            self.solve_counts[f"stage{stage}_lp"] += 1
            record['lp_status'] = int(relaxation.Status)
            record['lp_runtime'] = float(relaxation.Runtime)
            if relaxation.Status != gp.GRB.OPTIMAL:
                record['cut_added'] = False
                record['reason'] = 'fixed-state LP has no optimal dual certificate'
                return None
            label=f'LRP phase {self.phase} stage {stage} node {node_ind}'
            if stage == 3 and self.route_lp_separation_time_limit > 0:
                def optimize_separated_lp():
                    relaxation.Params.TimeLimit = bounded_solve_time(relaxation.Params.TimeLimit, deadline)
                    with backend_call("gurobi", "optimize", model=relaxation, phase=self.phase,
                                      path="backward", stage=3, kind="dfj_fixed_rhs_lp"):
                        relaxation.optimize()
                    self.solve_counts["stage3_lp"] += 1
                certificate, separation = separate_route_lp_dfj(
                    relaxation, fixed._lrp_spec, bindings,
                    time_limit=min(self.route_lp_separation_time_limit,
                        math.inf if deadline is None else max(0., deadline-time.monotonic()-.25)),
                    optimize=optimize_separated_lp, label=label)
                record['dfj_separation'] = separation
                if certificate is None:
                    record.update(cut_added=False, reason='DFJ route LP has no optimal dual certificate')
                    return None
                pi, intercept_lp = certificate['pi'], certificate['intercept']
                lp_objective = certificate['objective']
                record['dual_certificate_status'] = certificate['status']
                record['latest_lp_status'] = int(relaxation.Status)
            else:
                pi, intercept_lp = extract_verified_fixed_rhs_dual_cut(relaxation, bindings, label=label)
                lp_objective = float(relaxation.ObjVal)
            intercept = intercept_lp
            record.update(lp_objective=lp_objective, lp_intercept=intercept_lp,
                          slope=dict(pi), fixed_rhs_bindings=list(bindings))
            strengthen = self.strengthen_s2 if stage == 2 else self.strengthen
            try:
                if strengthen and stage == 3:
                    intercept = self._strengthen_route_cut(node, cut_lag, pi, intercept, record, deadline)
                elif strengthen and self.phase == 1:
                    intercept = self._strengthen_s2_phase1(node, cut_lag, pi, intercept, record, deadline)
                elif strengthen:
                    free = self.subproblem_builder.build_subproblem(stage, node, cut_lag, pi)
                    try:
                        bounded_solve_time(self.sub_time_limit, deadline)
                        with backend_call("gurobi", "optimize", model=free, phase=self.phase, path="backward", stage=stage, kind="free_state_mip"):
                            result = evaluate_model(free, time_limit=self.sub_time_limit, deadline=deadline,
                                                    mip_gap=0.0, threads=configured_gurobi_threads(), output=False)
                        self.solve_counts[f"stage{stage}_mip"] += 1
                        record['oracle'] = result.summary()
                        if result.certified_lower_bound is not None:
                            certificate = cut_from_backward(result)
                            intercept = max(intercept_lp, certificate.intercept)
                            record['oracle_certificate'] = certificate.to_dict()
                    finally:
                        free.dispose()
            except InvalidSolverPrimal as exc:
                # The verified LP cut already exists. This optional MIP
                # completed optimize but returned no accepted certificate;
                # discard every strengthening endpoint and keep only the LP.
                self.solve_counts[f"stage{stage}_mip"] += 1
                key = f"stage{stage}_mip_primal_rejections"
                self.solve_counts[key] = self.solve_counts.get(key, 0) + 1
                intercept = intercept_lp
                record.pop('oracle_certificate', None)
                record.update(oracle_stop_reason='invalid_solver_primal_keep_lp',
                              oracle_rejection_type=type(exc).__name__,
                              oracle_rejection_reason=str(exc),
                              oracle_bound_used=False,
                              strengthening_fallback='verified_fixed_rhs_lp')
            # Removing tiny slopes uses directed arithmetic on the full binary
            # parent box; a negative intercept is never clipped to zero.
            pi, intercept, removed, slack = clean_pi(pi, intercept)
            record.update(slope=dict(pi), intercept=intercept,
                          cleaned_coefficients=removed, cleaning_slack=slack,
                          cut_added=True)
            return pi, intercept
        except SolveDeadlineReached:
            record.update(cut_added=False, reason='shared_deadline_exhausted')
            return None
        finally:
            self.last_cut_diagnostics.append(record)
            if relaxation is not None:
                relaxation.dispose()
            fixed.dispose()

    def _generate_third_stage_cut(self, node, x_prev, cut_lag, third_ind, deadline=None):
        return self._generate_cut(3, node, x_prev, cut_lag, third_ind, deadline)

    def _generate_second_stage_cut(self, node, x_prev, cut_lag, second_ind, deadline=None):
        return self._generate_cut(2, node, x_prev, cut_lag, second_ind, deadline)

    @staticmethod
    def _zero_route_is_resolved(node_ind, parent, cost_star, cut_lag):
        """A trial with audited cost zero and envelope zero needs no separation.

        This only skips search at the current trial. It neither declares a
        node solved globally nor changes outer convergence or Stage-2 cuts.
        Require an explicitly recorded zero theta. An empty archive already
        has envelope zero because the physical model imposes theta >= 0.
        """
        actual=cost_star.get(3, {}).get(node_ind)
        theta=parent.get(f'theta[{node_ind}]')
        cuts=cut_lag.get(3, {}).get(node_ind, ())
        if actual is None or theta is None:
            return False
        if float(actual)!=0.0 or float(theta)!=0.0:
            return False
        envelope=0.0
        for pi,intercept in cuts:
            value=math.fsum([float(intercept)] + [float(c)*float(parent[k]) for k,c in pi.items()])
            if not math.isfinite(value):
                return False
            envelope=max(envelope,value)
        return envelope==0.0

    def backward_pass(self, scen_tree, x_star, cost_star, cut_lag, num_processes=1, pool=None, deadline=None):
        self.last_cut_diagnostics = []
        self.worker_pids = set()
        parallel = int(num_processes) > 1
        if parallel:
            from solvers.lrp_parallel import solve_sbc_layer
        third_jobs = []
        for stage in (2, 3):
            cut_lag.setdefault(stage, {})
        # Skip only audited zero-cost trials matching the current envelope,
        # including the initial theta >= 0 envelope. Every other route keeps
        # its fixed-LP/free-MIP SBC search; missing theta never permits a skip.
        for second_ind in scen_tree[1][0].successor:
            second = scen_tree[2][second_ind]
            for third_ind in second.successor:
                if self._zero_route_is_resolved(third_ind, x_star[2][second_ind], cost_star, cut_lag):
                    self.solve_counts['zero_route_skips'] += 1
                    self.last_cut_diagnostics.append({
                        'stage': 3, 'node': int(third_ind), 'phase': self.phase,
                        'context': scen_tree[3][third_ind].context.key,
                        'method': 'zero_route_search_skip', 'cut_added': False,
                        'archive_changed': False, 'reason': 'zero_cost_matches_zero_envelope',
                        'audited_route_cost': 0.0, 'theta_envelope': 0.0,
                        'global_convergence_claimed': False,
                    })
                    continue
                if parallel:
                    third_jobs.append((scen_tree[3][third_ind], x_star[2][second_ind]))
                    continue
                generated = self._generate_third_stage_cut(
                    scen_tree[3][third_ind], x_star[2][second_ind], cut_lag, third_ind, deadline=deadline)
                if generated is not None:
                    added = add_unique_cut(cut_lag[3].setdefault(third_ind, []), *generated)
                    self.last_cut_diagnostics[-1]['archive_changed'] = bool(added)
        if parallel:
            # All S3 rows are merged before any S2 job sees its child archive.
            solve_sbc_layer(self, 3, third_jobs, cut_lag, num_processes, pool=pool, deadline=deadline)
            second_jobs = [(scen_tree[2][q], x_star[1][0]) for q in scen_tree[1][0].successor]
            solve_sbc_layer(self, 2, second_jobs, cut_lag, num_processes, pool=pool, deadline=deadline)
            return cut_lag
        for second_ind in scen_tree[1][0].successor:
            generated = self._generate_second_stage_cut(
                scen_tree[2][second_ind], x_star[1][0], cut_lag, second_ind, deadline=deadline)
            if generated is not None:
                added = add_unique_cut(cut_lag[2].setdefault(second_ind, []), *generated)
                self.last_cut_diagnostics[-1]['archive_changed'] = bool(added)
        return cut_lag
