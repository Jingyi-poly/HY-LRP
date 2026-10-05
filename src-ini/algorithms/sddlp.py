"""Phase 2: the original forward / Level Set backward iteration for LRP.

SBC supplies the initial cuts; Level Set searches the Lagrangian multipliers.
The three computational stages share a two-stage information structure.
"""
from __future__ import annotations

import copy
from functools import wraps
from multiprocessing import get_context
import math
import time

from algorithms.base_algorithm import (
    SDDPAlgorithm, cut_archive_fingerprint, forward_values_are_finite,
    solver_diagnostics,
)
from core.backend_telemetry import capture_backend_run
from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent
from core.stage2_tolerance import Stage2ToleranceSchedule, stage2_weight_mass
from models.stage_builder import StageModelBuilder
from solvers.forward_solver import ForwardSolver
from solvers.lrp_forward_reuse import take_snapshot, solve_policy
from solvers.backward_solver_lag import BackwardSolverLagrangian
from solvers.forward_policy_certification import certify_policy

__all__ = ['SDDLP']



def _manage_phase2_pools(method):
    """Own the original independent forward/S2 and S3 pools for one solve."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._phase2_pool = self._phase2_s3_pool = None
        failed = True
        try:
            result = method(self, *args, **kwargs)
            failed = False
            return result
        finally:
            pools = (self._phase2_pool, self._phase2_s3_pool)
            self._phase2_pool = self._phase2_s3_pool = None
            error = None
            closed_ids = set()
            for pool in pools:
                if pool is None or id(pool) in closed_ids:
                    continue
                closed_ids.add(id(pool))
                try:
                    pool.terminate() if failed else pool.close()
                except BaseException as exc:
                    error = error or exc
                try:
                    pool.join()
                except BaseException as exc:
                    error = error or exc
            if error is not None and not failed:
                raise error
    return wrapped


class SDDLP(SDDPAlgorithm):
    def __init__(self, prob_data, scen_tree, config, cut_lag_init=None,
                 ub_init=None, lb_init=None, x_best_init=None,
                 last_forward_init=None, levelset_state_init=None, physical_seed_state_init=None,
                 route_dfj_state_init=None, scheduler_state_init=None):
        super().__init__(prob_data, scen_tree, config)
        config = self.config
        self.stage_builder = StageModelBuilder(
            prob_data, mip_gap=config.get('phase2_tol', 1e-6),
            lazy_threshold=config.get('phase2_lazy_threshold', 256))
        self.forward_solver = ForwardSolver(
            prob_data, self.stage_builder,
            sub_time_limit=config.get('phase2_time_limit', 1800.),
            period_dedup=config.get('phase2_forward_period_dedup', True))
        self.backward_solver = BackwardSolverLagrangian(
            prob_data, self.stage_builder,
            lambda_level=config.get('lambda_level', .3),
            mu_level=config.get('mu_level', .5),
            norm_option=config.get('norm_option', 1),
            tol=config.get('levelset_tol', 1e-3),
            iter_limit=config.get('levelset_iter_limit', 100),
            adaptive_alpha=config.get('adaptive_alpha', False),
            sub_time_limit=config.get('phase2_time_limit', 1800.),
            inner_s3_rounds=config.get('phase2_inner_s3_rounds', 30),
            inner_s3_rel_tol=config.get('phase2_tol', 1e-6),
            inner_s3_time_limit=config.get('phase2_inner_s3_time_limit', 300.),
            s2_cut_time_limit=config.get('phase2_s2_cut_time_limit', 0.),
        )
        if scheduler_state_init is not None:
            self.backward_solver.restore_scheduler_state(scheduler_state_init)
        self.cut_lag = copy.deepcopy(cut_lag_init) if cut_lag_init else self._initialize_cuts()
        # Preserve a supplied archive; only seed additional static bounds when
        # explicitly requested, as opposed to changing the original cold start.
        if config.get('lrp_static_bounds', False):
            from cuts.lrp_static_bounds import seed_lrp_static_cuts
            seed_lrp_static_cuts(prob_data, scen_tree, self.cut_lag)
        self.cut_Dict = self._initialize_cuts()
        self.route_dfj_restore = None
        if route_dfj_state_init is not None:
            from models.route_dfj_pool import restore_route_dfj_checkpoint
            self.route_dfj_restore = restore_route_dfj_checkpoint(
                self._physical_routes(), route_dfj_state_init)
        if levelset_state_init is not None:
            # S3 supports refer to a fixed physical route function. Preserve
            # their coordinate identity so the manager can recheck the actual
            # context/shifts on first use. S2 surrogate bundles are not reused.
            registry = self.backward_solver.cut_manager._bundle_coordinates = {}
            for node_id, saved in levelset_state_init.get('stage3', {}).items():
                node_id = int(node_id)
                if node_id not in self.cut_Dict[3]:
                    raise ValueError('Restored Level Set node is absent from this LRP tree')
                supports = copy.deepcopy(saved['supports'])
                self.cut_Dict[3][node_id] = supports
                registry[id(supports)] = copy.deepcopy(saved['coordinates'])
        self.initial_lb = lb_init
        self.best_ub = float('inf')
        self.x_best = None
        if x_best_init is not None:
            audit = certify_policy(prob_data, scen_tree, x_best_init)
            self.best_ub = float(audit['feasible_upper_bound'])
            self.x_best = copy.deepcopy(x_best_init)
            if ub_init is not None and float(ub_init) < self.best_ub:
                raise ValueError('Inherited UB is below the independently audited policy cost')
        elif ub_init is not None and math.isfinite(float(ub_init)):
            raise ValueError('A finite inherited UB needs its feasible policy')
        self._s2_tolerance = Stage2ToleranceSchedule.from_config(
            config, n_nodes=len(scen_tree[2]),
            weight_mass=stage2_weight_mass(scen_tree[2]))
        self._physical_seed_cache = copy.deepcopy(physical_seed_state_init or {})
        self._physical_seed_events = []
        self._policy_polish_available = False
        self._policy_polish_stats = []
        self._physical_seed_spent = 0.
        self._late_physical = None
        reuse_enabled = bool(config.get('phase_forward_reuse', True))
        self._initial_forward = copy.deepcopy(last_forward_init) if reuse_enabled else None
        self._forward_reuse = dict(enabled=reuse_enabled,
            available=reuse_enabled and last_forward_init is not None,
            used=False, saved_seconds=0., phase1_iteration=None,
            semantic_match=False, solve_policy_match=None, rejected_reason=None)

    def _take_initial_forward(self):
        payload, self._initial_forward = getattr(self, '_initial_forward', None), None
        if payload is None:
            return None
        values, reason = take_snapshot(payload, self.prob_data, self.scen_tree,
                                       self.cut_lag, self.forward_solver)
        if reason is not None:
            self._forward_reuse['rejected_reason'] = reason
            print(f'  [Phase2-ForwardReuse] rejected: {reason}', flush=True)
            return None
        self._forward_reuse.update(used=True, semantic_match=True,
            solve_policy_match=(payload.get('solve_policy_fingerprint') == solve_policy(self.forward_solver)),
            solve_policy_compatible=True, rejected_reason=None,
            saved_seconds=max(0., float(payload.get('forward_seconds', 0.))),
            phase1_iteration=payload.get('phase1_iteration'))
        self.forward_solver.last_forward_diagnostics = copy.deepcopy(
            payload.get('forward_diagnostics', []))
        self.forward_solver._last_eta_per_omega = {
            q: values[1][1][0][f'eta[{q}]'] for q in self.scen_tree[1][0].successor}
        self.forward_solver.last_policy_certificate = certify_policy(
            self.prob_data, self.scen_tree, values[1])
        print('  [Phase2-ForwardReuse] consumed Phase1 snapshot '
              f"from iter={payload.get('phase1_iteration')}; "
              f"saved_forward={self._forward_reuse['saved_seconds']:.6f}s", flush=True)
        return values

    def _initialize_cuts(self):
        return {stage: {node.index: [] for node in nodes}
                for stage, nodes in self.scen_tree.items()}

    def levelset_checkpoint(self):
        """Serializable S3 bundles; model objects and stale S2 supports excluded."""
        registry = getattr(self.backward_solver.cut_manager, '_bundle_coordinates', {})
        return {'stage3': {node_id: dict(supports=copy.deepcopy(supports),
                                       coordinates=copy.deepcopy(registry[id(supports)]))
                          for node_id, supports in self.cut_Dict.get(3, {}).items()
                          if supports and id(supports) in registry}}

    def physical_seed_checkpoint(self):
        return copy.deepcopy(self._physical_seed_cache)

    def _physical_routes(self):
        return [(node.context, int(node.info)) for node in self.scen_tree[3]]

    def route_dfj_checkpoint(self):
        from models.route_dfj_pool import route_dfj_checkpoint
        return route_dfj_checkpoint(self._physical_routes())

    def scheduler_checkpoint(self):
        return self.backward_solver.scheduler_checkpoint()

    def _maybe_physical_seed(self, iteration, trial, best_lb, deadline, stale):
        """Original budgeted physical seed scheduling, using LRP root cuts."""
        backend = self.config.get('phase15_backend', 'none')
        schedule = self.config.get('phase15_schedule', 'off')
        if backend == 'none' or schedule == 'off':
            return None
        from core.customized_subprob import ensure_import_path
        ensure_import_path()
        from s2backward.phase15 import run_phase15
        from s2backward.late_physical import LatePhysicalSchedule
        if self._late_physical is None:
            self._late_physical = LatePhysicalSchedule(
                total_seconds=self.config.get('phase15_late_total_time_limit', 0.),
                per_call_seconds=self.config.get('phase15_late_per_call_time_limit', 180.),
                window=self.config.get('phase15_late_window', 3),
                relative_gain=self.config.get('phase15_late_relative_gain', 1e-4))
        self._late_physical.observe(iteration, best_lb)
        outer_remaining = max(0., deadline-time.monotonic())
        available = max(0., self.config.get('phase15_total_time_limit', 0.)-self._physical_seed_spent)
        if not math.isfinite(outer_remaining):
            outer_remaining = available+self._late_physical.remaining
        trigger, budget = None, 0.
        if available and (iteration == 1 or (schedule == 'on_demand' and
                stale >= self.config.get('phase15_on_demand_streak', 2))):
            trigger = 'initial_gap' if iteration == 1 else 'stale_backward'
            budget = min(available, self.config.get('phase15_per_call_time_limit', 15.), outer_remaining)
        if trigger is None:
            budget = self._late_physical.request(outer_remaining)
            if budget:
                trigger = 'lb_stagnation'
        if not budget:
            return None
        started = time.monotonic()
        report = run_phase15(self.prob_data, self.scen_tree, self.cut_lag,
            backend=backend, time_limit_s=budget, initial_policy=trial,
            cache=self._physical_seed_cache, s3_bundles=self.cut_Dict.get(3, {}))
        elapsed = time.monotonic()-started
        if trigger == 'lb_stagnation':
            self._late_physical.charge(elapsed)
        else:
            self._physical_seed_spent += elapsed
        event = dict(iteration=iteration, trigger=trigger, requested_seconds=budget,
                     actual_seconds=elapsed, report=report)
        self._physical_seed_events.append(event)
        return report

    def _maybe_dump_state(self, iteration, x_star, cost_star, best_lb):
        """Keep the original optional Phase-2 state dump interface for LRP."""
        import os
        dump_dir = os.environ.get('LRP_PHASE2_DUMP_STATE_DIR',
                                  os.environ.get('VRP_PHASE2_DUMP_STATE_DIR', '')).strip()
        if not dump_dir:
            return
        import pickle
        import tempfile
        from pathlib import Path
        from core.run_snapshot import capture_run_snapshot
        run_snapshot = getattr(self, '_state_run_snapshot', None)
        if run_snapshot is None:
            run_snapshot = capture_run_snapshot(self.prob_data, self.scen_tree, self.config)
            self._state_run_snapshot = run_snapshot
        payload = {
            'iteration': int(iteration), 'cut_lag': self.cut_lag,
            'x_star': x_star, 'cost_star': cost_star,
            'best_lb': float(best_lb), 'best_ub': float(self.best_ub),
            'x_best': self.x_best, 'run_snapshot': run_snapshot,
            's2_piece_tables': None,
            'levelset_state': self.levelset_checkpoint(),
            'route_dfj_state': self.route_dfj_checkpoint(),
            'scheduler_state': self.scheduler_checkpoint(),
            'physical_seed_state': self.physical_seed_checkpoint(),
        }
        # Serialize before creating any target, then atomically replace each
        # file so readers never observe half a pickle. No live models enter it.
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        directory = Path(dump_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name in (f'phase2_state_iter{iteration:04d}.pkl', 'phase2_state_latest.pkl'):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix='.' + name + '.',
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, directory / name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def _best_lower_bound(self):
        bounds = list(self.lb_history)
        if self.initial_lb is not None:
            bounds.append(float(self.initial_lb))
        return max(bounds) if bounds else None

    def _schedule_stage2_tolerance(self, best_lb):
        schedule = self._s2_tolerance
        if schedule is None:
            return None
        eps = schedule.node_tolerance(self.best_ub, best_lb)
        self.forward_solver.set_stage2_tolerance(eps, rel_cap=schedule.rel_cap)
        return eps

    def _accept_policy(self, policy, best_lb):
        audit = certify_policy(self.prob_data, self.scen_tree, policy)
        upper = float(audit['feasible_upper_bound'])
        if best_lb is not None and minimization_bounds_inverted(best_lb, upper):
            raise RuntimeError(f'Phase2 certified LB {best_lb} exceeds audited policy UB {upper}')
        if upper < self.best_ub:
            self.best_ub, self.x_best = upper, copy.deepcopy(policy)
        return upper

    def _inner_policy_gap(self, inner_policy):
        if inner_policy is None:
            return None
        best_lb = self._best_lower_bound()
        prior_incumbent = self.x_best
        self._accept_policy(inner_policy['x_star'], best_lb)
        runtime = getattr(self, 'physical_forward_runtime', None)
        if runtime is not None:
            runtime.observe(self, phase=2, iteration=getattr(self, '_physical_iteration', 0),
                            boundary='inner_policy')
        gap = (minimization_gap_percent(best_lb, self.best_ub)
               if best_lb is not None else float('inf'))
        if (gap >= self.config.get('phase2_tol', 1e-6) * 100
                and getattr(self, '_policy_polish_available', False)):
            self._policy_polish_available = False
            from solvers.lrp_policy_polish import improve_inner_policy
            candidate = improve_inner_policy(
                self.prob_data, self.scen_tree, inner_policy, prior_incumbent,
                certified_lb=best_lb,
                budget=getattr(self.backward_solver, '_refinement_budget', None))
            if candidate is not None:
                if candidate.get('certified') is not True:
                    raise RuntimeError('Physical policy search returned an uncertified policy')
                previous = self.best_ub
                offered = self._accept_policy(candidate['policy'], best_lb)
                if offered != candidate['ub']:
                    raise RuntimeError('Physical policy search UB differs from its independent audit')
                if runtime is not None:
                    runtime.collect(candidate['policy'], source='route_polish',
                                    phase=2, iteration=getattr(self, '_physical_iteration', 0))
                    runtime.observe(self, phase=2, iteration=getattr(self, '_physical_iteration', 0),
                                    boundary='inner_policy')
                closed = (best_lb is not None and
                    minimization_gap_percent(best_lb, self.best_ub) < self.config.get('phase2_tol', 1e-6) * 100)
                self._policy_polish_stats.append(dict(candidate['stats'], ub=offered,
                    previous_ub=previous, improved=offered < previous, closed=closed))
                print(f'  [Phase2-RoutePolish] UB={offered:,.6f}; '
                      f'improved={offered < previous}; closed={closed}; '
                      f"time={candidate['stats']['seconds']:.3f}s", flush=True)
            gap = (minimization_gap_percent(best_lb, self.best_ub)
                   if best_lb is not None else float('inf'))
        if self.ub_history:
            self.ub_history[-1] = self.best_ub
        if gap < self.config.get('phase2_tol', 1e-6) * 100:
            return {'lb': best_lb, 'ub': self.best_ub, 'gap_percent': gap,
                    'x_star': copy.deepcopy(self.x_best)}
        return None

    def _print_iteration_bounds(self, iteration, total_start, time_limit,
                                best_lb, fwd_time, *, bwd_time=None):
        gap = minimization_gap_percent(best_lb, self.best_ub)
        gap_str = 'INVALID' if minimization_bounds_inverted(best_lb, self.best_ub) else (
            f'{gap:7.3f}%' if math.isfinite(gap) else '    inf')
        timings = f'fwd={fwd_time:.1f}s'
        checkpoint = 'forward'
        if bwd_time is not None:
            timings += f', bwd={bwd_time:.1f}s'
            checkpoint = 'backward'
        elapsed_h = (time.monotonic() - total_start) / 3600
        limit = f'{time_limit / 3600:g}h' if time_limit > 0 else 'unlimited'
        print(f'  [Phase2-SDDLP {iteration:>3}  {elapsed_h:.2f}h/{limit}]  '
              f'LB={best_lb:>16,.2f}  UB={self.best_ub:>16,.2f}  '
              f'Gap={gap_str}  ({timings}) [{checkpoint}]', flush=True)

    @_manage_phase2_pools
    @capture_backend_run(phase=2)
    def solve(self):
        if getattr(self, '_solve_started', False):
            raise RuntimeError('SDDLP.solve() is one-shot; create a new solver')
        self._solve_started = True
        runtime = getattr(self, 'physical_forward_runtime', None)
        if runtime is None and (self.config.get('physical_forward_profile', 'off') != 'off'
                or self.config.get('physical_cg_profile','off') in ('paced','pool','cg')):
            from solvers.lrp_physical_forward import PhysicalForwardRuntime
            runtime = self.physical_forward_runtime = PhysicalForwardRuntime(
                self.prob_data, self.scen_tree, self.config)
        if runtime is not None:
            self.forward_solver.record_root_certificate = runtime.record_root_certificate
        print('\n' + '=' * 80)
        print('SDDLP算法')
        print('=' * 80)
        from core.exact_solver_log import set_current_phase
        set_current_phase('phase2')
        tol = float(self.config.get('phase2_tol', 1e-6))
        time_limit = float(self.config.get('phase2_total_time_limit', 36000.))
        iter_limit = int(self.config.get('phase2_iter_limit', 0))
        import os
        legacy_limit_name = ('LRP_PHASE2_MAX_ITERS' if 'LRP_PHASE2_MAX_ITERS' in os.environ
                             else 'VRP_PHASE2_MAX_ITERS')
        legacy_iter_limit = int(os.environ.get(legacy_limit_name, '0'))
        num_processes = int(self.config.get('num_processes', 1))
        level_limit = int(self.config.get('levelset_iter_limit', 100))
        no_new_limit = int(self.config.get('phase2_no_new_cut_limit', 3))
        no_improve_limit = int(self.config.get('phase2_no_improve_limit', 0))
        no_improve_count = 0
        last_improved_lb, last_improved_ub = -math.inf, math.inf
        total_start = time.monotonic()
        deadline = None if time_limit <= 0 else total_start + time_limit
        if runtime is not None:
            deadline = runtime.clip_deadline(deadline)
        stop_reason, iteration, stale = 'iteration_limit', 0, 0
        forward_times, backward_times, cumulative_times, diagnostics = [], [], [], []
        master_refresh_times, refinement_history, new_cut_counts = [], [], []
        pool = self._phase2_pool = (
            get_context("spawn").Pool(processes=num_processes)
            if num_processes > 1 and not (runtime is not None and runtime.expired()) else None
        )
        while iter_limit <= 0 or iteration < iter_limit:
            if ((time_limit > 0 and time.monotonic() - total_start >= time_limit)
                    or runtime is not None and runtime.expired()):
                stop_reason = 'phase2_total_time_limit'
                break
            # The original debug cap counts complete outer iterations,
            # including their backward passes. The explicit config cap above
            # retains its existing forward-boundary behavior.
            if legacy_iter_limit > 0 and iteration >= legacy_iter_limit:
                stop_reason = 'debug_iteration_limit'
                print(f'\n  [debug] 达到 {legacy_limit_name}={legacy_iter_limit} 轮, '
                      'Phase2 提前终止.', flush=True)
                break
            iteration += 1
            self._physical_iteration = iteration
            completed_backward = False
            cuts_before = sum(len(rows) for stage in self.cut_lag.values() for rows in stage.values())
            started = time.monotonic()
            self._schedule_stage2_tolerance(self._best_lower_bound())
            reused = self._take_initial_forward() if iteration == 1 else None
            lb, trial, ub, costs = (reused if reused is not None else
                self.forward_solver.forward_pass(self.scen_tree, self.cut_lag,
                    num_processes=num_processes, pool=pool, deadline=deadline))
            forward_times.append(time.monotonic() - started)
            backward_times.append(0.)
            master_refresh_times.append(0.)
            if not forward_values_are_finite((lb, trial, ub, costs)):
                raise RuntimeError('Phase2 forward returned a nonfinite certificate or trial')
            previous = self._best_lower_bound()
            best_lb = float(lb) if previous is None else max(previous, float(lb))
            self.lb_history.append(best_lb)
            self._accept_policy(trial, best_lb)
            self.ub_history.append(self.best_ub)
            if runtime is not None:
                runtime.after_forward(self, phase=2, iteration=iteration, trial=trial, costs=costs,
                                      forward_wall=forward_times[-1])
                runtime.run_epoch(self, trial, costs, iteration=iteration, deadline=deadline, pool=pool)
                # The physical epoch may refresh the certified root bound.
                # Use that same bound for stopping and the next backward budget.
                best_lb = self._best_lower_bound()
                if minimization_bounds_inverted(best_lb, self.best_ub):
                    raise RuntimeError('Physical epoch LB exceeds audited policy UB')
                self.ub_history[-1] = self.best_ub
            gap = minimization_gap_percent(best_lb, self.best_ub)
            self._print_iteration_bounds(iteration, total_start, time_limit, best_lb, forward_times[-1])
            if gap >= tol*100 and (iter_limit <= 0 or iteration < iter_limit):
                physical = self._maybe_physical_seed(iteration, trial, best_lb,
                    (math.inf if deadline is None else deadline) if runtime is not None else
                    (math.inf if time_limit <= 0 else total_start+time_limit), stale)
                if physical is not None:
                    stale = 0
                    master = physical.get('master') or {}
                    if master.get('lb_certified') and master.get('lb') is not None:
                        best_lb = max(best_lb, float(master['lb']))
                        if minimization_bounds_inverted(best_lb, self.best_ub):
                            raise RuntimeError('Physical root cut LB exceeds audited policy UB')
                        self.lb_history[-1] = best_lb
                    gap = minimization_gap_percent(best_lb, self.best_ub)
                    print(f'    [Phase1.5] added={physical.get("added", 0)} '
                          f'LB={best_lb:.9f} gap={gap:.6g}%')
            if gap < tol * 100:
                self.converged, stop_reason = True, 'gap_tolerance'
            elif ((time_limit > 0 and time.monotonic() - total_start >= time_limit)
                    or runtime is not None and runtime.expired()):
                stop_reason = 'phase2_total_time_limit'
            elif iter_limit > 0 and iteration >= iter_limit:
                stop_reason = 'iteration_limit'
            else:
                before = cut_archive_fingerprint(self.cut_lag)
                if self.config.get('reset_levelset', True):
                    for node_id in self.cut_Dict.get(2, {}):
                        self.cut_Dict[2][node_id] = []
                bwd_started = time.monotonic()
                eps = self._schedule_stage2_tolerance(best_lb)
                # Pay for the final root certificate from the SAME total
                # budget. Without this reserve the last S2 cuts are saved,
                # but a deadline skip prevents their actual S1 evaluation.
                backward_deadline = deadline
                if deadline is not None:
                    remaining = max(0., deadline - time.monotonic())
                    backward_deadline = deadline - min(2., remaining)
                if runtime is not None:
                    backward_deadline = runtime.backward_deadline(backward_deadline)
                if num_processes > 1 and self._phase2_s3_pool is None:
                    if runtime is not None:
                        self._phase2_s3_pool = pool  # drained S3/S2 waves, one worker budget
                    else:
                        s3_processes = self.backward_solver._resolve_s3_num_processes(num_processes, 4)
                        self._phase2_s3_pool = get_context("spawn").Pool(processes=s3_processes)
                # Original Final boundary: once per backward, incumbent only.
                self._policy_polish_available = True
                self.backward_solver._refinement_budget = None
                try:
                    self.cut_lag, self.cut_Dict = self.backward_solver.backward_pass(
                        self.scen_tree, trial, costs, self.cut_lag, self.cut_Dict,
                        iter_limit=level_limit, num_processes=num_processes, pool=pool,
                        s3_pool=self._phase2_s3_pool,
                        outer_gap=gap, outer_gap_abs=self.best_ub-best_lb,
                        s2_abs_tol=eps,
                        stage1_eta=getattr(self.forward_solver, '_last_eta_per_omega', None),
                        s2_rel_cap=None if self._s2_tolerance is None else self._s2_tolerance.rel_cap,
                        policy_checkpoint=self._inner_policy_gap,
                        deadline=backward_deadline,
                        **({} if runtime is None else {'allow_shared_pool': True}))
                finally:
                    self._policy_polish_available = False
                    self.backward_solver._refinement_budget = None
                completed_backward = True
                backward_times[-1] = time.monotonic() - bwd_started
                handoff = getattr(self.backward_solver, 'last_stage2_handoff', None)
                self.backward_solver.last_stage2_handoff = None
                if handoff is not None:
                    self.forward_solver.set_stage2_handoff(handoff)
                inner = getattr(self.backward_solver, 'last_inner_policy', None)
                if inner is not None:
                    self._accept_policy(inner['x_star'], best_lb)
                    self.ub_history[-1] = self.best_ub
                diagnostics.append(copy.deepcopy(getattr(self.backward_solver, 'last_cut_diagnostics', {})))
                refinement_history.append(copy.deepcopy(self.backward_solver.last_refinement_diagnostic))
                # Refresh only while the shared deadline permits. If it is
                # exhausted, preserve the already certified LB and retain all
                # new cuts for the next resumed master solve.
                master_started = time.monotonic()
                _, master_result = self.forward_solver._solve_model(
                    1, self.scen_tree[1][0], self.cut_lag, {}, deadline=deadline)
                master_refresh_times[-1] = time.monotonic()-master_started
                if master_result.certified_lower_bound is not None:
                    best_lb = max(best_lb, master_result.certified_lower_bound)
                    self.lb_history[-1] = best_lb
                    if minimization_bounds_inverted(best_lb, self.best_ub):
                        raise RuntimeError('Refreshed master LB exceeds audited policy UB')
                if runtime is not None:
                    runtime.observe(self, phase=2, iteration=iteration, boundary='master_refresh',
                                    backward_wall=backward_times[-1],
                                    master_refresh_wall=master_refresh_times[-1], completed_backward=True)
                self._maybe_dump_state(iteration, trial, costs, best_lb)
                changed = before != cut_archive_fingerprint(self.cut_lag)
                stale = 0 if changed else stale + 1
                if no_new_limit > 0 and stale >= no_new_limit:
                    if getattr(self.backward_solver, 'last_refinement_limited', False):
                        self.backward_solver.request_refinement()
                    stale = 0  # unchanged cuts are never a convergence certificate
                if minimization_gap_percent(best_lb, self.best_ub) < tol * 100:
                    self.converged, stop_reason = True, 'gap_tolerance'
                elif deadline is not None and time.monotonic() >= deadline:
                    stop_reason = 'phase2_total_time_limit'
            if (completed_backward and not self.converged
                    and stop_reason != 'phase2_total_time_limit' and no_improve_limit > 0):
                # Original absolute 1e-6 best-LB/best-UB improvement test.
                # Check after the complete backward/master boundary so a
                # new cut's certified improvement is not discarded as stale.
                if best_lb > last_improved_lb + 1e-6 or self.best_ub < last_improved_ub - 1e-6:
                    no_improve_count = 0
                    last_improved_lb, last_improved_ub = best_lb, self.best_ub
                else:
                    no_improve_count += 1
                    if no_improve_count >= no_improve_limit:
                        stop_reason = 'no_improvement'
                        print(f'  LB/UB 均无改进连续 {no_improve_count} 轮, Phase2 终止.', flush=True)
            self.time_history.append(time.monotonic() - started)
            cumulative_times.append(time.monotonic() - total_start)
            cuts_after = sum(len(rows) for stage in self.cut_lag.values() for rows in stage.values())
            new_cut_counts.append(cuts_after - cuts_before)
            print(f'  bwd={backward_times[-1]:.1f}s  total={self.time_history[-1]:.1f}s  '
                  f'new_cuts={new_cut_counts[-1]}', flush=True)
            if backward_times[-1] > 0:
                self._print_iteration_bounds(iteration, total_start, time_limit, best_lb,
                                             forward_times[-1], bwd_time=backward_times[-1])

            callback = getattr(self, 'progress_callback', None)
            if callback is not None:
                callback(self, iteration, time.monotonic()-total_start)
            if self.converged or stop_reason in ('phase2_total_time_limit', 'no_improvement'):
                break
        result = {
            'Vstar': self._best_lower_bound(), 'initial_lb': self.initial_lb,
            'x_best': self.x_best, 'cut_lag': self.cut_lag,
            'levelset_state': self.levelset_checkpoint(),
            'route_dfj_state': self.route_dfj_checkpoint(),
            'scheduler_state': self.scheduler_checkpoint(),
            'physical_seed_state': self.physical_seed_checkpoint(),
            'phase15_history': copy.deepcopy(self._physical_seed_events),
            'phase15_spent_seconds': self._physical_seed_spent,
            'physical_policy_search': copy.deepcopy(self._policy_polish_stats),
            'phase15_late_schedule': (None if self._late_physical is None else self._late_physical.report()),
            'LB_list': self.lb_history, 'UB_list': self.ub_history,
            'time_list': self.time_history, 'cumulative_time_list': cumulative_times,
            'forward_time_list': forward_times, 'backward_time_list': backward_times,
            'master_refresh_time_list': master_refresh_times,
            'new_cut_count_list': new_cut_counts,
            'refinement_history': refinement_history,
            'converged': self.converged, 'iterations': iteration,
            'total_time': time.monotonic()-total_start, 'stop_reason': stop_reason,
            'backward_history': diagnostics,
            'policy_audit': certify_policy(self.prob_data, self.scen_tree, self.x_best)
                            if self.x_best is not None else None,
            'forward_reuse': dict(getattr(self, '_forward_reuse', {'enabled': False, 'used': False})),
        }
        result.update(solver_diagnostics(self.forward_solver, self.backward_solver))
        if runtime is not None:
            result['physical_forward'] = runtime.summary()
        return result
