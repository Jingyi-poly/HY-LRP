"""
SDDP with Strengthened Benders Cuts算法
"""

import copy
import math
import os
import time
from core.backend_telemetry import capture_backend_run
from functools import wraps
from multiprocessing import get_context
from algorithms.base_algorithm import (
    SDDPAlgorithm, cut_archive_fingerprint, forward_values_are_finite, solver_diagnostics,
)
from solvers.lrp_forward_reuse import make_snapshot
from solvers.forward_solver_phase1 import ForwardSolver
from solvers.backward_solver_sbc_phase1 import BackwardSolverSBC
from models.stage_builder_phase1 import StageModelBuilder
from core.solver_bounds import (
    minimization_bounds_inverted,
    minimization_gap_percent,
)



__all__ = ['SDDP_SBC']


def _manage_phase1_pool(solve_method):
    """Close workers cleanly on success and terminate them on failure.

    The Phase-1 loop has several normal ``break`` exits.  Wrapping the whole
    solve method keeps those exits unchanged while ensuring a persistent pool
    cannot leak into Phase 2 or a following A/B run.
    """
    @wraps(solve_method)
    def wrapped(self, *args, **kwargs):
        self._phase1_pool = None
        try:
            result = solve_method(self, *args, **kwargs)
        except BaseException:
            pool = self._phase1_pool
            self._phase1_pool = None
            if pool is not None:
                try:
                    pool.terminate()
                finally:
                    pool.join()
            raise
        else:
            pool = self._phase1_pool
            self._phase1_pool = None
            if pool is not None:
                try:
                    pool.close()
                finally:
                    pool.join()
            return result

    return wrapped


class SDDP_SBC(SDDPAlgorithm):
    """
    SDDP with Strengthened Benders Cuts算法

    Phase 1算法流程:
    1. 初始化切割字典
    2. 迭代执行:
       a. 前向传递:求解各阶段问题，计算上界
       b. 后向传递:生成SBC切割,更新下界
    3. 检查收敛:gap < tol
    4. 返回结果

    Attributes:
        prob_data: ProblemData实例
        scen_tree: 场景树
        config: 算法配置
        forward_solver: 前向求解器
        backward_solver: 后向求解器
        stage_builder: 阶段模型构建器
    """

    def __init__(self, prob_data, scen_tree, config):
        """
        初始化SDDP-SBC算法

        Args:
            prob_data: ProblemData实例
            scen_tree: 场景树 {stage: [nodes]}
            config: AlgorithmConfig实例或配置字典
        """
        super().__init__(prob_data, scen_tree, config)
        config = self.config

        _lazy_thr = config.get('phase1_lazy_threshold', 8)

        # 创建求解器
        self.stage_builder = StageModelBuilder(prob_data,
                                                  mip_gap=config.get('phase1_mip_gap', 1e-4),
                                                  lazy_threshold=_lazy_thr)
        phase1_time_limit = config.get('phase1_time_limit', 1800.0)
        self.forward_solver = ForwardSolver(prob_data, self.stage_builder,
                                               sub_time_limit=phase1_time_limit,
                                               stage2_sub_time_limit=config.get('phase1_forward_s2_time_limit'),
                                               stage2_mipgap_schedule=(
                                                   (config.get('phase1_s2_mipgap_early', 1e-3),
                                                    config.get('phase1_s2_mipgap_late', 1e-4),
                                                    config.get('phase1_s2_mipgap_switch', 30))
                                                   if config.get('phase1_s2_mipgap_schedule', False) else None),
                                               period_dedup=config.get(
                                                   'phase1_forward_period_dedup', True
                                               ))
        self.backward_solver = BackwardSolverSBC(
            prob_data,
            self.stage_builder,
            strengthen_s2=True,
            strengthen=True,
            sub_time_limit=phase1_time_limit,
        )

        # 初始化切割字典
        self.cut_lag = self._initialize_cuts()
        # Match the Investment cold start. Precomputed LRP bounds are an
        # optional strengthening, not part of the physical model constraints.
        if config.get('lrp_static_bounds', False):
            from cuts.lrp_static_bounds import seed_lrp_static_cuts
            seed_lrp_static_cuts(prob_data, scen_tree, self.cut_lag)

        # 上界追踪（历史最小值）
        self.best_ub = float('inf')

    def _initialize_cuts(self):
        """初始化切割字典"""
        cut_lag = {}
        for stage, nodes in self.scen_tree.items():
            cut_lag[stage] = {}
            for node_idx in range(len(nodes)):
                cut_lag[stage][node_idx] = []
        return cut_lag

    @_manage_phase1_pool
    @capture_backend_run(phase=1)
    def solve(self):
        """
        执行SDDP-SBC算法

        Returns:
            结果字典:
            - 'Vstar': 最优值（下界）
            - 'x_best': 最优解
            - 'cut_lag': 生成的切割
            - 'LB_list': 下界历史
            - 'UB_list': 上界历史
            - 'time_list': 每次迭代时间
            - 'cumulative_time_list': 每个完整迭代边界的累计墙钟时间
            - 'converged': 是否收敛
            - 'iterations': 迭代次数
            - 'stop_reason': 收敛、无改进、总时限或迭代上限
        """
        print("\n" + "=" * 80)
        print("SDDP-SBC算法")
        print("=" * 80)

        # One-shot: do not reuse solver across runs
        if getattr(self, '_solve_started', False):
            raise RuntimeError("SDDP_SBC.solve() is one-shot; create a new solver")
        self._solve_started = True

        runtime = getattr(self, 'physical_forward_runtime', None)
        if runtime is None and (self.config.get('physical_forward_profile', 'off') != 'off'
                or self.config.get('physical_cg_profile','off') in ('paced','pool','cg')):
            from solvers.lrp_physical_forward import PhysicalForwardRuntime
            runtime = self.physical_forward_runtime = PhysicalForwardRuntime(
                self.prob_data, self.scen_tree, self.config)
        if runtime is not None:
            self.forward_solver.record_root_certificate = runtime.record_root_certificate

        from core.exact_solver_log import set_current_phase
        set_current_phase("phase1")

        tol = self.config.get('phase1_tol', 1e-2)
        iter_limit = self.config.get('phase1_iter_limit', 100)
        num_processes = self.config.get('num_processes', 1)
        no_improve_limit = max(0, int(self.config.get(
            'phase1_no_improve_limit', self.config.get('no_improve_limit', 5)
        )))
        total_time_limit = max(
            0.0,
            float(self.config.get('phase1_total_time_limit', 0.0)),
        )
        stability_enabled = bool(
            self.config.get('phase1_stability_handoff', True)
        )
        stability_window = max(
            1, int(self.config.get('phase1_stability_window', 3))
        )
        stability_min_iteration = max(
            stability_window + 1,
            int(self.config.get('phase1_stability_min_iteration', 6)),
        )
        stagnation_window = max(
            0, int(self.config.get('phase1_stagnation_window', 0))
        )
        stagnation_rel_tol = max(
            0.0, float(self.config.get('phase1_stagnation_rel_tol', 0.0))
        )
        stagnation_min_iteration = max(
            stagnation_window + 1,
            int(self.config.get('phase1_stagnation_min_iteration', 8)),
        )
        best_lb_by_iteration = []
        best_ub_by_iteration = []
        stagnation_telemetry = {
            'enabled': stagnation_window > 0,
            'triggered': False,
            'window': stagnation_window,
            'rel_tol': stagnation_rel_tol,
            'min_iteration': stagnation_min_iteration,
            'iteration': None,
            'lb_rel_gain': None,
            'ub_rel_gain': None,
        }
        stability_samples = []
        stability_telemetry = {
            'enabled': stability_enabled,
            'triggered': False,
            # window = unchanged transitions (need W+1 samples)
            'window': stability_window,
            'min_iteration': stability_min_iteration,
            'iteration': None,
            'completed_boundaries': 0,
            'stable_transitions': 0,
        }
        # 主求解循环
        total_start = time.time()
        phase_deadline = (None if total_time_limit <= 0 else
                          time.monotonic() + total_time_limit)
        if runtime is not None:
            phase_deadline = runtime.clip_deadline(phase_deadline)
        preseed_report = None
        preseed_budget = self.config.get('phase1_preseed_time_limit', 0.)
        if preseed_budget > 0 and iter_limit > 0:
            from cuts.lrp_preseed import seed_lrp_prepass
            # Preheating is charged to the existing Phase-1 budget. Only its
            # verified cuts persist; normal forward supplies all LB/UB history.
            if total_time_limit > 0:
                preseed_budget = min(preseed_budget, total_time_limit)
            if runtime is not None and phase_deadline is not None:
                preseed_budget = min(preseed_budget, max(0., phase_deadline-time.monotonic()))
            if preseed_budget > 0:
                preseed_report = seed_lrp_prepass(
                    self.prob_data, self.scen_tree, self.cut_lag,
                    time_limit_s=preseed_budget,
                    max_rounds=self.config.get('phase1_preseed_rounds', 2))
        no_improve_count = 0
        last_lb = -float('inf')
        last_ub = float('inf')
        cumulative_time_history = []
        stop_reason = 'iteration_limit'

        # 创建持久化 Pool
        # Each worker owns its Gurobi environment.
        pool = self._phase1_pool = (
            get_context("spawn").Pool(processes=num_processes)
            if num_processes > 1 and not (runtime is not None and runtime.expired()) else None
        )
        # Cache last forward for Phase-2 iter-1 reuse
        self._last_forward = None
        for iteration in range(1, iter_limit + 1):
            if runtime is not None and (runtime.expired() or
                    phase_deadline is not None and time.monotonic() >= phase_deadline):
                stop_reason = 'phase1_total_time_limit'
                break
            iter_start = time.time()

            # ==================== 前向传递 ====================
            fwd_start = time.time()
            forward_options = {} if runtime is None else {'deadline': phase_deadline}
            lb, x_star, ub, cost_star = self.forward_solver.forward_pass(
                self.scen_tree, self.cut_lag, num_processes, pool=pool, iteration=iteration,
                **forward_options)
            fwd_time = time.time() - fwd_start
            forward_values = (lb, x_star, ub, cost_star)
            stage1_lb_certified = math.isfinite(float(lb))
            forward_incumbent_feasible = (
                forward_values_are_finite(forward_values)
                and not minimization_bounds_inverted(float(lb), float(ub))
            )
            if not stage1_lb_certified:
                raise RuntimeError(
                    "Phase1 forward returned no finite certified Stage-1 LB"
                )
            if not forward_incumbent_feasible:
                raise RuntimeError(
                    "Phase1 forward returned a non-finite or bound-inverted "
                    "incumbent payload"
                )
            # Preserve the original Phase-1 -> Phase-2 forward handoff. A
            # later archive change still invalidates it at the boundary below.
            self._last_forward = (make_snapshot(
                self.prob_data, self.scen_tree, self.cut_lag, self.forward_solver,
                forward_values, fwd_time, iteration)
                if self.config.get('phase_forward_reuse', True) else None)

            # 更新下界（取历史最大值）
            self.lb_history.append(lb)
            best_lb = max(self.lb_history)

            # 更新上界（历史最小值）
            if ub < self.best_ub:
                self.best_ub = ub
                self.x_best = copy.deepcopy(x_star)
            self.ub_history.append(self.best_ub)
            if runtime is not None:
                runtime.after_forward(self, phase=1, iteration=iteration,
                                      trial=x_star, costs=cost_star, forward_wall=fwd_time)
            # 单向 gap；LB>UB 是证书错误，绝不能被 abs() 伪装成收敛。
            bounds_inverted = minimization_bounds_inverted(
                best_lb,
                self.best_ub,
            )
            gap = minimization_gap_percent(best_lb, self.best_ub)

            if bounds_inverted:
                raise RuntimeError('Phase1 certified LB exceeds feasible policy UB')
            else:
                gap_str = f"{gap:7.3f}%" if gap < float('inf') else "    inf"
            print(f"  [Phase1-SBC  {iteration:>3}/{iter_limit}]  "
                    f"LB={best_lb:>16,.2f}  UB={self.best_ub:>16,.2f}  "
                    f"Gap={gap_str}  (fwd={fwd_time:.1f}s)", end="")

            # 检查收敛（gap）
            if gap < tol * 100:
                self.converged = True
                stop_reason = 'gap_tolerance'
                iter_end = time.time()
                iter_time = iter_end - iter_start
                self.time_history.append(iter_time)
                cumulative_time_history.append(iter_end - total_start)
                print()
                break

            # 显式启用的启发式早停：0 表示关闭，与 Phase 2 语义一致。
            # 正数表示历史最优 LB 与 UB 同时无改进的连续轮数。
            lb_improved = best_lb > last_lb + 1e-6
            ub_improved = self.best_ub < last_ub - 1e-6
            if lb_improved or ub_improved:
                no_improve_count = 0
                last_lb = best_lb
                last_ub = self.best_ub
            else:
                no_improve_count += 1
                if (
                    no_improve_limit > 0
                    and no_improve_count >= no_improve_limit
                ):
                    stop_reason = 'no_improvement'
                    iter_end = time.time()
                    iter_time = iter_end - iter_start
                    self.time_history.append(iter_time)
                    cumulative_time_history.append(iter_end - total_start)
                    print()
                    print(f"  LB/UB 均无改进连续 {no_improve_count} 轮, Phase1 终止.")
                    break

            # Relative-stagnation handoff.  Phase-1 SBC cuts use LP-dual
            # multipliers and typically stall a few percent above the true
            # value while the LB still creeps by a negligible amount every
            # iteration; the bitwise gate below never fires in that regime.
            # Checked before the backward pass so the certified forward
            # snapshot stays bitwise-consistent with the archive for Phase 2.
            best_lb_by_iteration.append(float(best_lb))
            best_ub_by_iteration.append(float(self.best_ub))
            if (
                stagnation_window > 0
                and iteration >= stagnation_min_iteration
                and len(best_lb_by_iteration) > stagnation_window
                and math.isfinite(best_lb_by_iteration[-1 - stagnation_window])
                and math.isfinite(best_ub_by_iteration[-1 - stagnation_window])
            ):
                lb_ref = best_lb_by_iteration[-1 - stagnation_window]
                ub_ref = best_ub_by_iteration[-1 - stagnation_window]
                scale = max(1.0, abs(float(self.best_ub)))
                lb_rel_gain = (float(best_lb) - lb_ref) / scale
                ub_rel_gain = (ub_ref - float(self.best_ub)) / scale
                stagnation_telemetry.update({
                    'lb_rel_gain': lb_rel_gain,
                    'ub_rel_gain': ub_rel_gain,
                })
                if (
                    lb_rel_gain <= stagnation_rel_tol
                    and ub_rel_gain <= stagnation_rel_tol
                ):
                    stagnation_telemetry.update({
                        'triggered': True,
                        'iteration': iteration,
                    })
                    stop_reason = 'phase1_stagnation_handoff'
                    iter_end = time.time()
                    iter_time = iter_end - iter_start
                    self.time_history.append(iter_time)
                    cumulative_time_history.append(iter_end - total_start)
                    print()
                    print(
                        "  [Phase1-StagnationHandoff] "
                        f"iter={iteration}: 最近 {stagnation_window} 轮 "
                        f"LB 相对改进 {lb_rel_gain:.2e}, UB 相对改进 "
                        f"{ub_rel_gain:.2e} 均 <= {stagnation_rel_tol:.1e}；"
                        "切换 Phase2（不标记 Phase1 收敛）."
                    )
                    break

            # ==================== 后向传递 ====================
            bwd_time = 0.0
            if iteration < iter_limit:
                bwd_start = time.time()
                backward_options = ({} if runtime is None else
                                    {'deadline': runtime.backward_deadline(phase_deadline)})
                self.cut_lag = self.backward_solver.backward_pass(
                    self.scen_tree, x_star, cost_star, self.cut_lag, num_processes, pool=pool,
                    **backward_options
                )

                bwd_time = time.time() - bwd_start

            iter_end = time.time()
            iter_time = iter_end - iter_start
            self.time_history.append(iter_time)
            cumulative_time_history.append(iter_end - total_start)
            print(f"  bwd={bwd_time:.1f}s  total={iter_time:.1f}s")
            if runtime is not None:
                runtime.observe(self, phase=1, iteration=iteration, boundary='backward',
                                backward_wall=bwd_time, completed_backward=iteration < iter_limit)

            # Stability handoff after backward (exact cut+bound fingerprint)
            stability_trigger_ready = False
            if stability_enabled and iteration < iter_limit:
                sample = {
                    'iteration': iteration,
                    'cut_fingerprint': cut_archive_fingerprint(self.cut_lag),
                    'best_lb_hex': float(best_lb).hex(),
                    'best_ub_hex': float(self.best_ub).hex(),
                }
                stability_samples.append(sample)
                if len(stability_samples) > stability_window + 1:
                    del stability_samples[0]

                stable_transitions = 0
                for left, right in zip(
                    reversed(stability_samples[:-1]),
                    reversed(stability_samples[1:]),
                ):
                    if (
                        left['cut_fingerprint'] != right['cut_fingerprint']
                        or left['best_lb_hex'] != right['best_lb_hex']
                        or left['best_ub_hex'] != right['best_ub_hex']
                    ):
                        break
                    stable_transitions += 1
                stability_telemetry.update({
                    'completed_boundaries': iteration,
                    'stable_transitions': stable_transitions,
                })

                if (
                    iteration >= stability_min_iteration
                    and stable_transitions >= stability_window
                ):
                    stability_trigger_ready = True

            # 外层总时限只在 forward/backward 和额外 multi-cut 都完成后
            # 检查。它是软时限：当前迭代不会在中途被中断。若同一边界也
            # 满足稳定窗口，显式总时限拥有停止原因优先级，但上面的边界
            # telemetry 仍如实计数。
            if (
                total_time_limit > 0.0
                and iteration < iter_limit
                and cumulative_time_history[-1] >= total_time_limit
            ):
                stop_reason = 'phase1_total_time_limit'
                # 本轮 backward 可能改变 cut_lag，而 _last_forward 来自改变前。
                # Phase 2 不得猜测，应用当前 cuts 重算 forward。
                self._last_forward = None
                print(
                    "  Phase1 外层总时限已到："
                    f"{cumulative_time_history[-1]:.1f}s >= "
                    f"{total_time_limit:.1f}s；在完整迭代边界终止。"
                )
                break

            if stability_trigger_ready:
                stability_telemetry.update({
                    'triggered': True,
                    'iteration': iteration,
                })
                stop_reason = 'phase1_stability_handoff'
                print(
                    "  [Phase1-StabilityHandoff] "
                    f"iter={iteration}, unchanged_transitions="
                    f"{stable_transitions}/{stability_window}; "
                    "完整 cut archive 与 best LB/UB 均逐位不变，"
                    "切换 Phase2（不标记 Phase1 收敛）."
                )
                break

        total_time = time.time() - total_start

        if (
            self._last_forward is not None
            and self._last_forward['cut_fingerprint']
            != cut_archive_fingerprint(self.cut_lag)
        ):
            # 防御式兜底：最后 forward 后任何 archive 改动都会使缓存失效。
            self._last_forward = None

        result = {
            'Vstar': max(self.lb_history) if self.lb_history else 0,
            'x_best': self.x_best,
            'cut_lag': self.cut_lag,
            'LB_list': self.lb_history,
            'UB_list': self.ub_history,
            'time_list': self.time_history,
            'cumulative_time_list': cumulative_time_history,
            'converged': self.converged,
            'iterations': len(self.lb_history),
            'total_time': total_time,
            'stop_reason': stop_reason,
            'stability_handoff': stability_telemetry,
            'stagnation_handoff': stagnation_telemetry,
            'last_forward': self._last_forward,
        }

        if preseed_report is not None:
            result['preseed_report'] = preseed_report
        from solvers.forward_policy_certification import certify_policy
        result.update(solver_diagnostics(self.forward_solver, self.backward_solver))
        result['policy_audit'] = (certify_policy(self.prob_data, self.scen_tree, self.x_best)
                                  if self.x_best is not None else None)
        if runtime is not None:
            result['physical_forward'] = runtime.summary()
        return result
