"""
算法配置和结果容器

该模块定义算法的配置参数和结果数据结构。
"""

PHYSICAL_FORWARD_DEFAULTS = {
    'epoch_budget_s': 60., 'pool_time_limit_s': 15.,
    'pool_max_routes_per_node_facility': 200, 'joint_time_limit_s': 30.,
    'joint_max_nodes_per_epoch': 2, 'joint_max_concurrent': 2,
    'route_lp_time_limit_s': 2., 'route_lp_max_calls_per_epoch': 4,
    'audit_reserve_s': 5., 'backward_epoch_budget_s': 60.,
    'node_accept_tol': 1., 'sep_atol': 1e-6, 'sep_rtol': 1e-9,
    'pool_mip_gap': 1e-2, 'joint_mip_gap': 1e-3,
}

PHYSICAL_CG_DEFAULTS = {
    'epoch_wall': 60., 'global_policy_pool_cap': 10.,
    'joint_shared_cap': 40., 'joint_node_cap': 30.,
    'max_joint_nodes': 2, 'pricing_call_cap': 3.,
    'cg_max_rounds': 50, 'commit_audit_refresh_reserve': 10.,
    'paced_backward_wall': 60.,
}


class AlgorithmConfig:
    """
    算法配置类

    封装SDDP算法的所有配置参数。

    Attributes:
        phase1_tol (float): Phase 1 收敛容差(gap)
        phase1_mip_gap (float): Phase 1 子问题 MIPGap
        phase2_tol (float): Phase 2 收敛容差(gap)
        phase1_iter_limit (int): Phase 1 最大迭代次数
        phase1_time_limit (float): Phase 1 单个受限子问题的统一时限(秒)
        phase1_total_time_limit (float): Phase 1 外层总时限(秒); 0 表示禁用
        phase2_time_limit (float): Phase 2 单个受限子问题的统一时限(秒)
        phase2_total_time_limit (float): Phase 2 外层算法总时限(秒)
        num_processes (int): 并行进程数
        phase1_no_improve_limit (int): Phase1 连续 LB/UB 均无改进早停次数；0=禁用
        phase2_no_improve_limit (int): Phase2 完整迭代后连续 LB/UB 均无改进早停次数；0=禁用
        no_improve_limit (int): 兼容旧字段，等同 phase1_no_improve_limit
        lambda_level (float): Level Set 方法的 lambda 参数
        mu_level (float): Level Set 方法的 mu 参数
        norm_option (int): Level Set 范数选项(1=L1, 2=L2)
        levelset_iter_limit (int): Level Set 内层最大迭代次数
        levelset_tol (float): Level Set 内层收敛容差
        reset_levelset (bool): Phase 2 每轮是否重置 Level Set cuts
        phase1_stability_handoff (bool): 状态连续逐位不变时切换到
            Phase 2；这是阶段预算分配，不是 Phase 1 收敛证明
        phase1_stability_window (int): 需要的连续不变 boundary transition 数
        phase1_stability_min_iteration (int): 允许状态稳定切换的最早迭代
    """

    def __init__(self, phase1_tol=1e-2, phase2_tol=1e-2,
                 phase1_iter_limit=100, phase2_total_time_limit=10 * 3600,
                 num_processes=1, no_improve_limit=5,
                 phase1_no_improve_limit=None,
                 lambda_level=0.3, mu_level=0.5, norm_option=1,
                 levelset_iter_limit=20, levelset_tol=1e-3,
                 reset_levelset=True,
                 adaptive_alpha=False,
                 phase1_time_limit=None, phase1_total_time_limit=0.0,
                 phase2_time_limit=None,
                 phase1_lazy_threshold=8,
                 phase1_forward_period_dedup=True,
                 phase1_stability_handoff=True,
                 phase1_stability_window=3,
                 phase1_stability_min_iteration=6,
                 phase1_stagnation_window=5,
                 phase1_stagnation_rel_tol=0.0,
                 phase1_stagnation_min_iteration=8,
                 phase2_lazy_threshold=256,
                 phase2_forward_period_dedup=True,
                 phase2_inner_s3_rounds=30,
                 phase2_inner_s3_time_limit=300.0,
                 phase2_s2_cut_time_limit=0.0,
                 phase2_no_new_cut_limit=3,
                 phase2_s2_abs_tol_kappa=0.5,
                 phase2_s2_abs_tol_floor_share=0.5,
                 phase2_s2_abs_tol_rel_cap=0.02,
                 phase_forward_reuse=True,
                 phase15_backend="none", phase15_total_time_limit=0.0,
                 phase15_schedule="off", phase15_on_demand_streak=2,
                 phase15_per_call_time_limit=15.0,
                 phase15_late_total_time_limit=0., phase15_late_per_call_time_limit=180.,
                 phase15_late_window=3, phase15_late_relative_gain=1e-4,
                 phase15_late_strategy='a_then_b',
                 phase1_mip_gap=1e-4, phase2_iter_limit=0, lrp_static_bounds=False,
                 *, sub_time_limit=None, phase1_sub_time_limit=None,
                 phase2_sub_time_limit=None, phase1_forward_s2_time_limit=None,
                 phase2_no_improve_limit=0, phase1_s2_mipgap_schedule=False,
                 phase1_s2_mipgap_early=1e-3, phase1_s2_mipgap_late=1e-4,
                 phase1_s2_mipgap_switch=30,
                 phase1_preseed_time_limit=0., phase1_preseed_rounds=2,
                 physical_forward_profile='off', physical_forward_options=None,
                 physical_cg_profile='off', physical_cg_options=None):
        """
        初始化算法配置

        Args:
            phase1_tol: Phase 1 外层收敛容差(默认1e-2,即gap≤1%)
            phase1_mip_gap: Phase 1 子问题 MIPGap(默认1e-4)
            phase1_s2_mipgap_schedule: 已解析的 S2 forward 分阶段容差模式。
                默认False以保持旧LRP配置/快照的常量行为；True时仅S2 forward
                在 iteration < phase1_s2_mipgap_switch 使用early，否则late。
                主入口loader中显式phase1_mip_gap优先关闭该模式；快照直接
                恢复这里的已解析字段，不再重新读取环境变量。
            phase2_tol: Phase 2 收敛容差(默认1e-2,即gap≤1%)
            phase1_iter_limit: Phase 1 最大迭代次数(默认100)
            phase1_time_limit: Phase 1 单个受限子问题统一时限(默认1800秒)
            phase1_total_time_limit: Phase 1 外层总时限(默认0=禁用)。
                只在完整迭代边界检查，因此可超出一个迭代的用时。
            phase2_time_limit: Phase 2 单个受限子问题统一时限(默认1800秒)
            phase2_total_time_limit: Phase 2 外层算法总时限(默认10小时)
            phase1_sub_time_limit, phase2_sub_time_limit: 原单问题时限参数别名。
                明确传入 phase1_time_limit / phase2_time_limit 时以新字段为准；
                旧别名优先于通用 sub_time_limit；均未指定则仍为1800秒。
                Python phase2_time_limit 保持当前 LRP 单问题含义，旧命令的
                VRP_PHASE2_TIME_LIMIT 总时限兼容由 setup.solver_config 处理。
            phase15_backend: 物理路径 eta 补强；routeopt / gurobi / none
            phase15_total_time_limit: 所有 Phase 1.5 调用共享的总时限，计入 Phase 2
            phase15_schedule: ``once`` 仅首轮调用；``on_demand`` 首轮调用后，
                在 backward archive 连续停滞时复用该模块；``off`` 禁用
            phase15_on_demand_streak: on_demand 再调用前所需的连续停滞 backward 数
            phase15_per_call_time_limit: on_demand 单次调用上限
            phase15_late_total_time_limit: 独立的 LB 停滞后期预算；Main profile 默认启用
            phase15_late_per_call_time_limit: 后期每次调用上限，仍受 Phase2 总时限约束
            phase15_late_window: 检查 LB 累计增量的连续轮数，不依赖 UB/cut 数
            phase15_late_relative_gain: 窗口内 LB 增量阈值，相对于 max(1,|LB|)
            num_processes: 并行进程数(默认1)
            no_improve_limit: 兼容旧字段；未单独指定 phase1/2 时限时用于 Phase1
            phase1_no_improve_limit: Phase1 连续 LB/UB 均无改进早停次数
                (默认同 no_improve_limit；0=禁用，正数为启发式阶段切换)
            phase2_no_improve_limit: Phase2 完整 backward/master 刷新后连续
                LB/UB 均无改进的早停次数；0=禁用。触发时不声明收敛。
            phase1_forward_s2_time_limit: Phase1 forward Stage-2 专用时限；
                None 使用 phase1_time_limit，仍受外层 deadline 约束。
            lambda_level: Level Set lambda 参数(默认0.3)
            mu_level: Level Set mu 参数(默认0.5)
            norm_option: 范数选项(默认1, 1=L1, 2=L2)
            levelset_iter_limit: Level Set 内层最大迭代次数(默认20)
            levelset_tol: Level Set 内层收敛容差(默认1e-3)
            reset_levelset: Phase 2 每轮是否重置 Level Set cuts(默认True)
            adaptive_alpha: Level Set α自适应(默认False=固定中点, True=Delta-based自适应)
            phase1_stability_handoff: 是否启用 Phase 1→Phase 2 状态稳定切换
                (默认 True；触发时不声明 Phase 1 收敛)
            phase1_stability_window: cut archive 与 best LB/UB 连续不变的
                completed-boundary transition 数(默认 3)
            phase1_stability_min_iteration: 最早切换迭代(默认 6，且至少为
                ``phase1_stability_window + 1``)
            phase1_stagnation_window: 相对停滞切换窗口(迭代数)。最近 W 轮
                best LB 与 best UB 的相对改进都不超过
                ``phase1_stagnation_rel_tol`` 时把当前合法状态交给 Phase 2
                (默认 5；0=禁用)。这是阶段预算分配，不是收敛证明。
            phase1_stagnation_rel_tol: 停滞判定的相对改进阈值(默认 0：W 轮内 LB、UB 均完全不变)
            phase1_stagnation_min_iteration: 允许停滞切换的最早迭代(默认 8)
            phase2_inner_s3_rounds: Phase 2 backward 中，固定 Stage-1 试探点
                时 Stage2↔Stage3 内层轮数上限(默认 30；0=禁用)。每轮在
                refresh 后的 alpha 上生成 S3 Lagrangian cut 并重解 S2。
                路径/theta 残差和固定车队 S2 的认证区间分别检查；前者小
                不代表后者已充分搜索。预算或轮数耗尽时可以受限返回，
                但不据此宣称收敛。
            phase2_inner_s3_time_limit: 内层轮总时限(秒，默认 300)
            phase2_s2_cut_time_limit: 每轮 S2 Level Set 分离共享时限，
                同时受 Phase 2 外层剩余预算约束(秒，默认 0 不另加整轮限制)。
            phase2_lazy_threshold: Phase 2 Stage-2 模型中，每个 Stage-3
                successor 的 S3->S2 cut 数不超过该阈值时装成显式行，否则走
                lazy callback(默认 256；0=全部 lazy)。theta 占 Stage-2 目标
                的大部分，lazy 时根 LP 对 theta 没有任何信息：C50 冻结状态
                (每个 successor 7 条 cut)同一节点显式行 4-7 s，全 lazy
                166 s 或 300 s 超时。只有 archive 很大时才需要 lazy 保护 LP。
            phase2_no_new_cut_limit: Phase 2 连续多少轮 backward 未改变 cut
                archive 后检查是否增加 refinement 预算(默认 3；0=禁用检查)。
                不作为终止条件；限时 oracle 后续仍可能找到更强的 cut。
            phase2_s2_abs_tol_kappa: Phase 2 所有 Stage-2 MIP(forward、
                refresh、S2->S1 fleet oracle piece)共用的绝对容差
                ε = max(floor_share·phase2_tol·|UB|, kappa·(UB−LB)) / W
                中的 kappa(默认 0.5，控制请求的精度而非实际误差保证)；
                W = Σ Stage-2 节点 multi_coeff(UB 累加与 Stage-1 目标对每个
                节点值函数用的权重)。kappa 与 floor_share 同时为 0 时关闭，
                恢复固定 MIPGap 合同。cut 只用认证下界，LB 始终合法；UB 是
                真实策略成本。限时或停滞缓存可以返回宽于 ε 的区间，
                必须检查实际端点；正的 ε 下限不保证最终收敛。C50 上难 piece
                的 MIP 根界离最优 230-440：kappa=0.1 在 tol=1e-6、gap 1.1%
                时 ε≈70，每个难节点 refresh/Level Set/forward 全部跑满时限；
                ε≈350 时同样的 piece 0.1-160 s 收敛，Level Set 与 forward
                直接查表。
            phase2_s2_abs_tol_floor_share: ε 下限占目标 phase2_tol·|UB| 的
                份额(默认 0.5，必须 < 1)。只有各节点实际认证区间宽度
                均不超过 ε 时，区间宽度加权和才不超过 W·ε；该界不包含
                theta 对真实路径的低估，也不等同于全局最优性 gap。
            phase2_s2_abs_tol_rel_cap: 单个 MIP 的 MIPGapAbs 另受
                rel_cap·|参考值| 限制(默认 0.02)，避免小目标节点被过度放松
                (C50 上难 piece 的 MIP 根界离最优约 230-270≈1% 目标值，
                1% 会卡在边界上)；
                只收紧停机条件，不影响 ε 的接受判定。
        """
        self.phase1_tol = phase1_tol
        if physical_forward_profile not in ('off', 'baseline', 'paced', 'pool', 'joint'):
            raise ValueError('physical_forward_profile must be off/baseline/paced/pool/joint')
        supplied = {} if physical_forward_options is None else dict(physical_forward_options)
        unknown = set(supplied) - set(PHYSICAL_FORWARD_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown physical-forward options: {sorted(unknown)}')
        physical = dict(PHYSICAL_FORWARD_DEFAULTS, **supplied)
        for name, default in PHYSICAL_FORWARD_DEFAULTS.items():
            value = physical[name]
            if isinstance(value, bool):
                raise ValueError(f'physical {name} must be numeric, not boolean')
            if isinstance(default, int):
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f'physical {name} must be a nonnegative integer')
            else:
                value = float(value)
                if not 0. <= value < float('inf'):
                    raise ValueError(f'physical {name} must be finite and nonnegative')
                physical[name] = value
        if physical['pool_max_routes_per_node_facility'] < 1:
            raise ValueError('physical pool route limit must be positive')
        if not 1 <= physical['joint_max_concurrent'] <= 2:
            raise ValueError('physical joint concurrency must be 1 or 2')
        if physical['joint_max_nodes_per_epoch'] > 2 or physical['route_lp_max_calls_per_epoch'] > 4:
            raise ValueError('V1 supports at most two joint nodes and four route LPs per epoch')
        if physical_forward_profile != 'off' and not 1 <= num_processes <= 12:
            raise ValueError('physical-forward profiles require a total worker budget of 1..12')
        self.physical_forward_profile = physical_forward_profile
        self.physical_forward_options = physical
        if physical_cg_profile not in ('off','baseline','paced','pool','cg'):
            raise ValueError('physical_cg_profile must be off/baseline/paced/pool/cg')
        cg_supplied={} if physical_cg_options is None else dict(physical_cg_options)
        unknown=set(cg_supplied)-set(PHYSICAL_CG_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown physical-CG options: {sorted(unknown)}')
        cg=dict(PHYSICAL_CG_DEFAULTS,**cg_supplied)
        for name,default in PHYSICAL_CG_DEFAULTS.items():
            value=cg[name]
            if isinstance(value,bool): raise ValueError(f'physical CG {name} must be numeric')
            if isinstance(default,int):
                if not isinstance(value,int) or value<0: raise ValueError(f'physical CG {name} must be nonnegative integer')
            else:
                value=float(value)
                if not 0<=value<float('inf'): raise ValueError(f'physical CG {name} must be finite and nonnegative')
                cg[name]=value
        if cg['max_joint_nodes']>2:
            raise ValueError('physical CG supports at most two nodes per epoch')
        if physical_cg_profile not in ('off','baseline') and not 1<=num_processes<=12:
            raise ValueError('physical CG profiles require 1..12 total workers')
        if physical_cg_profile!='off' and physical_forward_profile!='off':
            raise ValueError('legacy physical-forward and V2 physical-CG profiles are mutually exclusive')
        self.physical_cg_profile=physical_cg_profile
        self.physical_cg_options=cg
        self.lrp_static_bounds = bool(lrp_static_bounds)
        preseed_seconds = float(phase1_preseed_time_limit)
        if isinstance(phase1_preseed_time_limit, bool) or not 0. <= preseed_seconds < float('inf'):
            raise ValueError('phase1_preseed_time_limit must be finite and nonnegative')
        try:
            preseed_rounds = int(phase1_preseed_rounds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('phase1_preseed_rounds must be a nonnegative integer') from exc
        if isinstance(phase1_preseed_rounds, bool) or preseed_rounds < 0 or preseed_rounds != phase1_preseed_rounds:
            raise ValueError('phase1_preseed_rounds must be a nonnegative integer')
        self.phase1_preseed_time_limit = preseed_seconds
        self.phase1_preseed_rounds = preseed_rounds
        phase1_mip_gap = float(phase1_mip_gap)
        if not 0.0 <= phase1_mip_gap < float("inf"):
            raise ValueError("phase1_mip_gap must be finite and nonnegative")
        self.phase1_mip_gap = phase1_mip_gap
        self.phase1_s2_mipgap_schedule = bool(phase1_s2_mipgap_schedule)
        for name, value in (('phase1_s2_mipgap_early', phase1_s2_mipgap_early),
                            ('phase1_s2_mipgap_late', phase1_s2_mipgap_late)):
            value = float(value)
            if not 0.0 <= value < float('inf'):
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, value)
        if (isinstance(phase1_s2_mipgap_switch, bool)
                or int(phase1_s2_mipgap_switch) != phase1_s2_mipgap_switch):
            raise ValueError('phase1_s2_mipgap_switch must be an integer')
        self.phase1_s2_mipgap_switch = int(phase1_s2_mipgap_switch)
        self.phase2_tol = phase2_tol
        self.phase1_iter_limit = phase1_iter_limit
        if isinstance(phase2_iter_limit, bool) or int(phase2_iter_limit) != phase2_iter_limit or phase2_iter_limit < 0:
            raise ValueError("phase2_iter_limit must be a nonnegative integer; zero disables the limit")
        self.phase2_iter_limit = int(phase2_iter_limit)
        fallback_sub_time_limit = 1800.0 if sub_time_limit is None else sub_time_limit
        self.phase1_time_limit = (phase1_time_limit if phase1_time_limit is not None
                                 else phase1_sub_time_limit if phase1_sub_time_limit is not None
                                 else fallback_sub_time_limit)
        self.phase1_total_time_limit = phase1_total_time_limit
        self.phase2_time_limit = (phase2_time_limit if phase2_time_limit is not None
                                 else phase2_sub_time_limit if phase2_sub_time_limit is not None
                                 else fallback_sub_time_limit)
        self.phase2_total_time_limit = phase2_total_time_limit
        self.phase1_forward_s2_time_limit = phase1_forward_s2_time_limit
        if phase15_backend not in ("routeopt", "gurobi", "none"):
            raise ValueError("phase15_backend must be routeopt, gurobi or none")
        phase15_limit = float(phase15_total_time_limit)
        if not 0.0 <= phase15_limit < float("inf"):
            raise ValueError("phase15_total_time_limit must be finite and nonnegative")
        self.phase15_backend = phase15_backend
        self.phase15_total_time_limit = phase15_limit
        if phase15_schedule not in ("once", "on_demand", "off"):
            raise ValueError("phase15_schedule must be once, on_demand or off")
        phase15_streak = int(phase15_on_demand_streak)
        if phase15_streak < 1:
            raise ValueError("phase15_on_demand_streak must be positive")
        phase15_per_call = float(phase15_per_call_time_limit)
        if not 0.0 < phase15_per_call < float("inf"):
            raise ValueError("phase15_per_call_time_limit must be finite and positive")
        self.phase15_schedule = phase15_schedule
        self.phase15_on_demand_streak = phase15_streak
        self.phase15_per_call_time_limit = phase15_per_call
        if phase15_backend == 'routeopt':
            raise ValueError('LRP physical pricing supports gurobi; the investment RouteOpt adapter is incompatible')
        late_total = float(phase15_late_total_time_limit)
        if not 0.0 <= late_total < float('inf'):
            raise ValueError('phase15_late_total_time_limit must be finite and nonnegative')
        if phase15_backend == 'none' and (phase15_schedule != 'off' or phase15_limit or late_total):
            raise ValueError('A disabled physical seed requires schedule=off and zero budgets')
        self.phase15_late_total_time_limit = late_total
        self.phase15_late_per_call_time_limit = float(phase15_late_per_call_time_limit)
        self.phase15_late_window = int(phase15_late_window)
        self.phase15_late_relative_gain = float(phase15_late_relative_gain)
        if not 0.0 < self.phase15_late_per_call_time_limit < float('inf'):
            raise ValueError('phase15_late_per_call_time_limit must be positive and finite')
        if self.phase15_late_window < 1 or self.phase15_late_window != phase15_late_window:
            raise ValueError('phase15_late_window must be a positive integer')
        if not 0.0 <= self.phase15_late_relative_gain < float('inf'):
            raise ValueError('phase15_late_relative_gain must be finite and nonnegative')
        if phase15_late_strategy not in ('current', 'residual', 'a_then_b'):
            raise ValueError('invalid phase15_late_strategy')
        self.phase15_late_strategy = phase15_late_strategy
        self.num_processes = num_processes
        self.no_improve_limit = no_improve_limit
        self.phase1_no_improve_limit = (
            no_improve_limit if phase1_no_improve_limit is None else phase1_no_improve_limit
        )
        if (isinstance(phase2_no_improve_limit, bool)
                or int(phase2_no_improve_limit) != phase2_no_improve_limit
                or phase2_no_improve_limit < 0):
            raise ValueError('phase2_no_improve_limit must be a nonnegative integer; zero disables it')
        self.phase2_no_improve_limit = int(phase2_no_improve_limit)
        self.lambda_level = lambda_level
        self.mu_level = mu_level
        self.norm_option = norm_option
        self.levelset_iter_limit = levelset_iter_limit
        self.levelset_tol = levelset_tol
        self.reset_levelset = reset_levelset
        self.adaptive_alpha = adaptive_alpha
        # Phase1 专用（不影响 Phase2）
        self.phase1_lazy_threshold = phase1_lazy_threshold
        self.phase1_forward_period_dedup = bool(phase1_forward_period_dedup)
        # Phase-1→2 stability handoff
        self.phase1_stability_handoff = bool(phase1_stability_handoff)
        if isinstance(phase1_stability_window, bool):
            raise ValueError(
                "phase1_stability_window must be a positive integer"
            )
        stability_window = int(phase1_stability_window)
        if (
            stability_window < 1
            or stability_window != phase1_stability_window
        ):
            raise ValueError(
                "phase1_stability_window must be a positive integer"
            )
        if isinstance(phase1_stability_min_iteration, bool):
            raise ValueError(
                "phase1_stability_min_iteration must be a positive integer"
            )
        stability_min_iteration = int(phase1_stability_min_iteration)
        if (
            stability_min_iteration < 1
            or stability_min_iteration != phase1_stability_min_iteration
        ):
            raise ValueError(
                "phase1_stability_min_iteration must be a positive integer"
            )
        self.phase1_stability_window = stability_window
        # W unchanged transitions need W+1 boundary samples
        self.phase1_stability_min_iteration = max(
            stability_window + 1,
            stability_min_iteration,
        )
        # Relative-progress phase-allocation gate (complements bitwise gate)
        stagnation_window = int(phase1_stagnation_window)
        if stagnation_window < 0 or stagnation_window != phase1_stagnation_window:
            raise ValueError("phase1_stagnation_window must be a non-negative integer")
        stagnation_rel_tol = float(phase1_stagnation_rel_tol)
        if not (stagnation_rel_tol >= 0.0):
            raise ValueError("phase1_stagnation_rel_tol must be non-negative")
        stagnation_min_iteration = int(phase1_stagnation_min_iteration)
        if stagnation_min_iteration < 1:
            raise ValueError("phase1_stagnation_min_iteration must be positive")
        self.phase1_stagnation_window = stagnation_window
        self.phase1_stagnation_rel_tol = stagnation_rel_tol
        self.phase1_stagnation_min_iteration = max(
            stagnation_window + 1, stagnation_min_iteration
        )
        # Phase2 专用
        self.phase2_lazy_threshold = phase2_lazy_threshold
        self.phase2_forward_period_dedup = bool(phase2_forward_period_dedup)
        inner_rounds = int(phase2_inner_s3_rounds)
        if inner_rounds < 0 or inner_rounds != phase2_inner_s3_rounds:
            raise ValueError("phase2_inner_s3_rounds must be a non-negative integer")
        inner_time_limit = float(phase2_inner_s3_time_limit)
        if not (inner_time_limit >= 0.0):
            raise ValueError("phase2_inner_s3_time_limit must be non-negative")
        self.phase2_inner_s3_rounds = inner_rounds
        self.phase2_inner_s3_time_limit = inner_time_limit
        s2_cut_time_limit = float(phase2_s2_cut_time_limit)
        if not 0.0 <= s2_cut_time_limit < float('inf'):
            raise ValueError('phase2_s2_cut_time_limit must be nonnegative and finite')
        self.phase2_s2_cut_time_limit = s2_cut_time_limit
        no_new_cut_limit = int(phase2_no_new_cut_limit)
        if no_new_cut_limit < 0 or no_new_cut_limit != phase2_no_new_cut_limit:
            raise ValueError("phase2_no_new_cut_limit must be a non-negative integer")
        self.phase2_no_new_cut_limit = no_new_cut_limit
        kappa = float(phase2_s2_abs_tol_kappa)
        if not (kappa >= 0.0):
            raise ValueError("phase2_s2_abs_tol_kappa must be non-negative")
        floor_share = float(phase2_s2_abs_tol_floor_share)
        if not (0.0 <= floor_share < 1.0):
            raise ValueError("phase2_s2_abs_tol_floor_share must lie in [0, 1)")
        rel_cap = float(phase2_s2_abs_tol_rel_cap)
        if not (rel_cap >= 0.0):
            raise ValueError("phase2_s2_abs_tol_rel_cap must be non-negative")
        self.phase2_s2_abs_tol_kappa = kappa
        self.phase2_s2_abs_tol_floor_share = floor_share
        self.phase2_s2_abs_tol_rel_cap = rel_cap
        self.phase_forward_reuse = bool(phase_forward_reuse)

    @property
    def phase1_sub_time_limit(self):
        """Original spelling; serialization keeps only the canonical LRP key."""
        return self.phase1_time_limit

    @phase1_sub_time_limit.setter
    def phase1_sub_time_limit(self, value):
        self.phase1_time_limit = value

    @property
    def phase2_sub_time_limit(self):
        return self.phase2_time_limit

    @phase2_sub_time_limit.setter
    def phase2_sub_time_limit(self, value):
        self.phase2_time_limit = value

    def get(self, key, default=None):
        """支持字典式访问"""
        return getattr(self, key, default)

    def to_dict(self):
        """转换为字典"""
        return {
            'physical_cg_profile': getattr(self,'physical_cg_profile','off'),
            'physical_cg_options': dict(getattr(self,'physical_cg_options',PHYSICAL_CG_DEFAULTS)),
            'physical_forward_profile': getattr(self, 'physical_forward_profile', 'off'),
            'physical_forward_options': dict(getattr(self, 'physical_forward_options', PHYSICAL_FORWARD_DEFAULTS)),
            'lrp_static_bounds': self.lrp_static_bounds,
            'phase1_preseed_time_limit': getattr(self, 'phase1_preseed_time_limit', 0.),
            'phase1_preseed_rounds': getattr(self, 'phase1_preseed_rounds', 2),
            'phase1_tol': self.phase1_tol,
            'phase1_mip_gap': self.phase1_mip_gap,
            'phase1_s2_mipgap_schedule': self.phase1_s2_mipgap_schedule,
            'phase1_s2_mipgap_early': self.phase1_s2_mipgap_early,
            'phase1_s2_mipgap_late': self.phase1_s2_mipgap_late,
            'phase1_s2_mipgap_switch': self.phase1_s2_mipgap_switch,
            'phase2_tol': self.phase2_tol,
            'phase1_iter_limit': self.phase1_iter_limit,
            'phase2_iter_limit': self.phase2_iter_limit,
            'phase1_time_limit': self.phase1_time_limit,
            'phase1_forward_s2_time_limit': self.phase1_forward_s2_time_limit,
            'phase1_total_time_limit': self.phase1_total_time_limit,
            'phase2_time_limit': self.phase2_time_limit,
            'phase2_total_time_limit': self.phase2_total_time_limit,
            'phase15_backend': self.phase15_backend,
            'phase15_total_time_limit': self.phase15_total_time_limit,
            'phase15_schedule': self.phase15_schedule,
            'phase15_on_demand_streak': self.phase15_on_demand_streak,
            'phase15_per_call_time_limit': self.phase15_per_call_time_limit,
            'phase15_late_total_time_limit': self.phase15_late_total_time_limit,
            'phase15_late_per_call_time_limit': self.phase15_late_per_call_time_limit,
            'phase15_late_window': self.phase15_late_window,
            'phase15_late_relative_gain': self.phase15_late_relative_gain,
            'phase15_late_strategy': self.phase15_late_strategy,
            'num_processes': self.num_processes,
            'no_improve_limit': self.no_improve_limit,
            'phase1_no_improve_limit': self.phase1_no_improve_limit,
            'phase2_no_improve_limit': self.phase2_no_improve_limit,
            'lambda_level': self.lambda_level,
            'mu_level': self.mu_level,
            'norm_option': self.norm_option,
            'levelset_iter_limit': self.levelset_iter_limit,
            'levelset_tol': self.levelset_tol,
            'reset_levelset': self.reset_levelset,
            'adaptive_alpha': self.adaptive_alpha,
            'phase1_lazy_threshold': self.phase1_lazy_threshold,
            'phase1_forward_period_dedup': self.phase1_forward_period_dedup,
            'phase1_stability_handoff': self.phase1_stability_handoff,
            'phase1_stability_window': self.phase1_stability_window,
            'phase1_stability_min_iteration': (
                self.phase1_stability_min_iteration
            ),
            'phase1_stagnation_window': self.phase1_stagnation_window,
            'phase1_stagnation_rel_tol': self.phase1_stagnation_rel_tol,
            'phase1_stagnation_min_iteration': (
                self.phase1_stagnation_min_iteration
            ),
            'phase2_lazy_threshold': self.phase2_lazy_threshold,
            'phase2_forward_period_dedup': self.phase2_forward_period_dedup,
            'phase2_inner_s3_rounds': self.phase2_inner_s3_rounds,
            'phase2_inner_s3_time_limit': self.phase2_inner_s3_time_limit,
            'phase2_s2_cut_time_limit': self.phase2_s2_cut_time_limit,
            'phase2_no_new_cut_limit': self.phase2_no_new_cut_limit,
            'phase2_s2_abs_tol_kappa': self.phase2_s2_abs_tol_kappa,
            'phase2_s2_abs_tol_floor_share': self.phase2_s2_abs_tol_floor_share,
            'phase2_s2_abs_tol_rel_cap': self.phase2_s2_abs_tol_rel_cap,
            'phase_forward_reuse': self.phase_forward_reuse,
        }


class SDDPResult:
    """
    SDDP算法结果类

    封装算法求解的结果,与原始pickle格式兼容。

    Attributes:
        Vstar: 最优值(下界)
        x_best: 最优解(变量值字典)
        cut_lag: 生成的Lagrangian切割
        LB_list: 下界历史
        UB_list: 上界历史
        time_list: 每次迭代的时间
        converged: 是否收敛
        total_time: 总求解时间
    """

    def __init__(self, Vstar, x_best, cut_lag, LB_list, UB_list, time_list,
                 converged=False, total_time=0.0):
        """
        初始化结果对象

        Args:
            Vstar: 最优值
            x_best: 最优解
            cut_lag: 切割字典
            LB_list: 下界列表
            UB_list: 上界列表
            time_list: 时间列表
            converged: 是否收敛
            total_time: 总时间
        """
        self.Vstar = Vstar
        self.x_best = x_best
        self.cut_lag = cut_lag
        self.LB_list = LB_list
        self.UB_list = UB_list
        self.time_list = time_list
        self.converged = converged
        self.total_time = total_time

    def to_dict(self):
        """
        转换为字典格式(与原始pickle格式兼容)

        Returns:
            结果字典
        """
        return {
            'Vstar': self.Vstar,
            'x_best': self.x_best,
            'cut_lag': self.cut_lag,
            'LB_list': self.LB_list,
            'UB_list': self.UB_list,
            'time_list': self.time_list,
            'converged': self.converged,
            'total_time': self.total_time
        }

    @classmethod
    def from_dict(cls, result_dict):
        """
        从字典创建结果对象

        Args:
            result_dict: 结果字典

        Returns:
            SDDPResult实例
        """
        return cls(
            Vstar=result_dict.get('Vstar'),
            x_best=result_dict.get('x_best', {}),
            cut_lag=result_dict.get('cut_lag', {}),
            LB_list=result_dict.get('LB_list', []),
            UB_list=result_dict.get('UB_list', []),
            time_list=result_dict.get('time_list', []),
            converged=result_dict.get('converged', False),
            total_time=result_dict.get('total_time', 0.0)
        )
