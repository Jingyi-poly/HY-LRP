"""Compare the production LRP SBC/Level Set solver with the independent two-stage EF."""
from __future__ import annotations

import math
import os
from pathlib import Path

from main import (build_parser, config_from_args, _instance_int, _validate_lrp_data_settings,
                  load_algorithm_config, LRP_INSTANCE as DEFAULT_COMPARE_INSTANCE)
from core.instance import LRPConfig, LRPInstance
from core.run_logging import console_log_path, setup_run_logging
from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent
from models.extensive_model_builder import ExtensiveModelBuilder
from setup.reporting import (gap_pct, merge_results, phase_bounds, prepare_output_directory,
                             print_algorithm_config, print_instance_summary,
                             save_outputs, write_json)
from setup.runner import solve_phase1, solve_two_phase
from setup.solver_config import configure_compare_exact_backend


def _bound_tolerance(*values):
    return max(1e-6, 1e-9 * max([1.0] + [abs(float(v)) for v in values if v is not None]))


def compare_bounds(phase_results, ef, policy_audit, config):
    """Check actual certified bounds; an EF time limit is not an optimality proof."""
    ef_ub, ef_lb = ef.get("objective"), ef.get("ObjBound")
    violations = []
    for phase, result in phase_results.items():
        for iteration, lb in enumerate(result.get("LB_list", []), 1):
            if not math.isfinite(float(lb)):
                violations.append({"phase": phase, "iteration": iteration, "kind": "nonfinite_lower_bound"})
            elif ef_ub is not None and lb > ef_ub + _bound_tolerance(lb, ef_ub):
                violations.append({"phase": phase, "iteration": iteration, "kind": "sddp_lb_above_ef_feasible_ub", "value": lb})
        for iteration, ub in enumerate(result.get("UB_list", []), 1):
            if not math.isfinite(float(ub)):
                continue  # No feasible policy may yet have been returned.
            if ef_lb is not None and ub < ef_lb - _bound_tolerance(ub, ef_lb):
                violations.append({"phase": phase, "iteration": iteration, "kind": "sddp_ub_below_ef_certified_lb", "value": ub})
    result = phase_results["phase2"]
    lb, ub = phase_bounds(result)
    actual_ub = policy_audit.get("feasible_upper_bound") if policy_audit is not None else None
    policy_matches = actual_ub is not None and ub is not None and abs(actual_ub - ub) <= _bound_tolerance(actual_ub, ub)
    if not policy_matches:
        violations.append({"kind": "recorded_upper_bound_does_not_match_returned_policy",
                           "recorded": ub, "recomputed": actual_ub})
    if lb is not None and ub is not None and minimization_bounds_inverted(lb, ub):
        violations.append({"kind": "sddp_bounds_inverted", "lower": lb, "upper": ub})
    ef_optimal = bool(ef.get("optimality_certified", False))
    tolerance = max(_bound_tolerance(ef_ub, ub), config.phase2_tol * max(1.0, abs(lb or 0.0)))
    objectives_match = ef_optimal and ub is not None and abs(ub - ef_ub) <= tolerance
    converged = bool(result.get("converged", False)) and lb is not None and ub is not None and (
        minimization_gap_percent(lb, ub) <= config.phase2_tol * 100.0
    )
    if bool(result.get("converged", False)) and not converged:
        violations.append({"kind": "invalid_sddp_convergence_claim"})
    passed = not violations and ef_optimal and converged and objectives_match and policy_matches
    return {
        "status": "PASS" if passed else ("FAIL" if violations else "INCONCLUSIVE"),
        "passed": passed, "bounds_valid": not violations, "violations": violations,
        "ef_optimality_certified": ef_optimal, "sddp_converged": converged,
        "objectives_match_within_tolerance": objectives_match,
        "comparison_objective_tolerance": tolerance,
        "sddp_lower_bound": lb, "sddp_upper_bound": ub,
        "ef_lower_bound": ef_lb, "ef_upper_bound": ef_ub,
        "absolute_objective_difference": None if ef_ub is None or ub is None else abs(ub - ef_ub),
        "policy_upper_bound": actual_ub, "policy_matches_recorded_upper_bound": policy_matches,
        "all_reported_iterations_checked": True,
        "checked_lower_bounds": sum(len(r.get("LB_list", [])) for r in phase_results.values()),
        "checked_upper_bounds": sum(len(r.get("UB_list", [])) for r in phase_results.values()),
        "decision_equality_required": False,
    }


# 与原 investment compare 一样，在这里改比较参数；显式命令/环境设置优先。
# 数据源独立于 main，在此选择；None 保留源数据的全部客户/设施。
# berlin52 预生成最大约 I3_J8；I=3/J=10 用 kroA100 源再 subset。
LRP_INSTANCE = Path(DEFAULT_COMPARE_INSTANCE).with_name("kroA100_I5_J20_T3_S3_scale")
NUM_CUSTOMERS = 8
NUM_FACILITIES = 3
COMPARE_PARAMS = {
    # 恢复原 investment 的 instance 参数块：在这里改数据规模。
    # None 保留源数据值；T/S 增大时按原数据生成配方扩展。
    "instance": {
        "instance": LRP_INSTANCE,
        "num_customers": NUM_CUSTOMERS,
        "num_facilities": NUM_FACILITIES,
        "T": 5,              # delivery periods；改这里或设 VRP_T
        "num_scenarios": 3,  # scenarios；改这里或设 VRP_NUM_SCENARIOS
        "scenario_seed": 42,
        "location_periods": (1, 3, 5),  # 多次开仓决策日（须含1、严格递增、≤T）；None 保留源日期
    },
    "ef_mip_gap": 1e-8,
    "ef_time_limit_s": 7200.0,
    "num_processes": int(os.environ.get("LRP_NUM_PROCESSES", os.environ.get("VRP_NUM_PROCESSES",
                         os.environ.get("LRP_CPU_BUDGET", os.environ.get("VRP_CPU_BUDGET", "12"))))),
    "phase1_tol": 1e-2,
    "phase2_tol": 1e-6,
    "levelset_tol": 1e-6,
}
_COMPARE_TOLERANCE_ENV = {
    "phase1_tol": "VRP_PHASE1_TOL",
    "phase2_tol": "VRP_PHASE2_TOL",
    "levelset_tol": "VRP_LEVELSET_TOL",
}


def _comparison_env(name, default=""):
    return os.environ.get("LRP_" + name, os.environ.get("VRP_" + name, default))


def _comparison_flag(name):
    return str(_comparison_env(name, "0")).lower() not in ("", "0", "false", "no", "off")


def comparison_instance(num_customers=None, num_facilities=None, *, instance=None, out=None,
                        T=None, num_scenarios=None, scenario_seed=None, location_periods=None):
    """客户前缀与物理设施选择；数据参数集中在原 instance 参数块。"""
    _validate_lrp_data_settings()
    inst_params = dict(COMPARE_PARAMS["instance"])
    for key, explicit in (("T", T), ("num_scenarios", num_scenarios),
                          ("scenario_seed", scenario_seed),
                          ("num_customers", num_customers), ("num_facilities", num_facilities)):
        inst_params[key] = _instance_int(key, explicit, inst_params.get(key))
    for key, value in (("instance", instance), ("out", out),
                       ("num_customers", num_customers), ("num_facilities", num_facilities),
                       ("location_periods", location_periods)):
        if value is not None:
            inst_params[key] = value
    return LRPConfig(**inst_params)


def _apply_comparison_tolerances(config, *, explicit=None):
    explicit = explicit or {}
    for attribute, env_name in _COMPARE_TOLERANCE_ENV.items():
        if explicit.get(attribute) is not None:
            value = explicit[attribute]
        else:
            value = os.environ.get("LRP_" + env_name[4:],
                                   os.environ.get(env_name, COMPARE_PARAMS[attribute]))
        setattr(config, attribute, float(value))
    return config


def configure_comparison_solver(inst):
    """保留原配置调用点；LRP 使用已验证的物理设施后端。"""
    override = _comparison_env("COMPARE_S2_FORWARD_SOLVER").strip().lower()
    if override:
        if override not in ('gurobi', 'bpc', 'bp', 'native', 'auto'):
            raise ValueError('COMPARE_S2_FORWARD_SOLVER supports gurobi, bpc, or auto for LRP')
        os.environ['LRP_S2_FORWARD_SOLVER'] = override
    # A requested all-Gurobi comparison has priority over the S2-only switch.
    configure_compare_exact_backend(inst.prob_data.n)


def _comparison_config(args=None):
    if args is None:
        config = load_algorithm_config()
        explicit = {}
    else:
        config = config_from_args(args)
        explicit = dict(phase1_tol=args.phase1_tol, phase2_tol=args.tol,
                        levelset_tol=args.levelset_tol)
    _apply_comparison_tolerances(config, explicit=explicit)
    if args is None or args.processes is None:
        config.num_processes = int(os.environ.get("LRP_NUM_PROCESSES",
                            os.environ.get("VRP_NUM_PROCESSES", COMPARE_PARAMS["num_processes"])))
    return config


def _compare_size_tag(cfg):
    # Physical facility IDs are not interchangeable vehicle labels.
    return getattr(cfg, "_output_tag", cfg.instance_tag())


def _compare_log_path(out_dir, cfg):
    # _output_tag already includes any requested suffix, exactly once.
    return Path(out_dir) / ("compare_" + _compare_size_tag(cfg) + ".txt")


def _print_compare_metrics(*, phase1_time, phase2_time, sddp_lb, sddp_ub, ef=None, phase2_executed=True):
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print("\n  求解时间:")
    print(f"    SDDP Phase1:         {phase1_time:>10,.1f}s")
    if phase2_executed:
        print(f"    SDDP Phase2:         {phase2_time:>10,.1f}s")
    else:
        print("    SDDP Phase2:         本次未运行")
    print(f"    SDDP 合计:           {phase1_time + phase2_time:>10,.1f}s")
    print(f"  SDDP LB:             {sddp_lb if sddp_lb is not None else '无有效下界'}")
    print(f"  SDDP UB:             {sddp_ub if sddp_ub is not None else '无可行策略'}")
    label = 'Phase2' if phase2_executed else 'Phase1'
    print(f"  {label} Gap:          {gap_pct(sddp_lb, sddp_ub)}")
    if ef is not None:
        print(f"    EF (Gurobi):        {ef['Runtime']:>10,.1f}s")
        print(f"  EF status:           {ef['status']}")
        print(f"  EF raw ObjVal:       {ef.get('RawObjVal')}")
        print(f"  EF LB (ObjBound):    {ef.get('ObjBound')}")
        print(f"  EF certified UB:     {ef.get('objective')}")
        print(f"  EF Gap (LB-UB):      {gap_pct(ef.get('ObjBound'), ef.get('objective'))}")


def _print_policy_comparison(ef_audit, policy_audit):
    """按同一物理设施逐项展示；不同最优解不因此判为模型不一致。"""
    if not ef_audit or not policy_audit:
        return
    print("\n" + "=" * 80)
    print("Stage 1: 设施开仓状态（物理设施编号不可置换）")
    print("=" * 80)
    print(f"  {'设施':>6} {'区间':>6} {'EF_A':>8} {'SDDP_A':>8} {'相同':>6}")
    for i, (ef_row, sddp_row) in enumerate(zip(ef_audit['A'], policy_audit['A'])):
        for k, (a, b) in enumerate(zip(ef_row, sddp_row)):
            print(f"  {i:>6} {k:>6} {a:>8} {b:>8} {'是' if a == b else '否':>6}")
    print("\nStage 2+3: 客户分配、外包与路径成本（仅展开差异节点）")
    for ef_node in ef_audit['nodes']:
        t, s = ef_node['period'], ef_node['scenario']
        node = policy_audit['nodes'][f'{t},{s}']
        assigned = [[j for j, flag in enumerate(row) if flag] for row in node['alpha']]
        outsourced = [j for j, flag in enumerate(node['e']) if flag]
        differences = []
        for i, ef_assigned in enumerate(ef_node['assigned']):
            if ef_assigned != assigned[i]:
                differences.append(f"    设施 {i} 客户组: EF={ef_assigned}  SDDP={assigned[i]}")
            ef_arcs = sorted(zip(ef_node['routes'][i], ef_node['routes'][i][1:]))
            sddp_arcs = sorted(tuple(arc) for arc in node['tours'][i]['arcs'])
            if ef_arcs != sddp_arcs:
                differences.append(f"    设施 {i} 路径: EF={ef_node['routes'][i]}  SDDP arcs={sddp_arcs}")
        if ef_node['outsourced'] != outsourced:
            differences.append(f"    外包: EF={ef_node['outsourced']}  SDDP={outsourced}")
        for field, label in (('routing_cost', '路线成本'), ('outsourcing_cost', '外包成本')):
            if abs(ef_node[field] - node[field]) > _bound_tolerance(ef_node[field], node[field]):
                differences.append(f"    {label}: EF={ef_node[field]:.6f}  SDDP={node[field]:.6f}")
        ef_total = ef_node['routing_cost'] + ef_node['outsourcing_cost']
        sddp_total = node['routing_cost'] + node['outsourcing_cost']
        if differences:
            print(f"  ★ 节点 (t={t}, s={s}) — 决策/成本差异（不同最优解允许不同决策）:")
            for detail in differences:
                print(detail)
            print(f"    节点成本: EF={ef_total:.6f}  SDDP={sddp_total:.6f}")
        else:
            print(f"  节点 (t={t}, s={s}): ✓ 完全匹配  成本={ef_total:.6f}")
    print(f"  设施成本: EF={ef_audit['facility_cost']:.6f}  SDDP={policy_audit['facility_cost']:.6f}")
    print(f"  期望路线成本: EF={ef_audit['expected_routing_cost']:.6f}  SDDP={policy_audit['expected_routing']:.6f}")
    print(f"  期望外包成本: EF={ef_audit['expected_outsourcing_cost']:.6f}  SDDP={policy_audit['expected_outsourcing']:.6f}")


def run_comparison(instance=None, out=None, *, config=None, ef_time_limit=None,
                   ef_threads=None, ef_mip_gap=None, num_customers=None,
                   num_facilities=None, T=None, num_scenarios=None,
                   scenario_seed=None, location_periods=None):
    from solvers.forward_policy_certification import certify_policy
    import numpy as np

    # Preserve the original customer-count positional call, with an optional
    # physical-facility count, and run_comparison(instance_path, output_path).
    if isinstance(instance, int):
        if num_customers is not None or num_facilities is not None:
            raise TypeError('Do not mix positional sizes with named size overrides')
        if out is not None and not isinstance(out, int):
            raise TypeError('The second positional argument after customer count is facility count')
        num_customers, num_facilities, instance, out = instance, out, None, None
    cfg = comparison_instance(num_customers, num_facilities, instance=instance, out=out,
                              T=T, num_scenarios=num_scenarios, scenario_seed=scenario_seed,
                              location_periods=location_periods)
    output = prepare_output_directory(cfg, prefix="compare")
    suffix = _comparison_env("COMPARE_LOG_SUFFIX").strip()
    # The same suffix applies to all saved outputs, as well as the console log.
    if suffix:
        cfg._output_tag = _compare_size_tag(cfg) + "_" + suffix
    if getattr(cfg, '_legacy_output', False):
        # prepare_output_directory checked the unsuffixed tag; check the final
        # name too so repeating a named comparison cannot overwrite it.
        tag = cfg._output_tag
        if any((output / name).exists() for name in
               (f'compare_{tag}.txt', f'result_{tag}.pkl', f'report_{tag}.txt')):
            from datetime import datetime
            cfg._output_tag += '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = (_compare_log_path(output, cfg) if getattr(cfg, '_legacy_output', False)
                else console_log_path(cfg))
    diagnostic_output = (output / 'diagnostics' / _compare_size_tag(cfg)
                         if getattr(cfg, '_legacy_output', False) else output)
    ef_time_limit = float(COMPARE_PARAMS['ef_time_limit_s'] if ef_time_limit is None else ef_time_limit)
    ef_mip_gap = float(COMPARE_PARAMS['ef_mip_gap'] if ef_mip_gap is None else ef_mip_gap)
    ef_threads = int(_comparison_env('EF_GRB_THREADS', '1') if ef_threads is None else ef_threads)
    if not math.isfinite(ef_time_limit) or ef_time_limit <= 0:
        raise ValueError('EF time limit must be finite and positive')
    if not math.isfinite(ef_mip_gap) or ef_mip_gap < 0 or ef_threads < 1:
        raise ValueError('EF gap must be finite/nonnegative and EF threads must be positive')
    try:
        with setup_run_logging(log_path) as log_abs:
            print(f"[compare log] {log_abs}")
            inst = LRPInstance(cfg).build()
            configure_comparison_solver(inst)
            config = _comparison_config() if config is None else config
            print_instance_summary(cfg, inst)
            print_algorithm_config(config, num_customers=inst.prob_data.shape[1])
            phase1_only = _comparison_flag("COMPARE_PHASE1_ONLY")
            if phase1_only:
                r1, t1 = solve_phase1(inst.prob_data, inst.scen_tree, config)
                # Preserve the completed Phase 1 and the usual *_SBC fields;
                # Phase 2 is explicitly empty, never a duplicate Phase 1 run.
                r2, t2 = dict(r1), 0.0
                r2['stop_reason'] = 'phase1_only'
            else:
                r1, r2, t1, t2 = solve_two_phase(inst.prob_data, inst.scen_tree, config)
            result = merge_results(cfg, r1, r2, t1, t2, config=config)
            result['phase2_executed'] = not phase1_only
            if phase1_only:
                result['phase2'] = None
                result['phase2_result'] = None
                result['phase2_time'] = None
                result['execution']['observed_node_execution'].pop('phase2', None)
                result['backend_telemetry']['phase2'] = None
                result['phase2_backend_telemetry'] = None
            result["instance_sha256"] = inst.prob_data.logical_hash()
            result["dimensions"] = dict(zip(("I", "J", "T", "L", "S"), inst.prob_data.shape))
            result["data_selection"] = inst.prob_data.metadata.get("subset")
            result['report_data'] = {k: inst.prob_data.arrays[k].tolist() for k in (
                'facility_ids', 'customer_ids', 'capacity', 'opening_cost', 'continuation_cost',
                'closing_cost', 'active', 'demand', 'scenario_prob', 'location_periods', 'period_to_interval',
                'cost_fc', 'cost_cc', 'route_cost') if k in inst.prob_data.arrays}
            policy_audit = certify_policy(inst.prob_data, inst.scen_tree, r2["x_best"]) if r2.get("x_best") else None
            result["policy_audit"] = policy_audit
            save_outputs(cfg, result)
            if phase1_only or _comparison_flag("COMPARE_SKIP_EF"):
                switch = 'COMPARE_PHASE1_ONLY' if phase1_only else 'COMPARE_SKIP_EF'
                reason = ('LRP_' if 'LRP_' + switch in os.environ else 'VRP_') + switch + '=1'
                print(f"\n[2] 跳过 {'Phase 2 与 ' if phase1_only else ''}Extensive Form ({reason})。")
                result['comparison'] = {'status': 'SKIPPED', 'passed': False, 'reason': reason}
                _print_compare_metrics(phase1_time=t1, phase2_time=t2,
                                       sddp_lb=result['lower_bound'], sddp_ub=result['upper_bound'],
                                       phase2_executed=not phase1_only)
                save_outputs(cfg, result)
                return result
            print(f"\n[2] 运行独立两阶段 Extensive Form (Gurobi, TimeLimit={ef_time_limit:.0f}s)...")
            # LogToConsole so node progress tees into the compare log; keep a copy under diagnostics.
            ef_log = diagnostic_output / "ef_gurobi.log"
            print(f"  EF progress -> compare log + {ef_log}")
            ef = ExtensiveModelBuilder(inst.prob_data).solve(
                time_limit=ef_time_limit, mip_gap=ef_mip_gap, threads=ef_threads,
                output=True, log_path=ef_log, model_path=diagnostic_output / "ef_model.lp",
            )
            try:
                if ef.get("primal") is not None:
                    np.savez_compressed(diagnostic_output / "ef_primal.npz", x=ef["primal"])
                serial_ef = {k: v for k, v in ef.items() if k not in {"model", "primal"}}
                result["ef"] = serial_ef
                result["comparison"] = compare_bounds({"phase1": r1, "phase2": r2}, ef, policy_audit, config)
                write_json(diagnostic_output / "ef_result.json", serial_ef)
                if ef.get('optimality_certified'):
                    _print_policy_comparison(ef.get('solution'), policy_audit)
                else:
                    print('  EF 尚未证最优；保留有效 LB/可行 UB，跳过最优决策逐项比较。')
                _print_compare_metrics(phase1_time=t1, phase2_time=t2,
                    sddp_lb=result['lower_bound'], sddp_ub=result['upper_bound'], ef=ef)
            finally:
                ef["model"].dispose()
            save_outputs(cfg, result)
            write_json(diagnostic_output / "comparison.json", result["comparison"])
            print(f"Comparison: {result['comparison']['status']}; "
                  f"difference={result['comparison']['absolute_objective_difference']}")
            print(f"Result directory: {output}")
            return result
    except Exception as exc:
        write_json(diagnostic_output / "error.json", {"status": "ERROR", "type": type(exc).__name__, "message": str(exc)})
        raise


def _parse_size_args(argv=None):
    parser = build_parser(__doc__)
    # Leave data defaults to COMPARE_PARAMS['instance'], not main.py's parser.
    parser.set_defaults(instance=None, customers=None, facilities=None)
    parser.add_argument('num_customers', nargs='?', type=int, help='取前 N 个客户')
    parser.add_argument('num_facilities', nargs='?', type=int, help='取前 N 个物理设施（可选第二个位置参数）')
    parser.add_argument("--ef-time-limit", type=float, default=None)
    parser.add_argument("--ef-threads", type=int, default=None)
    parser.add_argument("--ef-mip-gap", type=float, default=None)
    args = parser.parse_args(argv)
    for positional, option in (('num_customers', 'customers'), ('num_facilities', 'facilities')):
        value = getattr(args, positional)
        current = getattr(args, option)
        if value is not None:
            if current is not None and current != value:
                parser.error(f'{positional} and --{option} disagree')
            setattr(args, option, value)
    return args


def main(argv=None):
    args = _parse_size_args(argv)
    result = run_comparison(args.instance, args.out, config=_comparison_config(args),
                            ef_time_limit=args.ef_time_limit, ef_threads=args.ef_threads,
                            ef_mip_gap=args.ef_mip_gap,
                            num_customers=args.customers, num_facilities=args.facilities,
                            T=args.T, num_scenarios=args.num_scenarios,
                            scenario_seed=args.scenario_seed, location_periods=args.location_periods)
    # INCONCLUSIVE includes a normal time/iteration limit. FAIL denotes
    # inconsistent bounds or an invalid reported policy and stays nonzero.
    return 0 if result["comparison"]["status"] in {"PASS", "SKIPPED", "INCONCLUSIVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
