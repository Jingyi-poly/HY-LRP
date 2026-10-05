"""LRP bounds, policy costs and reproducible result persistence."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import math
import pickle
from pathlib import Path

from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent


def describe_stop_reason(reason):
    """Describe the recorded exit, without guessing from elapsed time or gap."""
    if not reason or reason == 'unknown':
        return '未记录（无法确定停止原因）'
    labels = {
        'gap_tolerance': '达到目标 gap',
        'phase2_total_time_limit': 'Phase 2 总时限耗尽',
        'phase1_total_time_limit': 'Phase 1 总时限耗尽',
        'iteration_limit': '达到迭代上限',
        'debug_iteration_limit': '达到调试迭代上限',
        'no_improvement': '连续无改进，未宣称收敛',
        'phase1_stability_handoff': '状态稳定，交给 Phase 2（非收敛）',
        'phase1_stagnation_handoff': '相对改善停滞，交给 Phase 2（非收敛）',
    }
    return f'{labels[reason]} [{reason}]' if reason in labels else str(reason)


def phase_bounds(result):
    lower = [float(v) for v in result.get("LB_list", []) if v is not None and math.isfinite(float(v))]
    if result.get("initial_lb") is not None and math.isfinite(float(result["initial_lb"])):
        lower.append(float(result["initial_lb"]))
    if not lower and result.get("Vstar") is not None and math.isfinite(float(result["Vstar"])):
        lower.append(float(result["Vstar"]))
    upper = [float(v) for v in result.get("UB_list", []) if v is not None and math.isfinite(float(v))]
    if not upper:
        # A resumed Phase 2 can hit its deadline before the first forward
        # pass. Its independently audited inherited policy is still an UB;
        # no artificial history entry is needed to display that certificate.
        audited_upper = (result.get('policy_audit') or {}).get('feasible_upper_bound')
        if audited_upper is not None and math.isfinite(float(audited_upper)):
            upper.append(float(audited_upper))
    return max(lower, default=None), min(upper, default=None)


def relative_gap(lb, ub):
    if lb is None or ub is None or not (math.isfinite(lb) and math.isfinite(ub)):
        return None
    value = minimization_gap_percent(lb, ub) / 100.0
    return value if math.isfinite(value) else None


def gap_pct(lb, ub):
    if lb is not None and ub is not None and minimization_bounds_inverted(lb, ub):
        return 'LB>UB'
    value = relative_gap(lb, ub)
    return '  --' if value is None else f'{100 * value:.4f}%'


def print_phase_result(label, result, elapsed):
    print(f'\n{label} 结束')
    if result is None:
        print('  - 本次未运行；未记录该阶段的独立用时和 gap。')
        return
    lb, ub = phase_bounds(result)
    def number(value):
        return '--' if value is None else f'{value:.2f}'
    invalid = lb is not None and ub is not None and minimization_bounds_inverted(lb, ub)
    convergence = ('证书无效（LB > UB）' if invalid else
                   '内部 LB/UB 已收敛' if result.get('converged') and relative_gap(lb, ub) is not None else
                   '未收敛')
    print(f'  - 最佳下界: {number(lb)}')
    print(f"  - 迭代次数: {result.get('iterations', 0)}")
    print(f'  - 求解时间: {number(elapsed)} 秒')
    print(f"  - 停止原因: {describe_stop_reason(result.get('stop_reason'))}")
    print(f'  - 收敛状态: {convergence}')
    print(f'  - 最佳可行上界: {number(ub)}')
    print(f'  - 最终 gap: {gap_pct(lb, ub)}')


def print_algorithm_config(config, *, num_customers=None):
    """Keep the original Phase / Forward-Backward / Stage configuration map."""
    import os
    from core.solver_settings import (configured_backward_s3_backend,
                                     configured_forward_s2_backend, forward_s2_bpc_policy)
    requested = config.num_processes
    backend = os.environ.get('LRP_S3_BACKEND', 'native')
    s2_backend = configured_forward_s2_backend(num_customers)
    print('\n算法配置（按 Phase / Forward-Backward / Stage）:')
    print('  [公共]')
    print(f'    并行进程={requested}')
    print(f"    Gurobi Threads/model={os.environ.get('VRP_GRB_THREADS', '1')}")
    print('    两个信息阶段；三个算法层：设施计划 → 分配/外包 → 各设施 TSP')
    for phase, label in ((1, 'SDDP-SBC'), (2, 'SDDLP + Level Set')):
        prefix = f'phase{phase}_'
        get = lambda key, default=None: getattr(config, prefix + key, default)
        limit = get('total_time_limit', 0.)
        limit_text = '关闭' if limit <= 0 else f'{limit:g}s ({limit / 3600:g}h)'
        rounds = get('iter_limit', 0)
        rounds_text = '不设迭代上限' if rounds <= 0 else f'最多{rounds}轮'
        print(f'  [Phase {phase} · {label}]')
        print(f"    Stage2 lazy threshold={get('lazy_threshold')}；"
              f"forward period 去重={get('forward_period_dedup')}")
        print(f"    外层: gap<{get('tol') * 100:g}%  {rounds_text}  总时限={limit_text}")
        print(f"    子问题基础 TimeLimit={get('time_limit'):g}s")
        if phase == 1:
            print(f"    无改进切换: {get('no_improve_limit', 0)} 轮（0 表示关闭）")
            print(f"    稳定切换: {get('stability_handoff', False)}；窗口={get('stability_window', 0)}；"
                  f"最早第{get('stability_min_iteration', 0)}轮")
            print(f"    相对停滞: 窗口={get('stagnation_window', 0)}；阈值={get('stagnation_rel_tol', 0):g}；"
                  f"最早第{get('stagnation_min_iteration', 0)}轮")
            prefix_text = '基础 ' if get('s2_mipgap_schedule', False) else ''
            print(f"    {prefix_text}MIPGap={get('mip_gap', 0):g}")
        print('    Forward:')
        print('      Stage1 设施开关主问题 ....... Gurobi MIP')
        s2_label = {'bpc': 'C++ BPC', 'gurobi': 'Gurobi MIP',
                    'auto': 'auto（客户数>20用 C++ BPC）'}[s2_backend]
        print(f'      Stage2 客户分配与外包 ....... {s2_label}；各物理设施容量约束')
        if s2_backend in ('bpc', 'auto'):
            bpc = forward_s2_bpc_policy(phase)
            print(f"        BPC forward gap={bpc['forward_gap']:g}；软时限={bpc['time_limit_s']:g}s；"
                  '失败时在剩余预算内回退 Gurobi')
        if phase == 1 and get('s2_mipgap_schedule', False):
            switch = get('s2_mipgap_switch')
            print(f"        S2 forward MIPGap 调度: iteration<{switch} 用 {get('s2_mipgap_early'):g}；"
                  f"iteration>={switch} 用 {get('s2_mipgap_late'):g}（Gurobi 分支）")
        if phase == 1 and get('forward_s2_time_limit') is not None:
            print(f"        S2 forward 专用 TimeLimit={get('forward_s2_time_limit'):g}s；受外层剩余预算约束")
        print(f'      Stage3 各设施 TSP ........... {backend}')
        print('    Backward:')
        print(f'      Stage3 自由路线 oracle ....... {configured_backward_s3_backend(phase)}')
        if phase == 1:
            print('      Stage3→Stage2 / Stage2→Stage1 ... Strengthened Benders cuts')
        else:
            print('      Stage3→Stage2 / Stage2→Stage1 ... Level Set / Lagrangian cuts')
            print(f"      无改进停止: {get('no_improve_limit', 0)} 轮（0 表示关闭；不视为收敛）")
            print(f'      Level Set: norm={config.norm_option}  lambda={config.lambda_level:g}  '
                  f'mu={config.mu_level:g}  轮数={config.levelset_iter_limit}  tol={config.levelset_tol:g}')
            print(f"      S2↔S3 内层精化: 最多{max(1, get('inner_s3_rounds', 0))}轮；"
                  f"共享预算={get('inner_s3_time_limit', 0):g}s")
            s2_limit = get('s2_cut_time_limit', 0)
            s2_limit_text = f'{s2_limit:g}s' if s2_limit > 0 else '不另设整轮时限'
            print(f"      S2 cut 整轮预算={s2_limit_text}；"
                  f"绝对容差 kappa={get('s2_abs_tol_kappa', 0):g}；"
                  f"floor share={get('s2_abs_tol_floor_share', 0):g}")
    print('  [Phase 1.5 · 物理路径补强]')
    print(f'    schedule={config.phase15_schedule}  backend={config.phase15_backend}')


def print_instance_summary(cfg, inst):
    data = inst.prob_data
    m, n, H, L, S = data.shape
    print(f"LRP {data.name}: facilities={m}, customers={n}, delivery_periods={H}, "
          f"facility_intervals={L}, scenarios={S}")
    print(f"  两个信息阶段；三个计算层: 设施 -> 分配/外包 -> 各设施路线")
    print(f"  period_to_interval={data.period_to_interval.tolist()}, pi={data.scenario_prob.tolist()}")
    print(f"  数据: {inst.path}; SHA256={data.logical_hash()}")


def _phase_summary(result, seconds):
    lb, ub = phase_bounds(result)
    return {
        "lower_bound": lb, "upper_bound": ub, "relative_gap": relative_gap(lb, ub),
        "iterations": result.get("iterations", 0), "seconds": seconds,
        "converged": bool(result.get("converged", False)) and relative_gap(lb, ub) is not None,
        "stop_reason": result.get("stop_reason", "unknown"),
        "solve_counts": result.get("solve_counts"),
        "backend_telemetry": result.get("backend_telemetry"),
    }


def merge_results(cfg, result_p1, result_p2, phase1_time, phase2_time, *, config=None):
    result = dict(result_p2)
    result.update({
        "schema": "lrp_production_sddp_v1", "information_stages": 2, "algorithm_layers": 3,
        "execution": execution_metadata(config, phase1=result_p1, phase2=result_p2),
        "instance": str(cfg.instance), "instance_name": cfg.instance_tag(),
        "phase1": _phase_summary(result_p1, phase1_time),
        "phase2": _phase_summary(result_p2, phase2_time),
        "phase1_result": result_p1, "phase2_result": result_p2,
        "phase1_time": phase1_time, "phase2_time": phase2_time,
        "total_time": phase1_time + phase2_time,
        "algorithm_config": None if config is None else config.to_dict(),
    })
    # Preserve the original fixed result schema, including inactive Phase 1.5.
    # Empty compatibility fields do not claim that a physical seed was run.
    for key, default in (('initial_lb', None), ('forward_time_list', []),
                         ('backward_time_list', []), ('stop_reason', 'unknown'),
                         ('physical_route_seed', {}), ('physical_policy_search', []),
                         ('phase15_events', []), ('phase15_telemetry', {})):
        result.setdefault(key, default)
    # Preserve the established investment result-reader interface.
    for key in ('Vstar', 'x_best', 'cut_lag', 'LB_list', 'UB_list', 'time_list',
                'forward_time_list', 'backward_time_list', 'initial_lb',
                'stop_reason', 'converged', 'iterations'):
        result[key + '_SBC'] = result_p1.get(key, 'unknown' if key == 'stop_reason' else None)
    result['cfg'] = cfg
    result['phase2_backend_telemetry'] = result_p2.get('backend_telemetry')
    result['backend_telemetry'] = {'phase1': result_p1.get('backend_telemetry'),
                                 'phase2': result_p2.get('backend_telemetry')}
    lb, ub = phase_bounds(result_p2)
    result["lower_bound"] = lb
    result["upper_bound"] = ub
    result["relative_gap"] = relative_gap(lb, ub)
    result["converged"] = bool(result.get("converged", False)) and result["relative_gap"] is not None
    return result


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        # Preserve an intentionally unbounded solve setting in strict JSON.
        # Bounds and other observations retain the usual nonfinite -> null
        # conversion; only configuration uses the snapshot's tagged encoding.
        from core.run_snapshot import _config_dict
        return {str(k): jsonable(_config_dict(v) if k == 'algorithm_config' and v is not None else v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_output_directory(cfg, *, prefix="lrp"):
    from core.run_logging import results_root, compare_ef_results_root
    cfg._legacy_output = cfg.out is None
    if cfg.out is None:
        path = compare_ef_results_root() if prefix == 'compare' else results_root()
        tag = cfg.instance_tag()
        names = [f'{tag}.txt', f'result_{tag}.pkl', f'report_{tag}.txt']
        if prefix == 'compare':
            names.append(f'compare_{tag}.txt')
        if any((path / name).exists() for name in names):
            tag += '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cfg._output_tag = tag
    else:
        path = Path(cfg.out).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        # An existing output directory is usable; existing run files are not
        # overwritten accidentally. Directory existence alone is no conflict.
        occupied = [name for name in ('result.pkl', 'result.json', 'report.txt', 'console.log')
                    if (path / name).exists()]
        if occupied:
            raise FileExistsError(f'Output files already exist in {path}: {occupied}')
    cfg.out = path
    return path


def save_outputs(cfg, final_result):
    from setup.result_report import generate_report_from_result
    if cfg.out is None:
        prepare_output_directory(cfg)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    from core.run_logging import result_tag
    tag = result_tag(cfg)
    legacy = getattr(cfg, "_legacy_output", False)
    pkl_path = out / (f"result_{tag}.pkl" if legacy else "result.pkl")
    with pkl_path.open("wb") as stream:
        pickle.dump(final_result, stream)
    write_json(out / (f"result_{tag}.json" if legacy else "result.json"), final_result)
    _, report_path = generate_report_from_result(final_result, str(out / (f"report_{tag}.txt" if legacy else "report.txt")))
    return pkl_path, report_path


def print_summary(cfg, config, log_path, pkl_path, report_path,
                  result_p1, result_p2, phase1_time, phase2_time):
    print('\n' + '=' * 78)
    print(f'算例: {cfg.instance_tag()}')
    print(f'  Phase1 gap={config.phase1_tol * 100:g}%  '
          f'Phase2 gap={config.phase2_tol * 100:g}%  并行（配置）={config.num_processes}')
    print('=' * 78)
    print(f"{'阶段':<12} {'迭代':>6} {'时间(s)':>9} {'LB':>16} {'UB':>16} {'Gap':>10} {'收敛':>6}")
    print('-' * 78)
    for label, result, elapsed in (('Phase1 SBC', result_p1, phase1_time),
                                   ('Phase2 SDDLP', result_p2, phase2_time)):
        if result is None:
            print(f'{label:<12} 本次未运行（没有独立用时或 gap）')
            continue
        lb, ub = phase_bounds(result)
        lb_text = '--' if lb is None else f'{lb:,.2f}'
        ub_text = '--' if ub is None else f'{ub:,.2f}'
        seconds = '--' if elapsed is None else f'{elapsed:.2f}'
        converged = '✓' if result.get('converged') and relative_gap(lb, ub) is not None else '✗'
        print(f"{label:<12} {result.get('iterations', 0):>6} {seconds:>9} "
              f'{lb_text:>16} {ub_text:>16} {gap_pct(lb, ub):>10} {converged:>6}')
    print('=' * 78)
    total = sum(value for value in (phase1_time, phase2_time) if value is not None)
    print(f'总时间: {total:.2f} 秒')
    print(f'控制台日志: {log_path}')
    print(f'结果文件: {pkl_path}')
    print(f'报告文件: {report_path}')


def continuation_report_result(cfg, result, elapsed, inst, config, *, warm_path, warm_sha):
    """Add presentation metadata without altering the raw Phase-2 checkpoint.

    A continuation is not a new Phase-1 experiment. Its warm file remains
    separate provenance, and no *_SBC histories or Phase-1 time are invented.
    """
    report = dict(result)
    lb, ub = phase_bounds(result)
    report.update({
        'schema': 'lrp_production_sddp_v1', 'information_stages': 2, 'algorithm_layers': 3,
        'instance': str(inst.path), 'instance_name': cfg.instance_tag(), 'cfg': cfg,
        'execution': execution_metadata(config, phase2=result),
        'instance_sha256': inst.prob_data.logical_hash(),
        'dimensions': dict(zip(('I', 'J', 'T', 'L', 'S'), inst.prob_data.shape)),
        'data_selection': inst.prob_data.metadata.get('subset'),
        'phase1': None, 'phase1_result': None, 'phase1_time': None,
        'phase2': _phase_summary(result, elapsed), 'phase2_result': result,
        'phase2_time': elapsed, 'total_time': elapsed,
        'algorithm_config': config.to_dict(),
        'lower_bound': lb, 'upper_bound': ub, 'relative_gap': relative_gap(lb, ub),
        'phase2_backend_telemetry': result.get('backend_telemetry'),
        'backend_telemetry': {'phase1': None, 'phase2': result.get('backend_telemetry')},
        'report_provenance': {'mode': 'Phase 2 continuation', 'warm_checkpoint': str(warm_path),
                              'warm_sha256': warm_sha, 'phase1_run_in_this_experiment': False,
                              'raw_result': str(Path(cfg.out) / 'result.pkl')},
    })
    report['report_data'] = {key: inst.prob_data.arrays[key].tolist() for key in (
        'facility_ids', 'customer_ids', 'capacity', 'opening_cost', 'continuation_cost',
        'closing_cost', 'active', 'demand', 'scenario_prob', 'location_periods', 'period_to_interval',
        'cost_fc', 'cost_cc', 'route_cost') if key in inst.prob_data.arrays}
    report['diagnostic_files'] = {name: str(Path(cfg.out) / name) for name in (
        'request.json', 'source_hashes.json', 'summary.json', 'progress.json',
        'checkpoint.pkl', 'checkpoint_events.jsonl', 'completed_cut_diagnostics.jsonl')
        if (Path(cfg.out) / name).exists()}
    return report


def execution_metadata(config, **phase_results):
    """Record configuration separately from process observations."""
    import os
    from core.solver_settings import configured_backward_s3_backend, configured_forward_s2_backend
    route_backend = os.environ.get('LRP_S3_BACKEND', 'native')
    s2_backend = configured_forward_s2_backend()
    backward_routes = {f'phase{phase}': configured_backward_s3_backend(phase)
                       for phase in (1, 2)}
    observed = {name: dict(result.get('node_execution') or {})
                for name, result in phase_results.items() if result is not None}
    policy = {}
    if config is not None:
        for phase in (1, 2):
            policy[f'phase{phase}'] = {
                'lazy_threshold': getattr(config, f'phase{phase}_lazy_threshold'),
                'forward_period_dedup': getattr(config, f'phase{phase}_forward_period_dedup'),
            }
    return {
        'backend': 'gurobi' if route_backend == s2_backend == 'gurobi' else 'mixed',
        'configured_stage_backends': {'stage1': 'gurobi', 'stage2': s2_backend,
                                     'stage3': route_backend},
        'configured_backward_stage2_backend': 'gurobi',
        'configured_backward_stage3_backends': backward_routes,
        'requested_node_processes': None if config is None else config.num_processes,
        'observed_node_execution': observed,
        'stage2_cut_policy': policy,
    }
