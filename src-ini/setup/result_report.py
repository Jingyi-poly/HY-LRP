"""Restore investment-style report sections using recorded LRP decisions only."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
import pickle
import json
import math
from pathlib import Path

# Support the original direct-script command as well as imported use.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if (_SRC_ROOT / 'core').is_dir():
    sys.path.insert(0, str(_SRC_ROOT))

from setup.reporting import describe_stop_reason, gap_pct, jsonable, phase_bounds
from core.run_snapshot import _config_dict, _decode_config_dict


def _sep(char='=', width=72):
    return char * width


def _section(title, char='=', width=72):
    return '\n'.join([_sep(char, width), f'  {title}', _sep(char, width)])


def _subsection(title, width=72):
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _readable_items(value, *, indent='  ', configuration=False):
    """Readable parameter/count listing; full diagnostics stay in the pickle."""
    value = jsonable(_config_dict(value) if configuration else value)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict) and item != {'__config_float__': '+inf'}:
                lines.append(f'{indent}{key}:')
                lines.extend(_readable_items(item, indent=indent + '  '))
            elif isinstance(item, list) and any(isinstance(x, dict) for x in item):
                lines.append(f'{indent}{key}:')
                for index, entry in enumerate(item, 1):
                    lines.append(f'{indent}  第 {index} 项:')
                    lines.extend(_readable_items(entry, indent=indent + '    '))
            else:
                lines.append(f'{indent}{key}: {_readable_value(item)}')
        return lines
    return [indent + _readable_value(value)]


def _readable_value(value):
    if isinstance(value, dict) and value == {'__config_float__': '+inf'}:
        return 'inf（不设上限）'
    if value is None:
        return '未记录'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, list):
        return ', '.join(_readable_value(item) for item in value) or '（无）'
    return str(value)


def _fmt(value):
    return '--' if value is None else f'{float(value):.6f}'



def _recorded_arc_cost(data, t, i, v, w):
    """Use stored input coefficients only, including asymmetric own-root arcs."""
    if 'route_cost' in data:
        return float(data['route_cost'][t][i][v][w])
    if 'cost_fc' not in data or 'cost_cc' not in data:
        return None
    if v == 0:
        return float(data['cost_fc'][t][i][w - 1])
    if w == 0:
        return float(data['cost_fc'][t][i][v - 1])
    return float(data['cost_cc'][t][v - 1][w - 1])


def _check_cost(actual, expected, label):
    if not math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f'{label}: 保存的弧成本合计 {actual} 与审计成本 {expected} 不一致')


def generate_report_from_result(result, output_path):
    if result.get('algorithm_config') is not None:
        # Strict result JSON and pickle share the same visible settings. Do
        # not mutate the saved result or interpret tagged values as bounds.
        result = dict(result, algorithm_config=_decode_config_dict(result['algorithm_config']))
    lines = [_sep(), '  LRP4SDDP 优化结果报告',
             f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", _sep(), '']
    def section(title):
        lines.extend(['', _section(title)])
    def listing(value):
        lines.extend(_readable_items(value))
    section('1. 问题规模')
    lines.extend([f"实例: {result.get('instance_name', result.get('instance', '未记录'))}",
                  f"数据 SHA256: {result.get('instance_sha256', '未记录')}",
                  '两个信息阶段，三个算法层：设施计划 → 分配/外包 → 各设施 TSP',
                  '索引 i、j、t、s、k 从 0 开始；physical_id 是输入数据中的原始点编号。',
                  'gap 使用原算法 LB 分母；时间和阶段只按实际记录展示。'])
    dimensions = result.get('dimensions') or {}
    for key, label in (('J', '客户数'), ('I', '设施数'), ('S', '场景数'),
                       ('T', '配送周期'), ('L', '设施决策周期')):
        if key in dimensions:
            lines.append(f'  {label}: {dimensions[key]}')
    for key, title in (('data_selection', '数据选择'), ('report_provenance', '本次运行来源'),
                       ('algorithm_config', '算法参数')):
        if result.get(key) is not None:
            lines.append(_subsection(title))
            lines.extend(_readable_items(result[key], configuration=key == 'algorithm_config'))
    execution = result.get('execution') or {}
    if execution.get('configured_stage_backends'):
        lines.append('  各层配置后端（实际调用统计见求解次数）:')
        listing(execution['configured_stage_backends'])
    if execution.get('configured_backward_stage3_backends'):
        lines.append('  Backward Stage 3 自由路线后端（按阶段）:')
        listing(execution['configured_backward_stage3_backends'])
    requested = (result.get('algorithm_config') or {}).get('num_processes')
    if requested is not None:
        lines.append(f'  节点并行进程配置: {requested}')
    for phase, observed in execution.get('observed_node_execution', {}).items():
        if not observed:
            continue
        forward = len(set(observed.get('forward_worker_pids', [])))
        backward = len(set(observed.get('backward_worker_pids', [])))
        lines.append(f'  {phase} 已记录的子进程数: 最近 forward={forward}，backward={backward}')
    for phase, policy in execution.get('stage2_cut_policy', {}).items():
        lines.append(f"  {phase}: Stage 2 lazy threshold={policy['lazy_threshold']}；"
                     f"forward period 去重={policy['forward_period_dedup']}")
    reuse = (result.get('phase2_result') or result).get('forward_reuse')
    if reuse is not None:
        lines.append('  Phase 1→Phase 2 forward 快照复用:')
        listing(reuse)
    data = result.get('report_data') or {}
    if data:
        lines.append('设施位置及逐期容量:')
        for i, physical_id in enumerate(data.get('facility_ids', ['--'] * len(data['capacity']))):
            lines.append(f"i={i} physical_id={physical_id} capacity={data['capacity'][i]}")
        lines.append(f"location_periods={data['location_periods']}; period_to_interval={data['period_to_interval']}")
        lines.append(f"scenario_prob={data['scenario_prob']}")

    section('2. 算法收敛情况')
    for phase, label in (('phase1', 'Phase 1 (SDDP-SBC)'), ('phase2', 'Phase 2 (SDDLP)')):
        raw = result.get(phase + '_result')
        # Explicit None means the phase was not run. Only legacy flat files
        # lack the nested key; inherited histories must not override a skip.
        if phase + '_result' not in result:
            if phase == 'phase2':
                raw = result
            elif 'LB_list_SBC' in result:
                raw = {key[:-4]: value for key, value in result.items() if key.endswith('_SBC')}
        lines.append(_subsection(label))
        if raw is None or (phase == 'phase2' and result.get('phase2_executed') is False):
            lines.append('本次未运行；没有该阶段的独立用时和 gap，不能用续跑结果代替。')
            continue
        # Old flat Phase-2 files prepended the handoff LB without a UB/time.
        lower, upper = raw.get('LB_list') or [], raw.get('UB_list') or []
        if (phase == 'phase2' and len(lower) == raw.get('iterations', 0) + 1
                and len(upper) == raw.get('iterations', 0)):
            raw = dict(raw)
            initial = raw.get('initial_lb')
            raw['initial_lb'] = lower[0] if initial is None else max(initial, lower[0])
            raw['LB_list'] = lower[1:]
        lb, ub = phase_bounds(raw)
        seconds = result.get(phase + '_time', raw.get('total_time'))
        algorithm = result.get('algorithm_config') or {}
        target = algorithm.get(phase + '_tol')
        limit = algorithm.get(phase + '_total_time_limit')
        limit_text = '未记录' if limit is None else ('未启用' if limit <= 0 else f'{limit:g} 秒 ({limit / 3600:g} 小时)')
        convergence = ('✗ 证书无效（LB > UB）' if gap_pct(lb, ub) == 'LB>UB' else
                       '✓ 内部 LB/UB 已收敛' if raw.get('converged') and lb is not None and ub is not None else
                       '✗ 未收敛')
        lines.extend([f'  收敛状态:  {convergence}',
                      f"  停止原因:  {describe_stop_reason(raw.get('stop_reason'))}",
                      '  目标 gap:  ' + ('未记录' if target is None else f'{target * 100:g}%'),
                      f'  外层总时限: {limit_text}',
                      f"  迭代次数:  {raw.get('iterations', '未记录')}",
                      f'  求解时间:  {_fmt(seconds)} 秒',
                      f'  最终下界:  {_fmt(lb)}', f'  最终上界:  {_fmt(ub)}',
                      f'  最优间隙:  {gap_pct(lb, ub)}'])
        if raw.get('initial_lb') is not None:
            if (result.get('report_provenance') or {}).get('mode') == 'Phase 2 continuation':
                initial_label = '续跑传入初始下界'
            elif 'phase1_result' in result and result['phase1_result'] is None:
                initial_label = '传入初始下界'
            else:
                initial_label = 'Phase 1 传入初始下界'
            lines.append(f"  {initial_label}（不计迭代）: {raw['initial_lb']:,.2f}")
        if phase == 'phase2' and raw.get('stop_reason') == 'phase2_total_time_limit':
            lines.append('  时限在主要操作之间检查；在途求解可能使实际耗时超过配置上限。')
        if phase == 'phase2':
            precision = raw.get('theta_precision') or result.get('theta_precision') or {}
            if precision:
                changes = sum(record.get('decision') is not None
                              for record in precision.get('history', []))
                lines.extend([
                    '  历史记录：动态 theta 精度（现已删除）: '
                    f"{'启用' if precision.get('enabled') else '关闭'}；"
                    f"收紧 {changes} 次；最终 scale={precision.get('final_scale', 1):g}",
                    '  （仅展示保存的历史字段；不启用已删除的精度控制。）'])
        for key, title in (('forward_time_list', '前向累计'), ('backward_time_list', '后向累计'),
                           ('master_refresh_time_list', '主问题刷新累计')):
            values = raw.get(key) or []
            if values:
                lines.append(f'  {title}:  {sum(values):.2f} 秒')
        lines.extend(['', '  迭代历史:',
                      '迭代 | 下界 | 上界 | gap | 本轮(s) | 累计(s) | forward(s) | backward(s) | master refresh(s) | new cuts'])
        histories = [raw.get(k) or [] for k in ('LB_list', 'UB_list', 'time_list',
                     'cumulative_time_list', 'forward_time_list', 'backward_time_list', 'master_refresh_time_list', 'new_cut_count_list')]
        for index in range(max(map(len, histories), default=0)):
            values = [h[index] if index < len(h) else None for h in histories]
            lines.append(' | '.join([str(index + 1), _fmt(values[0]), _fmt(values[1]),
                                    gap_pct(values[0], values[1])] + [_fmt(v) for v in values[2:]]))
        lines.append('  求解次数:')
        listing(raw.get('solve_counts'))
    total_time = result.get('total_time')
    lines.extend(['', f'  总求解时间: {_fmt(total_time)} 秒'])

    audit = result.get('policy_audit') or {}
    policy = result.get('x_best') or {}
    root = policy.get(1, policy.get('1', {}))
    root = root.get(0, root.get('0', {}))
    section('3. Stage 1 — 设施开仓、延续与关仓决策')
    lines.append('i | physical_id | interval k | A | o | h | b | 本期设施成本')
    for i, availability in enumerate(audit.get('A', [])):
        prev = 0
        for k, curr in enumerate(availability):
            opening, continuing, closing = curr * (1-prev), curr * prev, prev * (1-curr)
            cost = None
            if data:
                cost = (opening * data['opening_cost'][i][k] + continuing * data['continuation_cost'][i][k]
                        + closing * data['closing_cost'][i][k])
            lines.append(f"{i} | {data.get('facility_ids', ['--'] * len(audit['A']))[i]} | {k} | {curr} | "
                         f"{root.get(f'o[{i},{k}]', opening)} | {root.get(f'h[{i},{k}]', continuing)} | "
                         f"{root.get(f'b[{i},{k}]', closing)} | {_fmt(cost)}")
            prev = curr
    if not audit:
        lines.append('未保存政策审计；不从缺失数据推断可行解。')

    nodes = audit.get('nodes', {})
    nodes = list(nodes.values()) if isinstance(nodes, dict) else nodes
    nodes = sorted(nodes, key=lambda z: (z['scenario'], z['period']))
    section('4. Stage 2 — 客户分配与外包')
    for node in nodes:
        t, s = node['period'], node['scenario']
        lines.append(f'\nperiod={t}, scenario={s}; availability={node.get("availability")}')
        lines.append('客户 j | physical_id | active | demand | assigned facility i | outsourced e')
        for j, outsourced in enumerate(node['e']):
            assigned = [i for i, row in enumerate(node['alpha']) if row[j]]
            active = data['active'][t][s][j] if data else '--'
            demand = data['demand'][t][s][j] if data else None
            physical_id = data['customer_ids'][j] if 'customer_ids' in data else '--'
            lines.append(f'{j} | {physical_id} | {active} | {_fmt(demand)} | {assigned} | {outsourced}')
        lines.append('设施 i | dispatch u | 客户数 | load | capacity')
        for i, row in enumerate(node['alpha']):
            load = sum(data['demand'][t][s][j] * bit for j, bit in enumerate(row)) if data else None
            capacity = data['capacity'][i][t] if data else None
            lines.append(f"{i} | {node['u'][i]} | {sum(row)} | {_fmt(load)} | {_fmt(capacity)}")

    section('5. Stage 3 — 各设施路径规划')
    lines.append('local_route 中 0 是该设施，客户 j 对应路线点 j+1；空路线表示闲置。')
    lines.append('逐弧列为原始 route cost；输入未保存公里数时，不将计价成本标为距离。')
    weighted_arc_costs = []
    all_arc_costs_recorded = True
    for node in nodes:
        node_arc_costs = []
        node_costs_recorded = True
        for tour in node['tours']:
            route, i = tour['local_route'], tour['facility']
            physical = [data['facility_ids'][i] if v == 0 else data['customer_ids'][v-1]
                        for v in route] if 'facility_ids' in data and 'customer_ids' in data else None
            lines.append(f"t={node['period']} s={node['scenario']} i={i} customers={tour['customers']} "
                         f"local_route={route} physical_route={physical} cost={_fmt(tour['cost'])}")
            if not route:
                lines.append('  路径弧列表: （无，设施未出行）')
                _check_cost(0., tour['cost'], '闲置设施路线')
                node_arc_costs.append(0.)
                continue
            arcs = list(zip(route, route[1:]))
            costs = [_recorded_arc_cost(data, node['period'], i, v, w) for v, w in arcs]
            if any(cost is None for cost in costs):
                lines.append('  未保存原始弧成本；无法恢复逐弧成本表，未从路线总成本猜测。')
                node_costs_recorded = False
                continue
            lines.append('  路径弧列表:')
            lines.append('  local 起点 → 终点 | physical 起点 → 终点 | 弧成本')
            for index, ((v, w), cost) in enumerate(zip(arcs, costs)):
                physical_arc = ('未记录' if physical is None else
                                f'{physical[index]} → {physical[index + 1]}')
                lines.append(f'  {v} → {w} | {physical_arc} | {cost:.12g}')
            arc_total = math.fsum(costs)
            _check_cost(arc_total, tour['cost'], f"路线 t={node['period']} s={node['scenario']} i={i}")
            node_arc_costs.append(arc_total)
            lines.append(f'  弧成本合计: {arc_total:.12g}；与保存的路线成本一致。')
        if node_costs_recorded:
            node_total = math.fsum(node_arc_costs)
            _check_cost(node_total, node['routing_cost'], '节点路线成本')
            if 'scenario_prob' in data:
                weighted_arc_costs.append(data['scenario_prob'][node['scenario']] * node_total)
            else:
                all_arc_costs_recorded = False
        else:
            all_arc_costs_recorded = False
    if nodes and all_arc_costs_recorded:
        expected_arc_cost = math.fsum(weighted_arc_costs)
        _check_cost(expected_arc_cost, audit['expected_routing'], '期望路线成本')
        total_from_arcs = math.fsum([audit['facility_cost'], expected_arc_cost, audit['expected_outsourcing']])
        _check_cost(total_from_arcs, audit['feasible_upper_bound'], '完整政策成本')
        lines.append(f'  逐弧重算期望路线成本: {expected_arc_cost:.12g}；与成本汇总一致。')
        lines.append(f'  设施 + 逐弧路线 + 外包: {total_from_arcs:.12g}；与保存可行 UB 一致（浮点容差内）。')

    section('6. 求解下界与成本汇总')
    for key in ('facility_cost', 'expected_routing', 'expected_outsourcing', 'feasible_upper_bound',
                'all_original_nodes_audited', 'cost_rounding'):
        lines.append(f'{key}: {audit.get(key, "未记录")}')
    lines.append('各节点成本未乘概率；期望成本按 scenario_prob 加权，所有期间求和。')
    lines.append('t | s | probability | routing | outsourcing | node total')
    for node in nodes:
        s = node['scenario']
        p = data['scenario_prob'][s] if data else None
        lines.append(' | '.join([str(node['period']), str(s), _fmt(p), _fmt(node['routing_cost']),
                                _fmt(node['outsourcing_cost']), _fmt(node['true_feasible_cost'])]))
    section('7. 变量说明')
    lines.extend([
        '  Stage 1 变量:',
        '    A[i,k]        设施 i 在设施周期 k 是否可用（二元）',
        '    o[i,k]        本期开仓；h[i,k] 延续；b[i,k] 关仓（二元）',
        '    eta[succ]     后续节点成本下界近似',
        '', '  Stage 2 变量:',
        '    u[i]          本期设施是否出车服务非空客户集合（二元）',
        '    alpha[i,j]    客户 j 分配给设施 i（二元）',
        '    e[j]          客户 j 是否外包（二元）',
        '    theta[succ]   对应设施路线成本下界近似',
        '', '  Stage 3 变量:',
        '    r[i,v,w]      设施 i 路径弧 v→w 是否使用（二元）；0 为该设施位置',
        '', '  两阶段算法:',
        '    Phase 1: SDDP-SBC（Strengthened Benders Cuts）',
        '    Phase 2: SDDLP（Level Set / Lagrangian cuts）',
        '    两个信息阶段、三个计算层；独立 EF 仍为两信息阶段。',
        _subsection('Cut 数量与完整结果索引'),
    ])
    for stage, by_node in result.get('cut_lag', {}).items():
        if isinstance(by_node, dict):
            for node, rows in by_node.items():
                lines.append(f'stage={stage} node={node} cuts={len(rows)}')
    lines.append('完整 cut 系数、变量、计时、缓存和后端诊断保存在原始 result pickle；未从报告删除其数据。')
    for key in ('diagnostic_files', 'comparison'):
        if key in result:
            lines.append(key + ':'); listing(result[key])
    lines.extend([_sep(), '  报告生成完毕', _sep()])
    text = '\n'.join(lines) + '\n'
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return text, str(path.resolve())

def generate_report(result_path: str, output_path: str):
    path = Path(result_path)
    if path.suffix == ".json":
        result = json.loads(path.read_text())
    else:
        with path.open("rb") as stream:
            result = pickle.load(stream)
    return generate_report_from_result(result, output_path)


def main(argv=None):
    from core.run_logging import results_root

    parser = argparse.ArgumentParser(description=__doc__)
    # Original --result / --out interface; newer positional calls also work.
    parser.add_argument('result_path', nargs='?')
    parser.add_argument('output_path', nargs='?')
    parser.add_argument('--result', dest='result_option')
    parser.add_argument('--out', dest='output_option')
    args = parser.parse_args(argv)
    if args.result_path and args.result_option:
        parser.error('Supply the result either positionally or with --result')
    if args.output_path and args.output_option:
        parser.error('Supply the report either positionally or with --out')
    result_path = args.result_option or args.result_path or str(results_root() / 'result.pkl')
    output_path = args.output_option or args.output_path or str(results_root() / 'report.txt')
    print(f'读取结果: {result_path}')
    report_text, out_path = generate_report(result_path, output_path)
    print(f'报告已写入: {out_path}')
    print()
    print(report_text)


if __name__ == "__main__":
    main()
