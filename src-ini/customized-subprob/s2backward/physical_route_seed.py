"""Budgeted physical route pricing in the original Phase-1.5 seed slot.

Only cut_lag[2] is changed. A restricted route dual proposes customer rewards;
independent physical pricing bounds certify the resulting root eta cut. No
physical bound enters the cheaper assignment oracle or its Level-Set bundle.
The serializable cache is scoped by complete immutable physical contexts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy, deepcopy
from fractions import Fraction
import math
import os
import re
import time

import gurobipy as gp
import numpy as np

from cuts.benders_cuts import add_unique_cut
from core.solver_bounds import certified_gurobi_minimization_lower_bound
from models.stage_builder import StageModelBuilder, _instance, _node_context
from models.subproblem_builder import SubproblemBuilder
from models.stage_model_core import (
    audit_gurobi_matrix, audit_tour, evaluate_model, matrix_primal_check,
    tour_from_evaluation,
)
from models.route_dfj_pool import remember_route_dfj_row
from solvers.forward_policy_certification import certify_stage1_forward_policy
from solvers.route_lp_separation import directed_mincut
from .physical_route_cut import build_physical_pricing_cut, round_down

_CACHE_VERSION = 'lrp-physical-route-pricing-v1'
_F = lambda x: Fraction.from_float(float(x))


def _trace(node, iteration, best, remaining, columns, *, final=False):
    if os.environ.get('LRP_PHYSICAL_TRACE', '0') == '1':
        print(f'[Phase1.5 physical] node={node} round={iteration} '
              f'certified_cut_at_A={best:.9f} remaining={max(0.,remaining):.2f}s '
              f'columns={columns} final={int(final)}', flush=True)


def _positive(value, name, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return value


def _up(value):
    return -round_down(-value)


def _options(model):
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.MIPGap = 0.
    model.Params.MIPGapAbs = 0.
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9
    model.Params.OptimalityTol = 1e-9


def _guarded_bound(model):
    value = certified_gurobi_minimization_lower_bound(model)
    return None if value is None else math.nextafter(value - 1e-9 - 1e-12*max(1., abs(value)), -math.inf)


def _separate_free_routes(model, deadline, budget):
    """Use only OPTIMAL free LP bounds; install valid DFJ rows in both specs.

    The LP phase has at most 40% of this pricing slice. A row uses the
    variable a_copy[j] RHS, so it is valid over the whole free-parent domain.
    No LP multiplier or incumbent is used as a physical pricing certificate.
    """
    started = time.perf_counter()
    result = dict(lp_solves=0, added_rows=0, certified_lower_bound=None,
                  certificates=[], seconds=0.)
    if budget < 2. or deadline - started < .3:
        return result
    stop = min(deadline - .2*budget, started + .4*budget)
    lp = model.relax()
    _options(lp)
    spec, M = model._lrp_spec, model._lrp_spec.linear
    ctx, i = spec.context, spec.facility
    try:
        arcs = {(v, w): col for (fi, v, w), col in M.groups['r'].items() if fi == i}
        vertices = {0} | {v for arc in arcs for v in arc}
        variables, lpvars = model.getVars(), lp.getVars()
        seen, audit_reserve = set(), .06
        while stop - time.perf_counter() > audit_reserve + .02:
            lp.Params.TimeLimit = max(1e-6, stop-time.perf_counter()-audit_reserve)
            lp.optimize()
            result['lp_solves'] += 1
            if lp.Status != gp.GRB.OPTIMAL:
                result['last_status'] = int(lp.Status)
                break
            begun = time.perf_counter()
            continuous = copy(spec)
            continuous.linear = copy(M)
            continuous.linear.integer = [0]*len(M.integer)
            matrix = audit_gurobi_matrix(continuous.linear, lp, lpvars)
            primal = matrix_primal_check(continuous, lp.getAttr('X', lpvars))
            lower = _guarded_bound(lp)
            certificate = dict(status='OPTIMAL', raw_bound=float(lp.ObjBound),
                               lower_bound=lower, matrix_audit=matrix, primal_audit=primal,
                               rows=len(M.row_names))
            result['certificates'].append(certificate)
            if lower is not None:
                prior = result['certified_lower_bound']
                result['certified_lower_bound'] = lower if prior is None else max(prior, lower)
            audit_reserve = max(.06, 1.3*(time.perf_counter()-begun))
            if stop-time.perf_counter() <= .03:
                break
            alpha = [lpvars[M.groups['a_copy'][j,]].X for j in range(ctx.n)]
            capacity = np.zeros((ctx.n+1, ctx.n+1))
            for (v, w), col in arcs.items():
                capacity[v, w] = max(0., float(lpvars[col].X))
            found = []
            for j in sorted((j for j, a in enumerate(alpha) if a > 1e-8), key=lambda j: -alpha[j]):
                answer = directed_mincut(capacity, j+1, deadline=stop-.03)
                if answer is None:
                    break
                value, original = answer
                key = tuple(sorted(original)), j
                if value < alpha[j]-1e-7 and key not in seen:
                    seen.add(key)
                    found.append(key)
            if not found:
                break
            for original, j in found:
                U = frozenset(original) & vertices
                if len(U) <= 1 or j+1 not in U:
                    continue
                inside = [(col, -1.) for (v, w), col in arcs.items() if v in U and w in U]
                inside += [(M.groups['a_copy'][v-1,], 1.) for v in sorted(U) if v != j+1]
                crossing = [(col, 1.) for (v, w), col in arcs.items() if v in U and w not in U]
                crossing.append((M.groups['a_copy'][j,], -1.))
                terms = inside if len(inside) < len(crossing) else crossing
                name = f'physical_pricing_dfj_{i}_{len(M.row_names)}'
                M.row(name, terms, lb=0.)
                for native, nativevars in ((model, variables), (lp, lpvars)):
                    expr = gp.LinExpr([v for _, v in terms], [nativevars[c] for c, _ in terms])
                    native.addConstr(expr >= 0., name=name+'_lb')
                remember_route_dfj_row(ctx, i, original, j)
                result['added_rows'] += 1
            model.update()
            lp.update()
    finally:
        lp.dispose()
    result['seconds'] = time.perf_counter()-started
    return result


def _column(ctx, i, subset, arcs):
    """Validate the exact capacity/active domain and the original physical tour."""
    subset = tuple(sorted(set(int(j) for j in subset)))
    if any(j < 0 or j >= ctx.n for j in subset):
        raise ValueError('route column customer outside physical context')
    alpha = tuple(int(j in subset) for j in range(ctx.n))
    ctx.check_route_state(i, alpha, int(bool(subset)))
    tour = audit_tour(ctx, i, alpha, int(bool(subset)), arcs)
    cost = _up(sum((_F(ctx.route_cost[i,v,w]) for v, w in tour['arcs']), Fraction()))
    return dict(customers=subset, arcs=tuple(tuple(a) for a in tour['arcs']),
                cost=cost, route_key=ctx.route_key(i))


def _store_column(state, ctx, i, column):
    if column['route_key'] != ctx.route_key(i):
        raise ValueError('route column belongs to another physical context')
    archive = state['columns'].setdefault(i, {})
    subset = tuple(column['customers'])
    if column['cost'] < archive.get(subset, {}).get('cost', math.inf):
        archive[subset] = column
        return 1
    return 0


def _subsets(initial_policy, bundles, node, ctx, i):
    result = set()
    if isinstance(initial_policy, Mapping):
        stage = initial_policy.get(2, {})
        previous = stage.get(node.index, {}) if isinstance(stage, Mapping) else {}
        if isinstance(previous, Mapping):
            values = [previous.get(f'alpha[{i},{j}]', 0.) for j in range(ctx.n)]
            if all(v in (0., 1.) for v in values):
                result.add(tuple(j for j, v in enumerate(values) if v))
    if isinstance(bundles, Mapping):
        stage = bundles.get(3, bundles)
        # The outer cut_Dict[3] is also accepted directly. Its integer key 3
        # may be a route node, not another stage dictionary.
        if not isinstance(stage, Mapping):
            stage = bundles
        rows = stage.get(node.index*ctx.m+i, [])
        if isinstance(rows, Mapping):
            rows = rows.get('supports', [])
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2 or not isinstance(row[0], Mapping):
                continue
            coefficients = row[0]
            values = [-float(coefficients.get(f'alpha[{i},{j}]', 0.)) for j in range(ctx.n)]
            if all(v in (0., 1.) for v in values):
                result.add(tuple(j for j, v in enumerate(values) if v))
    return sorted(result, key=lambda subset: (len(subset), subset))


def _initialize_columns(data, tree, node, ctx, state, initial_policy, bundles, opened, deadline):
    report = dict(reaudited=0, added=0, original_tsp_solves=0, native_failures=[], seconds=0.)
    start = time.perf_counter()
    for i in opened:
        # A serializable cache never stores a residual objective as a cost.
        old = state['columns'].get(i, {})
        checked = {}
        for subset, column in old.items():
            fresh = _column(ctx, i, subset, column['arcs'])
            checked[tuple(subset)] = fresh
            report['reaudited'] += 1
        state['columns'][i] = checked
        for j in range(ctx.n):
            if not ctx.active[j]:
                continue
            try:
                col = _column(ctx, i, (j,), ((0,j+1),(j+1,0)))
            except ValueError:
                continue
            report['added'] += _store_column(state, ctx, i, col)
        native = None
        for subset in _subsets(initial_policy, bundles, node, ctx, i):
            if not subset or subset in checked or time.perf_counter() >= deadline:
                continue
            alpha = tuple(int(j in subset) for j in range(ctx.n))
            try:
                ctx.check_route_state(i, alpha, 1)
            except ValueError:
                continue
            # Even without a native engine, an audited own-root cycle is a
            # valid column upper cost. Shortening it affects candidates only.
            path = (0,)+tuple(j+1 for j in subset)+(0,)
            col = _column(ctx, i, subset, tuple(zip(path[:-1],path[1:])))
            try:
                from solvers.lrp_native_oracle import LRPNativeRouteOracle, NativeUnavailable
                if native is None:
                    native = LRPNativeRouteOracle(data, tree[3][node.index*ctx.m+i])
                remaining = deadline-time.perf_counter()
                if remaining > .005:
                    ans = native.solve_fixed(alpha, 1, time_limit=min(.5, remaining))
                    report['original_tsp_solves'] += 1
                    arcs = []
                    for key, value in (ans.get('x') or {}).items():
                        match = re.fullmatch(r'r\[(\d+),(\d+),(\d+)\]', key)
                        if match and value:
                            fi, v, w = map(int, match.groups())
                            if fi != i or value != 1.:
                                raise ValueError('invalid original physical route incumbent')
                            arcs.append((v, w))
                    improved = _column(ctx, i, subset, arcs)
                    if improved['cost'] < col['cost']:
                        col = improved
            except (NativeUnavailable, ValueError) as exc:
                report['native_failures'].append(str(exc))
            report['added'] += _store_column(state, ctx, i, col)
    report['seconds'] = time.perf_counter()-start
    return report


def _coverage_rewards(data, node, archive, root, p, deadline):
    """Customer coverage duals are candidates, never physical lower bounds."""
    started = time.perf_counter()
    record = dict(status='budget_exhausted', seconds=0., objective_NOT_a_physical_bound=None)
    if deadline-started <= .02:
        return None, record
    model = StageModelBuilder(data).build_stage_problem(2, node, archive, root,
                                                       learned_cut_purpose='dual_lp')
    lp = None
    try:
        if deadline-time.perf_counter() <= .02:
            return None, record
        lp = model.relax()
        _options(lp)
        lp.Params.TimeLimit = max(1e-6, deadline-time.perf_counter()-.01)
        lp.optimize()
        record['status'] = int(lp.Status)
        if lp.Status != gp.GRB.OPTIMAL:
            return None, record
        lam = [min(p[j], max(0., float(lp.getConstrByName(f'R1_service_{j}').Pi)))
               if p[j] else 0. for j in range(len(p))]
        record['objective_NOT_a_physical_bound'] = float(lp.ObjVal)
        return lam, record
    finally:
        if lp is not None:
            lp.dispose()
        model.dispose()
        record['seconds'] = time.perf_counter()-started


def _restricted_rewards(ctx, state, opened, p, center, box, deadline):
    """Optimize the restricted dual. Its objective is not certified Q(A)."""
    if deadline-time.perf_counter() <= .01:
        return None, None
    model = gp.Model('lrp_physical_restricted_route_dual')
    try:
        _options(model)
        active = [j for j in range(ctx.n) if ctx.active[j]]
        lower = {j:max(0.,center[j]-box) if center is not None else 0. for j in active}
        upper = {j:min(p[j],center[j]+box) if center is not None else p[j] for j in active}
        reward = model.addVars(active, lb=lower, ub=upper, name='lambda')
        beta = model.addVars(opened, lb=-gp.GRB.INFINITY, ub=0., name='beta')
        model.setObjective(gp.quicksum(reward[j] for j in active)+gp.quicksum(beta[i] for i in opened),gp.GRB.MAXIMIZE)
        for i in opened:
            for column in state['columns'].get(i, {}).values():
                model.addConstr(gp.quicksum(reward[j] for j in column['customers'])+beta[i] <= column['cost'])
        remaining = deadline-time.perf_counter()
        if remaining <= .01:
            return None, None
        model.Params.TimeLimit = remaining
        model.optimize()
        # A limited LP incumbent could be clamped into the reward box too,
        # but an optimal restricted candidate makes the diagnostics clearer.
        if model.Status != gp.GRB.OPTIMAL:
            return None, None
        lam = [min(p[j], max(0.,float(reward[j].X))) if j in reward else 0. for j in range(ctx.n)]
        return lam, float(model.ObjVal)
    finally:
        model.dispose()


def _analytic_beta(lam):
    # Nonnegative physical cost and binary service: pricing >= -sum(lambda).
    return round_down(-sum((_F(v) for v in lam), Fraction()))


def _prior_beta(state, i, lam):
    value, source = _analytic_beta(lam), dict(source='nonnegative_cost_minus_all_rewards', certified=True)
    for certificate_index, entry in enumerate(state.get('certificates', [])):
        if all(old >= new for old, new in zip(entry['rewards'], lam)):
            candidate = entry['beta'][i]
            if candidate > value:
                value = candidate
                source = dict(source='cached_coordinatewise_larger_rewards', certified=True,
                              rewards=list(entry['rewards']), certified_lower_bound=candidate,
                              certificate_index=certificate_index,
                              original_source=entry['evidence'][i].get('source'))
    return value, source


def _pricing(data, tree, node, ctx, state, i, lam, deadline, model_cache):
    """Build/continue one true physical pricing model and audit both channels."""
    started = time.perf_counter()
    lower, evidence = _prior_beta(state, i, lam)
    record = dict(facility=i, route_key=ctx.route_key(i), beta=lower, evidence=evidence,
                  pricing_solves=0, new_columns=0, seconds=0., status='analytic_bound',
                  build_seconds=0., reused_same_rewards=False)
    remaining = deadline-started
    if remaining <= .02 or not any(lam):
        if not any(lam):
            record.update(beta=0., evidence=dict(source='nonnegative_cost_empty_route', certified=True))
        return record
    key = tuple(lam)
    cached = model_cache.get(i)
    if cached is not None and cached[0] != key:
        cached[1].dispose()
        del model_cache[i]
        cached = None
    if cached is None:
        pi = {f'alpha[{i},{j}]':lam[j] for j in range(ctx.n)}
        pi[f'u[{i}]'] = 0.
        model = SubproblemBuilder(data).build_subproblem(3, tree[3][node.index*ctx.m+i], {}, pi)
        model_cache[i] = (key, model)
        record['build_seconds'] = time.perf_counter()-started
        if deadline-time.perf_counter() > .1:
            record['free_lp'] = _separate_free_routes(model, deadline, remaining)
            lp_lower = record['free_lp']['certified_lower_bound']
            if lp_lower is not None and lp_lower > lower:
                lower = min(0., lp_lower)
                evidence = dict(source='optimal_free_LP_ObjBound', certified=True,
                                detail=record['free_lp'])
    else:
        model = cached[1]
        record['reused_same_rewards'] = True
    # Matrix/primal readback and route auditing are part of the cooperative
    # deadline; reserve a measured-size allowance before optimizer time.
    audit_reserve = max(.025, min(.35, .0000015*model.NumVars))
    remaining = deadline-time.perf_counter()-audit_reserve
    if remaining > .001:
        ev = evaluate_model(model, time_limit=remaining, mip_gap=0., threads=1)
        record['pricing_solves'] = 1
        record['status'] = ev.report['status']
        record['oracle'] = ev.summary()
        if ev.certified_lower_bound is not None and ev.certified_lower_bound > lower:
            lower = min(0., ev.certified_lower_bound)
            evidence = dict(source='physical_MIP_ObjBound', certified=True, detail=ev.summary())
        # Incumbents only create physically audited upper-cost columns.
        # They are never beta, nor lower bounds for the true node recourse.
        for solution in range(int(model.SolCount)):
            if time.perf_counter() >= deadline:
                break
            model.Params.SolutionNumber = solution
            candidate = copy(ev)
            candidate.x = np.asarray(model.getAttr('Xn', model.getVars()))
            try:
                matrix_primal_check(candidate.problem, candidate.x)
                a = candidate.values('a_copy')
                alpha, used = ctx.check_route_state(i, [a[j,] for j in range(ctx.n)],
                                                   candidate.values('u_copy')[()])
                tour = tour_from_evaluation(candidate, i)
                column = _column(ctx, i, tuple(j for j,b in enumerate(alpha) if b), tour['arcs'])
                if used:
                    record['new_columns'] += _store_column(state, ctx, i, column)
            except ValueError as exc:
                record.setdefault('discarded_incumbents', []).append(str(exc))
    record.update(beta=lower, evidence=evidence, seconds=time.perf_counter()-started)
    return record


def _value(cut, ctx, A):
    pi, intercept = cut
    return round_down(_F(intercept)+sum((_F(pi.get(f'A[{i},{ctx.interval}]',0.))*A[i]
                                      for i in range(ctx.m)), Fraction()))


def _remember_certificate(state, rewards, beta, evidence):
    entry = dict(rewards=list(rewards), beta=list(beta), evidence=deepcopy(evidence))
    for old in state['certificates']:
        if old['rewards'] == entry['rewards']:
            for i in range(len(beta)):
                if beta[i] > old['beta'][i]:
                    old['beta'][i], old['evidence'][i] = beta[i], deepcopy(evidence[i])
            return
    state['certificates'].append(entry)
    # Keep the strongest-candidate evidence separately; this bounded reward
    # history is an optional source of valid monotonic pricing lower bounds.
    del state['certificates'][:-80]


def _seed_node(data, tree, archive, node, state, root, initial_policy, bundles,
               *, deadline, pricing_limit, final_limit):
    started = time.perf_counter()
    ctx = _node_context(_instance(data), node, stage=2)
    A = [int(root[f'A[{i},{ctx.interval}]']) for i in range(ctx.m)]
    opened = [i for i in range(ctx.m) if A[i]]
    closed = [i for i in range(ctx.m) if not A[i]]
    p = [float(ctx.outsourcing[j]) if ctx.active[j] else 0. for j in range(ctx.n)]
    report = dict(node=int(node.index), context_key=ctx.key, A=A, opened=opened,
                  rounds=[], pricing_solves=0, added=0, status='budget_exhausted',
                  policy_UB=None, objective_space='unweighted_true_physical_recourse')
    budget = max(0., deadline-started)
    if budget <= .01:
        return report
    reserve = min(.35*budget, len(closed)*final_limit + len(opened)*pricing_limit)
    search_deadline = deadline-reserve
    report['initialization'] = _initialize_columns(data, tree, node, ctx, state,
        initial_policy, bundles, opened, min(search_deadline, started+.15*budget))
    lam = state.get('rewards')
    if lam is None:
        lam, report['coverage_candidate'] = _coverage_rewards(data, node, archive, root, p,
            min(search_deadline, time.perf_counter()+max(.1,.08*budget)))
    if lam is None:
        lam = list(p)
    center = list(lam)
    box = float(state.get('box', min(5., max(p, default=0.))))
    best = None
    for old in state.get('certificates', []):
        cut = build_physical_pricing_cut(data, node, old['rewards'], old['beta'], beta_certified=True)
        candidate = dict(rewards=old['rewards'], beta=old['beta'], evidence=old['evidence'],
                         cut=cut, value=_value(cut, ctx, A))
        if best is None or candidate['value'] > best['value']:
            best = deepcopy(candidate)
    model_cache, stalled, iteration = {}, 0, 0
    try:
        while search_deadline-time.perf_counter() > .02:
            kind, rmp = 'coverage_or_cached_rewards', None
            if iteration:
                unrestricted = iteration % 6 == 0
                lam, rmp = _restricted_rewards(ctx, state, opened, p,
                    None if unrestricted else center, None if unrestricted else box, search_deadline)
                kind = 'unrestricted_RMP' if unrestricted else 'stabilized_RMP'
                if lam is None:
                    break
            beta, evidence = zip(*[_prior_beta(state, i, lam) for i in range(ctx.m)])
            beta, evidence = list(beta), list(evidence)
            rows = []
            for position, i in enumerate(opened):
                remaining = search_deadline-time.perf_counter()
                if remaining <= .02:
                    break
                share = remaining/max(1,len(opened)-position)
                row = _pricing(data, tree, node, ctx, state, i, lam,
                    min(search_deadline,time.perf_counter()+min(pricing_limit,share)), model_cache)
                beta[i], evidence[i] = row['beta'], row['evidence']
                rows.append(row)
                report['pricing_solves'] += row['pricing_solves']
            cut = build_physical_pricing_cut(data, node, lam, beta, beta_certified=True)
            value = _value(cut, ctx, A)
            _remember_certificate(state, lam, beta, evidence)
            if best is None or value > best['value']+1e-7:
                best = deepcopy(dict(rewards=list(lam), beta=beta, evidence=evidence, cut=cut, value=value))
                center, stalled = list(lam), 0
            else:
                stalled += 1
            if stalled >= 4:
                box = min(max(p,default=0.), max(.01,box*1.5))
                stalled = 0
            report['rounds'].append(dict(iteration=iteration, kind=kind,
                restricted_objective_NOT_a_bound=rmp, certified_cut_at_A=value,
                best_certified_at_A=best['value'], pricing=rows, box=box,
                column_counts={i:len(state['columns'].get(i,{})) for i in opened},
                elapsed=time.perf_counter()-started))
            _trace(node.index, iteration, best['value'], deadline-time.perf_counter(),
                   sum(len(c) for c in state['columns'].values()))
            iteration += 1
            if not opened or not any(p):
                break
            # Certified pricing + restricted dual close the route relaxation
            # at this A, not the integral physical recourse or global LRP.
            if rmp is not None and rmp-value <= 1e-7+1e-9*max(1.,abs(rmp)):
                report['route_relaxation_closed'] = True
                break
        if best is None:
            lam = list(p)
            beta, evidence = zip(*[_prior_beta(state, i, lam) for i in range(ctx.m)])
            cut = build_physical_pricing_cut(data, node, lam, beta, beta_certified=True)
            best = dict(rewards=lam,beta=list(beta),evidence=list(evidence),cut=cut,value=_value(cut,ctx,A))
        # Certify coefficients for every physical facility. Open coefficients
        # tighten the current anchor; closed coefficients strengthen other A.
        report['final_pricing'] = []
        order = opened+closed
        for position, i in enumerate(order):
            remaining = deadline-time.perf_counter()
            if remaining <= .025:
                break
            limit = min(final_limit, remaining/max(1,len(order)-position))
            row = _pricing(data,tree,node,ctx,state,i,best['rewards'],
                           min(deadline,time.perf_counter()+limit),model_cache)
            if row['beta'] > best['beta'][i]:
                best['beta'][i], best['evidence'][i] = row['beta'], row['evidence']
            report['pricing_solves'] += row['pricing_solves']
            report['final_pricing'].append(row)
        best['cut'] = build_physical_pricing_cut(data,node,best['rewards'],best['beta'],beta_certified=True)
        best['value'] = _value(best['cut'],ctx,A)
        _remember_certificate(state,best['rewards'],best['beta'],best['evidence'])
        _trace(node.index, iteration, best['value'], deadline-time.perf_counter(),
               sum(len(c) for c in state['columns'].values()), final=True)
        state.update(rewards=list(best['rewards']),box=box)
        # The zero eta row already dominates a nonpositive cut at every A
        # when its intercept is zero. Other signed affine rows remain useful.
        if best['cut'][1] > 0.:
            target = archive.setdefault(2,{}).setdefault(node.index,[])
            report['added'] = int(add_unique_cut(target,*best['cut']))
        report.update(status='certified_physical_cut', certificate=best,
                      certified_cut_at_A=best['value'], route_relaxation_only=True,
                      cache_columns={i:len(c) for i,c in state['columns'].items()})
    finally:
        for _, model in model_cache.values():
            model.dispose()
    report['seconds'] = time.perf_counter()-started
    report['deadline_exhausted'] = time.perf_counter() >= deadline
    return report


def seed_full_fleet_route_bounds(prob_data, tree, cut_lag, *, time_limit_s,
        per_node_limit_s=5., partial_node_limit_s=8., initial_policy=None,
        s3_bundles=None, cache=None, max_nodes=1):
    """Budgeted physical pricing; retain the original public seed entry point.

    Default one context per call, rotating through the complete Stage-2 tree.
    per_node_limit_s is the search pricing slice, partial_node_limit_s the
    final certification slice. Cache contains only serializable data, never
    native model handles. All construction, column auditing and solver time
    count against the cooperative deadline; their overrun is reported.
    """
    started = time.perf_counter()
    budget = _positive(time_limit_s,'time_limit_s',allow_zero=True)
    pricing = _positive(per_node_limit_s,'per_node_limit_s')
    final = _positive(partial_node_limit_s,'partial_node_limit_s')
    if isinstance(max_nodes,bool) or int(max_nodes) != max_nodes or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    if cache is None:
        cache = {}
    if not isinstance(cache,dict):
        raise ValueError('physical seed cache must be a mutable dictionary')
    data = _instance(prob_data)
    instance_hash = data.logical_hash()
    if cache and (cache.get('version') != _CACHE_VERSION or cache.get('instance_sha256') != instance_hash):
        cache.clear()
    cache.setdefault('version',_CACHE_VERSION)
    cache.setdefault('instance_sha256',instance_hash)
    cache.setdefault('contexts',{})
    cache.setdefault('cursor',0)
    nodes = tree[2]
    nodes = sorted(nodes.values() if isinstance(nodes,Mapping) else nodes,key=lambda n:n.index)
    report = dict(attempted=False,attempts=0,solves=0,pricing_solves=0,added=0,
                  copied=0,bounds=[],failures=[],nodes=len(nodes),groups=len(nodes),
                  skipped=0,seconds=0.,deadline_exhausted=False,instance_sha256=instance_hash,
                  cache_version=_CACHE_VERSION,status='budget_exhausted')
    if budget <= 0. or not nodes:
        return report
    m,_,_,L,_ = data.shape
    policy = initial_policy.get(1,{}).get(0,{}) if isinstance(initial_policy,Mapping) else {}
    if not policy and isinstance(initial_policy,Mapping) and any(str(k).startswith('A[') for k in initial_policy):
        policy = initial_policy
    if not policy:
        policy = {f'A[{i},{k}]':1. for i in range(m) for k in range(L)}
        report['anchor_source'] = 'full_physical_availability_no_policy_supplied'
    else:
        report['anchor_source'] = 'audited_initial_policy_availability'
    root,_ = certify_stage1_forward_policy(prob_data,policy)
    deadline = started+budget
    count = min(int(max_nodes),len(nodes))
    cursor = int(cache['cursor'])%len(nodes)
    for offset in range(count):
        remaining = deadline-time.perf_counter()
        if remaining <= .02:
            break
        node = nodes[(cursor+offset)%len(nodes)]
        ctx = _node_context(data,node,stage=2)
        state = cache['contexts'].setdefault(ctx.key,dict(columns={},certificates=[]))
        report['attempted'] = True
        report['attempts'] += 1
        limit = remaining/(count-offset)
        try:
            row = _seed_node(prob_data,tree,cut_lag,node,state,root,initial_policy,s3_bundles,
                deadline=min(deadline,time.perf_counter()+limit),pricing_limit=pricing,final_limit=final)
            report['bounds'].append(row)
            report['solves'] += 1
            report['pricing_solves'] += row['pricing_solves']
            report['added'] += row['added']
        except gp.GurobiError as exc:
            if exc.errno not in (gp.GRB.Error.NO_LICENSE,gp.GRB.Error.SIZE_LIMIT_EXCEEDED):
                raise
            report['failures'].append(dict(node=int(node.index),reason=str(exc)))
            break
        finally:
            cache['cursor'] = (cursor+offset+1)%len(nodes)
    report.update(status='finished',seconds=time.perf_counter()-started,
                  next_node=int(nodes[int(cache['cursor'])].index))
    report['deadline_exhausted'] = report['seconds'] >= budget
    report['skipped'] = len(nodes)-report['attempts']
    return report


__all__ = ['seed_full_fleet_route_bounds']
