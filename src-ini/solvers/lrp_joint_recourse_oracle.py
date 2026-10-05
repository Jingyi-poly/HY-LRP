"""Bounded, full-domain physical recourse at one fixed availability vector.

Both modes use the same unit-customer SCF graph.  Demand appears only in the
separate warehouse-capacity constraint, so active zero-demand customers cannot
form disconnected cycles.  No route pool, previous assignment, metric shortcut
or nearest-neighbour restriction defines this model's domain.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import math
import time

import numpy as np

from core.solve_deadline import SolveDeadlineReached
from cuts.lrp_static_bounds import basic_route_cuts
from models.stage_builder import _as_cut, _instance, _node_context, _route_pools
from models.stage_model_core import (
    LinearMILP, Subproblem, InvalidSolverPrimal, audit_tour, binary_vector,
    evaluate_model, index, matrix_primal_check, service_block,
)
from solvers.lrp_physical_policy_pool import (
    _build_native, _remaining, audit_node_policy, audit_route,
)
from solvers.lrp_physical_types import (
    NodePhysicalCertificate, node_signature, validate_node_certificate,
)


class PhysicalRecourseValidityError(ValueError):
    """A physical-domain/certificate contradiction, with serializable evidence."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence

    def __reduce__(self):
        # BaseException pickles only ``args``.  Preserve the second required
        # constructor argument so a spawned worker's failure reaches its parent
        # intact instead of becoming an unpickle TypeError/hung result queue.
        return type(self), (str(self), self.evidence)


def add_physical_route_network(M, ctx, facility_id, alpha_columns, u_column, *,
                               continuous_arcs=False, cost_in_objective=True,
                               deadline=None):
    """Add complete-active own-root degrees and unit-count connectivity flow.

    ``alpha_columns`` includes every original customer, even inactive ones.
    This reusable block does not add service, capacity or nonempty constraints;
    the joint model uses the original service block, and a route LP must supply
    its own alpha/u domain.  Every real tour extends with flows m,m-1,...,1,0.
    """
    i = index(facility_id, ctx.m, "facility")
    if len(alpha_columns) != ctx.n:
        raise ValueError("full original alpha dimension is required")
    active = tuple(j for j in range(ctx.n) if ctx.active[j])
    vertices = (0,) + tuple(j+1 for j in active)
    count = len(active)
    r, flow = {}, {}
    for v in vertices:
        _remaining(deadline)
        for w in vertices:
            if v == w:
                continue
            r[i,v,w] = M.var("r", (i,v,w),
                float(ctx.route_cost[i,v,w]) if cost_in_objective else 0.,
                integer=not continuous_arcs)
            flow[i,v,w] = M.var("connectivity_flow", (i,v,w), ub=count, integer=False)
            M.row(f"SCF_link_{i}_{v}_{w}", [(flow[i,v,w],1.),(r[i,v,w],-count)], ub=0.)
    for j, alpha in enumerate(alpha_columns):
        _remaining(deadline)
        v = j+1
        outgoing = [(r[i,v,w],1.) for w in vertices if (i,v,w) in r]
        incoming = [(r[i,w,v],1.) for w in vertices if (i,w,v) in r]
        M.row(f"SCF_out_{i}_{j}", outgoing+[(alpha,-1.)], lb=0., ub=0.)
        M.row(f"SCF_in_{i}_{j}", incoming+[(alpha,-1.)], lb=0., ub=0.)
        if j in active:
            terms = [(flow[i,w,v],1.) for w in vertices if w != v]
            terms += [(flow[i,v,w],-1.) for w in vertices if w != v]
            M.row(f"SCF_consume_{i}_{j}", terms+[(alpha,-1.)], lb=0., ub=0.)
    M.row(f"SCF_root_out_{i}", [(r[i,0,j+1],1.) for j in active]+[(u_column,-1.)], lb=0., ub=0.)
    M.row(f"SCF_root_in_{i}", [(r[i,j+1,0],1.) for j in active]+[(u_column,-1.)], lb=0., ub=0.)
    M.row(f"SCF_root_supply_{i}",
        [(flow[i,0,j+1],1.) for j in active]+[(flow[i,j+1,0],-1.) for j in active]
        +[(alpha_columns[j],-1.) for j in active], lb=0., ub=0.)
    return {"r": r, "f": flow}


def build_joint_recourse_problem(prob_data, node, A_mask, cut_lag=None, *,
                                  mode="exact_mip", deadline=None):
    """Pure canonical builder.  Costs are physical, unweighted node units."""
    if mode not in {"exact_mip", "network_relax"}:
        raise ValueError("mode must be exact_mip or network_relax")
    _remaining(deadline)
    data = _instance(prob_data)
    ctx = _node_context(data, node, stage=2)
    node_signature(data, ctx)  # fail closed for unsupported fees/arc domains
    mask = binary_vector(A_mask, ctx.m, "availability")
    M = LinearMILP(connectivity="physical_count_scf")
    service_block(M, ctx, fixed_A=mask)
    _remaining(deadline)
    pools, _ = _route_pools(ctx, node, cut_lag or {})
    for i in range(ctx.m):
        _remaining(deadline)
        if mask[i]:
            add_physical_route_network(M, ctx, i,
                [M.groups['alpha'][i,j] for j in range(ctx.n)], M.groups['u'][i,],
                continuous_arcs=mode=="network_relax",
                cost_in_objective=mode=="exact_mip", deadline=deadline)
        route_terms = [(col,float(ctx.route_cost[i,v,w]))
                       for (owner,v,w),col in M.groups.get('r',{}).items() if owner==i]
        if mode == "network_relax":
            theta = M.var('theta',(i,),1.,ub=math.inf,integer=False)
            M.row(f"physical_route_epigraph_{i}", [(theta,1.)]+[(c,-a) for c,a in route_terms],lb=0.)
            lhs = [(theta,1.)]
        else:
            # Exact aggregation: one defining equality replaces expansion of
            # the same full arc-cost vector in every archived route cut.
            # Keep objective costs on r; this auxiliary has zero objective.
            route_value = M.var('physical_route_cost', (i,), ub=math.inf, integer=False)
            M.row(f"physical_route_cost_definition_{i}",
                  [(route_value,1.)]+[(col,-cost) for col,cost in route_terms],
                  lb=0., ub=0.)
            lhs = [(route_value,1.)]
        # Include mandatory original floors and the frozen complete S3 archive.
        for h, item in enumerate([*basic_route_cuts(ctx,i), *deepcopy(pools.get(i,()))]):
            if h % 32 == 0:
                _remaining(deadline)
            cut = _as_cut(item,ctx,i)
            terms = lhs+[(M.groups['alpha'][i,j],-a) for j,a in enumerate(cut.coefficients[:-1]) if a]
            if cut.coefficients[-1]:
                terms += [(M.groups['u'][i,],-cut.coefficients[-1])]
            M.row(f"physical_archive_{i}_{h}",terms,lb=cut.intercept)
    for group, values in M.groups.items():
        for key, col in values.items():
            M.names[col] = f"{group}[{','.join(map(str,key))}]"
    M.validate()
    _remaining(deadline)
    return Subproblem(M,"assignment","forward",
        "true_node" if mode=="exact_mip" else "physical_network_relaxation",
        ctx,fixed_state=mask,domain="complete_fixed_availability",
        uses_exact_routes=mode=="exact_mip",
        name=f"physical_{mode}_t{ctx.period}_s{ctx.scenario}")


def _outsourcing(node, ctx):
    state = {f'alpha[{i},{j}]':0. for i in range(ctx.m) for j in range(ctx.n)}
    state.update({f'u[{i}]':0. for i in range(ctx.m)})
    state.update({f'e[{j}]':float(ctx.active[j]) for j in range(ctx.n)})
    return {2:{node.index:state},3:{rid:{} for rid in node.successor}}


def _extract_policy(evaluation, tree, node):
    M, ctx = evaluation.problem.linear, evaluation.problem.context
    state = {}
    for group in ('alpha','u','e'):
        for key,value in evaluation.values(group).items():
            state[f"{group}[{','.join(map(str,key))}]"] = float(value)
    routes = {rid:{} for rid in node.successor}
    by_i = {int(tree[3][rid].info):rid for rid in node.successor}
    if set(by_i) != set(range(ctx.m)):
        raise ValueError("incorrect successor-facility mapping")
    for (i,v,w), value in evaluation.values('r').items():
        if value:
            routes[by_i[i]][f'r[{i},{v},{w}]'] = 1.
    return {2:{node.index:state},3:routes}


def _start_vector(problem, tree, node, policy):
    M,ctx = problem.linear,problem.context
    x = np.zeros(len(M.names))
    state = policy[2][node.index]
    for i,value in enumerate(problem.fixed_state):
        x[M.groups['z'][i,]] = value
    for group in ('alpha','u','e'):
        for key,col in M.groups[group].items():
            x[col] = state[f"{group}[{','.join(map(str,key))}]"]
    for rid in node.successor:
        i = int(tree[3][rid].info)
        arcs = [tuple(map(int,name[2:-1].split(',')))[1:]
                for name,value in policy[3][rid].items() if name.startswith('r[') and value]
        alpha = [state[f'alpha[{i},{j}]'] for j in range(ctx.n)]
        tour = audit_tour(ctx,i,alpha,state[f'u[{i}]'],arcs)
        count = len(tour['customers'])
        ordered = list(zip(tour['local_route'][:-1],tour['local_route'][1:]))
        for pos,(v,w) in enumerate(ordered):
            x[M.groups['r'][i,v,w]] = 1.
            x[M.groups['connectivity_flow'][i,v,w]] = count-pos
        if 'physical_route_cost' in M.groups:
            value = M.groups['physical_route_cost'][i,]
            x[value] = math.fsum(float(ctx.route_cost[i,v,w]) for v,w in ordered)
        if 'theta' in M.groups:
            theta = M.groups['theta'][i,]
            # Start is a proposal only; choose a value satisfying all theta rows.
            value = 0.
            for terms,lo in zip(M.rows,M.row_lb):
                if theta in terms and math.isfinite(lo):
                    value = max(value,(lo-sum(a*x[c] for c,a in terms.items() if c!=theta))/terms[theta])
            x[theta] = value
    matrix_primal_check(problem,x)
    return x


def solve_joint_recourse(prob_data, tree, node, A_mask, cut_lag=None, *,
                         mode="exact_mip", time_limit_s=30., deadline=None,
                         envelope_version=0, previous=None, incumbent_policy=None,
                         cached_policy=None, eta_ref=None, accept_tolerance=1.,
                         mip_gap=1e-3, env=None):
    """Return a typed physical certificate; never mutate archive/pool/history.

    A network incumbent cannot become a physical upper.  Separate physical
    starts/baselines remain available at any termination.  Bounds retain their
    provenance and are never clipped to a feasible upper.
    """
    started = time.monotonic()
    if mode not in {"exact_mip","network_relax"}:
        raise ValueError("invalid joint mode")
    limit = float(time_limit_s)
    if not math.isfinite(limit) or limit < 0 or not math.isfinite(mip_gap) or mip_gap < 0:
        raise ValueError("invalid joint time limit or MIP gap")
    if deadline is not None and (math.isnan(float(deadline)) or float(deadline)==-math.inf):
        raise ValueError("invalid joint deadline")
    stop = min(started+limit,math.inf if deadline is None else float(deadline))
    data = _instance(prob_data); ctx = _node_context(data,node,stage=2)
    signature = node_signature(data,ctx); mask = binary_vector(A_mask,ctx.m,"availability")
    if type(envelope_version) is not int or envelope_version < 0:
        raise ValueError("invalid envelope version")
    if cached_policy is not None:
        if incumbent_policy is not None:
            raise ValueError("supply only one physical policy start")
        incumbent_policy = cached_policy
    build_seconds = solve_seconds = audit_seconds = 0.
    tick = time.monotonic()
    policy,upper,audit = audit_node_policy(data,tree,node,mask,_outsourcing(node,ctx))
    lower,lower_source,lower_domain = 0.,"nonnegative_cost_floor",mode
    upper_source = "audited_all_outsourcing"
    if previous is not None:
        validate_node_certificate(previous,data,node,mask)
        if previous.q_lower is not None and previous.q_lower > lower:
            lower,lower_source,lower_domain = previous.q_lower,previous.lower_source,previous.domain_kind
        if previous.policy is not None:
            old,old_upper,old_audit = audit_node_policy(data,tree,node,mask,previous.policy)
            if previous.q_upper is not None and abs(old_upper-previous.q_upper)>2e-6:
                raise ValueError("cached physical upper does not equal policy cost")
            if old_upper < upper:
                policy,upper,audit,upper_source = old,old_upper,old_audit,"audited_cached_policy"
    if incumbent_policy is not None:
        part = {2:{node.index:deepcopy(incumbent_policy[2][node.index])},
                3:{rid:deepcopy(incumbent_policy[3][rid]) for rid in node.successor}}
        # Source can belong to another A, but it must itself be a valid node
        # policy.  Only whole routes on newly closed facilities are removed.
        part,_,_ = audit_node_policy(data,tree,node,(1,)*ctx.m,part)
        state = part[2][node.index]
        for rid in node.successor:
            i = int(tree[3][rid].info)
            if not mask[i]:
                for j in range(ctx.n):
                    if state[f'alpha[{i},{j}]']:
                        state[f'e[{j}]'] = 1.
                        state[f'alpha[{i},{j}]'] = 0.
                state[f'u[{i}]'] = 0.
                part[3][rid] = {}
        part,price,checked = audit_node_policy(data,tree,node,mask,part)
        if price < upper:
            policy,upper,audit,upper_source = part,price,checked,"audited_node_policy"
    audit_seconds += time.monotonic()-tick
    diagnostic = {"requested_mode":mode,"solver_executed":False,"model_built":False,
                  "node_unweighted":True,"node_signature":signature,
                  "envelope_version":envelope_version,"deadline":stop}
    model = None
    problem = None  # early cached-bound contradictions have no constructed model

    def finish(status):
        nonlocal model, audit_seconds
        if lower > upper + 2e-6+1e-10*max(1.,abs(lower),abs(upper)):
            raise PhysicalRecourseValidityError("physical lower exceeds audited upper",
                {"diagnostics":deepcopy(diagnostic),"lower":lower,"upper":upper,
                 "instance":data,"node":node,"policy":deepcopy(policy),
                 "A_mask":mask,"model_spec":None if problem is None else deepcopy(problem.linear),
                 "cut_lag":deepcopy(cut_lag or {}),"mode":mode,
                 "lower_source":lower_source,"upper_source":upper_source})
        audit_started = time.monotonic()
        routes = []
        for route in audit['tours']:
            if route['customers']:
                routes.append(audit_route(data,node,route['facility'],
                    [j-1 for j in route['local_route'][1:-1]],source=upper_source))
        audit_seconds += time.monotonic()-audit_started
        diagnostic['policy_audit_completed_monotonic'] = time.monotonic()
        if model is not None:
            model.dispose()
            model = None
        returned = time.monotonic()
        diagnostic.update(returned_monotonic=returned,
            finished_after_deadline=returned > stop,
            deadline_overrun_seconds=max(0.,returned-stop))
        return NodePhysicalCertificate((int(ctx.period),int(ctx.scenario)),signature,mask,
            lower,upper,deepcopy(policy),lower_source,upper_source,status,
            upper-lower <= 2e-6+1e-10*max(1.,abs(lower),abs(upper)),True,
            lower_domain,envelope_version,returned-started,build_seconds,
            solve_seconds,audit_seconds,deepcopy(diagnostic),tuple(routes))

    if not any(ctx.active):
        return finish("EMPTY_NODE")
    if time.monotonic() >= stop:
        return finish("NO_BUDGET")
    import gurobipy as gp
    try:
        tick = time.monotonic()
        problem = build_joint_recourse_problem(data,node,mask,cut_lag,mode=mode,deadline=stop)
        model,variables,matrix_audit = _build_native(problem.linear,env,stop)
        model.ModelName = problem.name
        diagnostic.update(model_built=True,matrix_audit=matrix_audit,
                          variables=len(problem.linear.names),rows=len(problem.linear.rows))
        # Both independent all-outsourcing and the best physical policy are
        # feasible starts; the second start is not called a solver incumbent.
        starts = [_outsourcing(node,ctx),policy]
        model.NumStart = len(starts)
        for slot,start in enumerate(starts):
            _remaining(stop)
            model.Params.StartNumber = slot
            model.setAttr('Start',variables,_start_vector(problem,tree,node,start).tolist())
        build_seconds += time.monotonic()-tick
        @contextmanager
        def before_optimize():
            nonlocal solve_seconds
            model.Params.Seed = 42
            model.Params.BestBdStop = math.inf
            model.Params.BestObjStop = -math.inf
            if eta_ref is not None:
                ref = float(eta_ref)
                if not math.isfinite(ref) or not math.isfinite(accept_tolerance) or accept_tolerance < 0:
                    raise ValueError("invalid physical stopping reference")
                sep = 1e-6+1e-9*max(1.,abs(ref))
                model.Params.BestBdStop = ref+sep+1e-8+1e-11*max(1.,abs(ref))
                # A network objective is not an actual physical upper.
                if mode == "exact_mip":
                    model.Params.BestObjStop = ref+accept_tolerance
            diagnostic['solver_executed'] = True
            diagnostic['solver_parameters'] = dict(Seed=42,Threads=1,
                FeasibilityTol=model.Params.FeasibilityTol,
                IntFeasTol=model.Params.IntFeasTol,
                OptimalityTol=model.Params.OptimalityTol)
            begun = time.monotonic()
            try:
                yield
            finally:
                solve_seconds += time.monotonic()-begun
        tick = time.monotonic()
        old_solve = solve_seconds
        try:
            ev = evaluate_model(model,problem=problem,time_limit=_remaining(stop),
                deadline=stop,mip_gap=mip_gap,threads=1,optimize_context=before_optimize)
        finally:
            audit_seconds += max(0.,time.monotonic()-tick-(solve_seconds-old_solve))
        diagnostic['evaluation'] = ev.summary()
        if (ev.report.get('lower_bound_source') == 'objective_bound_continuous'
                and ev.certified_lower_bound is not None and ev.certified_lower_bound > lower):
            lower,lower_source,lower_domain = ev.certified_lower_bound,mode+"_bound",mode
        elif ev.certified_lower_bound is not None:
            diagnostic['unpropagated_fallback_bound_source'] = ev.report.get('lower_bound_source')
        tick = time.monotonic()
        if mode == "exact_mip" and ev.x is not None:
            candidate = _extract_policy(ev,tree,node)
            candidate,value,checked = audit_node_policy(data,tree,node,mask,candidate)
            if value < upper:
                policy,upper,audit,upper_source = candidate,value,checked,"audited_joint_integer_policy"
        elif ev.x is not None:
            diagnostic['network_primal_is_not_physical_upper'] = True
        audit_seconds += time.monotonic()-tick
        return finish(ev.report['status'])
    except SolveDeadlineReached:
        if not diagnostic['model_built']:
            build_seconds = time.monotonic()-started-audit_seconds
        return finish("NO_BUDGET")
    except InvalidSolverPrimal as exc:
        diagnostic['rejected_solver_primal'] = str(exc)
        return finish("INVALID_SOLVER_PRIMAL")
    except gp.GurobiError as exc:
        diagnostic['solver_error'] = repr(exc)
        return finish("SOLVER_ERROR")
    finally:
        if model is not None:
            model.dispose()


__all__ = ['add_physical_route_network','build_joint_recourse_problem',
           'solve_joint_recourse','PhysicalRecourseValidityError']
