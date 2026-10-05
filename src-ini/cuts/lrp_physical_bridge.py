"""Certified physical information -> the ordinary LRP cut archive.

Costs are unweighted node/route costs.  This module never writes a LevelSet
support, target, UB or scalar historical LB.  Physical cuts survive monotone
archive additions; installation returns scopes for the caller to invalidate.
"""
from __future__ import annotations

from fractions import Fraction as F
import math
import time

from core.solve_deadline import SolveDeadlineReached
from core.solver_bounds import extract_verified_fixed_rhs_dual_cut
from cuts.benders_cuts import _fraction_to_finite_float_down, add_unique_cut
from cuts.lrp_static_bounds import basic_node_cuts, basic_route_cuts
from models.stage_builder import _as_cut, _instance, _node_context, _route_pools, _state_keys
from models.stage_model_core import AffineCut, LinearMILP, Subproblem, binary_vector, index
from solvers.forward_s2_reference import _CanonicalLpDualView
from solvers.lrp_joint_recourse_oracle import add_physical_route_network
from solvers.lrp_physical_policy_pool import _build_native, _remaining
from solvers.lrp_physical_types import NodePhysicalCertificate, NODE_LOWER_SOURCES, node_signature, validate_node_certificate

BRIDGE_SCHEMA = "lrp_physical_bridge_v1"
BRIDGE_SOURCES = frozenset({"physical_monotone_state", "physical_fixed_route_lp"})


def _down(value):
    return _fraction_to_finite_float_down(value, label="physical bridge payment")


def _ratio(value):
    return (value.numerator, value.denominator)


def _version(value):
    if type(value) is not int or value < 0:
        raise ValueError("envelope version must be a nonnegative integer")
    return value


def _exact_value(cut, state):
    return F(cut.intercept) + sum((F(a)*F(float(x)) for a,x in zip(cut.coefficients,state)),F())


def _paid_cut(level, scope, coefficients, intercept, certificate):
    # All parent coordinates lie in [0,1].  Directed slope rounding never
    # strengthens the exact row, including cancellation in ell - L0.
    paid = tuple(_down(x) for x in coefficients)
    beta = _down(intercept)
    payment = intercept-F(beta)+sum((min(F(),a-F(b)) for a,b in zip(coefficients,paid)),F())
    if payment < 0:
        raise ValueError("negative full-box payment")
    return AffineCut(level,scope,beta,paid,
        "availability_box" if level=="node" else "parent",
        {**certificate,"fullbox_payment_exact":_ratio(payment)})


def build_eta_state_cut(prob_data, node, certificate, *, global_lower=None):
    """Support true Q using its availability monotonicity, without metricity.

    Without a newly opened facility A <= Abar and Q(A) >= Q(Abar) >= ell.
    With at least one newly opened facility, ell-D*sum(new A) <= L0.
    The all-open anchor therefore gives a valid constant global lower bound.
    V1 proves L0=0 from nonnegative original costs; a bare positive proposed
    global floor is rejected.  Negative global floors are valid but weaker.
    """
    data = _instance(prob_data)
    ctx = _node_context(data,node,stage=2)
    if not isinstance(certificate,NodePhysicalCertificate):
        raise TypeError("eta requires a typed true-node lower certificate")
    validate_node_certificate(certificate,data,node,certificate.A_mask)
    _version(certificate.envelope_version)
    if certificate.q_lower is None:
        return None
    if any(not math.isfinite(float(x)) or float(x)<0
           for array in (ctx.route_cost.flat,ctx.outsourcing) for x in array):
        return None  # no proven global floor for this supported construction
    floor = 0. if global_lower is None else float(global_lower)
    if not math.isfinite(floor) or floor > 0:
        raise ValueError("a positive global floor needs a separate full-domain proof")
    if certificate.lower_source=="nonnegative_cost_floor" and certificate.q_lower>0:
        raise ValueError("nonnegative costs prove only the zero floor")
    ell = max(F(certificate.q_lower),F(floor))
    delta = ell-F(floor)
    meta = dict(schema=BRIDGE_SCHEMA,source="physical_monotone_state",
        node_signature=node_signature(data,ctx),node_index=int(node.index),
        envelope_version=certificate.envelope_version,A_mask=certificate.A_mask,
        lower_source=certificate.lower_source,lower_exact=_ratio(ell),
        global_floor_exact=_ratio(F(floor)),delta_exact=_ratio(delta),
        global_floor_source="nonnegative_original_physical_costs")
    return _paid_cut("node",ctx.key,
        tuple(-delta if not a else F() for a in certificate.A_mask),ell,meta)


def build_route_lp_problem(prob_data,node,facility_id,alpha_bar,u_bar,*,deadline=None):
    """Pure full-active SCF LP; pins change RHS, never variable bounds/domain."""
    _remaining(deadline)
    data = _instance(prob_data)
    ctx = _node_context(data,node,stage=2)
    node_signature(data,ctx)
    i = index(facility_id,ctx.m,"facility")
    alpha,u = ctx.check_route_state(i,alpha_bar,u_bar)
    M = LinearMILP(connectivity="physical_count_scf")
    a = [M.var("alpha_copy",(i,j),integer=False) for j in range(ctx.n)]
    use = M.var("u_copy",(i,),integer=False)
    for j in range(ctx.n):
        M.row(f"physical_activation_{j}",[(a[j],1.),(use,-1.)],ub=0.)
    M.row("physical_nonempty",[(use,1.)]+[(col,-1.) for col in a],ub=0.)
    M.row("physical_capacity",[(a[j],float(ctx.demand[j])) for j in range(ctx.n)]
          +[(use,-float(ctx.capacity[i]))],ub=0.)
    add_physical_route_network(M,ctx,i,a,use,continuous_arcs=True,
                               cost_in_objective=True,deadline=deadline)
    bindings = []
    for j,col in enumerate(a):
        name = f"physical_pin_alpha_{j}"
        M.row(name,[(col,1.)],lb=float(alpha[j]),ub=float(alpha[j]))
        bindings.append((f"alpha[{i},{j}]",name))
    M.row("physical_pin_u",[(use,1.)],lb=float(u),ub=float(u))
    bindings.append((f"u[{i}]","physical_pin_u"))
    for group,values in M.groups.items():
        for key,col in values.items():
            M.names[col] = f"{group}[{','.join(map(str,key))}]"
    M.validate()
    if any(M.integer) or not all(math.isfinite(x) for x in [*M.lower,*M.upper]):
        raise ValueError("physical route support requires a finite continuous box")
    _remaining(deadline)
    spec = Subproblem(M,"tsp","forward","physical_route_lp",ctx,
        fixed_state=(*alpha,u),facility=i,domain="complete_active_parent_box",
        uses_exact_routes=False,name=f"physical_route_lp_t{ctx.period}_s{ctx.scenario}_i{i}")
    return spec,tuple(bindings)


def _dual_diagnostics(view,bindings,intercept):
    """Record the same finite-box payment; never use ObjVal as a certificate."""
    pins = {row for _,row in bindings}
    variables = view.getVars()
    residual = [F(v.Obj) for v in variables]
    constant = F(view.ObjCon)
    signs = 0
    for con in view.getConstrs():
        raw = F(con.Pi)
        pi = min(F(),raw) if con.Sense=="<" else max(F(),raw) if con.Sense==">" else raw
        signs += int(pi!=raw)
        if con.ConstrName not in pins:
            constant += pi*F(con.RHS)
        row = view.getRow(con)
        for p in range(row.size()):
            residual[row.getVar(p).index] -= pi*F(row.getCoeff(p))
    payment = sum((min(r*F(v.LB),r*F(v.UB)) for r,v in zip(residual,variables)),F())
    exact = constant+payment
    if F(intercept)>exact:
        raise ValueError("verified dual exceeds finite-box exact intercept")
    return dict(sign_projections=signs,
        residual_max_abs_exact=_ratio(max(map(abs,residual),default=F())),
        residual_box_term_exact=_ratio(payment),
        intercept_before_rounding_exact=_ratio(exact),
        intercept_rounding_payment_exact=_ratio(exact-F(intercept)))


def _certify_route_lp(spec,bindings,model,*,signature,node_index,envelope_version):
    view = _CanonicalLpDualView(model,spec.linear)
    # Bindings must see the same canonical Pi/RHS as the complete residual audit.
    by_name = {c.ConstrName:c for c in view.getConstrs()}
    view.getConstrByName = by_name.get
    slope,beta = extract_verified_fixed_rhs_dual_cut(view,bindings,label="physical route support")
    diagnostics = _dual_diagnostics(view,bindings,beta)
    ctx,i = spec.context,spec.facility
    cut = AffineCut("route",ctx.route_key(i),beta,
        tuple(slope[key] for key in _state_keys(ctx,i)),"parent",
        dict(schema=BRIDGE_SCHEMA,source="physical_fixed_route_lp",
             node_signature=signature,node_index=node_index,facility=i,
             envelope_version=envelope_version,anchor=tuple(spec.fixed_state),
             complete_active_domain=True,finite_parent_bounds=True,
             dual_certificate="canonical_fixed_rhs_exact_residual",**diagnostics))
    return cut,diagnostics


def build_route_lp_support(prob_data,node,facility_id,alpha_bar,u_bar,*,deadline,
                           time_limit_s=2.,envelope_version=0,env=None):
    """At most two seconds including build, within caller's shared deadline.

    Only an OPTIMAL continuous LP's canonical, residual-paid dual is returned.
    Failure/expiry produces diagnostics, never a guessed lower endpoint.
    """
    started = time.monotonic()
    version = _version(envelope_version)
    limit = float(time_limit_s)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("time limit must be finite and nonnegative")
    if deadline is not None and math.isnan(float(deadline)):
        raise ValueError("deadline must not be NaN")
    stop = min(started+min(limit,2.),math.inf if deadline is None else float(deadline))
    diagnostic = dict(executed=False,certified=False,status="DEADLINE",build_seconds=0.,
                      solve_seconds=0.,audit_seconds=0.,envelope_version=version)
    model = None
    try:
        _remaining(stop)
        spec,bindings = build_route_lp_problem(prob_data,node,facility_id,alpha_bar,u_bar,deadline=stop)
        model,_,matrix_audit = _build_native(spec.linear,env,stop)
        diagnostic.update(build_seconds=time.monotonic()-started,matrix_audit=matrix_audit)
        model.Params.TimeLimit = _remaining(stop)
        from core.backend_telemetry import backend_call
        model.Params.Threads = 1
        model.Params.Seed = 42
        before = time.monotonic()
        diagnostic["executed"] = True
        with backend_call("gurobi", purpose="physical_route_support"):
            model.optimize()
        diagnostic.update(solve_seconds=time.monotonic()-before,status=int(model.Status))
        if int(model.Status)!=2:
            diagnostic["reason"]="NO_DUAL_CERTIFICATE"
            return None,diagnostic
        _remaining(stop)
        before = time.monotonic()
        try:
            cut,paid = _certify_route_lp(spec,bindings,model,
                signature=node_signature(prob_data,node),node_index=int(node.index),envelope_version=version)
        except (ValueError,KeyError) as exc:
            diagnostic.update(reason="NO_DUAL_CERTIFICATE",error=str(exc),
                              audit_seconds=time.monotonic()-before)
            return None,diagnostic
        completed = time.monotonic()
        diagnostic.update(paid,audit_seconds=completed-before,
                          audit_completed_monotonic=completed)
        # Exact residual payment can consume the remaining local LP budget.
        # Report its tail but do not return a support completed after the cap.
        _remaining(stop)
        diagnostic['certified'] = True
        return cut,diagnostic
    except SolveDeadlineReached:
        diagnostic["reason"]="DEADLINE"
        return None,diagnostic
    finally:
        if model is not None:
            model.dispose()
        returned = time.monotonic()
        diagnostic.update(wall_seconds=returned-started,returned_monotonic=returned,
                          deadline_monotonic=stop,overrun_seconds=max(0.,returned-stop))


def _state(ctx,facility,state):
    if isinstance(state,dict):
        state = tuple(state[key] for key in _state_keys(ctx,facility))
    if facility is None:
        return binary_vector(state,ctx.m,"availability")
    values = tuple(state)
    if len(values)!=ctx.n+1:
        raise ValueError("route candidate must have complete alpha/u coordinates")
    alpha,u = ctx.check_route_state(facility,values[:-1],values[-1])
    return (*alpha,u)


def physical_archive_value(prob_data,node,archive,state,*,facility_id=None):
    """Exact complete current envelope including mandatory basic cuts and zero."""
    ctx = _node_context(_instance(prob_data),node,stage=2)
    i = None if facility_id is None else index(facility_id,ctx.m,"facility")
    values = _state(ctx,i,state)
    if i is None:
        rows = [*basic_node_cuts(ctx),*archive.get(2,{}).get(node.index,())]
    else:
        pools,_ = _route_pools(ctx,node,archive)
        rows = [*basic_route_cuts(ctx,i),*pools[i]]
    return max([F(),*(_exact_value(_as_cut(row,ctx,i),values) for row in rows)])


def install_physical_cut(prob_data,node,archive,cut,*,envelope_version,expected_version,
                         candidate_states,sep_atol=1e-6,sep_rtol=1e-9):
    """Validate and install through add_unique_cut in the public archive.

    ``expected_version`` identifies this certified batch.  ``envelope_version``
    is its current, append-only version and may advance as sibling rows install.
    Scope/signature/expected batch must match; caller must never reuse a version
    after replacing/removing cuts.  Public tuple conversion is exact (no new
    cleaning).  The caller owns cache invalidation and same-A trial rescoring.
    """
    current,expected = _version(envelope_version),_version(expected_version)
    if current<expected:
        raise ValueError("envelope version regressed")
    ctx = _node_context(_instance(prob_data),node,stage=2)
    if not isinstance(cut,AffineCut):
        raise TypeError("only a certified bridge AffineCut can be installed")
    meta = cut.certificate
    if (meta.get("schema")!=BRIDGE_SCHEMA or meta.get("source") not in BRIDGE_SOURCES
        or meta.get("node_signature")!=node_signature(prob_data,node)
        or meta.get("node_index")!=int(node.index)
        or meta.get("envelope_version")!=expected):
        raise ValueError("physical bridge source, scope or version mismatch")
    i = None if cut.level=="node" else index(meta.get("facility"),ctx.m,"facility")
    _as_cut(cut,ctx,i)
    if i is None:
        if (meta["source"]!="physical_monotone_state"
            or meta.get("lower_source") not in NODE_LOWER_SOURCES
            or meta.get("global_floor_source")!="nonnegative_original_physical_costs"):
            raise ValueError("node cut requires a true-Q monotone certificate")
        stage,target = 2,node.index
    else:
        if (meta["source"]!="physical_fixed_route_lp"
            or meta.get("dual_certificate")!="canonical_fixed_rhs_exact_residual"
            or meta.get("complete_active_domain") is not True
            or meta.get("finite_parent_bounds") is not True):
            raise ValueError("route cut requires a complete finite-box LP certificate")
        _,mapping = _route_pools(ctx,node,archive)
        stage,target = 3,mapping[i]
    atol,rtol = float(sep_atol),float(sep_rtol)
    if not all(math.isfinite(v) and v>=0 for v in (atol,rtol)):
        raise ValueError("invalid separation tolerances")
    gains=[]
    for state in candidate_states:
        values = _state(ctx,i,state)
        old = physical_archive_value(prob_data,node,archive,values,facility_id=i)
        new = _exact_value(cut,values)
        tolerance = F(atol)+F(rtol)*max(F(1),abs(old),abs(new))
        gains.append((new-old,tolerance))
    report = dict(installed=False,stage=stage,node=target,parent_node=node.index,
        facility=i,version_before=current,version_after=current,affected_s2_nodes=[],
        requires_same_A_reprice=False,source=meta["source"],
        maximum_candidate_gain_exact=_ratio(max((g for g,_ in gains),default=F())))
    if not any(g>tol for g,tol in gains):
        report["reason"]="NO_SEPARATION"
        return report
    pi = {key:a for key,a in zip(_state_keys(ctx,i),cut.coefficients) if a}
    rows = archive.setdefault(stage,{}).setdefault(target,[])
    if add_unique_cut(rows,pi,cut.intercept):
        report.update(installed=True,version_after=current+1,reason="INSTALLED",
            affected_s2_nodes=[node.index] if i is not None else [],
            requires_same_A_reprice=i is not None)
    else:
        report["reason"]="DUPLICATE"
    return report
