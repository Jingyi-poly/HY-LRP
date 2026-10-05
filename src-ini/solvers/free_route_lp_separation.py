"""Free physical S3 DFJ separation with an audited last-OPTIMAL certificate.

The caller owns the MIP. The helper owns/disposes its relaxation. Intervening
fractional points only propose globally valid physical DFJ rows. No fractional
state or objective is returned as an integer oracle support.
"""
from copy import copy
import math
import time
import gurobipy as gp
import numpy as np

from models.stage_model_core import audit_gurobi_matrix,matrix_primal_check
from core.solver_bounds import certified_gurobi_minimization_lower_bound
from core.solver_settings import configured_gurobi_threads
from core.backend_telemetry import backend_call
from solvers.route_lp_separation import directed_mincut

def matrix_audit_reserve(nonzeros):
    """Conservative measured allowance for complete matrix/primal readback."""
    return .05 + 1.5e-6 * int(nonzeros)


def _guarded_bound(model):
    raw = certified_gurobi_minimization_lower_bound(model)
    return None if raw is None else math.nextafter(raw-1e-9-1e-12*max(1., abs(raw)), -math.inf)


def _options(model):
    model.Params.OutputFlag = 0
    model.Params.Threads = configured_gurobi_threads()
    model.Params.Seed = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9
    model.Params.OptimalityTol = 1e-9


from models.route_dfj_factor import append_factor_rows as _append_rows, projected_extra


def separate_free_route_lp(model, *, time_limit, deadline=None):
    """Deadline uses time.monotonic; total helper budget includes final audit."""
    if math.isnan(float(time_limit)) or time_limit<0:raise ValueError('Invalid LP budget')
    started=time.monotonic();end=started+float(time_limit)
    if deadline is not None:end=min(end,float(deadline))
    old=model._lrp_spec
    if old.layer!='tsp' or old.direction!='backward' or old.domain!='parent':
        raise ValueError('Only the original free parent-domain physical S3 oracle is supported')
    # Preserve all older Evaluation.problem row-prefix snapshots, including
    # those retained from another multiplier in the same Level Set call.
    problem=copy(old);M=copy(old.linear)
    for field in ('names','cost','lower','upper','integer','rows','row_names','row_lb','row_ub'):
        setattr(M,field,list(getattr(old.linear,field)))
    M.groups={g:dict(entries) for g,entries in old.linear.groups.items()}
    M.dfj_flow_map=dict(getattr(old.linear,'dfj_flow_map',{}))
    problem.linear=M;model._lrp_spec=problem;model._lrp_native.specification=problem
    ctx,i=problem.context,problem.facility
    record=dict(certificate_kind='free_physical_LP_lower_bound_only',time_limit=float(time_limit),
        context=ctx.key,route_key=ctx.route_key(i),multipliers=list(problem.multipliers),
        certified_lower_bound=None,fully_separated=False,rows=[],history=[],lp_solves=0,
        phase_seconds=dict(relax=0.,optimize=0.,flow=0.,append=0.,snapshot=0.,final_audit=0.))
    if end-time.monotonic()<.08:
        record.update(stop_reason='insufficient_audit_budget',wall_seconds=time.monotonic()-started);return record
    tick=time.monotonic();lp=model.relax();_options(lp)
    record['phase_seconds']['relax']=time.monotonic()-tick
    snapshot=None;reason='LP_budget';seen=set()
    nonzeros=sum(map(len,M.rows));reserve=matrix_audit_reserve(nonzeros)
    try:
        arcs={(v,w):col for (fi,v,w),col in M.groups['r'].items() if fi==i}
        vertices={0}|{v for arc in arcs for v in arc}
        alpha_columns=[M.groups['a_copy'][j,] for j in range(ctx.n)]
        while time.monotonic()<end-reserve-.005:
            lp.Params.TimeLimit=max(1e-6,end-time.monotonic()-reserve)
            tick=time.monotonic()
            with backend_call('gurobi','free_route_dfj_lp',model=lp):
                lp.optimize()
            record['phase_seconds']['optimize']+=time.monotonic()-tick
            record['lp_solves']+=1
            if lp.Status!=gp.GRB.OPTIMAL:
                reason='LP_not_optimal_use_last_OPTIMAL_snapshot';break
            tick=time.monotonic()
            variables=lp.getVars()
            current=dict(status=int(lp.Status),raw_objbound=float(lp.ObjBound),objective=float(lp.ObjVal),
                certified_lower_bound=_guarded_bound(lp),x=np.asarray(lp.getAttr('X',variables)),
                canonical_rows=len(M.rows),native_rows=int(lp.NumConstrs),canonical_vars=len(M.names),native_vars=int(lp.NumVars),multipliers=tuple(problem.multipliers))
            if current['certified_lower_bound'] is not None:
                snapshot=current
            record['phase_seconds']['snapshot']+=time.monotonic()-tick
            # These are explicitly unverified intermediate diagnostics. Only
            # the final saved snapshot receives the audit/certificate below.
            step=dict(iteration=record['lp_solves'],raw_unverified_lp_objective=current['objective'],
                      canonical_rows=current['canonical_rows'])
            record['history'].append(step)
            if time.monotonic()>=end-reserve:break
            raw=np.zeros((ctx.n+1,ctx.n+1))
            for (v,w),col in arcs.items():raw[v,w]=current['x'][col]
            capacity=np.maximum(raw,0.);alpha=current['x'][alpha_columns]
            active=sorted((j for j in range(ctx.n) if alpha[j]>1e-8),key=lambda j:(-alpha[j],j))
            found=[];violations=[];complete=True;tick=time.monotonic()
            for j in active:
                flow_deadline=time.perf_counter()+max(0.,end-time.monotonic()-reserve)
                answer=directed_mincut(capacity,j+1,deadline=flow_deadline)
                if answer is None:complete=False;break
                U=answer[1]&vertices;key=(tuple(sorted(U)),j)
                if 0 in U or j+1 not in U:raise AssertionError('Invalid physical DFJ candidate')
                outside=vertices-U
                crossing=math.fsum(float(raw[v,w]) for v in U for w in outside)
                if crossing<alpha[j]-1e-7 and key not in seen:
                    found.append(key);violations.append(float(alpha[j]-crossing))
            record['phase_seconds']['flow']+=time.monotonic()-tick
            step.update(positive_sources=len(active),new_violations=len(found),
                        max_violation=max(violations,default=0.),complete_source_pass=complete)
            if not found:
                record['fully_separated']=complete
                reason='no_violated_DFJ' if complete else 'flow_deadline';break
            if time.monotonic()>=end-reserve:
                reason='row_install_deadline';break
            # Every graph built by the current route builder is complete over
            # its retained physical vertices. Estimate the exact smaller DFJ
            # representation before installing a batch, reserving its audit.
            projected=nonzeros+projected_extra(M,vertices,found)
            if time.monotonic()+matrix_audit_reserve(projected)+.01>=end:
                reason='projected_final_audit_reserve';break
            seen.update(found);tick=time.monotonic()
            added=_append_rows(model,lp,found,len(M.rows))
            record['rows'].extend(added)
            nonzeros+=sum(row['nonzeros'] for row in added)
            reserve=matrix_audit_reserve(nonzeros)
            record['phase_seconds']['append']+=time.monotonic()-tick
        if snapshot is not None:
            tick=time.monotonic()
            # Recreate exactly the last actually OPTIMAL matrix. Rows appended
            # afterward remain in the owned MIP and physical pool; only this
            # temporary LP drops them before the full readback. No bound or
            # status is read after this mutation.
            extra=lp.getConstrs()[snapshot['native_rows']:]
            extra_vars=lp.getVars()[snapshot['native_vars']:]
            if extra:lp.remove(extra)
            if extra_vars:lp.remove(extra_vars)
            lp.update()
            variables=lp.getVars()
            continuous=copy(problem);linear=copy(M);continuous.linear=linear
            for field in ('rows','row_names','row_lb','row_ub'):
                setattr(linear,field,list(getattr(M,field)[:snapshot['canonical_rows']]))
            for field in ('names','cost','lower','upper'):
                setattr(linear,field,list(getattr(M,field)[:snapshot['canonical_vars']]))
            linear.integer=[0]*snapshot['canonical_vars']
            linear.groups={g:{k:c for k,c in entries.items() if c<snapshot['canonical_vars']} for g,entries in M.groups.items()}
            linear.dfj_flow_map={U:c for U,c in M.dfj_flow_map.items() if c<snapshot['canonical_vars']}
            try:
                if tuple(problem.multipliers)!=snapshot['multipliers']:
                    raise AssertionError('Physical objective changed during free-LP separation')
                matrix=audit_gurobi_matrix(linear,lp,variables)
                primal=matrix_primal_check(continuous,snapshot['x'])
                rebuilt=math.fsum(float(c)*float(x) for c,x in zip(linear.cost,snapshot['x']))
                if abs(rebuilt-snapshot['objective'])>2e-6+1e-10*max(1.,abs(rebuilt)):
                    raise AssertionError('Saved LP objective does not match saved canonical primal')
                record['certified_lower_bound']=snapshot['certified_lower_bound']
                record['certificate']=dict(status=snapshot['status'],source='saved_OPTIMAL_Gurobi_ObjBound',
                    raw_objbound=snapshot['raw_objbound'],objective=snapshot['objective'],
                    canonical_rows=snapshot['canonical_rows'],native_rows=snapshot['native_rows'],
                    rows_added_after_certificate=len(M.rows)-snapshot['canonical_rows'],
                    removed_LP_only_rows=len(extra),removed_LP_only_vars=len(extra_vars),
                    canonical_vars=snapshot['canonical_vars'],native_vars=snapshot['native_vars'],matrix_audit=matrix,primal_audit=primal,
                    rebuilt_objective=rebuilt,fractional_primal_used_as_integer_support=False)
            except (ValueError,AssertionError,gp.GurobiError) as exc:
                record['certificate_rejected']=str(exc)
            record['phase_seconds']['final_audit']=time.monotonic()-tick
        record['stop_reason']=reason
    finally:lp.dispose()
    record['canonical_nonzeros']=nonzeros
    record['flow_auxiliaries']=len(M.dfj_flow_map)
    record['final_audit_reserved_seconds']=reserve
    record['wall_seconds']=time.monotonic()-started
    record['budget_overrun_seconds']=max(0.,time.monotonic()-end)
    return record
