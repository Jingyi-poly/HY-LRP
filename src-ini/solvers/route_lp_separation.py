"""Directed DFJ separation for a fixed-state route LP and certified SBC slopes.

For every root-free U containing customer j, a route through j must leave U:
    sum(r[v,w], v in U, w outside U) >= a_copy[j].
The variable RHS is essential: no fixed anchor value is embedded in these rows.
Every row is valid over the whole binary assignment domain, including inactivity.
"""
from __future__ import annotations

from collections import deque
import math
import time

import gurobipy as gp
import numpy as np

from core.solver_bounds import extract_verified_fixed_rhs_dual_cut
from models.route_dfj_pool import remember_route_dfj_row, route_dfj_rows


def directed_mincut(capacity, source, sink=0, *, deadline=None):
    """Float-capacity augmenting paths; None means the time budget interrupted it."""
    residual=np.asarray(capacity,dtype=float).copy()
    n=len(residual)
    if residual.shape!=(n,n) or not np.isfinite(residual).all() or np.any(residual<0):
        raise ValueError('Mincut requires a finite nonnegative square capacity matrix')
    if source==sink or not 0<=source<n or not 0<=sink<n:
        raise ValueError('Mincut source and sink must be distinct valid nodes')
    flow=0.
    while True:
        if deadline is not None and time.perf_counter()>=deadline:
            return None
        parent=[-1]*n;parent[source]=source;queue=deque([source])
        while queue and parent[sink]<0:
            v=queue.popleft()
            for w in np.flatnonzero(residual[v]>1e-10):
                w=int(w)
                if parent[w]<0:
                    parent[w]=v;queue.append(w)
                    if w==sink:break
        if parent[sink]<0:
            return flow,{v for v in range(n) if parent[v]>=0}
        amount=math.inf;w=sink
        while w!=source:
            v=parent[w];amount=min(amount,float(residual[v,w]));w=v
        w=sink
        while w!=source:
            v=parent[w];residual[v,w]-=amount;residual[w,v]+=amount;w=v
        flow+=amount


def _certificate(lp,bindings,label):
    if lp.Status!=gp.GRB.OPTIMAL:
        return None
    pi,intercept=extract_verified_fixed_rhs_dual_cut(lp,bindings,label=label)
    return {'pi':dict(pi),'intercept':float(intercept),'objective':float(lp.ObjVal),
            'status':int(lp.Status),'constraints':int(lp.NumConstrs)}


def separate_route_lp_dfj(lp, problem, bindings, *, time_limit=1., max_rounds=20,
                          optimize=None, label='S3 DFJ fixed-RHS LP'):
    """Return the last OPTIMAL LP certificate within an additional time budget.

    The initial fixed LP must already be optimal. If a later LP reaches a
    limit, retain the previous certificate, which used a subset of these same
    globally valid rows. No dual from an unfinished solve is ever used.
    ``optimize`` may instrument each actual added-row LP solve for telemetry.
    """
    if not math.isfinite(float(time_limit)) or time_limit<0:
        raise ValueError('DFJ separation time_limit must be finite and nonnegative')
    if isinstance(max_rounds,bool) or int(max_rounds)!=max_rounds or max_rounds<0:
        raise ValueError('DFJ max_rounds must be a nonnegative integer')
    if problem.layer!='tsp' or problem.direction!='forward' or problem.context is None:
        raise ValueError('DFJ dual separation requires a fixed forward route LP')
    started=time.perf_counter();deadline=started+float(time_limit)
    certificate=_certificate(lp,bindings,label)
    report={'enabled':True,'time_limit':float(time_limit),'max_rounds':int(max_rounds),
            'additional_lp_solves':0,'rows_added':0,'certified_rows_added':0,
            'pooled_rows_added':0,
            'fully_separated':False,'retained_previous_optimal_certificate':False,
            'initial_objective':None if certificate is None else certificate['objective'],
            'latest_lp_status':int(lp.Status),'history':[]}
    if certificate is None:
        report.update(stop_reason='initial_lp_not_optimal',wall_seconds=time.perf_counter()-started)
        return None,report
    old_time_limit=float(lp.Params.TimeLimit)
    n=problem.context.n;i=problem.facility
    a=[lp.getVarByName(f'a_copy[{j}]') for j in range(n)]
    if any(var is None for var in a):
        raise ValueError('Missing full assignment copies in the fixed route LP')
    arc_keys=[(v,w) for fi,v,w in problem.linear.groups.get('r',{}) if fi==i]
    arc_variables=[lp.getVarByName(f'r[{i},{v},{w}]') for v,w in arc_keys]
    if any(var is None for var in arc_variables):
        raise ValueError('Missing route variables in the fixed route LP')
    seen=set();base_rows=lp.NumConstrs;reason='round_limit'
    optimize=lp.optimize if optimize is None else optimize
    try:
        for iteration in range(int(max_rounds)):
            if time.perf_counter()>=deadline:
                reason='separation_time_limit';break
            values=lp.getAttr('X',a)
            active=[j for j,value in enumerate(values) if value>1e-8]
            if not active:
                report['fully_separated']=True;reason='empty_route';break
            capacity=np.zeros((n+1,n+1))
            for (v,w),value in zip(arc_keys,lp.getAttr('X',arc_variables)):
                capacity[v,w]=max(0.,float(value))
            violated=[];interrupted=False
            for j in active:
                result=directed_mincut(capacity,j+1,deadline=deadline)
                if result is None:
                    interrupted=True;break
                value,U=result
                key=(tuple(sorted(U)),j)
                if value<values[j]-1e-7 and key not in seen:
                    violated.append((U,j,key,values[j]-value))
            if interrupted:
                reason='separation_time_limit';break
            if not violated:
                report['fully_separated']=True;reason='no_violated_cut';break
            # All rows use a_copy[j], including j with a fixed anchor value one.
            r=dict(zip(arc_keys,arc_variables))
            for U,j,key,violation in violated:
                seen.add(key)
                expression=gp.quicksum(var for (v,w),var in r.items() if v in U and w not in U)
                lp.addConstr(expression>=a[j],name=f'lrp_dfj_{i}_{base_rows}_{len(seen)}')
                report['pooled_rows_added'] += int(remember_route_dfj_row(problem.context,i,U,j))
            lp.update();report['rows_added']=len(seen)
            remaining=deadline-time.perf_counter()
            if remaining<=0:
                reason='separation_time_limit';break
            lp.Params.TimeLimit=remaining
            optimize();report['additional_lp_solves']+=1
            report['latest_lp_status']=int(lp.Status)
            report['history'].append({'round':iteration+1,'rows_added':len(seen),
                'status':int(lp.Status),'objective':float(lp.ObjVal) if lp.SolCount else None})
            updated=_certificate(lp,bindings,label)
            if updated is None:
                reason='added_row_lp_not_optimal';break
            certificate=updated;report['certified_rows_added']=len(seen)
        report['retained_previous_optimal_certificate']=(report['certified_rows_added']<report['rows_added'])
        report.update(stop_reason=reason,wall_seconds=time.perf_counter()-started,
                      certified_lp_status=certificate['status'],certified_objective=certificate['objective'],
                      pooled_rows_total=len(route_dfj_rows(problem.context,i)))
        return certificate,report
    finally:
        lp.Params.TimeLimit=old_time_limit
