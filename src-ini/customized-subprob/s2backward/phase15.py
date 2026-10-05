"""Budgeted physical LRP eta seeds and an independent root lower-bound solve."""
from __future__ import annotations

import math
import time

from core.backend_telemetry import backend_call, backend_scope, record_backend_event
from models.stage_builder import StageModelBuilder, _instance
from models.stage_model_core import evaluate_model
from solvers.forward_policy_certification import (
    InvalidForwardPolicy, certify_stage1_forward_policy,
)


@backend_scope(phase='1.5', path='physical_seed', stage=1)
def refresh_stage1_bound(prob_data, tree, cut_lag, *, time_limit_s):
    """Reoptimize free facility availability; never manufacture a policy UB."""
    started = time.monotonic()
    budget = float(time_limit_s)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('invalid master refresh budget')
    report = dict(lb=None, lb_certified=False, availability=None, fleet=None,
                  status='budget_exhausted', seconds=0., build_seconds=0.,
                  solve_seconds=0., skip_reason='budget_exhausted_before_build')
    if budget <= 0:
        record_backend_event('gurobi', 'skip', 'deadline_before_build')
        return report
    deadline = started+budget
    model = StageModelBuilder(prob_data,mip_gap=0.,lazy_threshold=0).build_stage_problem(
        1,tree[1][0],cut_lag,{})
    try:
        report['build_seconds'] = time.monotonic()-started
        remaining = deadline-time.monotonic()-.01
        if remaining <= 0:
            report['skip_reason'] = 'budget_exhausted_during_build'
            return report
        begun = time.monotonic()
        with backend_call('gurobi','master_mip',model=model):
            evaluated = evaluate_model(model,time_limit=remaining,mip_gap=0.,threads=1)
        report['solve_seconds'] = time.monotonic()-begun
        lower = evaluated.certified_lower_bound
        report.update(lb=lower,lb_certified=lower is not None,status=evaluated.report['status'],
                      skip_reason=None,certificate=evaluated.summary())
        if evaluated.x is not None:
            raw = {f'A[{i},{k}]':value for (i,k),value in evaluated.values('A').items()}
            try:
                availability,_ = certify_stage1_forward_policy(prob_data,raw)
                # Keep the legacy report field for old scheduling callers;
                # it contains normalized physical A/o/h/b, never fleet z.
                report.update(availability=availability,fleet=availability)
            except InvalidForwardPolicy as exc:
                report['policy_error'] = str(exc)
    finally:
        model.dispose()
        report['seconds'] = time.monotonic()-started
    return report


def run_phase15(prob_data, tree, cut_lag, *, backend, time_limit_s, initial_policy=None,
                master_reserve_s=None, full_node_limit_s=5., partial_node_limit_s=8.,
                cache=None, s3_bundles=None, max_nodes=1):
    """Original seed -> independent Stage-1 refresh, under one total budget.

    gurobi runs physical route-dual pricing; none is disabled. The obsolete
    investment RouteOpt backend cannot certify the physical LRP domain and is
    explicitly unsupported. Default one S2 context per call, with persistent
    cache cursor rotation. The seed writes only true-Q rows in cut_lag[2].
    Returned availability is a comparison state, not a complete feasible UB.
    """
    started = time.monotonic()
    budget = float(time_limit_s)
    if not math.isfinite(budget) or budget < 0 or backend not in ('routeopt','gurobi','none'):
        raise ValueError('invalid Phase-1.5 backend or time budget')
    for name,value in (('full_node_limit_s',full_node_limit_s),
                       ('partial_node_limit_s',partial_node_limit_s),
                       ('master_reserve_s',master_reserve_s)):
        if value is None and name == 'master_reserve_s':
            continue
        number = float(value)
        if isinstance(value,bool) or not math.isfinite(number) or (number < 0 if name == 'master_reserve_s' else number <= 0):
            raise ValueError(f'invalid Phase-1.5 {name}')
    reserve = min(8.,.1*budget) if master_reserve_s is None else min(budget,float(master_reserve_s))
    report = dict(backend=backend,solves=0,pricing_solves=0,added=0,seconds=0.,seed_seconds=0.,
                  master=None,requested_seconds=budget,status='disabled',
                  master_reserve_seconds=reserve,master_status='not_requested',
                  master_skip_reason='disabled',master_build_seconds=0.,master_solve_seconds=0.)
    if backend == 'none' or budget == 0:
        return report
    if backend == 'routeopt':
        report.update(status='unsupported_backend',reason='investment RouteOpt has no certified physical LRP adapter',
                      master_skip_reason='unsupported_backend')
        return report
    deadline = started+budget
    from .physical_route_seed import seed_full_fleet_route_bounds
    seed_budget = max(0.,deadline-time.monotonic()-reserve)
    seeded = seed_full_fleet_route_bounds(prob_data,tree,cut_lag,time_limit_s=seed_budget,
        per_node_limit_s=full_node_limit_s,partial_node_limit_s=partial_node_limit_s,
        initial_policy=initial_policy,cache=cache,s3_bundles=s3_bundles,max_nodes=max_nodes)
    report.update(seeded)
    report.update(backend=backend,seed_seconds=float(seeded.get('seconds',0.)),requested_seconds=budget)
    remaining = deadline-time.monotonic()
    report['master_skip_reason'] = 'no_added_cuts' if not report['added'] else 'seed_deadline_exhausted'
    if report['added'] and remaining > 0:
        master = refresh_stage1_bound(prob_data,tree,cut_lag,time_limit_s=remaining)
        report.update(master=master,master_status=master.get('status','finished'),
                      master_skip_reason=master.get('skip_reason'),
                      master_build_seconds=float(master.get('build_seconds',0.)),
                      master_solve_seconds=float(master.get('solve_seconds',0.)))
    report['seconds'] = time.monotonic()-started
    report['deadline_exhausted'] = report['seconds'] >= budget
    return report


def same_fleet(prob_data,left,right):
    """Legacy name: exact equality of all physical availability A[i,k] bits."""
    if not isinstance(left,dict) or not isinstance(right,dict):
        return False
    m,_,_,L,_ = _instance(prob_data).shape
    return all(left.get(f'A[{i},{k}]') in (0.,1.)
               and right.get(f'A[{i},{k}]') in (0.,1.)
               and left[f'A[{i},{k}]'] == right[f'A[{i},{k}]']
               for i in range(m) for k in range(L))


__all__ = ['run_phase15','refresh_stage1_bound','same_fleet']
