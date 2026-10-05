"""Run a bounded physical portfolio without intervening S2/S3 forwards.

The caller supplies an already-certified global lower bound. It is used only
for gap comparisons, never converted into a fabricated master certificate.
After an open-gap continuation the caller must run a real forward before any
backward pass. This helper does not alter forward histories or oracle state.
"""
from copy import deepcopy
import math
import time

from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent
from .portfolio_bounds import PortfolioBounds


def run_physical_continuation(pd, tree, archive, policy, *, portfolio=None,
                              global_lb, total_budget, per_batch_seconds,
                              deadline, mode='a_then_b', tolerance=.01,
                              progress=None):
    """Return ``(portfolio, aggregate_route_seed_report)``.

    ``deadline`` is an absolute ``time.monotonic()`` deadline. Each batch
    reserves up to 20 seconds, within its allowance, for the public free-z
    master. Existing private-branch masters remain separately identified.
    Constructor and initial policy checks count in the first executed batch;
    other time outside completed batches is returned as ``overhead_seconds``.
    The caller charges those durations once to its shared late schedule.
    """
    from .physical_portfolio import PhysicalPortfolio
    from .phase15 import refresh_stage1_bound
    from .routeopt.restricted_master import certify_complete_policy

    started = time.monotonic()
    for name, value in (('global_lb', global_lb), ('total_budget', total_budget),
                        ('per_batch_seconds', per_batch_seconds),
                        ('deadline', deadline), ('tolerance', tolerance)):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f'physical continuation requires finite {name}')
    if total_budget < 0 or per_batch_seconds <= 0 or tolerance < 0:
        raise ValueError('invalid physical continuation budget or tolerance')
    if mode not in ('current', 'residual', 'a_then_b'):
        raise ValueError('invalid physical continuation mode')
    chain_deadline = min(float(deadline), started + float(total_budget))
    initial_s3 = deepcopy(archive.get(3))
    events, rows, priors, failures = [], [], [], []
    best_master, best_master_source = None, None
    owner = PortfolioBounds(pd, tree)
    owner.offer_policy(policy, origin='main', semantic_fingerprint=owner.semantic_fingerprint)
    current_lb = float(global_lb)
    if minimization_bounds_inverted(current_lb, owner.ub):
        raise ValueError('physical continuation global LB exceeds verified policy UB')

    def gap():
        lower = max(float(global_lb), owner.lb) if owner.lb is not None else float(global_lb)
        if minimization_bounds_inverted(lower, owner.ub):
            raise ValueError('physical continuation LB exceeds verified policy UB')
        return lower, minimization_gap_percent(lower, owner.ub)

    def accept_master(master, source):
        nonlocal best_master, best_master_source
        if not master:
            return
        owner.offer_master(master, origin=source,
                           semantic_fingerprint=owner.semantic_fingerprint,
                           source='refresh_stage1_bound')
        if master.get('lb_certified') is True and (
                best_master is None or float(master['lb']) > float(best_master['lb'])):
            best_master, best_master_source = deepcopy(master), source

    current_lb, current_gap = gap()
    status = 'gap_tolerance' if current_gap < 100. * tolerance else None
    if status is None and min(per_batch_seconds, chain_deadline - started) >= 30.:
        if portfolio is None:
            portfolio = PhysicalPortfolio(pd, tree, archive, owner.policy, mode=mode)
    if portfolio is not None and (
            portfolio.pd is not pd or portfolio.tree is not tree
            or portfolio.archive is not archive
            or portfolio.identity != owner.semantic_fingerprint):
        raise ValueError('physical continuation portfolio belongs to a different model or archive')
    preparation_seconds = time.monotonic() - started
    batch_started = started
    while status is None:
        now = time.monotonic()
        if now >= deadline:
            status = 'outer_deadline'
            break
        if chain_deadline - now < 30.:
            status = 'remaining_budget_below_minimum'
            break
        if per_batch_seconds < 30.:
            status = 'batch_budget_below_minimum'
            break
        if portfolio is None or not portfolio.has_work(owner.policy):
            status = 'portfolio_plan_complete'
            break
        batch_deadline = min(chain_deadline, batch_started + per_batch_seconds)
        remaining = batch_deadline - time.monotonic()
        if remaining < 30.:
            status = 'preparation_budget_exhausted'
            break
        before_lb, before_gap = gap()
        before_ub = owner.ub
        before_position = portfolio.report().get('completed_steps')
        remaining = batch_deadline - time.monotonic()
        if remaining < 30.:
            status = 'preparation_budget_exhausted'
            break
        reserve = min(20., remaining / 3.)
        seed = portfolio.advance(budget=remaining - reserve, policy=owner.policy,
                                 progress=progress or (lambda row: None))
        if archive.get(3) != initial_s3:
            raise RuntimeError('physical continuation changed the public Stage-3 archive')
        candidate = seed.get('policy_candidate')
        if candidate is not None:
            if candidate.get('certified') is not True:
                raise ValueError('physical continuation received an uncertified policy')
            normalized, checked_ub = certify_complete_policy(pd, tree, candidate['policy'])
            reported_ub = float(candidate['ub'])
            if not math.isfinite(reported_ub) or abs(reported_ub - checked_ub) > 1e-6:
                raise ValueError('physical continuation policy objective failed recomputation')
            owner.offer_policy(normalized, origin='physical_batch',
                               semantic_fingerprint=owner.semantic_fingerprint)
        private_master = seed.get('master')
        accept_master(private_master, 'private_branch')
        public_master = None
        remaining = batch_deadline - time.monotonic()
        if remaining > 0.:
            public_master = refresh_stage1_bound(pd, tree, archive, time_limit_s=remaining)
            accept_master(public_master, 'public_union')
        if archive.get(3) != initial_s3:
            raise RuntimeError('physical continuation master changed the Stage-3 archive')
        current_lb, current_gap = gap()
        elapsed = time.monotonic() - batch_started
        event = dict(batch_index=len(events) + 1,
                     requested_seconds=max(0., batch_deadline - batch_started),
                     seconds=elapsed, seed_seconds=float(seed.get('seconds', 0.)),
                     solves=int(seed.get('solves', 0)), added=int(seed.get('added', 0)),
                     status=seed.get('status', 'finished'),
                     private_master=deepcopy(private_master), public_master=deepcopy(public_master),
                     lb_before=before_lb, lb_after=current_lb,
                     ub_before=before_ub, ub_after=owner.ub,
                     gap_before=before_gap, gap_after=current_gap,
                     portfolio=deepcopy(seed.get('portfolio')))
        events.append(event)
        rows.extend(seed.get('bounds', ()))
        priors.extend(seed.get('route_priors', ()))
        failures.extend(seed.get('failures', ()))
        if progress is not None:
            progress(dict(batch=len(events), stage=seed.get('status', 'finished'),
                          global_lb=current_lb, ub=owner.ub,
                          gap_percent=current_gap, seconds=elapsed))
        if current_gap < 100. * tolerance:
            status = 'gap_tolerance'
        elif (not event['solves'] and not event['added']
              and current_lb == before_lb and owner.ub == before_ub
              and portfolio.report().get('completed_steps') == before_position):
            status = 'no_progress'
        batch_started = time.monotonic()

    current_lb, current_gap = gap()
    master_build = sum(float((event[key] or {}).get('build_seconds', 0.))
                       for event in events for key in ('private_master', 'public_master'))
    master_solve = sum(float((event[key] or {}).get('solve_seconds', 0.))
                       for event in events for key in ('private_master', 'public_master'))
    closed = status == 'gap_tolerance'
    report = dict(backend='routeopt_portfolio', status=status, requested_seconds=total_budget,
                  solves=sum(event['solves'] for event in events),
                  added=sum(event['added'] for event in events), bounds=rows,
                  route_priors=priors, failures=failures, master=best_master,
                  master_source=best_master_source,
                  master_status=(best_master or {}).get('status', 'unavailable'),
                  master_build_seconds=master_build, master_solve_seconds=master_solve,
                  seed_seconds=sum(event['seed_seconds'] for event in events),
                  policy_candidate=dict(certified=True, policy=owner.policy, ub=owner.ub),
                  preparation_seconds=preparation_seconds,
                  batch_events=events, global_lb=current_lb, global_ub=owner.ub,
                  gap_percent=current_gap, closed_gap=closed, requires_forward=not closed,
                  portfolio=portfolio.report() if portfolio is not None else None)
    finished = time.monotonic()
    report['seconds'] = finished - started
    report['overhead_seconds'] = max(
        0., report['seconds'] - sum(event['seconds'] for event in events))
    report['deadline_exhausted'] = finished >= chain_deadline
    return portfolio, report
