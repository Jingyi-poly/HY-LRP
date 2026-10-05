"""Complementary physical refinements with private archives and global bounds.

The two branches search the same physical model. Only newly certified physical
eta rows are published to the caller; private Stage-3 pools are never merged.
Finishing the finite refinement plan is not an optimality certificate.
"""
from copy import deepcopy
import math
import time

from .portfolio_bounds import PortfolioBounds
from .targeted_physical import (current_fleet_targets, run_targeted, physical_target_key,
                                physical_target_key_from_row, PhysicalTargetsUnavailable)


def _fleet_key(policy):
    return tuple(sorted((name, float(value)) for name, value in policy[1][0].items()
                        if name.startswith('z[')))


def _row_key(row):
    return physical_target_key_from_row(row)


def _cut_key(cut):
    return tuple(sorted(cut[0].items())), cut[1]


class PhysicalPortfolio:
    """One same-model continuation; A and B share certified incumbents only.

    `archive` is the caller's live archive. Branches start from copies and emit
    only physical eta additions back to it. `initial_master`, when provided,
    must be an actual free-fleet `refresh_stage1_bound` result.
    """

    def __init__(self, pd, tree, archive, policy, *, mode='a_then_b', initial_master=None):
        if mode not in ('current', 'residual', 'a_then_b'):
            raise ValueError('invalid physical portfolio mode')
        self.pd, self.tree, self.archive = pd, tree, archive
        self.mode = mode
        self.bounds = PortfolioBounds(pd, tree)
        self.identity = self.bounds.semantic_fingerprint
        self.bounds.offer_policy(policy, origin='initial', semantic_fingerprint=self.identity)
        if initial_master is not None:
            self.bounds.offer_master(initial_master, origin='initial',
                                     semantic_fingerprint=self.identity,
                                     source='refresh_stage1_bound')
        self._base = deepcopy(archive)
        self._s3 = deepcopy(archive.get(3))
        self._branches = {}
        self.events = []
        self.spent = 0.
        self._reset_plan(_fleet_key(self.bounds.policy))

    def _reset_plan(self, fleet):
        self.fleet = fleet
        if self.mode == 'a_then_b':
            self.plan = [('current', 'focus'), ('residual', 'focus'),
                         ('residual', 'cover'), ('residual', 'refine'),
                         ('current', 'cover'), ('current', 'refine')]
        else:
            self.plan = [(self.mode, stage) for stage in ('focus', 'cover', 'refine')]
        self.position = 0
        self.visits = {'current': {}, 'residual': {}}
        self.incomplete = {'current': set(), 'residual': set()}

    def has_work(self, policy=None):
        # This is only a scheduling hint. advance independently verifies policies.
        return (policy is not None and _fleet_key(policy) != self.fleet
                or self.position < len(self.plan))

    def _branch(self, method):
        if method not in self._branches:
            self._branches[method] = deepcopy(self._base)
        return self._branches[method]

    def _empty_report(self, started, status, **extra):
        portfolio = self.report()
        elapsed = time.monotonic() - started
        self.spent += elapsed
        portfolio['seconds'] = self.spent
        return dict(backend='routeopt_portfolio', status=status, solves=0, added=0,
                    bounds=[], route_priors=[], master=None, seconds=elapsed,
                    portfolio=portfolio, **extra)

    def advance(self, *, budget, progress, policy=None):
        from cuts.benders_cuts import add_unique_cut
        from solvers.forward_period_dedup import forward_semantic_fingerprint
        from .routeopt.physical_policy_search import improve

        started = time.monotonic()
        if not math.isfinite(budget) or budget <= 0.:
            raise ValueError('physical portfolio requires a positive finite budget')
        if forward_semantic_fingerprint(self.pd, self.tree) != self.identity:
            raise ValueError('physical portfolio model changed')
        if policy is not None:
            self.bounds.offer_policy(policy, origin='main', semantic_fingerprint=self.identity)
        incumbent = self.bounds.policy
        fleet = _fleet_key(incumbent)
        if fleet != self.fleet:
            self._reset_plan(fleet)

        selected = None
        while self.position < len(self.plan):
            method, stage = self.plan[self.position]
            branch = self._branch(method)
            try:
                targets = current_fleet_targets(self.pd, self.tree, incumbent, branch)
            except PhysicalTargetsUnavailable as exc:
                self.position = len(self.plan)
                return self._empty_report(started, 'unsupported_physical_domain',
                                          failures=exc.failures)
            if stage == 'focus':
                keys = {physical_target_key(t) for t in targets[:6]}
                cap, count = 25., 6
            elif stage == 'cover':
                keys = {physical_target_key(t) for t in targets
                        if physical_target_key(t) not in self.visits[method]
                        and (method != 'residual' or not all(t[2].values()))}
                cap, count = 8., 0
            else:
                keys = {physical_target_key(t) for t in targets
                        if physical_target_key(t) in self.incomplete[method]}
                cap, count = 25., 6
            if keys:
                selected = method, stage, branch, keys, cap, count
                break
            self.position += 1
        if selected is None:
            return self._empty_report(started, 'portfolio_plan_complete')

        method, stage, branch, keys, cap, count = selected
        before = {i: {_cut_key(c) for c in cuts} for i, cuts in branch.get(2, {}).items()}
        # Both whole-policy polishing and its checks use the same call budget.
        remaining = max(0., budget - (time.monotonic() - started))
        polish_reserve = min(12., max(0., remaining - 30.))
        seed_budget = remaining - polish_reserve
        if seed_budget < 1.:
            return self._empty_report(started, 'preparation_deadline')
        visits_before = dict(self.visits[method])
        report = run_targeted(self.pd, self.tree, branch, incumbent,
            budget=seed_budget, reserve=min(20., seed_budget / 3.), progress=progress,
            max_targets=count, per_root_limit=cap, policy_grace_s=3.,
            pricing_method=method, target_keys=keys, target_visits=self.visits[method])
        if branch.get(3) != self._s3:
            raise RuntimeError('physical portfolio changed a private Stage-3 archive')
        for row in report['bounds']:
            key = _row_key(row)
            if row.get('pricing_search_complete') is True:
                self.incomplete[method].discard(key)
            else:
                self.incomplete[method].add(key)
        attempted = {_row_key(row) for row in report['bounds']}
        attempted.update(key for key, value in self.visits[method].items()
                         if value > visits_before.get(key, 0))
        if stage == 'refine':
            # This finite precision pass is not called a proof of Q optimality.
            self.incomplete[method].difference_update(attempted)
            if not (keys - attempted):
                self.position += 1
        elif stage == 'focus' or not (keys - set(self.visits[method])):
            self.position += 1
        if report.get('status') in ('backend_unavailable', 'solver_unavailable',
                                    'unsupported_physical_domain'):
            self.position = len(self.plan)

        offered = report.pop('policy_candidate', None)
        if offered is not None:
            if offered.get('certified') is not True:
                raise RuntimeError('physical portfolio received an uncertified policy')
            self.bounds.offer_policy(offered['policy'], origin=method,
                                     semantic_fingerprint=self.identity)
        master = report.get('master') or {}
        if master.get('lb_certified') is True:
            self.bounds.offer_master(master, origin=method,
                semantic_fingerprint=self.identity, source='refresh_stage1_bound')

        polish_seconds = min(12., max(0., budget - (time.monotonic() - started)))
        if polish_seconds > 0.:
            polished = improve(self.bounds.policy, self.pd, self.tree, polish_seconds)
            if polished.get('certified') is not True:
                raise RuntimeError('physical portfolio polish was not certified')
            self.bounds.offer_policy(polished['policy'], origin=method + ':polish',
                                     semantic_fingerprint=self.identity)
            report['polish'] = polished['stats']

        published = 0
        for i, cuts in branch.get(2, {}).items():
            for cut in cuts:
                if _cut_key(cut) not in before.get(i, set()):
                    published += int(add_unique_cut(
                        self.archive.setdefault(2, {}).setdefault(i, []), *deepcopy(cut)))
        report['branch_added'] = report['added']
        report['added'] = published
        report['backend'] = 'routeopt_portfolio'
        report['policy_candidate'] = dict(certified=True, policy=self.bounds.policy, ub=self.bounds.ub)
        summary = self.bounds.report()
        self.events.append(dict(method=method, stage=stage, seconds=0.,
                                published_eta=published, **summary))
        report['portfolio'] = self.report()
        report['seconds'] = time.monotonic() - started
        report['deadline_exhausted'] = report['seconds'] >= budget
        self.spent += report['seconds']
        self.events[-1]['seconds'] = report['seconds']
        report['portfolio']['seconds'] = self.spent
        report['portfolio']['events'][-1]['seconds'] = report['seconds']
        return report

    def report(self):
        return dict(mode=self.mode, completed_steps=self.position, total_steps=len(self.plan),
                    plan_complete=self.position >= len(self.plan), seconds=self.spent,
                    events=deepcopy(self.events), bounds=self.bounds.report())
