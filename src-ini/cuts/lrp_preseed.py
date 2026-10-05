"""Optional physical degree and verified LP cuts before the first forward pass.

This module has no solver imports or import-time side effects.
The caller owns the compatible physical archive. Only certified cuts mutate it;
root reports are observations, never history entries, policy UBs or targets.
"""
from __future__ import annotations

import math
import time
from numbers import Integral, Real


def _seed_degree(tree, archive, deadline):
    """Add exact original degree coefficients, preserving physical ownership."""
    from cuts.benders_cuts import add_unique_cut
    from models.stage_model_core import route_degree_bounds

    report = {'candidates': 0, 'changed': 0, 'duplicates_or_dominated': 0,
              'facilities_completed': 0, 'complete': True}
    for node in tree[2]:
        ctx = node.context
        for i, rid in enumerate(node.successor):
            if time.monotonic() >= deadline:
                report['complete'] = False
                return report
            degree = route_degree_bounds(ctx, i)
            # Minima preparation is inside the same allowance; do not start
            # more work after it has used the remaining time.
            if time.monotonic() >= deadline:
                report['complete'] = False
                return report
            if degree.eligible:
                for customer, root in ((degree.incoming, degree.return_cost),
                                       (degree.outgoing, degree.depart_cost)):
                    pi = {f'alpha[{i},{j}]': float(c)
                          for j, c in enumerate(customer) if c}
                    if root:
                        pi[f'u[{i}]'] = float(root)
                    if not pi:
                        continue
                    report['candidates'] += 1
                    changed = add_unique_cut(archive.setdefault(3, {}).setdefault(rid, []), pi, 0.)
                    report['changed'] += int(changed)
                    report['duplicates_or_dominated'] += int(not changed)
            report['facilities_completed'] += 1
    return report


def seed_lrp_prepass(prob_data, tree, archive, *, time_limit_s, max_rounds=2):
    """Seed two directed degree bounds, then a bounded number of fixed-S2 LP sweeps.

    Each cut is in unweighted physical node units. The ordinary root model
    applies scenario probabilities. Build/import time is included in the one
    shared deadline; every solver receives only remaining time. LP pricing,
    native calls, free-S2 strengthening, UB/target/bundle updates are absent.
    Existing model builders are not interruptible mid-build; their existing
    deadline check prevents optimization after an overlong build.
    """
    started = time.monotonic()
    if isinstance(time_limit_s, bool) or not isinstance(time_limit_s, Real):
        raise ValueError('time_limit_s must be finite and nonnegative')
    budget = float(time_limit_s)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('time_limit_s must be finite and nonnegative')
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, Integral) or max_rounds < 0:
        raise ValueError('max_rounds must be a nonnegative integer')
    report = {'elapsed': 0., 'requested_seconds': budget, 'max_rounds': int(max_rounds),
              'degree': {}, 'eta_cuts_changed': 0, 'lp_attempted': 0,
              'rounds_completed': 0, 'rounds': [], 'roots': [], 'diagnostics': [],
              'solve_counts': {}, 'stop_reason': 'budget_zero'}
    solver = None
    def finish(reason):
        report['stop_reason'] = reason
        report['elapsed'] = time.monotonic()-started
        if solver is not None:
            report['diagnostics'] = list(solver.last_cut_diagnostics)
            report['solve_counts'] = dict(solver.solve_counts)
        return report
    # Must remain before any lazy solver import, instance access or mutation.
    if budget == 0:
        return finish('budget_zero')
    deadline = started+budget
    report['degree'] = _seed_degree(tree, archive, deadline)
    if not report['degree']['complete']:
        return finish('deadline_during_degree')
    if max_rounds == 0:
        return finish('round_limit_zero')
    if time.monotonic() >= deadline:
        return finish('deadline_after_degree')

    from core.customized_subprob import ensure_import_path
    from core.solve_deadline import SolveDeadlineReached
    from cuts.benders_cuts import add_unique_cut
    from models.stage_builder import StageModelBuilder
    from solvers.backward_solver_sbc import BackwardSolverSBC
    ensure_import_path()
    from s2backward.phase15 import refresh_stage1_bound

    # Retain a small local allowance for recording/disposal. This is not a
    # second budget and never extends the caller's shared deadline.
    def root_refresh(label):
        remaining = deadline-time.monotonic()-.02
        if remaining <= 0:
            return None
        value = refresh_stage1_bound(prob_data, tree, archive,
                                     time_limit_s=min(5., remaining))
        report['roots'].append({'label': label, 'elapsed': time.monotonic()-started,
                                'result': value})
        return value

    try:
        # Static S3 rows cannot change S1 directly. One initial root solve
        # suffices; no redundant before/after-degree solve is performed.
        current = root_refresh('before_eta')
        if current is None:
            return finish('deadline_before_initial_root')
        seen = set()
        for round_index in range(int(max_rounds)):
            if time.monotonic() >= deadline-.02:
                return finish('shared_deadline')
            parent = current.get('availability')
            if parent is None:
                return finish('no_certified_root_availability')
            key = tuple(sorted((k, float(v)) for k, v in parent.items() if k.startswith('A[')))
            if key in seen:
                return finish('repeated_full_availability')
            seen.add(key)
            if solver is None:
                solver = BackwardSolverSBC(prob_data,
                    stage_builder=StageModelBuilder(prob_data, lazy_threshold=0),
                    strengthen_s2=False, strengthen=False, sub_time_limit=2.,
                    route_lp_separation_time_limit=0.)
                solver.phase = 1
            row = {'round': round_index+1, 'attempted': 0, 'changed': 0,
                   'complete': True, 'parent_A': dict(key)}
            report['rounds'].append(row)
            before = time.monotonic()
            for node in tree[2]:
                if time.monotonic() >= deadline-.02:
                    row['complete'] = False
                    break
                report['lp_attempted'] += 1
                row['attempted'] += 1
                # Existing SBC entry preserves verified fixed-RHS dual
                # extraction, full-box cleaning payment and model disposal.
                cut = solver._generate_second_stage_cut(node, parent, archive,
                                                        node.index, deadline-.02)
                if cut is not None:
                    changed = add_unique_cut(archive.setdefault(2, {}).setdefault(node.index, []), *cut)
                    row['changed'] += int(changed)
                    report['eta_cuts_changed'] += int(changed)
            row['elapsed'] = time.monotonic()-before
            if not row['complete']:
                return finish('shared_deadline_during_sweep')
            report['rounds_completed'] += 1
            current = root_refresh(f'after_eta_round_{round_index+1}')
            if current is None:
                return finish('shared_deadline_before_root_refresh')
            if row['changed'] == 0:
                return finish('no_new_verified_eta_cut')
        return finish('max_rounds_completed')
    except SolveDeadlineReached:
        return finish('shared_deadline')

