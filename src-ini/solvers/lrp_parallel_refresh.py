"""Process-isolated context refreshes, reusing the existing policy procedure."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
import time
import uuid

from core.backend_telemetry import capture_backend_task
from models.stage_builder import StageModelBuilder, _as_cut, _instance
from solvers.forward_policy_certification import (
    certify_stage2_forward_policy, certify_stage3_forward_policy,
)
from solvers.lrp_parallel import isolated_worker, map_worker_jobs, worker_environment
from solvers.forward_period_dedup import _hash_semantic_value


def _input_fingerprint(job):
    sub_limit = float(job['sub_time_limit'])
    if not sub_limit > 0:
        raise ValueError('Refresh sub_time_limit must be positive; +inf means no per-solve cap')
    digest = hashlib.sha256()
    # Main permits an unlimited per-solve cap under a finite shared deadline.
    # Encode only this permitted infinity; leave the actual job budget intact
    # and retain the generic hash's rejection of nonfinite policy data.
    _hash_semantic_value(digest, tuple(
        ('uncapped_subproblem_time',) if key == 'sub_time_limit' and math.isinf(sub_limit)
        else job[key] for key in
        ('x', 'costs', 'targets', 'builder_options', 'sub_time_limit',
         's2_abs_tol', 's2_rel_cap', 'environment', 'deadline', 'node_seconds')))
    return digest.hexdigest()


def _scope(data, tree, index, root, cuts):
    """Hash full physical input and every coefficient; no truncated array repr."""
    node = tree[2][index]
    if node.index != index or node.time != 2:
        raise ValueError('Refresh node has a foreign stage/index')
    children = []
    for child in node.successor:
        route = tree[3][child]
        if (route.index != child or route.time != 3 or route.predecessor != index
                or route.context.key != node.context.key):
            raise ValueError('Refresh successor has a foreign physical context')
        rows = []
        for row in cuts.get(3, {}).get(child, []):
            typed = _as_cut(row, node.context, int(route.info))
            rows.append((float(typed.intercept).hex(),
                         tuple(float(v).hex() for v in typed.coefficients)))
        children.append((child, int(route.info), route.context.key, tuple(rows)))
    value = (_instance(data).logical_hash(), index, node.context.key,
             node.info, node.predecessor, tuple(children),
             tuple(sorted((str(k), float(v).hex()) for k, v in root.items())))
    return hashlib.sha256(repr(value).encode()).hexdigest()


def make_refresh_job(solver, tree, index, x_star, cost_star, cut_lag,
                     deadline, node_seconds):
    node = tree[2][index]
    children = list(node.successor)
    local_tree = deepcopy({1: {0: tree[1][0]}, 2: {index: node},
                           3: {q: tree[3][q] for q in children}})
    local_x = {1: {0: deepcopy(x_star[1][0])},
               2: ({index: deepcopy(x_star[2][index])} if index in x_star[2] else {}),
               3: {q: deepcopy(x_star[3][q]) for q in children if q in x_star[3]}}
    local_cost = {1: deepcopy(cost_star.get(1, {})),
                  2: ({index: cost_star[2][index]} if index in cost_star[2] else {}),
                  3: {q: cost_star[3][q] for q in children if q in cost_star[3]}}
    cuts = {3: {q: deepcopy(cut_lag.get(3, {}).get(q, [])) for q in children}}
    # Include all known states at this context, because the assignment changes.
    # Only primal witnesses are used by _refresh; no bundle/DFJ state is merged.
    facilities = {int(tree[3][q].info) for q in children}
    targets = {key: {'fixed_trial_witness':deepcopy(value['fixed_trial_witness'])} for key, value in
        getattr(solver.cut_manager, '_fixed_target_cache', {}).items()
        if len(key) == 5 and key[0] == 3 and key[1] == node.context.key
        and key[2] in facilities and value.get('fixed_trial_witness') is not None}
    job = dict(job_id=uuid.uuid4().hex, index=index, context=node.context.key,
        prob_data=solver.prob_data, tree=local_tree, x=local_x, costs=local_cost,
        cuts=cuts, targets=targets,
        source_fingerprint=_scope(solver.prob_data, local_tree, index, local_x[1][0], cuts),
        builder_options=solver.stage_builder.worker_options(),
        sub_time_limit=solver.sub_time_limit,
        s2_abs_tol=getattr(solver, '_s2_abs_tol', None),
        s2_rel_cap=getattr(solver, '_s2_rel_cap', None),
        environment=worker_environment(), deadline=deadline, node_seconds=node_seconds)
    job['input_fingerprint'] = _input_fingerprint(job)
    return job


@capture_backend_task(phase=2, path='backward-refresh', stage=None)
def refresh_context_worker(job):
    """One complete fixed-A subtree; no mutations reach the parent process."""
    started = time.monotonic()
    if (not math.isfinite(float(job['deadline'])) or
            not math.isfinite(float(job['node_seconds'])) or job['node_seconds'] < 0):
        raise ValueError('Refresh job requires a finite shared deadline and nonnegative allowance')
    deadline = min(float(job['deadline']), started + float(job['node_seconds']))
    header = dict(job_id=job['job_id'], index=job['index'], context=job['context'],
        source_fingerprint=job['source_fingerprint'], worker_pid=os.getpid(),
        input_fingerprint=job['input_fingerprint'],
        worker_started=started, deadline=deadline, completed=False)
    if deadline <= started:
        return (dict(header, reason='deadline_before_start'),)
    if (job['tree'][2][job['index']].context.key != job['context'] or
            _input_fingerprint(job) != job['input_fingerprint'] or
            _scope(job['prob_data'], job['tree'], job['index'], job['x'][1][0],
                   job['cuts']) != job['source_fingerprint']):
        raise ValueError('Parallel refresh input scope changed')
    if time.monotonic() >= deadline:
        return (dict(header, reason='deadline_before_environment'),)
    from solvers.backward_solver_lag import BackwardSolverLagrangian
    with isolated_worker(job['environment']) as env:
        if time.monotonic() >= deadline:
            return (dict(header, reason='deadline_after_environment'),)
        builder = StageModelBuilder(job['prob_data'], env=env, **job['builder_options'])
        solver = BackwardSolverLagrangian(job['prob_data'], builder,
                                         sub_time_limit=job['sub_time_limit'])
        solver._s2_abs_tol = job['s2_abs_tol']
        solver._s2_rel_cap = job['s2_rel_cap']
        solver.cut_manager._fixed_target_cache = deepcopy(job['targets'])
        local_x, local_cost = deepcopy(job['x']), deepcopy(job['costs'])
        scores, complete = solver._refresh(job['tree'], local_x, local_cost,
            job['cuts'], deadline, [job['index']])
        if not complete:
            return (dict(header, reason='deadline_before_refresh'),)
        return (dict(header, completed=True, worker_finished=time.monotonic(),
            assignment=local_x[2][job['index']], routes=local_x[3],
            assignment_cost=local_cost[2][job['index']], route_costs=local_cost[3],
            score=scores[job['index']], counts=dict(solver.solve_counts)),)


def _same_number(actual, expected, label):
    if (not math.isfinite(float(actual)) or not math.isfinite(float(expected)) or
            abs(actual-expected) > 2e-6 + 1e-10*max(1., abs(actual), abs(expected))):
        raise ValueError(f'Parallel refresh {label} disagrees with its physical policy')


def validate_refresh_result(solver, tree, x_star, cut_lag, job, result):
    """Audit a complete subtree before any parent state is committed."""
    if not isinstance(result, tuple) or len(result) != 1:
        raise ValueError('Malformed parallel refresh packet')
    packet = result[0]
    for key in ('job_id', 'index', 'context', 'source_fingerprint', 'input_fingerprint'):
        if packet.get(key) != job[key]:
            raise ValueError(f'Parallel refresh result changed {key}')
    index = job['index']
    if _scope(solver.prob_data, tree, index, x_star[1][0], cut_lag) != job['source_fingerprint']:
        raise ValueError('Parallel refresh parent root/context/archive changed')
    if not math.isfinite(float(packet['deadline'])) or packet['deadline'] > job['deadline']:
        raise ValueError('Parallel refresh enlarged its shared deadline')
    if packet.get('completed') is False:
        return None
    if packet.get('completed') is not True:
        raise ValueError('Parallel refresh completion flag is missing')
    node = tree[2][index]
    if set(packet['routes']) != set(node.successor) or set(packet['route_costs']) != set(node.successor):
        raise ValueError('Parallel refresh packet does not contain its complete route subtree')
    assignment = deepcopy(packet['assignment'])
    normalized, outsourcing = certify_stage2_forward_policy(
        solver.prob_data, node, x_star[1][0], assignment)
    assignment.update(normalized)
    assignment['stage_cost'] = outsourcing
    routes, costs = {}, {}
    for child in node.successor:
        values, cost = certify_stage3_forward_policy(solver.prob_data, tree[3][child],
                                                    assignment, packet['routes'][child])
        _same_number(packet['route_costs'][child], cost, 'route cost')
        routes[child] = dict(values, stage_cost=cost)
        costs[child] = cost
    score = deepcopy(packet['score'])
    upper = float(score['envelope_upper'])
    lower = score['envelope_lower']
    if not math.isfinite(upper) or upper < 0 or (lower is not None and
            (not math.isfinite(float(lower)) or lower > upper)):
        raise ValueError('Parallel refresh contains an invalid assignment interval')
    _same_number(packet['assignment_cost'], upper, 'assignment cost')
    _same_number(outsourcing + math.fsum(costs.values()), score['physical_upper'], 'physical upper')
    for child in node.successor:
        theta = float(assignment[f'theta[{child}]'])
        if not math.isfinite(theta) or theta < 0:
            raise ValueError('Parallel refresh route epigraph is invalid')
    residual = math.fsum(max(0., costs[q]-assignment[f'theta[{q}]']) for q in node.successor)
    _same_number(residual, score['route_residual'], 'route residual')
    if score['fixed_A_gap'] is not None:
        if lower is None:
            raise ValueError('Parallel refresh claims a fixed-A gap without a lower bound')
        _same_number(score['physical_upper']-lower, score['fixed_A_gap'], 'fixed-A gap')
    counts = packet['counts']
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
        raise ValueError('Parallel refresh solve counts must be nonnegative integers')
    score.update(worker_pid=packet['worker_pid'],
                 worker_started=packet['worker_started'],
                 worker_finished=packet['worker_finished'])
    # These flags and the actual LB are retained, never inferred from a route.
    return dict(index=index, assignment=assignment, routes=routes, costs=costs,
        assignment_cost=packet['assignment_cost'], score=score, counts=dict(counts),
        worker_pid=packet['worker_pid'])


def parallel_refresh(solver, tree, x_star, cost_star, cut_lag, deadline,
                     second_indices, workers, pool=None):
    """Queue frozen contexts; atomically audit then commit in requested order."""
    indices = list(second_indices)
    remaining = deadline-time.monotonic()
    if remaining <= 0 or not indices:
        return {}, []
    slots = min(max(1, int(workers)), len(indices))
    seconds = remaining / math.ceil(len(indices)/slots)
    jobs = [make_refresh_job(solver, tree, q, x_star, cost_star, cut_lag,
                            deadline, seconds) for q in indices]
    results = map_worker_jobs(refresh_context_worker, jobs, workers, pool=pool)
    if len(results) != len(jobs):
        raise ValueError('Parallel refresh returned the wrong packet count')
    audited = [validate_refresh_result(solver, tree, x_star, cut_lag, job, result)
               for job, result in zip(jobs, results)]
    scores, completed = {}, []
    for packet in audited:
        if packet is None:
            continue
        q = packet['index']
        x_star[2][q] = packet['assignment']
        x_star[3].update(packet['routes'])
        cost_star[2][q] = packet['assignment_cost']
        cost_star[3].update(packet['costs'])
        scores[q] = packet['score']
        completed.append(q)
        for key, value in packet['counts'].items():
            solver.solve_counts[key] = solver.solve_counts.get(key, 0) + value
        solver.solve_counts['parallel_refresh_jobs'] = solver.solve_counts.get('parallel_refresh_jobs', 0)+1
        solver.worker_pids.add(packet['worker_pid'])
        if not hasattr(solver, '_refresh_worker_pids'):
            solver._refresh_worker_pids = set()
        solver._refresh_worker_pids.add(packet['worker_pid'])
    return scores, completed
