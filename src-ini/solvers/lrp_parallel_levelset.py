"""Process-safe Level Set jobs in the original node coordinates."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
import time
import uuid

from core.backend_telemetry import capture_backend_task
from models.stage_builder import StageModelBuilder, _state_keys
from models.route_dfj_pool import route_dfj_checkpoint, restore_route_dfj_checkpoint
from solvers.lrp_parallel import isolated_worker, worker_environment


def _fingerprint(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _scoped_targets(manager, stage, context, facility=None):
    return {key: deepcopy(value) for key, value in
            getattr(manager, '_fixed_target_cache', {}).items()
            if key[0] == stage and key[1] == context and key[2] == facility}


def _merge_fixed_target(prob_data, node, key, old, incoming):
    """Retain the strongest proof and cheapest audited primal independently."""
    best = deepcopy(incoming if old is None or incoming['used_lower_bound'] >=
                    old['used_lower_bound'] else old)
    if key[0] != 3:
        return best
    from cuts.lagrangian_cuts import _reaudit_fixed_trial_witness
    witness = None
    for record in (old, incoming):
        if record is None or record.get('fixed_trial_witness') is None:
            continue
        try:
            candidate = _reaudit_fixed_trial_witness(
                prob_data, node, dict(key[3]), record['fixed_trial_witness'])
        except (ValueError, KeyError):
            # A malformed optional route does not strengthen or invalidate
            # the separately certified scalar lower bound.
            continue
        if witness is None or candidate['physical_upper'] < witness['physical_upper']:
            witness = candidate
    best.pop('fixed_trial_witness', None)
    if witness is not None:
        if best['used_lower_bound'] > witness['physical_upper']:
            raise ValueError('Fixed-target lower bound exceeds the audited route upper bound')
        best['fixed_trial_witness'] = witness
    return best


def make_lagrangian_job(solver, stage, node, index, parent, target,
                        cut_lag, cut_Dict, options, deadline,
                        node_seconds=None, seed_only=False, skip_seed=False):
    """Export only the selected node's archives; no live models cross a PID."""
    index = int(index)
    archive = cut_Dict.get(stage, {}).get(index, [])
    own = deepcopy(cut_lag.get(stage, {}).get(index, []))
    cuts = {stage: {index: own}}
    if stage == 2:
        cuts[3] = {child: deepcopy(cut_lag.get(3, {}).get(child, []))
                   for child in node.successor}
    routes = ([(node.context, int(node.info))] if stage == 3 else
              [(node.context, i) for i in range(node.context.m)])
    coordinates = getattr(solver.cut_manager, '_bundle_coordinates', {}).get(id(archive))
    return dict(job_id=uuid.uuid4().hex, stage=stage, index=index, node=node, prob_data=solver.prob_data,
                context=node.context.key, source_cuts=own,
                source_fingerprint=_fingerprint(cuts), cut_lag=cuts,
                supports=deepcopy(archive), coordinates=deepcopy(coordinates),
                support_fingerprint=_fingerprint((archive, coordinates)),
                targets=_scoped_targets(solver.cut_manager, stage, node.context.key,
                    int(node.info) if stage == 3 else None),
                route_dfj=route_dfj_checkpoint(routes),
                builder_options=solver.stage_builder.worker_options(),
                parent=deepcopy(parent), target=float(target), options=dict(options),
                adaptive_alpha=solver.adaptive_alpha,
                sub_time_limit=solver.sub_time_limit,
                s2_abs_tol=getattr(solver, '_s2_abs_tol', None),
                s2_rel_cap=getattr(solver, '_s2_rel_cap', None),
                s2_tight_exit_abs=getattr(solver, '_s2_tight_exit_abs', None) if stage == 2 else None,
                sbc_prepass_enabled=solver.sbc_prepass_enabled,
                route_witness_limits=(dict(solver._route_witness_buffer.limits)
                    if stage == 3 and getattr(solver, '_route_witness_buffer', None) is not None else None),
                deadline=deadline, node_seconds=node_seconds,
                seed_only=bool(seed_only), skip_seed=bool(skip_seed),
                environment=worker_environment())


@capture_backend_task(phase=2, path='backward', stage=None)
def lagrangian_worker(job):
    """Run one original _generate_cut, returning its portable evidence/state."""
    stage, index, node = job['stage'], job['index'], job['node']
    if node.context.key != job['context']:
        raise ValueError('Parallel Level Set node context changed')
    if _fingerprint(job['cut_lag']) != job['source_fingerprint']:
        raise ValueError('Parallel Level Set source archive changed')
    deadline = job['deadline']
    if job['node_seconds'] is not None:
        local = time.monotonic() + max(0., float(job['node_seconds']))
        deadline = local if deadline is None else min(deadline, local)
    if deadline is not None and time.monotonic() >= deadline:
        return (dict(job_id=job['job_id'], stage=stage, index=index, context=job['context'],
                     source_fingerprint=job['source_fingerprint'],
                     started=False, worker_pid=os.getpid()),)
    from solvers.backward_solver_lag import BackwardSolverLagrangian
    with isolated_worker(job['environment']) as env:
        builder = StageModelBuilder(job['prob_data'], env=env, **job['builder_options'])
        solver = BackwardSolverLagrangian(job['prob_data'], builder,
            adaptive_alpha=job['adaptive_alpha'], sub_time_limit=job['sub_time_limit'])
        if job.get('route_witness_limits') is not None:
            if stage != 3:
                raise ValueError('native route witness collection requires an S3 job')
            solver.configure_route_witness_collection(True, **job['route_witness_limits'])
        from models.subproblem_builder import SubproblemBuilder
        solver._route_sbc.subproblem_builder = SubproblemBuilder(job['prob_data'], env=env,
            lazy_threshold=builder.lazy_threshold, connectivity=builder.connectivity)
        solver.cut_manager._model_env = env
        solver.sbc_prepass_enabled = job['sbc_prepass_enabled']
        solver._s2_abs_tol, solver._s2_rel_cap = job['s2_abs_tol'], job['s2_rel_cap']
        solver._s2_tight_exit_abs = job.get('s2_tight_exit_abs') if stage == 2 else None
        cuts = deepcopy(job['cut_lag'])
        supports = deepcopy(job['supports'])
        bundles = {stage: {index: supports}}
        if stage == 3 and job['coordinates'] is not None:
            solver.cut_manager._bundle_coordinates = {id(supports): deepcopy(job['coordinates'])}
        solver.cut_manager._fixed_target_cache = deepcopy(job['targets'])
        routes = ([(node.context, int(node.info))] if stage == 3 else
                  [(node.context, i) for i in range(node.context.m)])
        restore_route_dfj_checkpoint(routes, job['route_dfj'])
        if deadline is not None and time.monotonic() >= deadline:
            return (dict(job_id=job['job_id'], stage=stage, index=index, context=job['context'],
                         source_fingerprint=job['source_fingerprint'],
                         started=False, worker_pid=os.getpid()),)
        changed = solver._generate_cut(stage, node, index, job['parent'], job['target'],
            cuts, bundles, job['options'], deadline,
            seed_only=job['seed_only'], skip_seed=job['skip_seed'])
        solver._record_counts_only()
        return (dict(job_id=job['job_id'], stage=stage, index=index, context=job['context'],
            source_fingerprint=job['source_fingerprint'], started=True,
            worker_pid=os.getpid(), changed=bool(changed),
            cuts=deepcopy(cuts.get(stage, {}).get(index, [])), supports=deepcopy(supports),
            coordinates=deepcopy(getattr(solver.cut_manager, '_bundle_coordinates', {}).get(id(supports))),
            targets=_scoped_targets(solver.cut_manager, stage, node.context.key,
                    int(node.info) if stage == 3 else None),
            route_dfj=route_dfj_checkpoint(routes),
            diagnostics=deepcopy(solver.last_cut_diagnostics),
            counts=dict(solver.solve_counts),
            manager_counts=dict(getattr(solver.cut_manager, 'solve_counts', {})),
            sbc_counts=dict(solver._route_sbc.solve_counts),
            **({'route_witnesses': solver.drain_route_witnesses()}
               if job.get('route_witness_limits') is not None else {})),)


def _preserves_inherited_rows(inherited, returned, keys, *, upper):
    """Same-slope dominance, without rescanning every returned row.

    The existing payload validator runs first. Float tuples preserve its
    exact coordinate equality, including missing and signed zeros. Keep
    original comparisons for unusual endpoint types accepted by old callers.
    """
    if not inherited:
        return True
    if any(type(value) not in (int, float)
           for rows in (inherited, returned) for _, value in rows):
        return all(any(
            all(float(pi.get(k, 0.)) == float(old_pi.get(k, 0.)) for k in keys)
            and (value <= old_value if upper else value >= old_value)
            for pi, value in returned) for old_pi, old_value in inherited)
    strongest = {}
    for pi, value in returned:
        slope = tuple(float(pi.get(k, 0.)) for k in keys)
        previous = strongest.get(slope)
        if previous is None or (value < previous if upper else value > previous):
            strongest[slope] = value
    for old_pi, old_value in inherited:
        slope = tuple(float(old_pi.get(k, 0.)) for k in keys)
        value = strongest.get(slope)
        if value is None or not (value <= old_value if upper else value >= old_value):
            return False
    return True


def merge_lagrangian_result(solver, job, result, cut_lag, cut_Dict):
    """Commit one disjoint node result in the parent's deterministic order."""
    payload = result[0]
    stage, index, node = job['stage'], job['index'], job['node']
    for key in ('job_id', 'stage', 'index', 'context', 'source_fingerprint'):
        if payload.get(key) != job[key]:
            raise ValueError('Parallel Level Set result has a foreign ' + key)
    merged = getattr(solver, '_merged_parallel_levelset_jobs', set())
    if job['job_id'] in merged:
        return False
    if not payload['started']:
        return False
    current = cut_lag.get(stage, {}).get(index, [])
    scoped = {stage: {index: current}}
    if stage == 2:
        scoped[3] = {q: cut_lag.get(3, {}).get(q, []) for q in node.successor}
    if _fingerprint(scoped) != job['source_fingerprint']:
        raise ValueError('Parallel Level Set source/child archive changed before merge')
    supports = cut_Dict.get(stage, {}).get(index, [])
    coordinates = getattr(solver.cut_manager, '_bundle_coordinates', {}).get(id(supports))
    if _fingerprint((supports, coordinates)) != job['support_fingerprint']:
        raise ValueError('Parallel Level Set parent bundle changed before merge')
    rows = payload['cuts']
    keys = set(_state_keys(node.context, int(node.info) if stage == 3 else None))
    for pi, intercept in rows:
        if set(pi) - keys or not math.isfinite(float(intercept)) or any(
                not math.isfinite(float(value)) for value in pi.values()):
            raise ValueError('Parallel Level Set result has invalid cut coordinates')
    # add_unique_cut legitimately strengthens a same-slope row in place.
    # Every inherited slope must survive with at least its old intercept.
    if not _preserves_inherited_rows(current, rows, keys, upper=False):
        raise ValueError('Parallel Level Set result dropped or weakened an inherited cut')
    for coefficients, upper in payload['supports']:
        if set(coefficients) - keys or not math.isfinite(float(upper)) or any(
                not math.isfinite(float(value)) for value in coefficients.values()):
            raise ValueError('Parallel Level Set support has invalid coordinates')
    if stage == 3 and payload['coordinates'] is not None:
        ctx, i = node.context, int(node.info)
        shifts = dict.fromkeys(_state_keys(ctx, i), 0.)
        centered = job['environment'].get('LRP_RESIDUAL_CENTERING', '1') not in ('0', 'false', 'False')
        if centered:
            vertices = [0] + [j+1 for j in range(ctx.n) if ctx.active[j]]
            for j in range(ctx.n):
                if ctx.active[j]:
                    shifts[f'alpha[{i},{j}]'] = min(float(ctx.route_cost[i,v,j+1])
                                                   for v in vertices if v != j+1)
        expected = (ctx.route_key(i), tuple(shifts.items()))
        if payload['coordinates'] != expected:
            raise ValueError('Parallel Level Set support coordinate scope changed')
        if coordinates == payload['coordinates']:
            if not _preserves_inherited_rows(supports, payload['supports'], keys, upper=True):
                raise ValueError('Parallel Level Set result dropped an inherited support')
    elif stage == 3 and payload['supports'] and payload['coordinates'] != coordinates:
        raise ValueError('Parallel Level Set support coordinates are missing')
    expected_names = tuple(_state_keys(node.context, int(node.info) if stage == 3 else None))
    facility = int(node.info) if stage == 3 else None
    cache = getattr(solver.cut_manager, '_fixed_target_cache', {})
    merged_targets = {}
    for key, value in payload['targets'].items():
        if (len(key) != 5 or key[:3] != (stage, job['context'], facility)
                or tuple(name for name, _ in key[3]) != expected_names
                or any(v not in (0., 1.) for _, v in key[3])
                or (stage == 3 and key[4] != ())):
            raise ValueError('Parallel Level Set target cache has a foreign physical scope')
        if not math.isfinite(float(value['used_lower_bound'])) or value['used_lower_bound'] < 0.:
            raise ValueError('Parallel Level Set target cache has an invalid lower bound')
        merged_targets[key] = _merge_fixed_target(solver.prob_data, node, key,
                                                 cache.get(key), value)
    for group in ('counts', 'manager_counts', 'sbc_counts'):
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in payload[group].values()):
            raise ValueError('Parallel Level Set counts must be nonnegative integers')
    if any(d.get('stage') != stage for d in payload['diagnostics']):
        raise ValueError('Parallel Level Set diagnostic stage mismatch')
    witnesses = payload.get('route_witnesses', ())
    if witnesses:
        from models.stage_builder import _instance
        from solvers.lrp_backward_route_witnesses import validate_route_witness
        limits = job.get('route_witness_limits')
        if (stage != 3 or limits is None or not isinstance(witnesses, (tuple, list))
                or len(witnesses) > min(limits['max_routes'], limits['max_routes_per_scope'])):
            raise ValueError('Parallel Level Set route witness packet exceeds its requested scope/budget')
        data_hash = _instance(solver.prob_data).logical_hash()
        for witness in witnesses:
            validate_route_witness(solver.prob_data, node, witness, instance_sha256=data_hash)
    routes = ([(node.context, int(node.info))] if stage == 3 else
              [(node.context, i) for i in range(node.context.m)])
    # This validator checks every row before changing the physical DFJ pool.
    restore_route_dfj_checkpoint(routes, payload['route_dfj'])
    current = cut_lag.setdefault(stage, {}).setdefault(index, [])
    current[:] = deepcopy(rows)
    supports = cut_Dict.setdefault(stage, {}).setdefault(index, [])
    supports[:] = deepcopy(payload['supports'])
    if stage == 3:
        registry = getattr(solver.cut_manager, '_bundle_coordinates', None)
        if registry is None:
            solver.cut_manager._bundle_coordinates = registry = {}
        if payload['coordinates'] is None:
            registry.pop(id(supports), None)
        else:
            registry[id(supports)] = payload['coordinates']
    cache = getattr(solver.cut_manager, '_fixed_target_cache', None)
    if cache is None:
        solver.cut_manager._fixed_target_cache = cache = {}
    cache.update(merged_targets)
    for target, incoming in ((solver.solve_counts, payload['counts']),
                             (solver._route_sbc.solve_counts, payload['sbc_counts'])):
        for key, value in incoming.items():
            target[key] = target.get(key, 0) + value
    counts = getattr(solver.cut_manager, 'solve_counts', None)
    if counts is None:
        solver.cut_manager.solve_counts = counts = {}
    for key, value in payload['manager_counts'].items():
        counts[key] = counts.get(key, 0) + value
    solver.last_cut_diagnostics.extend(payload['diagnostics'])
    solver.last_s2_cut_diagnostics.extend(d for d in payload['diagnostics'] if d['stage'] == 2)
    sink = getattr(solver, '_route_witness_buffer', None)
    if sink is not None:
        for witness in witnesses:
            sink.accept(node, witness)
    merged.add(job['job_id'])
    solver._merged_parallel_levelset_jobs = merged
    solver.solve_counts['parallel_levelset_jobs'] = solver.solve_counts.get('parallel_levelset_jobs', 0) + 1
    return payload['changed']
