"""Process-isolated LRP node jobs behind the original Pool interface."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import multiprocessing as mp
import os
import sys
from multiprocessing.connection import wait
import time
import traceback

from core.backend_telemetry import (backend_scope, capture_backend_task,
                                    merge_worker_result)


def worker_environment():
    """Send effective settings explicitly even to a pool created earlier."""
    return {name: value for name, value in os.environ.items()
            if name.startswith(('LRP_', 'VRP_'))}


@contextmanager
def isolated_worker(environment):
    """Own a quiet Gurobi Env; never pickle or inherit a caller's live Env."""
    import gurobipy as gp
    previous = worker_environment()
    for name in previous:
        os.environ.pop(name, None)
    os.environ.update(environment)
    env = None
    captured = _RecentOutput()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
        # Hide only solver startup. Keep useful Python progress visible and
        # preserve a bounded tail on exceptions, as in the original worker API.
        from core.run_logging import _Tee
        with redirect_stdout(_Tee(sys.stdout, captured)), redirect_stderr(_Tee(sys.stderr, captured)):
            yield env
    except Exception as exc:
        exc.worker_output = captured.getvalue()
        raise
    finally:
        if env is not None:
            env.dispose()
        for name in worker_environment():
            os.environ.pop(name, None)
        os.environ.update(previous)


class _RecentOutput(io.StringIO):
    """Bound exception context without retaining every Level Set iteration."""
    def write(self, text):
        super().write(text)
        if self.tell() > 65536:
            tail = self.getvalue()[-65536:]
            self.seek(0)
            self.truncate(0)
            super().write(tail)
        return len(text)


class _PipeOutput:
    def __init__(self, connection, stream):
        self.connection, self.stream, self.pending = connection, stream, ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending or len(self.pending) >= 65536:
            newline = self.pending.find('\n')
            end = min(newline + 1, 65536) if newline >= 0 else 65536
            chunk, self.pending = self.pending[:end], self.pending[end:]
            self._send(chunk)
        return len(text)

    def _send(self, text):
        try:
            self.connection.send((self.stream, text))
        except (BrokenPipeError, EOFError, OSError):
            pass  # The parent may already be cleaning up another failed job.

    def flush(self):
        if self.pending:
            self._send(self.pending)
            self.pending = ''

    def isatty(self):
        return False


def _worker_with_output(worker, job, connection):
    stdout, stderr = _PipeOutput(connection, 'stdout'), _PipeOutput(connection, 'stderr')
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return worker(job)
    finally:
        stdout.flush()
        stderr.flush()
        connection.close()


def _drain_worker_output(readers, streams, timeout=0):
    ready = wait(readers, timeout) if readers else []
    while ready:
        for reader in ready:
            try:
                stream, text = reader.recv()
            except EOFError:
                readers.remove(reader)
                reader.close()
                continue
            streams[stream].write(text)
            streams[stream].flush()
        ready = wait(readers, 0) if readers else []


class WorkerPoolFailure(RuntimeError):
    """A lost process or result channel left submitted jobs without results.

    This is infrastructure failure, never a solver timeout or a certificate.
    ``completed_results`` retains the already returned prefix for diagnosis;
    callers must not mistake the incomplete batch for a completed cut pass.
    """
    def __init__(self, reason, context, completed_results=()):
        self.reason = str(reason)
        self.worker_context = dict(context)
        self.completed_results = tuple(completed_results)
        super().__init__(f"LRP worker pool failure: {self.reason}; {self.worker_context}")

    def __reduce__(self):
        return type(self), (self.reason, self.worker_context, self.completed_results)


def _check_pool_health(pool, observed_workers, completed_results, submitted_count):
    # Keep Process references after Pool automatically replaces a crashed
    # worker; consulting only pool._pool would lose the old exit code.
    for process in getattr(pool, '_pool', ()):
        observed_workers[id(process)] = process
    recycled = getattr(pool, '_maxtasksperchild', None) is not None
    exited = [(process.pid, process.exitcode) for process in observed_workers.values()
              if process.exitcode is not None
              and (process.exitcode != 0 or not recycled)]
    dead_handlers = [name for name in ('_result_handler', '_task_handler', '_worker_handler')
                     if getattr(pool, name, None) is not None
                     and not getattr(pool, name).is_alive()]
    if exited or dead_handlers:
        reason = 'worker_exited' if exited else 'pool_handler_exited'
        context = {'exited_workers': exited, 'dead_handlers': dead_handlers,
                   'submitted_count': submitted_count,
                   'completed_count': len(completed_results),
                   'incomplete_batch': True}
        raise WorkerPoolFailure(reason, context, completed_results)


def _terminate_failed_pool(pool, failure):
    """Stop an unusable shared/owned pool without the dead-handler assertion.

    CPython Pool.terminate() refuses a nonempty result cache when its result
    handler died (e.g. while unpickling an exception). Mark outstanding async
    results as failed first; otherwise owner cleanup can leave live workers and
    hang in join(). No successful result or numerical bound is manufactured.
    """
    if getattr(pool, '_lrp_worker_failure', None) is not None:
        return
    pool._lrp_worker_failure = failure
    cache = getattr(pool, '_cache', None)
    if cache is not None:
        for result in list(cache.values()):
            try:
                result._set(0, (False, failure))
            except BaseException:
                # An arbitrary user callback may itself fail. The failed pool
                # is no longer reusable, so clear its bookkeeping regardless.
                pass
        cache.clear()
    try:
        pool.terminate()
    except BaseException as cleanup_error:
        failure.worker_context['cleanup_error'] = repr(cleanup_error)
        raise failure from cleanup_error


def map_worker_jobs(worker, jobs, num_processes, pool=None):
    """Stream worker progress through the parent's Tee; retain result order."""
    jobs = list(jobs)
    if not jobs:
        return []
    if int(num_processes) < 1:
        raise ValueError('num_processes must be positive')
    owned = pool is None
    if owned:
        pool = mp.get_context('spawn').Pool(processes=min(int(num_processes), len(jobs)))
    else:
        context = getattr(pool, '_ctx', None)
        if context is None or context.get_start_method() not in ('spawn', 'forkserver'):
            raise ValueError('LRP solver workers require a spawn/forkserver Pool; a live Gurobi Env must not be forked')
    if getattr(pool, '_lrp_worker_failure', None) is not None:
        raise pool._lrp_worker_failure
    observed_workers = {id(process): process for process in getattr(pool, '_pool', ())}
    readers, senders = [], []
    streams = {'stdout': sys.stdout, 'stderr': sys.stderr}
    try:
        pending = []
        for job in jobs:
            reader, sender = mp.Pipe(duplex=False)
            readers.append(reader)
            senders.append(sender)
            pending.append(pool.apply_async(_worker_with_output, (worker, job, sender)))
        results = []
        for result in pending:
            while not result.ready():
                _check_pool_health(pool, observed_workers, results, len(pending))
                _drain_worker_output(readers, streams, timeout=.05)
            _drain_worker_output(readers, streams)
            try:
                value = result.get()
            except Exception as exc:
                merge_worker_result(exc)
                raise
            results.append(merge_worker_result(value))
        if owned:
            pool.close()
        return results
    except WorkerPoolFailure as failure:
        _terminate_failed_pool(pool, failure)
        raise
    except BaseException:
        if owned:
            pool.terminate()
        raise
    finally:
        if owned:
            pool.join()
        for sender in senders:
            sender.close()
        _drain_worker_output(readers, streams)
        for reader in readers:
            reader.close()


@capture_backend_task(phase=None, path='forward', stage=None)
def forward_node_worker(job):
    """Solve one fixed-root S2 node and every one of its facility routes."""
    from models.stage_builder import StageModelBuilder
    from solvers.forward_solver import ForwardSolver
    started = time.monotonic()
    pid = os.getpid()
    try:
        with isolated_worker(job['environment']) as env:
            builder = StageModelBuilder(job['data'], env=env, **job['builder_options'])
            solver = ForwardSolver(job['data'], builder, sub_time_limit=job['sub_time_limit'],
                                   stage2_sub_time_limit=job.get('stage2_sub_time_limit'),
                                   stage2_mip_gap=job.get('stage2_mip_gap'),
                                   period_dedup=False)
            solver.phase = job['phase']
            solver.set_stage2_tolerance(job['s2_abs_tol'], job['s2_rel_cap'])
            with backend_scope(phase=job['phase'], path='forward'):
                packet = solver._forward_recourse_node(job['node'], job['third_nodes'],
                    job['cuts'], job['root_values'], deadline=job['deadline'],
                    stage2_result=job.get('stage2_result'))
            packet.update(worker_pid=pid, solve_counts=dict(solver.solve_counts),
                          diagnostics=solver.last_forward_diagnostics,
                          worker_started=started, worker_finished=time.monotonic())
            for record in packet['diagnostics']:
                record['worker_pid'] = pid
            return (packet,)
    except Exception as exc:
        error = RuntimeError(f"LRP forward worker pid={pid}, S2 node={job['node'].index}: "
                             f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        error.worker_context = {'pid': pid, 'stage': 2, 'node': job['node'].index,
                                'traceback': traceback.format_exc(),
                                'captured_output': getattr(exc, 'worker_output', '')}
        raise error from exc


def _copy_equivalent_subtree(solver, packet, source, target, tree, root_values):
    from solvers.forward_period_dedup import remap_stage2_forward_values
    from solvers.forward_policy_certification import (certify_stage2_forward_policy,
                                                      certify_stage3_forward_policy)
    values = remap_stage2_forward_values(packet['x2'], source, target, tree)
    normalized, cost = certify_stage2_forward_policy(solver.prob_data, target, root_values, values)
    values.update(normalized)
    values['stage_cost'] = cost
    routes, route_costs = {}, {}
    source_by_facility = {tree[3][r].info: r for r in source.successor}
    for r in target.successor:
        third = tree[3][r]
        source_r = source_by_facility[third.info]
        raw = deepcopy(packet['x3'][source_r])
        normalized, cost = certify_stage3_forward_policy(solver.prob_data, third, values, raw)
        raw.update(normalized)
        raw['stage_cost'] = cost
        routes[r], route_costs[r] = raw, cost
    solver.period_dedup_stats['stage2_hits'] += 1
    solver.period_dedup_stats['stage3_hits'] += len(target.successor)
    solver.last_forward_diagnostics.append(dict(stage=2, node=target.index,
        source_node=source.index, status='period_dedup', gurobi_executed=False,
        optimization_completed=False, policy_reaudited=True,
        reason='identical_physical_subtree'))
    return dict(node=target.index, x2=values, cost2=packet['cost2'], x3=routes, cost3=route_costs)


def solve_forward_nodes(solver, tree, cut_lag, root_values, num_processes, *, pool=None, deadline=None):
    options = solver.stage_builder.worker_options()
    environment = worker_environment()
    groups, representative = {}, {}
    dedup = solver.period_dedup and hasattr(solver, '_period_solve_policy')
    if dedup:
        from solvers.forward_period_dedup import stage2_period_key
    for q in tree[1][0].successor:
        key = (stage2_period_key(tree[2][q], root_values, cut_lag, solver.prob_data,
                                tree, solve_policy=solver._period_solve_policy()) if dedup else q)
        representative[q] = groups.setdefault(key, q)
    jobs = []
    for q in groups.values():
        node = tree[2][q]
        jobs.append(dict(data=solver.prob_data, node=node,
            third_nodes={r: tree[3][r] for r in node.successor},
            cuts={3: {r: cut_lag.get(3, {}).get(r, []) for r in node.successor}},
            root_values=root_values, builder_options=options,
            sub_time_limit=solver.sub_time_limit, phase=solver.phase,
            stage2_sub_time_limit=solver.stage2_sub_time_limit,
            stage2_mip_gap=solver.stage2_mip_gap,
            s2_abs_tol=solver._s2_abs_tol, s2_rel_cap=solver._s2_rel_cap,
            deadline=deadline, environment=environment,
            stage2_result=solver._take_stage2_handoff(tree, node, root_values,
                                                     cut_lag, deadline)))
    results = map_worker_jobs(forward_node_worker, jobs, num_processes, pool=pool)
    solved = {}
    for wrapped in results:
        packet = wrapped[0]
        solved[packet['node']] = packet
        for key, count in packet['solve_counts'].items():
            solver.solve_counts[key] = solver.solve_counts.get(key, 0) + count
        solver.last_forward_diagnostics.extend(packet['diagnostics'])
    packets = []
    for q in tree[1][0].successor:
        source = representative[q]
        packet = solved[source]
        if source != q:
            packet = _copy_equivalent_subtree(solver, packet, tree[2][source], tree[2][q], tree, root_values)
        packets.append(packet)
    return packets


@capture_backend_task(phase=None, path='backward', stage=None)
def sbc_node_worker(job):
    from models.stage_builder import StageModelBuilder
    from models.subproblem_builder import SubproblemBuilder
    from solvers.backward_solver_sbc import BackwardSolverSBC
    pid = os.getpid()
    try:
        with isolated_worker(job['environment']) as env:
            solver = BackwardSolverSBC(job['data'],
                StageModelBuilder(job['data'], env=env, **job['builder_options']),
                SubproblemBuilder(job['data'], env=env, **job['subproblem_options']),
                strengthen_s2=job['strengthen_s2'], strengthen=job['strengthen'],
                sub_time_limit=job['sub_time_limit'],
                route_lp_separation_time_limit=job['route_lp_separation_time_limit'])
            solver.phase = job['phase']
            with backend_scope(phase=job['phase'], path='backward', stage=job['stage']):
                generated = solver._generate_cut(job['stage'], job['node'], job['parent'],
                    job['cuts'], job['node'].index, deadline=job['deadline'])
            for record in solver.last_cut_diagnostics:
                record['worker_pid'] = pid
            return (dict(node=job['node'].index, stage=job['stage'], generated=generated,
                         worker_pid=pid, diagnostics=solver.last_cut_diagnostics,
                         solve_counts=dict(solver.solve_counts)),)
    except Exception as exc:
        error = RuntimeError(f"LRP SBC worker pid={pid}, stage={job['stage']}, node={job['node'].index}: "
                             f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        error.worker_context = {'pid': pid, 'stage': job['stage'], 'node': job['node'].index,
                                'traceback': traceback.format_exc(),
                                'captured_output': getattr(exc, 'worker_output', '')}
        raise error from exc


def solve_sbc_layer(solver, stage, nodes_and_parents, cut_lag, num_processes, *, pool=None, deadline=None):
    from cuts.benders_cuts import add_unique_cut
    options = solver.stage_builder.worker_options()
    sub_options = solver.subproblem_builder.worker_options()
    environment = worker_environment()
    jobs = []
    for node, parent in nodes_and_parents:
        jobs.append(dict(data=solver.prob_data, stage=stage, node=node, parent=parent,
            cuts={3: {r: cut_lag.get(3, {}).get(r, []) for r in node.successor}} if stage == 2 else {},
            builder_options=options, subproblem_options=sub_options,
            strengthen_s2=solver.strengthen_s2, strengthen=solver.strengthen,
            sub_time_limit=solver.sub_time_limit,
            route_lp_separation_time_limit=solver.route_lp_separation_time_limit,
            phase=solver.phase, deadline=deadline, environment=environment))
    for wrapped in map_worker_jobs(sbc_node_worker, jobs, num_processes, pool=pool):
        packet = wrapped[0]
        for key, count in packet['solve_counts'].items():
            solver.solve_counts[key] = solver.solve_counts.get(key, 0) + count
        generated = packet['generated']
        if generated is not None:
            changed = add_unique_cut(cut_lag[stage].setdefault(packet['node'], []), *generated)
            packet['diagnostics'][-1]['archive_changed'] = bool(changed)
        solver.last_cut_diagnostics.extend(packet['diagnostics'])
        solver.worker_pids.add(packet['worker_pid'])
