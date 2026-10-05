"""Facility-specific native PCTSP and fixed-TSP adapters for stochastic LRP.

A physical root is split into native start/end copies; directed costs are
preserved exactly. No fleet grouping, matching, metricity, symmetry, inactive
knapsack items, or approximate objective substitution is used.

The extension is loaded only from a source/binary-hash-verified build manifest.
This deliberately rejects the repository's old prebuilt extension, which can
predate the certified directed-rounding C++ source.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import time

import numpy as np

from core.backend_telemetry import backend_call, record_backend_event
from core.solve_deadline import SolveDeadlineReached, bounded_solve_time
from models.stage_builder import _instance, _node_context, _state_keys
from models.stage_model_core import audit_tour, index

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src-ini/customized-subprob/s3backward/espprc.cpp'


class NativeUnavailable(RuntimeError):
    """There is no verified current native extension for this interpreter."""


def _directed(value, upward):
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError('native oracle arithmetic overflow')
    represented = Fraction.from_float(answer)
    if (represented < value) if upward else (represented > value):
        answer = math.nextafter(answer, math.inf if upward else -math.inf)
    return answer


def _fraction(value):
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError('nonfinite native oracle coefficient')
    return Fraction.from_float(answer)


def _finite(value):
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and abs(value) < 1e100 else None


def _bounded_free_native_time(time_limit, deadline):
    """Only explicit None permits an uncapped free-oracle request.

    Numeric zero and nonfinite durations retain their former rejection. An
    expired deadline is never converted into native's zero=unlimited sentinel.
    Fixed-tour entry points keep their separate positive-finite contract.
    """
    if time_limit is not None:
        if not math.isfinite(float(time_limit)) or time_limit <= 0:
            raise ValueError('time_limit must be positive and finite, or explicitly None')
        return bounded_solve_time(time_limit, deadline)
    if deadline is None:
        return None
    deadline = float(deadline)
    if math.isnan(deadline):
        raise ValueError('deadline must not be NaN')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SolveDeadlineReached('global solve deadline reached')
    return remaining if math.isfinite(remaining) else None


def _native_manifest_path(manifest_name=None):
    """Resolve an explicit override once; never bypass a broken override."""
    explicit = manifest_name if manifest_name is not None else os.environ.get('LRP_NATIVE_MANIFEST')
    if explicit is not None:
        if not str(explicit).strip():
            raise NativeUnavailable('LRP native manifest override is empty')
        return Path(explicit).expanduser().resolve()
    local = SOURCE.parent / '.native-build' / sys.implementation.cache_tag / 'native_build.json'
    if local.is_file():
        return local
    # Preserve existing audited experiment manifests until their next build.
    return ROOT / 'artifacts/lrp_native_oracle/native_build.json'


def _file_stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _load_native(manifest_name=None):
    # Resolve before caching so an environment override changed between worker
    # jobs cannot silently reuse a different native build. Stat changes also
    # invalidate a cached certificate; changed files are hashed again below.
    manifest_path = _native_manifest_path(manifest_name)
    try:
        encoded = manifest_path.read_text()
        manifest = json.loads(encoded)
        binary = Path(manifest['binary'])
        if not binary.is_absolute():
            binary = manifest_path.parent / binary
        binary = binary.resolve()
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if not suffix or not binary.name.endswith(suffix):
            raise NativeUnavailable(f'native extension does not match this Python ABI: {binary}')
        if manifest.get('python_cache_tag', sys.implementation.cache_tag) != sys.implementation.cache_tag:
            raise NativeUnavailable('native build uses a different Python ABI')
        if manifest.get('extension_suffix', suffix) != suffix:
            raise NativeUnavailable('native build uses a different extension suffix')
        return _load_verified_native(str(manifest_path), encoded, str(binary),
                                     str(SOURCE.resolve()), _file_stamp(SOURCE), _file_stamp(binary))
    except NativeUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NativeUnavailable(f'invalid or unavailable verified native build {manifest_path}: {exc}') from exc


@lru_cache(maxsize=4)
def _load_verified_native(manifest_path, encoded, binary_name, source_name, source_stamp, binary_stamp):
    manifest = json.loads(encoded)
    binary, source = Path(binary_name), Path(source_name)
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise NativeUnavailable('native build predates the current C++ source')
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256']:
        raise NativeUnavailable('native extension hash differs from build manifest')
    # The final component must match PyInit_espprc_cpp. A private module prefix
    # prevents accidentally reusing a stale legacy import with that basename.
    name = '_lrp_verified_native_' + manifest['binary_sha256'] + '.espprc_cpp'
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise NativeUnavailable(f'cannot load {binary}')
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise NativeUnavailable(f'cannot import native extension for this interpreter: {binary}: {exc}') from exc
    if not hasattr(module, 'solve_pctsp'):
        raise NativeUnavailable('extension lacks the certified solve_pctsp ABI')
    sys.modules[name] = module
    return module, manifest


# Keep the former private helper's cache-reset hook available to diagnostics.
_load_native.cache_clear = _load_verified_native.cache_clear


class LRPNativeRouteOracle:
    """Reusable data preparation for exactly one (period, scenario, facility)."""

    def __init__(self, prob_data, node, *, manifest=None):
        self.data = _instance(prob_data)
        self.context = _node_context(self.data, node, stage=3)
        self.facility = index(int(node.info), self.context.m, 'facility')
        self.keys = _state_keys(self.context, self.facility)
        self.active = tuple(j for j in range(self.context.n) if self.context.active[j])
        if len(self.active) > 256:
            raise NativeUnavailable('native PCTSP supports at most 256 active customers')
        self.manifest_name = None if manifest is None else str(manifest)
        self._route_cost = self.context.route_cost[self.facility]
        self.objective_shifts = dict.fromkeys(self.keys, 0.)
        self._is_residual_view = False
        self.cost, self.volume = self._layout(self.active)
        self.solve_count = 0

    def residual_view(self, shifts):
        """Use downward residual incoming costs with independently scored support.

        ``shifts`` maps original parent keys (or customer integer indices) to
        nonnegative incoming lower bounds. The dispatch shift must be zero.
        Native Q' underestimates Q_res=route_original-shifts*alpha-rho*alpha;
        its certified LB is valid directly for Q_res. The returned incumbent
        objective is rescored in Q_res, never taken from the lower-cost model.
        """
        if self._is_residual_view:
            raise ValueError('create a residual view from the original oracle only')
        parsed=dict.fromkeys(self.keys,0.)
        for key,value in dict(shifts).items():
            if isinstance(key,(int,np.integer)) and not isinstance(key,bool):
                key=f'alpha[{self.facility},{index(int(key),self.context.n,"customer")}]'
            if key not in parsed:
                raise ValueError(f'unknown residual shift {key}')
            parsed[key]=float(value)
        if any(not math.isfinite(value) or value<0. for value in parsed.values()):
            raise ValueError('incoming residual shifts must be finite and nonnegative')
        if parsed[f'u[{self.facility}]']!=0.:
            raise ValueError('this incoming-cost view does not shift dispatch cost')
        view=object.__new__(type(self))
        view.data,view.context,view.facility=self.data,self.context,self.facility
        view.keys,view.active=self.keys,self.active
        view.manifest_name=self.manifest_name
        view._route_cost=np.array(self._route_cost,copy=True)
        allowed=[0]+[j+1 for j in self.active]
        for j in range(self.context.n):
            amount=parsed[f'alpha[{self.facility},{j}]']
            if not self.context.active[j]:
                if amount:raise ValueError('inactive coordinates must have zero incoming shift')
                continue
            for v in allowed:
                if v==j+1:continue
                residual=_fraction(self._route_cost[v,j+1])-_fraction(amount)
                if residual<0:raise ValueError('shift exceeds an allowed incoming arc cost')
                view._route_cost[v,j+1]=_directed(residual,False)
        view._route_cost.setflags(write=False)
        view.objective_shifts=parsed
        view._is_residual_view=True
        view.cost,view.volume=view._layout(view.active)
        view.solve_count=0
        return view

    def _layout(self, customers):
        """[selected original customers, start(root 0), end(root 0)]."""
        physical_local = np.array([j + 1 for j in customers] + [0, 0], dtype=int)
        cost = np.ascontiguousarray(self._route_cost[
            np.ix_(physical_local, physical_local)], dtype=np.float64)
        volume = np.zeros(len(customers) + 2, dtype=np.float64)
        volume[:len(customers)] = self.context.demand[list(customers)]
        return cost, volume

    def _native(self, customers, cost, volume, rho, sigma, *, time_limit,
                label_budget=0, ng_size=8, deadline=None, top_k=1):
        time_limit = _bounded_free_native_time(time_limit, deadline)
        if type(top_k) is not int or not 1 <= top_k <= 64:
            raise ValueError('top_k must be an integer in 1..64')
        if type(label_budget) is not int or label_budget < 0:
            raise ValueError('label_budget must be a nonnegative integer')
        module, manifest = _load_native(self.manifest_name)
        n = len(customers)
        prize = np.zeros(n + 2, dtype=np.float64)
        prize[:n] = rho
        # Hash verification/import can use time. Recheck immediately before
        # native starts, and translate only an explicit uncapped request.
        time_limit = _bounded_free_native_time(time_limit, deadline)
        kernel_time_limit = 0. if time_limit is None else time_limit
        self.solve_count += 1
        with backend_call('espprc', 'pctsp', stage=3, facility=self.facility,
                          active_customers=n, time_limit=time_limit,
                          native_unbounded=time_limit is None) as event:
            raw = dict(module.solve_pctsp(cost, prize, volume, n, n, n + 1,
                float(self.context.capacity[self.facility]), float(sigma),
                np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
                cutoff=float('inf'), top_k=top_k, label_budget=label_budget,
                time_limit_s=kernel_time_limit, ng_size=int(ng_size),
                bound_ng_size=min(int(ng_size), 8)))
            event.update(status=raw.get('status'), native_lb_certified=raw.get('lb_certified'),
                         timed_out=raw.get('timed_out'), optimality_proven=raw.get('status') == 0)
        raw['source_sha256'] = manifest['source_sha256']
        raw['binary_sha256'] = manifest['binary_sha256']
        raw['effective_time_limit_s'] = time_limit
        raw['native_unbounded'] = time_limit is None
        return raw

    def _decode(self, raw, customers, pi):
        """Independent strict original-domain primal certification."""
        y = raw.get('y')
        if type(y) is not int or y not in (0, 1):
            raise ValueError('invalid native dispatch bit')
        if raw.get('alpha_inactive') != []:
            raise ValueError('inactive-customer assignments are forbidden in LRP')
        path = raw.get('path')
        if not isinstance(path, list) or any(type(j) is not int for j in path):
            raise ValueError('invalid native path indices')
        n = len(customers)
        if y:
            if len(path) < 3 or path[0] != n or path[-1] != n + 1:
                raise ValueError('native path has incorrect own-root copies')
            visited = path[1:-1]
            if len(visited) != len(set(visited)) or any(j < 0 or j >= n for j in visited):
                raise ValueError('native path is not elementary over active customers')
            physical_path = [0] + [customers[j] + 1 for j in visited] + [0]
        else:
            if path:
                raise ValueError('idle native facility has a nonempty path')
            visited, physical_path = [], []
        selected = {customers[j] for j in visited}
        alpha = tuple(int(j in selected) for j in range(self.context.n))
        self.context.check_route_state(self.facility, alpha, y)
        arcs = list(zip(physical_path[:-1], physical_path[1:]))
        tour = audit_tour(self.context, self.facility, alpha, y, arcs)
        xcp = {f'alpha[{self.facility},{j}]': float(alpha[j]) for j in range(self.context.n)}
        xcp[f'u[{self.facility}]'] = float(y)
        route_exact = sum((_fraction(self.context.route_cost[self.facility, v, w])
                           for v, w in arcs), Fraction())
        reward = sum((_fraction(pi[key]) * int(xcp[key]) for key in self.keys), Fraction())
        native_route_exact = sum((_fraction(self._route_cost[v,w]) for v,w in arcs), Fraction())
        native_objective = _directed(native_route_exact-reward, True)
        if _finite(raw.get('obj_val')) != native_objective:
            raise ValueError('native objective does not equal independently rebuilt directed-up cost')
        shift = sum((_fraction(self.objective_shifts[key])*int(xcp[key]) for key in self.keys), Fraction())
        objective = _directed(route_exact-shift-reward, True)
        return dict(inner_value=objective, inner_xcp=xcp,
                    route_cost=_directed(route_exact, True), centered_route_cost=_directed(route_exact-shift,True), tour=tour,
                    x={f'r[{self.facility},{v},{w}]': 1. for v, w in arcs})

    def solve(self, pi_value, *, time_limit=1., label_budget=0, ng_size=8, deadline=None,
              top_k=1):
        """min route - rho*alpha - sigma*u over the legal free parent domain.

        Returns inner_value/inner_xcp for bundle upper supports and outer_lb
        for cut intercepts. Neither channel is substituted for the other.
        Explicit time_limit=None removes the per-call cap; any finite shared
        deadline still bounds native execution. Numeric zero is invalid.
        """
        started = time.perf_counter()
        if type(top_k) is not int or not 1 <= top_k <= 64:
            raise ValueError('top_k must be an integer in 1..64')
        time_limit = _bounded_free_native_time(time_limit, deadline)
        unknown = set(pi_value) - set(self.keys)
        if unknown:
            raise ValueError(f'unknown parent multipliers: {sorted(unknown)}')
        pi = {key: float(pi_value.get(key, 0.)) for key in self.keys}
        if any(not math.isfinite(v) for v in pi.values()):
            raise ValueError('nonfinite parent multipliers')
        sigma = pi[f'u[{self.facility}]']
        rho = [pi[f'alpha[{self.facility},{j}]'] for j in self.active]
        # Nonpositive rewards + nonnegative original costs make the empty
        # route an exact optimum. No metric shortcut is being applied.
        if sigma <= 0. and all(value <= 0. for value in rho):
            xcp = dict.fromkeys(self.keys, 0.)
            record_backend_event('espprc', 'analytic_result', 'nonpositive_rewards', stage=3)
            return dict(inner_value=0., inner_xcp=xcp, outer_lb=0., exact=True,
                status='OPTIMAL', source='lrp_analytic_nonpositive_rewards',
                route_cost=0., tour=audit_tour(self.context,self.facility,[0]*self.context.n,0,[]),
                x={}, native_executed=False, wall_seconds=time.perf_counter()-started,
                raw_bound=0., incumbent_policy_certified=True)
        raw = self._native(self.active, self.cost, self.volume, rho, sigma,
                           time_limit=time_limit, label_budget=label_budget, ng_size=ng_size,
                           deadline=deadline, **({'top_k': top_k} if top_k > 1 else {}))
        status = raw.get('status')
        accepted_status = type(status) is int and status in (0, 2)
        primal, reason = None, None
        if accepted_status:
            try:
                primal = self._decode(raw, self.active, pi)
            except ValueError as exc:
                reason = str(exc)
        bound = _finite(raw.get('lb'))
        if not (accepted_status and raw.get('lb_certified') is True):
            bound = None
        # Empty is always feasible with objective zero, independently of the
        # native incumbent payload. Reject rather than clamp an inverted bound.
        if bound is not None and (bound > 0. or (primal is not None and bound > primal['inner_value'])):
            bound = None
        result = dict(inner_value=None, inner_xcp=None, route_cost=None, tour=None, x=None)
        if primal is not None:
            result.update(primal)
        upper = result['inner_value']
        result.update(outer_lb=bound, raw_bound=_finite(raw.get('lb')),
            exact=bool(status == 0 and upper is not None and bound is not None
                       and upper-bound <= 1e-7 + 1e-12*max(1.,abs(upper),abs(bound))),
            status='OPTIMAL' if status == 0 else 'LIMIT' if status == 2 else 'INVALID',
            source='lrp_native_pctsp_residual' if any(self.objective_shifts.values()) else 'lrp_native_pctsp',
            objective_shifts=dict(self.objective_shifts), native_executed=True, native_diagnostics=raw,
            incumbent_policy_certified=primal is not None, incumbent_rejection_reason=reason,
            wall_seconds=time.perf_counter()-started, context=self.context.route_key(self.facility))
        if top_k > 1:
            # Additional columns are primal witnesses only. Each uses exactly
            # the same original-domain audit as the primary incumbent; no
            # candidate's objective is ever installed as a pricing lower bound.
            candidates, rejected = [], []
            payload = raw.get('candidate_routes', ())
            if not isinstance(payload, (list, tuple)) or len(payload) > top_k:
                payload = ()
                rejected.append('invalid candidate route batch')
            if accepted_status:
                seen = set()
                for candidate in payload:
                    if deadline is not None and time.monotonic() >= deadline:
                        rejected.append('candidate decode deadline'); break
                    try:
                        if not isinstance(candidate, dict):
                            raise ValueError('invalid candidate route payload')
                        decoded = self._decode(candidate, self.active, pi)
                        if deadline is not None and time.monotonic() >= deadline:
                            rejected.append('candidate decode deadline'); break
                        identity = tuple(decoded['inner_xcp'][key] for key in self.keys)
                        if decoded['inner_xcp'][f'u[{self.facility}]'] and identity not in seen:
                            candidates.append(decoded); seen.add(identity)
                    except (ValueError, TypeError, KeyError, OverflowError) as exc:
                        rejected.append(str(exc))
            result['candidate_primals'] = tuple(candidates)
            result['candidate_rejections'] = tuple(rejected)
            if result['outer_lb'] is not None and any(
                    result['outer_lb'] > primal['inner_value'] for primal in candidates):
                result['outer_lb'] = None
                result['exact'] = False
                result['bound_rejection_reason'] = 'additional audited route contradicts native lower bound'
            result['wall_seconds'] = time.perf_counter()-started
        return result

    def _solve_fixed_pctsp(self, alpha, u, *, time_limit=1., label_budget=0, ng_size=8, deadline=None):
        """Fixed TSP via a rigorously forced-all-customer native PCTSP.

        A reward M greater than a known feasible full-tour cost U makes every
        full tour dominate every strict subset, without triangle inequality.
        Thus TSP* = PCTSP* + |assigned| M. The transformed bound is rounded
        downward; an independently audited full tour supplies the upper bound.
        """
        started = time.perf_counter()
        alpha, u = self.context.check_route_state(self.facility, alpha, u)
        customers = tuple(j for j, chosen in enumerate(alpha) if chosen)
        n = len(customers)
        path = [0] + [j+1 for j in customers] + [0] if n else []
        arcs = list(zip(path[:-1],path[1:]))
        known_exact = sum((_fraction(self.context.route_cost[self.facility,v,w]) for v,w in arcs),Fraction())
        known_cost = _directed(known_exact,True)
        diagnostic = {'backend':'analytic' if n <= 1 else 'lrp_native_forced_pctsp',
                      'assigned_customers':n,'native_executed':False}
        bound = _directed(known_exact, False) if n <= 1 else 0.
        if n > 1:
            reward = math.nextafter(known_cost,math.inf)
            if not math.isfinite(reward):
                raise ValueError('cannot represent a finite force-visit reward')
            cost,volume = self._layout(customers)
            raw = self._native(customers,cost,volume,[reward]*n,0.,
                time_limit=time_limit,label_budget=label_budget,ng_size=ng_size,deadline=deadline)
            pi = {f'alpha[{self.facility},{j}]':reward if j in customers else 0.
                  for j in range(self.context.n)}
            pi[f'u[{self.facility}]']=0.
            diagnostic.update(native_executed=True,native_diagnostics=raw,
                              force_visit_reward=reward,known_full_tour_upper=known_cost)
            if raw.get('status') in (0,2):
                try:
                    primal = self._decode(raw,customers,pi)
                    if any(primal['inner_xcp'][f'alpha[{self.facility},{j}]'] != alpha[j]
                           for j in range(self.context.n)):
                        raise ValueError('native time-limited incumbent omits required customers')
                    arcs = primal['tour']['arcs']
                    known_cost = primal['route_cost']
                except ValueError as exc:
                    diagnostic['fallback'] = str(exc)
                native_lb = _finite(raw.get('lb'))
                if raw.get('lb_certified') is True and native_lb is not None:
                    candidate = _directed(_fraction(native_lb)+n*_fraction(reward),False)
                    if candidate <= known_cost:
                        bound=max(0.,candidate)
                    else:
                        diagnostic['rejected_native_bound']='transformed bound exceeds known feasible full tour'
        audit_tour(self.context,self.facility,alpha,u,arcs)
        route={f'r[{self.facility},{v},{w}]':1. for v,w in arcs}
        route['stage_cost']=known_cost
        return {'x':route,'objective':known_cost,'stage_cost':known_cost,'lower_bound':bound,
                'exact':known_cost-bound <= 1e-7 + 1e-12*max(1.,known_cost),
                'diagnostic':{**diagnostic,'wall_seconds':time.perf_counter()-started,
                              'context':self.context.route_key(self.facility)}}


    def _solve_fixed_concorde(self, alpha, u, *, time_limit=1., deadline=None):
        """Integer symmetric TSP with a rigorous bound for original float costs.

        For every tour C_float = C_int/scale + sum(edge rounding errors).
        Sum of per-tail minimum errors bounds the latter for every tour, so a
        proven integer optimum gives a valid original-objective lower bound.
        Exact symmetry is required; no triangle inequality is assumed.
        """
        customers=tuple(j for j,b in enumerate(alpha) if b)
        local=[0]+[j+1 for j in customers]
        q=len(local)
        matrix=self.context.route_cost[self.facility][np.ix_(local,local)]
        if not np.array_equal(matrix,matrix.T):
            return None
        binary=Path(os.environ.get('LRP_CONCORDE_BIN',ROOT/'src-ini/tools/concorde/concorde'))
        if not binary.is_file() or not os.access(binary,os.X_OK):
            return None
        max_edge=float(matrix.max())
        safe_edge=(2**30-1)//q
        # This bundled CLI's small Held-Karp path packs edges into signed
        # 16 bits (larger values return 'edge too long'). The scaling error
        # below remains explicit and valid when this guard reduces precision.
        if q <= 12:
            safe_edge=min(safe_edge,32767)
        scale=min(1_000_000,max(1,int(safe_edge/max_edge))) if max_edge else 1
        integer=np.array([[int(round(_fraction(value)*scale)) for value in row]
                          for row in matrix],dtype=np.int64)
        if np.max(integer)>safe_edge:
            return None
        initial_path=list(range(q))
        best_path=initial_path
        best_exact=sum((_fraction(matrix[initial_path[k],initial_path[(k+1)%q]])
                        for k in range(q)),Fraction())
        # Universal directed minimum-outgoing-edge relaxation, valid even if
        # the CLI times out before proving an optimum or writing a tour.
        basic_lb=sum((min(_fraction(matrix[v,w]) for w in range(q) if w!=v)
                      for v in range(q)),Fraction())
        objective_lb=_directed(basic_lb,False)
        proved=False; reason=None; raw_objective=None; stdout=''; returncode=None
        started=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='lrp_concorde_') as td:
            tsp=Path(td)/'route.tsp'; sol=Path(td)/'route.sol'
            tsp.write_text('NAME: lrp_fixed_route\nTYPE: TSP\nDIMENSION: '+str(q)
                +'\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n'
                +'\n'.join(' '.join(map(str,row)) for row in integer.tolist())+'\nEOF\n')
            time_limit = bounded_solve_time(time_limit, deadline)
            with backend_call('concorde','fixed_tsp',stage=3,facility=self.facility,
                              assigned_customers=len(customers),time_limit=time_limit) as event:
                try:
                    cp=subprocess.run([str(binary.resolve()),'-s','0','-o',str(sol),str(tsp)],
                        cwd=td,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,timeout=float(time_limit),check=False)
                    stdout=cp.stdout;returncode=cp.returncode
                    if returncode != 0:reason=f'concorde_exit_{returncode}'
                except subprocess.TimeoutExpired as exc:
                    reason='concorde_time_limit'
                    stdout=exc.stdout or ''
                    if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
                event['status']=returncode
                event['timed_out']=reason=='concorde_time_limit'
            if sol.is_file():
                tokens=sol.read_text().split()
                try:
                    count=int(tokens[0]);tour=[int(v) for v in tokens[1:]]
                    if count!=q or len(tour)!=q or set(tour)!=set(range(q)):
                        raise ValueError('invalid Concorde permutation')
                    exact=sum((_fraction(matrix[tour[k],tour[(k+1)%q]]) for k in range(q)),Fraction())
                    if exact<best_exact:best_path,best_exact=tour,exact
                    integer_cost=sum(int(integer[tour[k],tour[(k+1)%q]]) for k in range(q))
                    matches=re.findall(r'Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)',stdout)
                    if returncode==0 and matches and Fraction(matches[-1])==integer_cost:
                        raw_objective=integer_cost
                        error_lb=sum((min(_fraction(matrix[v,w])-Fraction(int(integer[v,w]),scale)
                                          for w in range(q) if w!=v) for v in range(q)),Fraction())
                        transformed=_directed(Fraction(integer_cost,scale)+error_lb,False)
                        if transformed<=_directed(best_exact,True):
                            objective_lb=max(objective_lb,transformed);proved=True
                except (ValueError,IndexError) as exc:
                    reason=str(exc)
        # Orient the returned cycle at this one physical root, keeping all
        # original local customer indices when rebuilding its cost and arcs.
        root_position=best_path.index(0)
        best_path=best_path[root_position:]+best_path[:root_position]
        physical=[local[j] for j in best_path]+[0]
        arcs=list(zip(physical[:-1],physical[1:]))
        audit_tour(self.context,self.facility,alpha,u,arcs)
        upper=_directed(best_exact,True)
        route={f'r[{self.facility},{v},{w}]':1. for v,w in arcs};route['stage_cost']=upper
        return {'x':route,'objective':upper,'stage_cost':upper,'lower_bound':objective_lb,
            'exact':upper-objective_lb<=1e-7+1e-12*max(1.,upper),
            'diagnostic':{'backend':'lrp_concorde_fixed_tsp','native_executed':True,
                'concorde_executed':True,'integer_optimality_proven':proved,
                'integer_objective':raw_objective,'integer_scale':scale,
                'objective_rounding_error_bound':upper-objective_lb,
                'reason':reason,'returncode':returncode,'stdout_tail':stdout.splitlines()[-12:],'assigned_customers':len(customers),
                'lower_bound_source':'integer_optimum_plus_directed_rounding_error' if proved else 'minimum_outgoing_edges',
                'wall_seconds':time.perf_counter()-started,'context':self.context.route_key(self.facility)}}

    def solve_fixed(self, alpha, u, *, time_limit=1., label_budget=0, ng_size=8,
                    backend='auto', deadline=None):
        """Use certified Concorde scaling for symmetric fixed tours, PCTSP otherwise.

        ``exact`` concerns the original floating-cost objective; a certified
        integer optimum can leave a small explicit rounding interval.
        """
        if self._is_residual_view:
            raise ValueError('fixed TSP must use the original-cost oracle, not a residual view')
        if backend not in ('auto','pctsp','concorde'):
            raise ValueError('fixed native backend must be auto, pctsp, or concorde')
        alpha,u=self.context.check_route_state(self.facility,alpha,u)
        if not math.isfinite(float(time_limit)) or time_limit<=0:
            raise ValueError('time_limit must be positive and finite')
        if (backend=='concorde' and sum(alpha)>=2) or (backend=='auto' and sum(alpha)>8):
            result=self._solve_fixed_concorde(alpha,u,time_limit=time_limit,deadline=deadline)
            if result is not None:
                return result
            if backend=='concorde':
                raise NativeUnavailable('Concorde unavailable or fixed route matrix not exactly symmetric')
        return self._solve_fixed_pctsp(alpha,u,time_limit=time_limit,
                                       label_budget=label_budget,ng_size=ng_size,deadline=deadline)


def solve_stage3_backward(prob_data,node,pi_value,*,time_limit=1.,oracle=None,**options):
    oracle = oracle or LRPNativeRouteOracle(prob_data,node)
    return oracle.solve(pi_value,time_limit=time_limit,**options)


def solve_stage3_fixed(prob_data,node,x_prev,*,time_limit=1.,oracle=None,**options):
    oracle=oracle or LRPNativeRouteOracle(prob_data,node)
    i=oracle.facility
    alpha=[x_prev[f'alpha[{i},{j}]'] for j in range(oracle.context.n)]
    return oracle.solve_fixed(alpha,x_prev[f'u[{i}]'],time_limit=time_limit,**options)
