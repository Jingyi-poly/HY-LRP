"""Optional physical-forward boundary controller, never a replacement algorithm.

SBC/LevelSet own the optimistic A_L forward/backward. This parent-owned adapter
keeps physical policies A_P separate, installs certified rows before jobs are
created, and shares the caller's absolute clock. A restricted pool has no LB
adapter. All node values remain unweighted until a whole policy is audited.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
from fractions import Fraction as F
import math
from pathlib import Path
import time

from algorithms.base_algorithm import cut_archive_fingerprint
from core.solution import PHYSICAL_FORWARD_DEFAULTS
from core.solve_deadline import SolveDeadlineReached
from core.solver_bounds import minimization_gap_percent, minimization_bounds_inverted
from solvers.forward_policy_certification import certify_policy, certify_stage1_forward_policy
from solvers.forward_ub import round_fraction_up


def _verified_ng_price_proposal(data, node, A, routes, *, deadline, **kwargs):
    """Verify the local build within the proposal's budget, never auto-build."""
    from importlib import import_module
    from solvers.lrp_ng_price_proposal import PriceProposal, propose_ng_price
    if time.monotonic() >= deadline:
        return PriceProposal(None, (), dict(status='NO_BUDGET', proposal_only=True))
    provider_module = import_module('customized-subprob.s2backward.routeopt.verified_pricing')
    pricing, report = provider_module.load_verified_pricing(deadline=deadline)
    if pricing is None:
        return PriceProposal(None, (), dict(status=report['status'],
            proposal_only=True, physical_bound=False, provider=report))
    result = propose_ng_price(data, node, A, routes, deadline=deadline,
                              pricing=pricing, **kwargs)
    result.diagnostics['provider'] = report
    return result


def _mask(root, ctx):
    return tuple(int(round(root[f'A[{i},{ctx.interval}]'])) for i in range(ctx.m))


def _joint_worker(job):
    """An ordinary existing-pool job; no nested pool or live model is sent."""
    from solvers.lrp_parallel import isolated_worker
    from solvers.lrp_joint_recourse_oracle import solve_joint_recourse
    import os
    started, cpu = time.monotonic(), time.process_time()
    with isolated_worker(job['environment']) as env:
        certificate = solve_joint_recourse(job['data'], job['tree'], job['node'],
            job['mask'], job['archive'], env=env, **job['options'])
    return {'certificate': certificate, 'pid': os.getpid(),
            'wall_seconds': time.monotonic()-started,
            'cpu_seconds': time.process_time()-cpu}


def reprice_same_trial(algorithm, trial, costs, affected_nodes):
    """Reprice Qhat at unchanged A/alpha/e/u; discard stale S2 upper supports.

    The ordinary canonical score includes every basic and learned route row.
    This does not optimize a new assignment, certify Qhat optimality, change a
    physical route, or raise a scalar global LB. True TSP witnesses stay valid.
    """
    from cuts.lrp_static_bounds import basic_route_cuts
    from models.stage_builder import _as_cut, _route_pools
    from solvers.forward_stage2_bpc import _score
    from cuts.lrp_physical_bridge import physical_archive_value
    affected = set(affected_nodes)
    data, tree, archive = algorithm.prob_data, algorithm.scen_tree, algorithm.cut_lag
    root = trial[1][0]
    for q in sorted(affected):
        node, ctx = tree[2][q], tree[2][q].context
        pools, mapping = _route_pools(ctx, node, archive)
        cuts = {i: [*basic_route_cuts(ctx, i),
                    *(_as_cut(row, ctx, i) for row in pools[i])] for i in range(ctx.m)}
        scored = _score(data, node, ctx, root, trial[2][q], cuts, mapping,
                        {'source': 'physical_cut_same_assignment_reprice',
                         'fresh_solve': False})
        trial[2][q], costs[2][q] = scored['x'], scored['objective']
        algorithm.cut_Dict.setdefault(2, {})[q] = []
    manager = algorithm.backward_solver.cut_manager
    cache = getattr(manager, '_fixed_target_cache', None)
    contexts = {tree[2][q].context.key for q in affected}
    if cache is not None:
        for key in list(cache):
            if isinstance(key, tuple) and len(key)>1 and key[0]==2 and key[1] in contexts:
                del cache[key]
    if affected:
        algorithm.forward_solver.set_stage2_handoff(None)
        algorithm.backward_solver.last_stage2_handoff = None
        algorithm._initial_forward = None
    # Same-plan eta reference includes the original mandatory facility floors.
    _, facility = certify_stage1_forward_policy(data, root)
    total = F(facility)
    for q in tree[1][0].successor:
        node = tree[2][q]
        eta = physical_archive_value(data, node, archive, _mask(root, node.context))
        root[f'eta[{q}]'] = round_fraction_up(eta)
        total += F(float(data.arrays['scenario_prob'][node.context.scenario])) * F(root[f'eta[{q}]'])
    costs[1][0] = round_fraction_up(total)
    algorithm.forward_solver._last_eta_per_omega = {
        q: root[f'eta[{q}]'] for q in tree[1][0].successor}


class _CGProfileDefault:
    def __repr__(self):
        return 'CG_PROFILE_DEFAULT'


_CG_PROFILE_DEFAULT = _CGProfileDefault()


class PhysicalForwardRuntime:
    """One run's state; profiles derive features without independent booleans."""
    def __init__(self, prob_data, tree, config, *, deadline=None, event_sink=None,
                 progress_sink=None, diagnostic_dir=None, started_at=None,
                 cg_closed_pricing_call_cap=_CG_PROFILE_DEFAULT, cg_coverage_price_seed=_CG_PROFILE_DEFAULT,
                 collect_backward_routes=True,cg_persistent_prices=_CG_PROFILE_DEFAULT):
        cg_profile=config.get('physical_cg_profile','off')
        # The existing cg profile selects the verified persistent-price path.
        # Omitted arguments follow the profile; explicit False/None retain
        # their disabling/legacy meanings. No new public config key is needed.
        if cg_persistent_prices is _CG_PROFILE_DEFAULT:
            cg_persistent_prices = cg_profile == 'cg'
        if cg_closed_pricing_call_cap is _CG_PROFILE_DEFAULT:
            cg_closed_pricing_call_cap = 0. if cg_profile == 'cg' and cg_persistent_prices else None
        if cg_coverage_price_seed is _CG_PROFILE_DEFAULT:
            # Seed from this run's current S3 cuts, never a saved policy/cut file.
            cg_coverage_price_seed = cg_profile == 'cg' and cg_persistent_prices
        self.profile = cg_profile if cg_profile!='off' else config.get('physical_forward_profile', 'off')
        if self.profile not in {'off', 'baseline', 'paced', 'pool', 'joint', 'cg'}:
            raise ValueError('unknown physical forward profile')
        self.options = dict(PHYSICAL_FORWARD_DEFAULTS)
        self.options.update(config.get('physical_forward_options', {}) or {})
        self._v2_pool_budget = cg_profile != 'off'
        self._pool_audit_reserve = min(self.options['audit_reserve_s'],
                                      self.options['pool_time_limit_s'])
        if cg_profile!='off':
            cg=dict(config.get('physical_cg_options',{}) or {})
            self.options.update(epoch_budget_s=cg.get('epoch_wall',60.),
                pool_time_limit_s=cg.get('global_policy_pool_cap',10.),
                joint_time_limit_s=cg.get('joint_node_cap',30.),
                joint_max_nodes_per_epoch=cg.get('max_joint_nodes',2),
                audit_reserve_s=cg.get('commit_audit_refresh_reserve',10.),
                backward_epoch_budget_s=cg.get('paced_backward_wall',60.))
            # The epoch's commit/audit/root reserve is separate from the
            # restricted pool's own cap. Reusing the full default 10s here
            # consumed the entire 10s pool allowance before model building.
            self._pool_audit_reserve = min(2., .2*self.options['pool_time_limit_s'])
            self.cg_options=cg
        else:
            self.cg_options={}
        if cg_closed_pricing_call_cap is not None:
            if isinstance(cg_closed_pricing_call_cap,bool) or not math.isfinite(float(cg_closed_pricing_call_cap)) or float(cg_closed_pricing_call_cap)<0:
                raise ValueError('closed pricing cap must be finite/nonnegative or None')
            cg_closed_pricing_call_cap=float(cg_closed_pricing_call_cap)
        self.cg_closed_pricing_call_cap=cg_closed_pricing_call_cap
        if any(not isinstance(flag,bool) for flag in
               (cg_coverage_price_seed,collect_backward_routes,cg_persistent_prices)):
            raise TypeError('CG seed and route collection switches must be boolean')
        self.cg_coverage_price_seed=cg_coverage_price_seed
        self.collect_backward_routes=collect_backward_routes
        self.cg_persistent_prices=cg_persistent_prices
        # Pure data only: no Gurobi model/basis crosses an epoch or checkpoint.
        self._cg_price_states={}
        self.data, self.tree, self.config = prob_data, tree, config
        self.deadline = deadline
        if deadline is not None and not math.isfinite(float(deadline)):
            raise ValueError('external experiment deadline must be finite or None')
        self.started = time.monotonic() if started_at is None else float(started_at)
        self.event_sink, self.progress_sink = event_sink, progress_sink
        self.diagnostic_dir = None if diagnostic_dir is None else Path(diagnostic_dir)
        self.envelope_version, self._archive_fingerprint = 0, None
        self.pool = None
        self.node_cache, self.last_probe, self.route_lp_cache = {}, {}, {}
        self._probe_clock = 0
        self._route_hashes = set()
        self._last_pool_version = None
        self._singletons_done = False
        self._current_iteration = 0
        self.last_A_L = self.last_A_P = None
        self.last_escape = None
        self.last_progress = None
        self.known_lb = self.known_ub = None
        self.known_policy = None
        self.blocked = False
        self._active_epoch_started = None
        self.stats = dict(physical_pool_wall=0., joint_wall=0., route_lp_wall=0.,
            cg_wall=0., pricing_wall=0., lp_certified_node_count=0,
            pricing_unresolved_count=0,
            audit_wall=0., physical_epoch_wall=0., forward_wall=0., backward_wall=0.,
            eta_generated=0, eta_installed=0, theta_generated=0, theta_installed=0,
            new_audited_routes=0, joint_calls=0, route_lp_calls=0, pool_calls=0,
            completed_iterations=0, joint_worker_cpu=0.)
        if self.pool_enabled:
            from solvers.lrp_physical_policy_pool import PhysicalRoutePool
            self.pool = PhysicalRoutePool(prob_data, tree,
                max_routes_per_node_facility=self.options['pool_max_routes_per_node_facility'])

    @property
    def pool_enabled(self):
        return self.profile in {'pool', 'joint', 'cg'}

    @property
    def joint_enabled(self):
        return self.profile == 'joint'

    @property
    def cg_enabled(self):
        return self.profile == 'cg'

    def clip_deadline(self, existing=None):
        values = [float(v) for v in (self.deadline, existing) if v is not None]
        return min(values) if values else None

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def backward_deadline(self, existing=None):
        result = self.clip_deadline(existing)
        if self.profile in {'paced', 'pool', 'joint', 'cg'}:
            local = time.monotonic()+self.options['backward_epoch_budget_s']
            result = local if result is None else min(result, local)
        return result

    def _emit(self, event, **payload):
        record = dict(event=event, profile=self.profile,
            monotonic=time.monotonic(), elapsed=time.monotonic()-self.started, **payload)
        if self.event_sink is not None:
            self.event_sink(record)
        return record

    def _sync_archive(self, archive):
        fingerprint = cut_archive_fingerprint(archive)
        if self._archive_fingerprint is not None and fingerprint != self._archive_fingerprint:
            self.envelope_version += 1
        self._archive_fingerprint = fingerprint

    def _next_probe_tick(self, iteration):
        """Advance internal scheduling time even when no node needs a probe.

        Public iteration numbering may restart on resume. Older checkpoints
        carry only last_probe; newer ones also preserve epochs without visits.
        """
        self._probe_clock = max(int(iteration),
            int(getattr(self, '_probe_clock', 0))+1,
            max(self.last_probe.values(), default=0)+1)
        return self._probe_clock

    def _failure(self, error, **evidence):
        self.blocked = True
        if self.diagnostic_dir is not None:
            from models.stage_builder import _instance
            import json
            import pickle
            directory = self.diagnostic_dir/'physical_validity_failure'
            _instance(self.data).save(directory)
            packet = {'error':repr(error),'evidence':deepcopy(evidence)}
            with (directory/'failure.pkl').open('wb') as stream:
                pickle.dump(packet,stream,protocol=5)
            with (directory/'cut_archive.pkl').open('wb') as stream:
                pickle.dump(packet['evidence'].get('archive',{}),stream,protocol=5)
            details = packet['evidence'].get('details')
            specification = details.get('model_spec') if isinstance(details,dict) else None
            if specification is not None:
                # Pure canonical export: never retain/serialize a Gurobi object,
                # and never construct or solve another model on this path.
                specification.save_matrix(directory/'matrix.npz')
                specification.write_lp(directory/'model.lp')
            (directory/'failure.json').write_text(json.dumps(
                packet,default=repr,indent=2)+'\n')
        # Persist the full reproducible packet first.  Event consumers receive
        # compact JSON fields, never an arbitrary matrix/instance object.
        self._emit('PHYSICAL_VALIDITY_FAILURE',error=repr(error),
            iteration=evidence.get('iteration'),A_L=evidence.get('A_L'),
            diagnostic_dir=None if self.diagnostic_dir is None else
                str(self.diagnostic_dir/'physical_validity_failure'))

    def record_root_certificate(self, bound, *, root=None):
        """Record a real root bound even before a long forward finishes."""
        if self.profile == 'off' or self.expired() or bound is None:
            return
        bound = float(bound)
        if not math.isfinite(bound) or abs(bound)>=1e99:
            return
        if self.known_ub is not None and bound > self.known_ub:
            raise ValueError('root lower exceeds independently audited global upper')
        self.known_lb = bound if self.known_lb is None else max(self.known_lb, bound)
        self._publish(phase=None, iteration=self._current_iteration, boundary='root_certificate')

    def _publish(self, *, phase, iteration, boundary, algorithm=None, **timings):
        if self.expired():
            return None
        progress_stats = dict(self.stats)
        if self._active_epoch_started is not None:
            progress_stats['physical_epoch_wall'] += time.monotonic()-self._active_epoch_started
        record = dict(profile=self.profile, phase=phase, iteration=iteration,
            boundary=boundary, certified_at=time.monotonic(), elapsed=time.monotonic()-self.started,
            LB=self.known_lb, audited_UB=self.known_ub,
            gap_lb_denominator=(None if self.known_lb is None or self.known_ub is None else
                                minimization_gap_percent(self.known_lb, self.known_ub)),
            envelope_version=self.envelope_version, stats=progress_stats, **timings)
        self.last_progress = record
        if self.progress_sink is not None:
            self.progress_sink(deepcopy(record), policy=deepcopy(self.known_policy),
                cut_archive=None if algorithm is None else deepcopy(algorithm.cut_lag))
        return record

    def observe(self, algorithm, *, phase, iteration, boundary, **timings):
        """Only certificates whose independent audit completes before the clock."""
        for metric in ('forward_wall', 'backward_wall'):
            if metric in timings:
                self.stats[metric] += float(timings[metric])
        if self.profile == 'off' or self.expired():
            return None
        self._current_iteration = iteration
        for diagnostic in getattr(algorithm.forward_solver, 'last_forward_diagnostics', ()):
            bound = diagnostic.get('certified_lower_bound')
            if diagnostic.get('stage')==1 and bound is not None and math.isfinite(float(bound)):
                if self.known_lb is None or bound > self.known_lb:
                    self.known_lb = float(bound)
        policy = getattr(algorithm, 'x_best', None)
        if policy is not None:
            before = time.monotonic()
            audit = certify_policy(self.data, self.tree, policy)
            self.stats['audit_wall'] += time.monotonic()-before
            if self.expired():
                self._emit('late_policy_audit', boundary=boundary, accepted=False)
                return None
            upper = float(audit['feasible_upper_bound'])
            if self.known_lb is not None and self.known_lb > upper:
                raise ValueError('certified root lower exceeds audited physical policy')
            if self.known_ub is None or upper < self.known_ub:
                self.known_ub, self.known_policy = upper, deepcopy(policy)
            self._emit('policy_audit', phase=phase, iteration=iteration,
                upper=upper, all_original_nodes_audited=True, boundary=boundary)
        if timings.get('completed_backward'):
            self.stats['completed_iterations'] += 1
        return self._publish(phase=phase, iteration=iteration, boundary=boundary,
                             algorithm=algorithm, **timings)

    def _within_deadline(self, deadline=None):
        stop = self.clip_deadline(deadline)
        return stop is None or time.monotonic()<stop

    def _staged_pool(self):
        """Isolate pending pool writes while sharing immutable physical inputs.

        Pool mutators replace snapshot entries, not their existing contents;
        route records are frozen dataclasses.  Their dictionaries and pin sets
        are copied so a late original-array audit cannot leak state changes.
        """
        pending = copy(self.pool)
        pending._routes = dict(self.pool._routes)
        pending._pins = {key:set(values) for key,values in self.pool._pins.items()}
        pending._snapshots = dict(self.pool._snapshots)
        pending.diagnostics = dict(self.pool.diagnostics)
        return pending

    def _record_new_routes(self):
        if self.pool is None:
            return
        hashes = {r.audit_signature for r in self.pool.routes()}
        self.stats['new_audited_routes'] += len(hashes-self._route_hashes)
        self._route_hashes.update(hashes)

    def _harvest_backward_routes(self, algorithm, *, deadline, iteration=0):
        """Move only completed, scoped physical witnesses into the route pool.

        This changes columns only, not the oracle trial, approximation values,
        cut archive, or any bound. Unconsumed witnesses survive the deadline.
        """
        solver=getattr(algorithm,'backward_solver',None)
        configure=getattr(solver,'configure_route_witness_collection',None)
        if configure is None:
            return
        enabled=self.pool_enabled and self.collect_backward_routes
        configure(enabled=enabled)
        if not enabled or not self._within_deadline(deadline):
            return
        count=getattr(solver,'route_witness_count',0)
        if not count:
            return
        from solvers.lrp_backward_route_witnesses import validate_route_witness
        begun=time.monotonic();accepted=0
        pending=self._staged_pool()
        old_routes=pending.routes()
        old_signatures={r.audit_signature for r in old_routes}
        # A resumed run's iteration may restart at 1 while retained columns
        # have larger generations. Fresh evidence must not be pruned as old
        # merely because its native transport record used generation zero.
        generation=max(int(iteration),max((r.generation_id for r in old_routes),default=-1)+1)
        consumed=[]
        for witness in solver.peek_route_witnesses(max_routes=512,deadline=deadline):
            if not self._within_deadline(deadline):
                break
            child=self.tree[3][witness.stage3_node_index]
            route=validate_route_witness(self.data,child,witness).route
            q=child.predecessor
            pending.add_audited_route(self.tree[2][q],replace(route,generation_id=generation))
            if not self._within_deadline(deadline):
                break
            accepted+=1;consumed.append(witness)
        # Keep the entire collection transaction outside the bound/UB path.
        # A batch finishing too late is retried, not backdated to this epoch.
        self.stats['audit_wall']+=time.monotonic()-begun
        if not self._within_deadline(deadline):
            self._emit('backward_route_harvest',accepted=0,new_columns=0,
                       pending=count,status='DEFERRED_DEADLINE')
            return
        self.pool=pending
        solver.discard_route_witnesses(consumed)
        self._record_new_routes()
        new=len({r.audit_signature for r in pending.routes()}-old_signatures)
        self._emit('backward_route_harvest',accepted=accepted,new_columns=new,
                   pending=solver.route_witness_count,status='AUDITED',
                   elapsed_wall=time.monotonic()-begun)

    def _add_pool_policy(self, policy, *, source, iteration=0, pin=None, deadline=None):
        if not self._within_deadline(deadline):
            return None
        pending = self._staged_pool()
        before = time.monotonic()
        result = pending.add_policy(policy,source=source,generation_id=max(0,int(iteration)),pin=pin)
        self.stats['audit_wall'] += time.monotonic()-before
        if not self._within_deadline(deadline):
            self._emit('late_policy_audit',boundary='pool_collection',accepted=False)
            return None
        self.pool = pending
        self._record_new_routes()
        return result

    def _accept_policy_before_deadline(self, algorithm, policy, *, deadline=None, boundary):
        """The original _accept_policy contract, with an audit-before-commit gate."""
        if not self._within_deadline(deadline):
            return None
        before = time.monotonic()
        candidate = deepcopy(policy)
        audit = certify_policy(self.data,self.tree,candidate)
        self.stats['audit_wall'] += time.monotonic()-before
        if not self._within_deadline(deadline):
            self._emit('late_policy_audit',boundary=boundary,accepted=False)
            return None
        upper = float(audit['feasible_upper_bound'])
        lower = algorithm._best_lower_bound()
        if lower is not None and minimization_bounds_inverted(lower,upper):
            raise RuntimeError(f'Phase2 certified LB {lower} exceeds audited policy UB {upper}')
        if upper<algorithm.best_ub:
            algorithm.best_ub,algorithm.x_best = upper,candidate
        return upper

    def _merge_node(self, certificate, node, mask, *, deadline=None):
        from solvers.lrp_physical_types import validate_node_certificate
        from solvers.lrp_physical_policy_pool import audit_node_policy
        if not self._within_deadline(deadline):
            return None
        before = time.monotonic()
        validate_node_certificate(certificate, self.data, node, mask)
        if certificate.policy is not None:
            _, audited_upper, _ = audit_node_policy(self.data, self.tree, node, mask, certificate.policy)
            if certificate.q_upper != audited_upper:
                raise ValueError('node upper does not equal original-array audited cost')
        key = (certificate.node_signature, tuple(mask))
        old = self.node_cache.get(key)
        lower_record = certificate
        if old is not None and old.q_lower is not None and (
                certificate.q_lower is None or old.q_lower > certificate.q_lower):
            lower_record = old
        upper_record = certificate
        if old is not None and old.q_upper is not None and (
                certificate.q_upper is None or old.q_upper < certificate.q_upper):
            upper_record = old
        low, high = lower_record.q_lower, upper_record.q_upper
        tol = 2e-6+1e-10*max(1., abs(low or 0.), abs(high or 0.))
        if low is not None and high is not None and low > high+tol:
            raise ValueError('merged physical node lower exceeds audited upper')
        merged = replace(lower_record, q_upper=high, policy=deepcopy(upper_record.policy),
            upper_source=upper_record.upper_source, audited_routes=upper_record.audited_routes,
            closed_within_tolerance=low is not None and high is not None and high-low<=tol)
        pending = self.pool
        if self.pool is not None:
            pending = self._staged_pool()
            owner = 'node_best:'+repr(key)
            pairs = [(node.index, route) for route in merged.audited_routes]
            pending.replace_route_pins(owner,pairs)
        self.stats['audit_wall'] += time.monotonic()-before
        if not self._within_deadline(deadline):
            self._emit('late_node_audit',node_id=certificate.node_id,A_mask=mask,accepted=False)
            return None
        self.node_cache[key] = merged
        self.pool = pending
        self._record_new_routes()
        return merged

    def collect(self, policy, *, source, iteration=0, phase=None, deadline=None):
        """Collect audited entries only; a partial deadline never implies all nodes exist."""
        if not self.pool_enabled or not self._within_deadline(deadline) or self.blocked:
            return None
        from solvers.lrp_physical_policy_pool import audit_node_policy,audit_route
        from solvers.lrp_physical_types import NodePhysicalCertificate
        pin = 'current_forward' if 'forward' in source else None
        result = self._add_pool_policy(policy,source=source,iteration=iteration,pin=pin,deadline=deadline)
        if result is None:
            return None
        normalized,upper,audit = result
        for q in self.tree[1][0].successor:
            if not self._within_deadline(deadline):
                break
            before = time.monotonic()
            node,ctx = self.tree[2][q],self.tree[2][q].context
            mask = _mask(normalized[1][0],ctx)
            part,price,node_audit = audit_node_policy(self.data,self.tree,node,mask,normalized)
            routes = tuple(audit_route(self.data,node,tour['facility'],
                [v-1 for v in tour['local_route'][1:-1]],source=source,
                generation_id=max(0,int(iteration))) for tour in node_audit['tours'] if tour['customers'])
            cert = NodePhysicalCertificate((ctx.period,ctx.scenario),self.pool.signatures[q],
                mask,0.,price,part,'nonnegative_cost_floor','audited_'+source,
                'AUDITED_POLICY',price==0.,True,'existing_s2',self.envelope_version,
                0.,0.,0.,0.,audited_routes=routes)
            self.stats['audit_wall'] += time.monotonic()-before
            if self._merge_node(cert,node,mask,deadline=deadline) is None:
                break
        return normalized,upper,audit

    def ingest_stage2_forward_certificates(self, algorithm, trial, *, phase, iteration, deadline=None):
        """Consume only the just-returned original fixed-A lower certificate.

        The forward hook is the provenance boundary: its last policy audit
        binds every node context and A mask; forward_pass reset the diagnostic
        list before solving.  A restored snapshot is conservatively skipped.
        Feasible objective/target values never substitute for a missing bound.
        """
        from models.stage_model_core import finite_number
        from solvers.lrp_physical_types import NodePhysicalCertificate,node_signature
        report = dict(accepted=0,skipped=0,reasons=[])
        if not self.pool_enabled or not self._within_deadline(deadline):
            report['reasons'].append('NO_BUDGET_OR_DISABLED')
            return report
        reused = phase==2 and iteration==1 and getattr(algorithm,'_forward_reuse',{}).get('used',False)
        policy_audit = getattr(algorithm.forward_solver,'last_policy_certificate',None)
        scoped = (isinstance(policy_audit,dict)
                  and policy_audit.get('all_original_nodes_audited') is True
                  and isinstance(policy_audit.get('nodes'),dict))
        for diagnostic in getattr(algorithm.forward_solver,'last_forward_diagnostics',()):
            if diagnostic.get('stage')!=2:
                continue
            q = diagnostic.get('node')
            reason = None
            if type(q) is not int or q not in self.tree[1][0].successor:
                reason = 'UNKNOWN_NODE'
            elif reused:
                reason = 'REUSED_FORWARD_SNAPSHOT'
            elif not scoped:
                reason = 'MISSING_FORWARD_SCOPE_AUDIT'
            if reason is not None:
                report['skipped'] += 1; report['reasons'].append(reason)
                self._emit('original_s2_diagnostic',phase=phase,iteration=iteration,
                    diagnostic=deepcopy(diagnostic),accepted=False,reason=reason)
                continue
            node,ctx = self.tree[2][q],self.tree[2][q].context
            mask = _mask(trial[1][0],ctx)
            record = policy_audit['nodes'].get(f'{ctx.period},{ctx.scenario}',{})
            bound = diagnostic.get('certified_lower_bound')
            if (record.get('context')!=ctx.key
                    or tuple(record.get('availability',()))!=mask):
                reason = 'FORWARD_SCOPE_MISMATCH'
            elif diagnostic.get('direction','forward')!='forward' or diagnostic.get(
                    'objective_space','node_underestimator')!='node_underestimator':
                reason = 'NOT_FIXED_A_UNDERESTIMATOR'
            elif not finite_number(bound):
                reason = 'NO_CERTIFIED_LOWER'
            elif not self._within_deadline(deadline):
                reason = 'DEADLINE'
            if reason is None:
                cert = NodePhysicalCertificate((ctx.period,ctx.scenario),node_signature(self.data,node),
                    mask,float(bound),None,None,'existing_s2_bound',None,'ORIGINAL_FIXED_A_BOUND',
                    False,True,'existing_s2',self.envelope_version,0.,0.,0.,0.,
                    diagnostics={'source':'original_forward_fixed_A',
                                 'original_diagnostic':deepcopy(diagnostic)})
                accepted = self._merge_node(cert,node,mask,deadline=deadline) is not None
                if not accepted:
                    reason = 'DEADLINE'
            else:
                accepted = False
            report['accepted' if accepted else 'skipped'] += 1
            if reason is not None:
                report['reasons'].append(reason)
            self._emit('original_s2_diagnostic',phase=phase,iteration=iteration,
                node_id=(ctx.period,ctx.scenario),A_mask=mask,node_signature=node_signature(self.data,node),
                diagnostic=deepcopy(diagnostic),accepted=accepted,reason=reason)
        return report

    def after_forward(self, algorithm, *, phase, iteration, trial, costs, forward_wall=0.):
        if self.profile=='off':
            return
        self._sync_archive(algorithm.cut_lag)
        self._current_iteration = iteration
        self.observe(algorithm, phase=phase, iteration=iteration, boundary='forward',
                     forward_wall=forward_wall)
        # Identical original S2 diagnostics for every enabled profile.  These
        # raw records are metrics only; typed ingestion below has its own gates.
        for diagnostic in getattr(algorithm.forward_solver,'last_forward_diagnostics',()):
            if diagnostic.get('stage')==2:
                self._emit('original_s2_forward_record',phase=phase,iteration=iteration,
                           diagnostic=deepcopy(diagnostic),certificate_claimed=False)
        if self.pool_enabled and not self.expired():
            self.collect(trial, source=f'phase{phase}_forward', iteration=iteration, phase=phase)
            self.ingest_stage2_forward_certificates(algorithm,trial,phase=phase,iteration=iteration)
            if algorithm.x_best is not None:
                self._add_pool_policy(algorithm.x_best,source='global_incumbent',
                                      iteration=iteration,pin='global_incumbent')

    def _install(self, node, archive, cut, states, affected, *, deadline=None):
        from cuts.lrp_physical_bridge import install_physical_cut
        if not self._within_deadline(deadline):
            return {'installed':False,'reason':'DEADLINE'}
        pending = dict(archive)
        stage = 2 if cut.level=='node' else 3
        pending[stage] = dict(archive.get(stage,{}))
        if stage==2:
            target = node.index
        else:
            from models.stage_builder import _route_pools
            _,mapping = _route_pools(node.context,node,archive)
            target = mapping[cut.certificate['facility']]
        pending[stage][target] = deepcopy(archive.get(stage,{}).get(target,[]))
        report = install_physical_cut(self.data, node, pending, cut,
            envelope_version=self.envelope_version,
            expected_version=cut.certificate['envelope_version'], candidate_states=states,
            sep_atol=self.options['sep_atol'], sep_rtol=self.options['sep_rtol'])
        if not self._within_deadline(deadline):
            self._emit('late_cut_audit',cut=cut.to_dict(),accepted=False)
            return {'installed':False,'reason':'DEADLINE'}
        if report['installed']:
            archive.setdefault(stage,{})[target] = pending[stage][target]
        self.envelope_version = report['version_after']
        kind = 'eta' if cut.level=='node' else 'theta'
        self.stats[kind+'_installed'] += int(report['installed'])
        affected.update(report['affected_s2_nodes'])
        self._emit('cut_install', cut=cut.to_dict(), **report)
        return report

    def _eta(self, node, archive, certificate, mask, affected, *, deadline=None):
        from cuts.lrp_physical_bridge import build_eta_state_cut
        cut = build_eta_state_cut(self.data, node, certificate)
        if cut is None:
            return
        self.stats['eta_generated'] += 1
        return self._install(node, archive, cut, [mask], affected,deadline=deadline)

    def _compose(self, algorithm, trial, iteration, *, deadline=None):
        """All cached improvements are composed under this entire fixed A_L."""
        policy = deepcopy(trial)
        for q in self.tree[1][0].successor:
            node = self.tree[2][q]
            key = (self.pool.signatures[q], _mask(trial[1][0], node.context))
            cert = self.node_cache.get(key)
            if cert is not None and cert.policy is not None:
                policy[2].update(deepcopy(cert.policy[2]))
                policy[3].update(deepcopy(cert.policy[3]))
        upper = self._accept_policy_before_deadline(algorithm,policy,deadline=deadline,
                                                     boundary='joint_composition')
        if upper is None:
            return None
        self._add_pool_policy(policy,source='joint_composition',iteration=iteration,deadline=deadline)
        if algorithm.x_best is not None:
            self._add_pool_policy(algorithm.x_best,source='global_incumbent',iteration=iteration,
                                  pin='global_incumbent',deadline=deadline)
        return policy

    def _run_cg_nodes(self, algorithm, trial, archive, affected, *, iteration,
                      work_stop, envelope_version, probe_tick=None):
        """Run at most two sequential V2 route-CG nodes under one shared cap."""
        from cuts.lrp_physical_bridge import physical_archive_value
        from cuts.lrp_physical_price_certificates import install_price_cut
        from solvers.lrp_joint_route_cg import JointRouteCG
        from solvers.lrp_physical_types import NodePhysicalCertificate
        if probe_tick is None:
            probe_tick = self._next_probe_tick(iteration)
        shared=min(work_stop,time.monotonic()+float(self.cg_options.get('joint_shared_cap',40.)))
        ranked=[]
        for q in self.tree[1][0].successor:
            node,ctx=self.tree[2][q],self.tree[2][q].context
            mask=_mask(trial[1][0],ctx)
            eta=float(physical_archive_value(self.data,node,archive,mask))
            cached=self.node_cache.get((self.pool.signatures[q],mask))
            upper=(cached.q_upper if cached is not None and cached.q_upper is not None else
                   sum(float(ctx.outsourcing[j]) for j in range(ctx.n) if ctx.active[j]))
            unprobed=(self.pool.signatures[q],mask) not in self.last_probe
            last=self.last_probe.get((self.pool.signatures[q],mask),-10**9)
            score=float(self.data.arrays['scenario_prob'][ctx.scenario])*max(0.,upper-eta)
            ranked.append((not unprobed,last,-score,ctx.period,ctx.scenario,q,mask,eta))
        ranked.sort()
        maximum=min(2,int(self.cg_options.get('max_joint_nodes',2)))
        proposal_options={}
        # Existing public profiles and constructor stay unchanged. Verification
        # occurs inside the already-budgeted proposal call; no new option.
        if self.cg_persistent_prices and self.cg_closed_pricing_call_cap==0.:
            proposal_options['price_proposal']=_verified_ng_price_proposal
        owner=JointRouteCG(self.data,self.tree,self.pool,**proposal_options)
        # Resume can restart the outer iteration counter. New physical
        # columns must remain newer than the inherited pool's generations.
        generation=max(int(iteration),max((r.generation_id for r in self.pool.routes()),default=-1)+1)
        results=[]
        for *_,q,mask,eta_before in ranked[:maximum]:
            if time.monotonic()>=shared: break
            node,ctx=self.tree[2][q],self.tree[2][q].context
            key=(self.pool.signatures[q],mask);self.last_probe[key]=probe_tick
            node_deadline=min(shared,time.monotonic()+float(self.cg_options.get('joint_node_cap',30.)))
            price_options={}
            if self.cg_persistent_prices:
                from solvers.lrp_cg_price_state import PhysicalCGPriceState
                if key not in self._cg_price_states:
                    # Only price proposals/evidence are evicted. Installed
                    # cuts and accepted policies have independent ownership.
                    if len(self._cg_price_states)>=128:
                        self._cg_price_states.pop(next(iter(self._cg_price_states)))
                    self._cg_price_states[key]=PhysicalCGPriceState(self.data,node,mask)
                price_options=dict(price_state=self._cg_price_states[key],stabilize_prices=True,
                    pricing_top_k=8,capacity_price_bound=True,pricing_ng_size=32)
            result=owner.evaluate(node,mask,deadline=node_deadline,
                envelope_version=envelope_version,
                existing_node_lower=eta_before,
                max_rounds=int(self.cg_options.get('cg_max_rounds',50)),
                pricing_call_cap=float(self.cg_options.get('pricing_call_cap',3.)),
                generation_id=generation,
                closed_pricing_call_cap=self.cg_closed_pricing_call_cap,
                **({'coverage_archive':archive} if self.cg_coverage_price_seed else {}),
                **price_options)
            results.append(result)
            self._emit('physical_cg_node',node_id=result.node_id,A_mask=result.facility_mask,
                status=result.status,route_lp_lower=result.route_lp_lower,
                rmp_lp_upper=result.rmp_lp_upper,audited_node_upper=result.audited_node_upper,
                combined_node_lower=result.combined_node_lower,timings=result.timings,
                closed_pricing_call_cap=self.cg_closed_pricing_call_cap,
                round_trace=result.round_trace)
            self.stats['cg_wall']+=result.timings.get('cg',0.)
            self.stats['pricing_wall']+=result.timings.get('pricing',0.)
            self.stats['lp_certified_node_count']+=int(result.status=='LP_CERTIFIED')
            self.stats['pricing_unresolved_count']+=int(result.status=='PRICING_UNRESOLVED')
            for certificate in result.eta_certificates:
                report=install_price_cut(self.data,node,archive,certificate.eta_cut,
                    envelope_version=self.envelope_version,expected_version=envelope_version,
                    candidate_states=[mask])
                if report['installed']:
                    self.envelope_version=report['version_after'];self.stats['eta_installed']+=1
                self.stats['eta_generated']+=1
                for cut in certificate.theta_cuts:
                    i=int(cut.certificate['facility'])
                    state=tuple(int(trial[2][q][f'alpha[{i},{j}]']) for j in range(ctx.n))+(int(trial[2][q][f'u[{i}]']),)
                    report=install_price_cut(self.data,node,archive,cut,
                        envelope_version=self.envelope_version,expected_version=envelope_version,
                        candidate_states=[state])
                    if report['installed']:
                        self.envelope_version=report['version_after'];self.stats['theta_installed']+=1
                        affected.add(q)
                    self.stats['theta_generated']+=1
            if result.audited_node_upper is not None and result.audited_node_policy is not None:
                cert=NodePhysicalCertificate((ctx.period,ctx.scenario),self.pool.signatures[q],mask,
                    result.route_lp_lower,result.audited_node_upper,result.audited_node_policy,
                    'route_lp_price_bound' if result.route_lp_lower is not None else None,
                    'restricted_route_pool_audit',result.status,
                    result.route_lp_lower is not None and
                    result.audited_node_upper-result.route_lp_lower<=1e-6,
                    True,'route_cg',self.envelope_version,result.timings.get('total',0.),0.,
                    result.timings.get('cg',0.),0.,diagnostics={'round_trace':result.round_trace})
                self._merge_node(cert,node,mask,deadline=shared)
            print(f'  [PhysicalCG] q={result.node_id} A={result.facility_mask} '
                  f'Lroute={result.route_lp_lower} RMP={result.rmp_lp_upper} '
                  f'U={result.audited_node_upper} rounds={len(result.round_trace)} '
                  f'new={len(result.added_route_ids)} status={result.status}',flush=True)
        return results

    def _weighted_eta_envelope(self, archive, root):
        """Evaluate the operating eta envelope at one complete facility plan."""
        from cuts.lrp_physical_bridge import physical_archive_value
        total=F()
        probabilities=self.data.arrays['scenario_prob']
        for q in self.tree[1][0].successor:
            node=self.tree[2][q]
            total += (F(float(probabilities[node.context.scenario])) *
                      F(float(physical_archive_value(self.data,node,archive,
                                                    _mask(root,node.context)))))
        return float(total)

    def _refresh_root_after_cg(self, algorithm, trial, archive_before, *, deadline,
                               iteration, eta_installed):
        """At most one certified root re-solve after a useful CG archive commit."""
        if eta_installed <= 0 or time.monotonic() >= deadline:
            return None
        started=time.monotonic()
        values,result=algorithm.forward_solver._solve_model(
            1,self.tree[1][0],algorithm.cut_lag,{},deadline=deadline)
        wall=time.monotonic()-started
        bound=result.certified_lower_bound
        if bound is not None:
            bound=max(0.,float(bound))
            if algorithm.best_ub < math.inf and minimization_bounds_inverted(bound,algorithm.best_ub):
                raise ValueError('physical CG root refresh LB exceeds audited global UB')
            if algorithm.lb_history:
                algorithm.lb_history[-1]=max(float(algorithm.lb_history[-1]),bound)
            self.record_root_certificate(bound,root=values)
        old_root=trial[1][0]
        anchor_before=self._weighted_eta_envelope(archive_before,old_root)
        anchor_after=self._weighted_eta_envelope(algorithm.cut_lag,old_root)
        next_before=next_after=None;changed=None;next_A=None
        if values is not None:
            next_A={k:v for k,v in values.items() if k.startswith('A[')}
            changed=next_A!={k:v for k,v in old_root.items() if k.startswith('A[')}
            next_before=self._weighted_eta_envelope(archive_before,values)
            next_after=self._weighted_eta_envelope(algorithm.cut_lag,values)
        self.last_escape=dict(iteration=int(iteration),eta_installed=int(eta_installed),
            anchor_gain=anchor_after-anchor_before,
            next_candidate_gain=None if next_after is None else next_after-next_before,
            A_changed=changed,A_next=next_A,root_certified_LB=bound,
            root_refresh_wall=wall,theta_next_assignment_gain=None,
            theta_next_assignment_reason='NEXT_S2_ASSIGNMENT_NOT_SOLVED_IN_EXTRA_ROOT_REFRESH')
        self._emit('physical_root_refresh',**self.last_escape)
        return self.last_escape

    def run_epoch(self, algorithm, trial, costs, *, iteration, deadline=None, pool=None):
        if not self.pool_enabled or self.expired() or self.blocked:
            return None
        from cuts.lrp_physical_bridge import physical_archive_value, build_route_lp_support
        from solvers.lrp_physical_policy_pool import solve_pool_policy
        from solvers.lrp_parallel import map_worker_jobs, worker_environment
        started = time.monotonic()
        stop = self.clip_deadline(deadline)
        stop = min(started+self.options['epoch_budget_s'], math.inf if stop is None else stop)
        work_stop = stop-self.options['audit_reserve_s']
        self._active_epoch_started = started
        archive, affected = algorithm.cut_lag, set()
        self._sync_archive(archive)
        archive_before_cg={2:{q:list(archive.get(2,{}).get(q,()))
                              for q in self.tree[1][0].successor}}
        self.last_A_L = {k: v for k,v in trial[1][0].items() if k.startswith('A[')}
        before_stats, old_version = dict(self.stats), self.envelope_version
        probe_tick = self._next_probe_tick(iteration)
        candidates = [deepcopy(trial)]
        try:
            if time.monotonic() >= work_stop:
                return {'status': 'NO_BUDGET'}
            self._harvest_backward_routes(algorithm,deadline=work_stop,iteration=iteration)
            if not self._singletons_done:
                try:
                    self.pool.add_singletons(generation_id=0, deadline=work_stop)
                    self._singletons_done = True
                except SolveDeadlineReached:
                    return {'status': 'NO_BUDGET'}
                finally:
                    self._record_new_routes()
            if time.monotonic()<work_stop and self._last_pool_version != self.pool.version:
                pool_stop = (min(work_stop, time.monotonic()+self.options['pool_time_limit_s'])
                             if self._v2_pool_budget else work_stop)
                pool_accept_stop = pool_stop if self._v2_pool_budget else stop
                result = solve_pool_policy(self.pool, trial[1][0],
                    incumbent_policy=algorithm.x_best, deadline=pool_stop,
                    time_limit=self.options['pool_time_limit_s'], mip_gap=self.options['pool_mip_gap'],
                    audit_reserve=self._pool_audit_reserve, seed=42)
                self.stats['pool_calls'] += 1
                self.stats['physical_pool_wall'] += result.wall_seconds
                self.stats['audit_wall'] += result.audit_seconds
                if result.status != 'NO_BUDGET':
                    self._last_pool_version = result.pool_version
                accepted = result.policy is not None and self._within_deadline(pool_accept_stop)
                changed = False
                improved = False
                if accepted:
                    self.last_A_P = {k:v for k,v in result.policy[1][0].items() if k.startswith('A[')}
                    changed = self.last_A_P != self.last_A_L
                    prior = algorithm.best_ub
                    offered = self._accept_policy_before_deadline(algorithm,result.policy,
                        deadline=pool_accept_stop,boundary='physical_pool')
                    accepted = offered is not None
                    if accepted and offered != result.audited_global_ub:
                        raise ValueError('pool audited upper changed in original policy audit')
                    improved = algorithm.best_ub < prior
                    if accepted:
                        self.collect(result.policy,source='pool_policy',iteration=iteration,phase=2,deadline=pool_accept_stop)
                        candidates.append(deepcopy(result.policy))
                self._emit('physical_pool', pool_version=result.pool_version, routes=self.pool.count,
                    A_L=self.last_A_L, A_P=self.last_A_P, A_changed=changed,
                    audited_UB=result.audited_global_ub, accepted=accepted, improved=improved,
                    status=result.status, wall=result.wall_seconds, diagnostics=result.diagnostics)
                print(f'  [PhysicalPool] pool_version={result.pool_version} routes={self.pool.count} '
                    f'A_changed={changed} audited_UB={result.audited_global_ub} '
                    f'improved={improved} status={result.status} wall={result.wall_seconds:.3f}s', flush=True)
            cg_eta_start=self.stats['eta_installed']
            if self.cg_enabled and time.monotonic()<work_stop:
                self._run_cg_nodes(algorithm,trial,archive,affected,iteration=iteration,
                                   work_stop=work_stop,envelope_version=old_version,
                                   probe_tick=probe_tick)
                self._refresh_root_after_cg(algorithm,trial,archive_before_cg,
                    deadline=stop,iteration=iteration,
                    eta_installed=self.stats['eta_installed']-cg_eta_start)
            if self.joint_enabled and time.monotonic()<work_stop:
                selection = []
                for q in self.tree[1][0].successor:
                    if time.monotonic()>=work_stop:
                        break
                    node, ctx = self.tree[2][q], self.tree[2][q].context
                    mask = _mask(trial[1][0], ctx)
                    key = (self.pool.signatures[q], mask)
                    cert = self.node_cache.get(key)
                    if cert is None:
                        self._emit('node_skip',node_id=(ctx.period,ctx.scenario),A_mask=mask,
                                   reason='NO_AUDITED_NODE_CACHE')
                        continue
                    eta = round_fraction_up(physical_archive_value(self.data,node,archive,mask))
                    tolerance = self.options['sep_atol']+self.options['sep_rtol']*max(1.,abs(eta),abs(cert.q_lower or 0.))
                    if cert.q_lower is not None and cert.q_lower > eta+tolerance:
                        self._eta(node, archive, cert, mask, affected,deadline=stop)
                        continue
                    if cert.q_upper is None:
                        self._emit('node_skip',node_id=cert.node_id,A_mask=mask,reason='NO_AUDITED_NODE_UPPER')
                        continue
                    accepted = cert.q_upper <= eta+self.options['node_accept_tol']
                    if accepted:
                        self._emit('node_skip', node_id=cert.node_id, A_mask=mask,
                            accepted_for_forward=True, closed_within_tolerance=cert.closed_within_tolerance)
                        continue
                    last = self.last_probe.get(key)
                    aged = last is not None and probe_tick-last>=5
                    score = float(self.data.arrays['scenario_prob'][ctx.scenario])*max(
                        0., cert.q_upper-max(eta,cert.q_lower if cert.q_lower is not None else eta))
                    selection.append(((-int(aged),-score,-int(last is None),ctx.period,ctx.scenario),
                                      node,mask,cert,eta,key))
                selection.sort(key=lambda item:item[0])
                maximum = min(2,int(self.options['joint_max_nodes_per_epoch']))
                jobs = []
                for _,node,mask,cert,eta,key in selection[:maximum]:
                    if time.monotonic() >= work_stop:
                        break
                    self.last_probe[key] = probe_tick
                    jobs.append(dict(data=self.data,tree=self.tree,node=node,mask=mask,
                        archive=deepcopy(archive),environment=worker_environment(),options=dict(
                            mode='exact_mip',time_limit_s=self.options['joint_time_limit_s'],
                            deadline=min(work_stop,time.monotonic()+self.options['joint_time_limit_s']),
                            envelope_version=self.envelope_version,previous=deepcopy(cert),
                            eta_ref=eta,accept_tolerance=self.options['node_accept_tol'],
                            mip_gap=self.options['joint_mip_gap'])))
                before = time.monotonic()
                concurrency = min(2,int(self.options['joint_max_concurrent']),int(self.config.get('num_processes',1)))
                packets = []
                for start in range(0,len(jobs),max(1,concurrency)):
                    batch = jobs[start:start+max(1,concurrency)]
                    if pool is not None and concurrency>1:
                        packets.extend(map_worker_jobs(_joint_worker,batch,concurrency,pool=pool))
                    else:
                        packets.extend(_joint_worker(job) for job in batch)
                self.stats['joint_wall'] += time.monotonic()-before
                for job,packet in zip(jobs,packets):
                    certificate, node = packet['certificate'],job['node']
                    self.stats['joint_calls'] += 1
                    self.stats['joint_worker_cpu'] += packet['cpu_seconds']
                    self.stats['audit_wall'] += certificate.audit_seconds
                    accepted = time.monotonic()<stop and not self.expired()
                    if accepted:
                        merged = self._merge_node(certificate,node,job['mask'],deadline=stop)
                        accepted = merged is not None
                        if accepted:
                            certificate = merged
                            self._eta(node,archive,certificate,job['mask'],affected,deadline=stop)
                    self._emit('joint_node', node_id=certificate.node_id, A_mask=certificate.A_mask,
                        q_lower=certificate.q_lower,q_upper=certificate.q_upper,
                        domain_kind=certificate.domain_kind,model_domain_complete=certificate.model_domain_complete,
                        status=certificate.status,accepted=accepted,closed_within_tolerance=certificate.closed_within_tolerance,
                        accepted_for_forward=certificate.q_upper is not None and certificate.q_upper<=job['options']['eta_ref']+self.options['node_accept_tol'],
                        wall=packet['wall_seconds'],build_seconds=certificate.build_seconds,
                        solve_seconds=certificate.solve_seconds,audit_seconds=certificate.audit_seconds,
                        node_signature=certificate.node_signature,diagnostics=certificate.diagnostics)
                    print(f'  [JointNode] q={certificate.node_id} A={certificate.A_mask} mode=exact_mip '
                        f'L={certificate.q_lower} U={certificate.q_upper} status={certificate.status} '
                        f'wall={packet["wall_seconds"]:.3f}s', flush=True)
                if packets and time.monotonic()<stop and not self.expired():
                    composed = self._compose(algorithm,trial,iteration,deadline=stop)
                    if composed is not None:
                        candidates.append(composed)
                # Prioritize largest actual-tour minus complete-theta residual.
                route_states = {}
                for policy in candidates:
                    if time.monotonic() >= work_stop:
                        break
                    for q in self.tree[1][0].successor:
                        if time.monotonic() >= work_stop:
                            break
                        node,ctx = self.tree[2][q],self.tree[2][q].context
                        physical = {}
                        for rid in node.successor:
                            if time.monotonic() >= work_stop:
                                break
                            owner = int(self.tree[3][rid].info)
                            physical[owner] = sum((F(float(ctx.route_cost[owner,v,w]))
                                for name,bit in policy[3][rid].items()
                                if name.startswith('r[') and bit
                                for _,v,w in [tuple(map(int,name[2:-1].split(',')))]),F())
                        for i in range(ctx.m):
                            if time.monotonic() >= work_stop:
                                break
                            state = tuple(int(policy[2][q][f'alpha[{i},{j}]']) for j in range(ctx.n))+(int(policy[2][q][f'u[{i}]']),)
                            # Audited empty routes cost zero; the complete
                            # theta envelope includes zero even with negative
                            # cuts, so their scheduling residual cannot be >0.
                            if not any(state):
                                continue
                            key = (self.pool.signatures[q],i,state)
                            old = physical_archive_value(self.data,node,archive,state,facility_id=i)
                            # Physical policy may be a different feasible order from the cached best.
                            # Its cost is only a scheduling hint, never a lower/target certificate.
                            score = float(physical.get(i,F())-old)
                            if score>self.options['sep_atol']:
                                route_states[key] = (score,node,i,state)
                route_calls = 0
                for key,(_,node,i,state) in sorted(route_states.items(),key=lambda item:(-item[1][0],item[0])):
                    if time.monotonic()>=work_stop:
                        break
                    cut = self.route_lp_cache.get(key)
                    if cut is None:
                        if route_calls>=min(4,int(self.options['route_lp_max_calls_per_epoch'])):
                            continue
                        cut,diagnostic = build_route_lp_support(self.data,node,i,state[:-1],state[-1],
                            deadline=work_stop,time_limit_s=self.options['route_lp_time_limit_s'],
                            envelope_version=self.envelope_version)
                        route_calls += 1; self.stats['route_lp_calls'] += 1
                        self.stats['route_lp_wall'] += diagnostic['wall_seconds']
                        self.stats['audit_wall'] += diagnostic.get('audit_seconds',0.)
                        self._emit('route_lp',node_id=(node.context.period,node.context.scenario),
                                   facility=i,state=state,diagnostics=diagnostic)
                        if cut is not None and self._within_deadline(stop):
                            self.route_lp_cache[key]=cut
                            self.stats['theta_generated'] += 1
                    if cut is not None and time.monotonic()<stop and not self.expired():
                        compatible = [entry[3] for entry in route_states.values()
                                      if entry[1].index==node.index and entry[2]==i]
                        self._install(node,archive,cut,compatible,affected,deadline=stop)
            if self.envelope_version != old_version:
                reprice_same_trial(algorithm,trial,costs,affected)
            if {k:v for k,v in trial[1][0].items() if k.startswith('A[')} != self.last_A_L:
                raise ValueError('physical forward changed the original A_L trial')
            self._archive_fingerprint = cut_archive_fingerprint(archive)
            self.observe(algorithm,phase=2,iteration=iteration,boundary='physical_epoch')
            return {'status':'COMPLETE' if time.monotonic()<stop else 'BUDGET_EXHAUSTED',
                    'affected_s2_nodes':sorted(affected),'envelope_version':self.envelope_version}
        except (ValueError, RuntimeError) as exc:
            self._failure(exc, iteration=iteration,A_L=self.last_A_L,
                          details=getattr(exc,'evidence',None),archive=deepcopy(archive))
            raise
        finally:
            self.stats['physical_epoch_wall'] += time.monotonic()-started
            self._active_epoch_started = None
            changes = {key:self.stats[key]-before_stats[key] for key in self.stats}
            self._emit('physical_budget',total=time.monotonic()-started,
                deadline=stop,overrun_seconds=max(0.,time.monotonic()-stop),changes=changes)
            print(f'  [PhysicalCuts] eta_installed={changes["eta_installed"]} '
                f'theta_installed={changes["theta_installed"]} envelope={old_version}->{self.envelope_version}',flush=True)
            print(f'  [PhysicalBudget] pool={changes["physical_pool_wall"]:.3f}s '
                f'joint_wall={changes["joint_wall"]:.3f}s route_lp={changes["route_lp_wall"]:.3f}s '
                f'audit={changes["audit_wall"]:.3f}s total={time.monotonic()-started:.3f}s',flush=True)

    def summary(self):
        return dict(profile=self.profile,options=dict(self.options),envelope_version=self.envelope_version,
            cg_closed_pricing_call_cap=self.cg_closed_pricing_call_cap,
            cg_coverage_price_seed=self.cg_coverage_price_seed,
            cg_persistent_prices=self.cg_persistent_prices,
            cg_pricing_top_k=8 if self.cg_persistent_prices else 1,
            cg_pricing_ng_size=32 if self.cg_persistent_prices else 8,
            cg_capacity_price_bound=self.cg_persistent_prices,
            collect_backward_routes=self.collect_backward_routes,
            route_count=0 if self.pool is None else self.pool.count,node_cache_count=len(self.node_cache),
            A_L=deepcopy(self.last_A_L),A_P=deepcopy(self.last_A_P),stats=dict(self.stats),
            last_progress=deepcopy(self.last_progress),blocked=self.blocked,
            last_escape=deepcopy(self.last_escape),
            pool_diagnostics=None if self.pool is None else dict(self.pool.diagnostics))
