"""Persistent facility-indexed route column generation for one LRP node."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import math
import time

import gurobipy as gp

from cuts.lrp_physical_price_certificates import build_price_certificate
from models.stage_builder import _instance, _node_context
from solvers.lrp_facility_pricing_adapter import FacilityPricingAdapter,transported_lower
from solvers.lrp_facility_pricing_adapter import price_hash
from solvers.lrp_physical_cg_types import JointResult
from solvers.lrp_physical_policy_pool import audit_node_policy
from solvers.lrp_physical_types import node_signature


def _left(deadline):
    return max(0., float(deadline)-time.monotonic())


def _mask(values, m):
    result=tuple(int(v) for v in values)
    if len(result)!=m or any(v not in (0,1) for v in result):
        raise ValueError("invalid facility mask")
    return result


def _fraction_down(value):
    answer=float(value)
    return math.nextafter(answer,-math.inf) if F(answer)>value else answer


def coverage_price_proposal(data, tree, node, archive, A, *, deadline):
    """Reuse the original coverage LP only to propose prices, never a bound.

    The complete current S3 envelope is included; S2 physical eta rows are
    excluded. The existing Phase-1.5 [0, outsourcing] repair is a proposal
    choice, not a change to the signed prices of the ordinary route master.
    """
    if _left(deadline) <= .02:
        return None, {'status': 'NO_BUDGET'}
    from core import customized_subprob  # registers the existing solver path
    from s2backward.physical_route_seed import _coverage_rewards
    ctx = node.context
    facilities = [int(tree[3][rid].info) for rid in node.successor]
    if len(facilities) != ctx.m or set(facilities) != set(range(ctx.m)):
        raise ValueError('coverage proposal requires every physical route child')
    scoped = {3: {rid: tuple(archive.get(3, {}).get(rid, ())) for rid in node.successor}}
    root = {f'A[{i},{ctx.interval}]': int(A[i]) for i in range(ctx.m)}
    cap = tuple(float(ctx.outsourcing[j]) if ctx.active[j] else 0. for j in range(ctx.n))
    lam, report = _coverage_rewards(data, node, scoped, root, cap,
                                    time.perf_counter()+_left(deadline))
    report = dict(report, source='coverage_lp_price_proposal',
                  retained_s3_rows=sum(map(len, scoped[3].values())),
                  included_s2_rows=0, objective_is_physical_bound=False)
    if lam is None or _left(deadline) <= 0:
        return None, report
    if len(lam) != ctx.n or any(not math.isfinite(float(v)) for v in lam):
        raise ValueError('invalid complete coverage price proposal')
    return tuple(min(float(v), cap[j]) if ctx.active[j] else 0.
                 for j, v in enumerate(lam)), report


class JointRouteCG:
    """Parent-owned route pool plus stateless per-round LP/pricing models."""
    def __init__(self, prob_data, tree, pool, *, pricing_backend="native", price_proposal=None):
        self.data,self.tree,self.pool=_instance(prob_data),tree,pool
        self.pricing_backend=pricing_backend
        self.price_proposal=price_proposal

    def _routes(self,node):
        return tuple(self.pool.routes(node))

    def _solve_master(self,node,A,*,integer,deadline,mip_gap=0.,route_snapshot=None):
        if _left(deadline)<=0: return None
        ctx=_node_context(self.data,node,stage=2)
        routes=self._routes(node) if route_snapshot is None else tuple(route_snapshot)
        model=gp.Model("lrp_node_route_master")
        try:
            model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.Seed=42
            model.Params.TimeLimit=max(1e-3,_left(deadline))
            if integer:
                model.Params.MIPGap=float(mip_gap)
            kind=gp.GRB.BINARY if integer else gp.GRB.CONTINUOUS
            x=[model.addVar(lb=0.,ub=1. if integer else gp.GRB.INFINITY,
                            vtype=kind,obj=float(r.cost),name=f"x[{k}]")
               for k,r in enumerate(routes)]
            e={j:model.addVar(lb=0.,ub=1. if integer else gp.GRB.INFINITY,
                              vtype=kind,obj=float(ctx.outsourcing[j]),name=f"e[{j}]")
               for j in range(ctx.n) if ctx.active[j]}
            cover={j:model.addConstr(gp.quicksum(x[k] for k,r in enumerate(routes)
                         if j in r.customers_in_order)+e[j]==1.,name=f"cover[{j}]") for j in e}
            fleet={i:model.addConstr(gp.quicksum(x[k] for k,r in enumerate(routes)
                         if r.facility_id==i)<=A[i],name=f"facility[{i}]") for i in range(ctx.m)}
            model.ModelSense=gp.GRB.MINIMIZE;model.optimize()
            if (not integer and model.Status!=gp.GRB.OPTIMAL) or (integer and model.SolCount<1):
                return dict(status=int(model.Status),routes=routes)
            result=dict(status=int(model.Status),objective=float(model.ObjVal),routes=routes,
                        x=tuple(float(v.X) for v in x),e={j:float(v.X) for j,v in e.items()})
            if not integer:
                result.update(lambda_vector=tuple(float(cover[j].Pi) if j in cover else 0.
                    for j in range(ctx.n)),beta=tuple(float(fleet[i].Pi) for i in range(ctx.m)))
            return result
        finally:
            model.dispose()

    def _policy(self,node,A,master):
        ctx=_node_context(self.data,node,stage=2)
        state={f"alpha[{i},{j}]":0. for i in range(ctx.m) for j in range(ctx.n)}
        state.update({f"u[{i}]":0. for i in range(ctx.m)})
        state.update({f"e[{j}]":float(ctx.active[j]) for j in range(ctx.n)})
        policy={2:{node.index:state},3:{rid:{} for rid in node.successor}}
        for value,route in zip(master["x"],master["routes"]):
            if value<.5: continue
            i=route.facility_id;state[f"u[{i}]"]=1.
            for j in route.customers_in_order:
                if state[f"e[{j}]"]!=1.: raise ValueError("restricted policy double covers customer")
                state[f"e[{j}]"]=0.;state[f"alpha[{i},{j}]"]=1.
            rid=next(r for r in node.successor if int(self.tree[3][r].info)==i)
            path=(0,)+tuple(j+1 for j in route.customers_in_order)+(0,)
            policy[3][rid]={f"r[{i},{v},{w}]":1. for v,w in zip(path,path[1:])}
        part,upper,audit=audit_node_policy(self.data,self.tree,node,A,policy)
        return part,upper,audit

    def _coverage_seed(self, node, A, archive, adapter, *, deadline,
                       pricing_call_cap, generation_id, envelope_version,
                       incremental_master=None):
        """One bounded price proposal before genuine RMP column generation.

        There is no RMP beta for this proposal. Every audited native route
        may be added as a feasible column, without claiming reduced cost.
        No seed outcome (including zero columns) terminates the CG loop.
        """
        begun = time.monotonic()
        lam, proposal = coverage_price_proposal(self.data, self.tree, node, archive, A,
                                                deadline=min(deadline,begun+2.))
        detail = dict(round=-1, price_source='coverage_seed', rmp_solved=False,
                      proposal=proposal, new_columns=0, pricing_detail=[])
        if lam is None or _left(deadline)<=0:
            detail.update(status='NO_PRICE_PROPOSAL', elapsed=time.monotonic()-begun)
            return None, (), (), detail
        ctx = node.context
        opened = [i for i in range(ctx.m) if A[i]]
        order = opened+[i for i in range(ctx.m) if not A[i]]
        results, added = [], []
        for position,i in enumerate(order):
            started = time.monotonic()
            allowance = (min(float(pricing_call_cap),_left(deadline)/max(1,len(opened)-position))
                         if A[i] else 0.)
            if allowance<=0:
                result=adapter.price(i,lam,deadline=started-1,generation_id=generation_id)
            else:
                result=adapter.price(i,lam,deadline=deadline,time_limit_s=allowance,
                                     generation_id=generation_id)
            results.append(result)
            # A seed route is useful independently of whether its seed price
            # would yield negative reduced cost at some unrelated RMP dual.
            candidates=(() if result.incumbent_route is None else (result.incumbent_route,))
            for route in candidates+result.additional_routes:
                if _left(deadline)<=0: break
                before=self.pool.version
                stored=self.pool.add_audited_route(node,route,deadline=deadline)
                changed = self.pool.version != before
                if changed:
                    added.append(stored.audit_signature)
                if incremental_master is not None:
                    # Pool insertion/eviction is not necessarily an RMP change.
                    changed = False
                    if stored is not None and _left(deadline)>0:
                        delta = incremental_master.sync((stored,))
                        changed = bool(delta['added'] or delta['replaced'])
                detail['new_columns'] += int(changed)
            detail['pricing_detail'].append(dict(facility=i,available=bool(A[i]),
                allowance_s=allowance,status=result.status,safe_lower=result.safe_lower,
                incumbent_price_value=result.incumbent_price_value,
                additional_route_count=len(result.additional_routes),
                lambda_hash=result.price_vector_hash,certificate_source=result.certificate_source,
                elapsed_wall_s=time.monotonic()-started,
                fallback=bool(result.error_accounting.get('fallback') or
                              result.certificate_source=='nonnegative_cost_fallback')))
        # Full-domain fallbacks finish the physical proof even if the native
        # allowance has expired. This is not a late policy/UB acceptance.
        certificate=build_price_certificate(self.data,node,A,lam,results,
                                             envelope_version=envelope_version)
        from cuts.lrp_physical_bridge import _exact_value
        detail.update(status='CERTIFIED_PRICE_SEED',lambda_vector=lam,
            route_lp_lower=_fraction_down(_exact_value(certificate.eta_cut,A)),
            pool_version=self.pool.version,
            route_addition_reason='audited_seed_route_not_rmp_reduced_cost',
            elapsed=time.monotonic()-begun)
        return certificate,tuple(results),tuple(added),detail

    def _level_price(self,node,A,center,lower,upper,*,deadline,routes):
        """L1 level projection in the restricted route dual, prices only.

        This model chooses a point between the certified physical lower and
        the restricted LP upper. Its objective/level is NEVER a physical LB.
        Full-domain pricing still has to certify every proposed vector.
        """
        begun=time.monotonic()
        if _left(deadline)<=.01 or center is None or upper<=lower:
            return None,dict(status='NO_LEVEL_PROPOSAL')
        ctx=node.context
        model=gp.Model('lrp_physical_price_level')
        try:
            model.Params.OutputFlag=0;model.Params.Threads=1
            level=lower+.5*(upper-lower)
            lam={j:model.addVar(lb=-gp.GRB.INFINITY,ub=float(ctx.outsourcing[j]),
                               name=f'lambda[{j}]') for j in range(ctx.n) if ctx.active[j]}
            beta={i:model.addVar(lb=-gp.GRB.INFINITY,ub=0.,name=f'beta[{i}]')
                  for i in range(ctx.m) if A[i]}
            dev={j:model.addVar(lb=0.,obj=1.,name=f'dev[{j}]') for j in lam}
            for j in lam:
                model.addConstr(dev[j]>=lam[j]-float(center[j]))
                model.addConstr(dev[j]>=float(center[j])-lam[j])
            for route in routes:
                if route.facility_id in beta:
                    model.addConstr(beta[route.facility_id]+
                        gp.quicksum(lam[j] for j in route.customers_in_order)<=float(route.cost))
            model.addConstr(gp.quicksum(lam.values())+gp.quicksum(beta.values())>=level)
            model.ModelSense=gp.GRB.MINIMIZE
            if _left(deadline)<=.01:
                return None,dict(status='BUILD_DEADLINE',elapsed=time.monotonic()-begun)
            model.Params.TimeLimit=max(.001,_left(deadline))
            model.optimize()
            detail=dict(status=int(model.Status),elapsed=time.monotonic()-begun,
                        level_proposal=level,certified_lower=lower,restricted_upper=upper,
                        norm='L1',objective_is_physical_bound=False)
            if model.SolCount<1 or _left(deadline)<=0:
                return None,detail
            proposal=tuple(min(float(lam[j].X),float(ctx.outsourcing[j])) if j in lam else 0.
                           for j in range(ctx.n))
            if any(not math.isfinite(v) for v in proposal):
                return None,dict(detail,invalid_prices=True)
            return proposal,detail
        finally:
            model.dispose()

    def evaluate(self,node,A_mask,*,deadline,envelope_version=0,existing_node_lower=None,
                 max_rounds=50,pricing_call_cap=3.,policy_reserve_share=.2,
                 generation_id=0,closed_pricing_call_cap=None,coverage_archive=None,
                 price_state=None,stabilize_prices=False,pricing_top_k=1,
                 capacity_price_bound=False,pricing_ng_size=8):
        """Keep the original path unless persistent physical CG is requested.

        The incrementally extended LP lives for this call, like Investment's
        RouteOpt root LP. Only independently scoped pure pricing evidence is
        allowed to survive the call or be checkpointed.
        """
        wrapper_started=time.monotonic()
        options=dict(deadline=deadline,envelope_version=envelope_version,
            existing_node_lower=existing_node_lower,max_rounds=max_rounds,
            pricing_call_cap=pricing_call_cap,policy_reserve_share=policy_reserve_share,
            generation_id=generation_id,closed_pricing_call_cap=closed_pricing_call_cap,
            coverage_archive=coverage_archive,pricing_top_k=pricing_top_k,
            capacity_price_bound=capacity_price_bound,pricing_ng_size=pricing_ng_size)
        if price_state is None and not stabilize_prices:
            return self._evaluate(node,A_mask,**options)
        if not isinstance(stabilize_prices,bool):
            raise TypeError('stabilize_prices must be boolean')
        if _left(deadline)<=0:
            return self._evaluate(node,A_mask,**options)
        from solvers.lrp_cg_master import IncrementalRouteMaster
        from solvers.lrp_cg_price_state import PhysicalCGPriceState
        ctx=_node_context(self.data,node,stage=2);A=_mask(A_mask,ctx.m)
        if price_state is None:
            price_state=PhysicalCGPriceState(self.data,node,A)
        price_state.validate(self.data,node,A)
        if _left(deadline)<=0:
            return self._evaluate(node,A_mask,**options)
        master=IncrementalRouteMaster(ctx,self._routes(node),A)
        try:
            result=self._evaluate(node,A,price_state=price_state,
                stabilize_prices=stabilize_prices,incremental_master=master,**options)
        finally:
            master.close()
        elapsed=time.monotonic()-wrapper_started
        return replace(result,timings={**result.timings,'total':elapsed,'cg':elapsed,
                       'incremental_lp_solves':master.diagnostics['lp_solves']})

    def _evaluate(self,node,A_mask,*,deadline,envelope_version=0,existing_node_lower=None,
                 max_rounds=50,pricing_call_cap=3.,policy_reserve_share=.2,
                 generation_id=0,closed_pricing_call_cap=None,coverage_archive=None,
                 price_state=None,stabilize_prices=False,incremental_master=None,pricing_top_k=1,
                 capacity_price_bound=False,pricing_ng_size=8):
        started=time.monotonic();ctx=_node_context(self.data,node,stage=2)
        A=_mask(A_mask,ctx.m)
        if not math.isfinite(float(deadline)) or not 0<policy_reserve_share<1:
            raise ValueError("joint CG needs finite deadline and a proper reserve")
        if closed_pricing_call_cap is not None:
            if isinstance(closed_pricing_call_cap,bool) or not math.isfinite(float(closed_pricing_call_cap)) or float(closed_pricing_call_cap)<0:
                raise ValueError("closed pricing cap must be finite/nonnegative or None")
            closed_pricing_call_cap=float(closed_pricing_call_cap)
        adapter=FacilityPricingAdapter(self.data,self.tree,node,backend=self.pricing_backend,
                                      top_k=pricing_top_k,capacity_bound=capacity_price_bound,
                                      ng_size=pricing_ng_size)
        trace=[];eta=[];theta=[];added=[];all_results=[];best_lb=0.
        prior_pricing={}
        stalled_prices=set()
        rmp_upper=None;stop_reason="NO_BUDGET";lp_closed=False
        cached=None
        if price_state is not None and _left(deadline)>.01:
            cached=price_state.best(self.data,node,A,envelope_version=envelope_version)
            if cached is not None and capacity_price_bound and _left(deadline)>.01:
                from solvers.lrp_facility_pricing_adapter import strengthen_capacity_bound
                old_certificate,old_results=cached
                upgraded=tuple(strengthen_capacity_bound(ctx,old_certificate.lambda_vector,r)
                               for r in old_results)
                price_state.consider(self.data,node,A,old_certificate.lambda_vector,upgraded,
                                     envelope_version=envelope_version,deadline=deadline)
                cached=price_state.best(self.data,node,A,envelope_version=envelope_version)
            if cached is not None and _left(deadline)>0:
                certificate,cached_results=cached
                from cuts.lrp_physical_bridge import _exact_value
                best_lb=max(0.,_fraction_down(_exact_value(certificate.eta_cut,A)))
                eta.append(certificate);theta.extend(certificate.theta_cuts)
                trace.append(dict(round=-2,price_source='cached_physical_price',cached_price=True,
                    lambda_vector=certificate.lambda_vector,route_lp_lower=best_lb,
                    new_columns=0,rmp_solved=False,completed_price_rounds=price_state.rounds))
        if cached is None and coverage_archive is not None and _left(deadline)>.05:
            # Bound once from the original remaining budget, preserving the
            # original policy/audit share and time for the actual RMP loop.
            seed_stop=time.monotonic()+.4*(1-policy_reserve_share)*_left(deadline)
            certificate,seed_results,seed_routes,seed_trace=self._coverage_seed(
                node,A,coverage_archive,adapter,deadline=seed_stop,
                pricing_call_cap=pricing_call_cap,generation_id=generation_id,
                envelope_version=envelope_version,incremental_master=incremental_master)
            trace.append(seed_trace)
            if certificate is not None:
                if price_state is not None:
                    price_state.consider(self.data,node,A,certificate.lambda_vector,seed_results,
                                         envelope_version=envelope_version,deadline=deadline)
                if price_state is not None and _left(deadline)<=0:
                    certificate=None
            if certificate is not None:
                eta.append(certificate);theta.extend(certificate.theta_cuts)
                all_results.extend(seed_results);added.extend(seed_routes)
                best_lb=max(best_lb,seed_trace['route_lp_lower'])
                prior_pricing={r.facility_id:(certificate.lambda_vector,r) for r in seed_results}
        for round_id in range(int(max_rounds)):
            previous_best=best_lb
            remaining=_left(deadline)
            if remaining<=max(.01,policy_reserve_share*(float(deadline)-started)):
                stop_reason="NO_BUDGET";break
            lp_deadline=deadline-max(.01,policy_reserve_share*remaining)
            if incremental_master is None:
                lp=self._solve_master(node,A,integer=False,deadline=lp_deadline)
            else:
                incremental_master.sync(self._routes(node))
                lp=incremental_master.solve(deadline=lp_deadline)
            if lp is None or "objective" not in lp:
                stop_reason="RMP_LP_NOT_SOLVED";break
            from solvers.lrp_cg_lp_certificate import certify_route_lp_upper
            lp_certificate=certify_route_lp_upper(ctx,A,lp,
                expected_node_signature=node_signature(self.data,node))
            rmp_upper=float(lp_certificate['upper'])
            # Exact cover duals are free. Repair only numerical violations of
            # the outsourcing-column cap, then price this repaired vector.
            lam=tuple(min(float(lp["lambda_vector"][j]),float(ctx.outsourcing[j]))
                      if ctx.active[j] else 0. for j in range(ctx.n))
            rmp_lam=lam;price_detail=dict(source='rmp_dual')
            if stabilize_prices and price_state is not None and price_state.price_center is not None:
                proposal,level_detail=self._level_price(node,A,price_state.price_center,best_lb,
                    rmp_upper,deadline=min(lp_deadline,time.monotonic()+2.),routes=lp['routes'])
                price_detail=dict(source='rmp_dual',level_projection=level_detail)
                if proposal is not None:
                    if price_hash(proposal) not in stalled_prices:
                        lam=proposal;price_detail['source']='level_projection'
                    else:
                        price_detail['source']='rmp_dual_after_stalled_level'
            proposal_columns=[]
            proposer=getattr(self,'price_proposal',None)
            proposal_floor=max(best_lb,0. if existing_node_lower is None else float(existing_node_lower))
            if (round_id==0 and proposer is not None and closed_pricing_call_cap==0.
                    and proposal_floor<rmp_upper-1e-6):
                # Preserve the original physical master and reserve time for
                # original-domain pricing and the integer policy/audit tail.
                left=_left(deadline)
                reserve=max(.01,policy_reserve_share*left)
                native_reserve=min(float(pricing_call_cap)*sum(A),.45*left)
                proposal_cap=min(40.,.4*left,left-reserve-native_reserve)
                if proposal_cap>.05:
                    proposed=proposer(self.data,node,A,self._routes(node),
                        deadline=time.monotonic()+proposal_cap,
                        existing_lower=proposal_floor,
                        generation_id=generation_id)
                    price_detail['auxiliary_proposal']=proposed.diagnostics
                    if proposed.lambda_vector is not None and _left(deadline)>reserve:
                        candidate=tuple(float(x) for x in proposed.lambda_vector)
                        if (len(candidate)!=ctx.n or any(not math.isfinite(v) for v in candidate)
                                or any(ctx.active[j] and candidate[j]>float(ctx.outsourcing[j])
                                       or not ctx.active[j] and candidate[j]!=0. for j in range(ctx.n))):
                            raise ValueError('Invalid auxiliary signed customer prices')
                        lam=candidate;price_detail['source']='auxiliary_ng_proposal'
                    for route in proposed.routes:
                        if _left(deadline)<=reserve:break
                        before=self.pool.version
                        stored=self.pool.add_audited_route(node,route,deadline=deadline)
                        changed=self.pool.version!=before
                        if changed:added.append(stored.audit_signature)
                        if incremental_master is not None:
                            changed=False
                            if stored is not None and _left(deadline)>reserve:
                                delta=incremental_master.sync((stored,))
                                changed=bool(delta['added'] or delta['replaced'])
                        if changed:proposal_columns.append(stored.audit_signature)
            order=sorted(range(ctx.m),key=lambda i:(-A[i],i));results=[];new=list(proposal_columns)
            pricing_detail=[]
            caps={i:float(pricing_call_cap) if A[i] or closed_pricing_call_cap is None
                  else closed_pricing_call_cap for i in order}
            price_stop=deadline-max(.01,policy_reserve_share*_left(deadline))
            for pos,i in enumerate(order):
                price_started=time.monotonic()
                remaining_price=price_stop-price_started
                allowance=0.
                if remaining_price<=0 or caps[i]==0.:
                    # Still obtain the complete-domain safe coefficient for a
                    # closed facility. Never pass native's zero=unlimited cap.
                    result=adapter.price(i,lam,deadline=time.monotonic()-1,
                                         generation_id=generation_id)
                else:
                    # None keeps the original fair sharing exactly. Explicitly
                    # skipped closed calls do not reserve time away from opens.
                    calls_left=(len(order)-pos if closed_pricing_call_cap is None else
                                sum(caps[j]>0. for j in order[pos:]))
                    fair=max(.001,remaining_price/calls_left)
                    allowance=min(caps[i],fair)
                    result=adapter.price(i,lam,deadline=price_stop,
                        time_limit_s=allowance,generation_id=generation_id)
                pricing_wall=time.monotonic()-price_started
                # Persistent state stores raw complete-domain evidence only.
                # No transported scalar can become an unaudited cache origin.
                old=prior_pricing.get(i) if price_state is None else None
                if old is not None and old[1].safe_lower is not None:
                    transported=_fraction_down(transported_lower(
                        old[1].safe_lower,old[0],lam,ctx.active))
                    if result.safe_lower is None or transported>result.safe_lower:
                        result=replace(result,safe_lower=transported,
                            certificate_source='eq4_transported_'+old[1].certificate_source,
                            error_accounting={**result.error_accounting,'transported':True,
                                'transported_from_price_hash':old[1].price_vector_hash,
                                'transport_formula':'eq4'})
                prior_pricing[i]=(lam,result)
                results.append(result)
                pricing_detail.append(dict(facility=i,available=bool(A[i]),cap_s=caps[i],
                    allowance_s=allowance,elapsed_wall_s=pricing_wall,
                    elapsed_solve_s=result.elapsed_solve,
                    fallback=bool(result.error_accounting.get("fallback") or
                                  result.certificate_source=="nonnegative_cost_fallback"),
                    transported=bool(result.error_accounting.get("transported")),
                    status=result.status,certificate_source=result.certificate_source,
                    incumbent_price_value=result.incumbent_price_value,
                    additional_route_count=len(result.additional_routes),
                    safe_lower=result.safe_lower,lambda_hash=result.price_vector_hash))
                candidates=(() if result.incumbent_route is None else (result.incumbent_route,))
                for route in candidates+result.additional_routes:
                    if _left(deadline)<=0: break
                    rc=F(*route.cost_exact)-sum((F(rmp_lam[j]) for j in route.customers_in_order),F())-F(lp["beta"][i])
                    if rc < F(-1e-7) or price_detail['source']=='level_projection':
                        before=self.pool.version
                        stored=self.pool.add_audited_route(node,route,deadline=deadline)
                        changed = self.pool.version != before
                        if changed:
                            added.append(stored.audit_signature)
                        if incremental_master is not None:
                            changed = False
                            if stored is not None and _left(deadline)>0:
                                delta = incremental_master.sync((stored,))
                                changed = bool(delta['added'] or delta['replaced'])
                        if changed:
                            new.append(stored.audit_signature)
            certificate=build_price_certificate(self.data,node,A,lam,results,
                                                 envelope_version=envelope_version)
            if price_state is not None:
                price_state.consider(self.data,node,A,lam,results,
                                     envelope_version=envelope_version,deadline=deadline)
                if _left(deadline)<=0:
                    stop_reason='PRICE_AUDIT_DEADLINE'
                    break
            eta.append(certificate);theta.extend(certificate.theta_cuts);all_results.extend(results)
            from cuts.lrp_physical_bridge import _exact_value
            best_lb=max(best_lb,_fraction_down(_exact_value(certificate.eta_cut,A)))
            tolerance=1e-6+1e-9*max(1.,abs(rmp_upper),abs(best_lb))
            if best_lb>rmp_upper+tolerance:
                raise ValueError('VALIDITY_COUNTEREXAMPLE: physical price LB exceeds restricted LP upper')
            gap=max(0.,rmp_upper-best_lb)
            trace.append(dict(round=round_id,rmp_lp_upper=rmp_upper,route_lp_lower=best_lb,
                              lp_certificate_gap=gap,new_columns=len(new),pool_version=self.pool.version,
                              lambda_vector=lam,beta=lp["beta"],pricing=[r.status for r in results],
                              pricing_detail=pricing_detail,
                              rmp_lp_certificate=lp_certificate,
                              **(dict(price_source=price_detail['source'],price_proposal=price_detail,
                                  raw_rmp_lambda_vector=rmp_lam,
                                  route_addition_reason='audited_level_routes_or_negative_raw_rmp_cost',
                                  completed_price_rounds=price_state.rounds,
                                  incremental_master=True,
                                  master_diagnostics=lp.get('master_diagnostics'))
                                  if price_state is not None else {})))
            if gap<=tolerance:
                stop_reason="LP_CERTIFIED";lp_closed=True;break
            if not new and not stabilize_prices:
                stop_reason="PRICING_UNRESOLVED" if any(
                    r.status not in {"OPTIMAL"} or r.error_accounting.get("fallback") for r in results
                    ) else "NO_COLUMNS_WITHOUT_CLOSURE"
                break
            if stabilize_prices and not new and best_lb<=previous_best+1e-7:
                if price_hash(lam) in stalled_prices and price_detail['source']!='level_projection':
                    stop_reason='PRICE_STALLED';break
                stalled_prices.add(price_hash(lam))
            stop_reason="ROUND_LIMIT"
        policy=upper=None
        if _left(deadline)>.01:
            master=self._solve_master(node,A,integer=True,deadline=deadline,mip_gap=1e-3,
                **({} if incremental_master is None else {'route_snapshot':incremental_master.routes}))
            if master is not None and "objective" in master and _left(deadline)>0:
                try:
                    policy,upper,_=self._policy(node,A,master)
                except (ValueError,RuntimeError):
                    policy=upper=None;stop_reason="POLICY_AUDIT_FAILURE"
        combined=max([best_lb,0.]+([] if existing_node_lower is None else [float(existing_node_lower)]))
        if upper is not None and combined>upper+2e-6+1e-10*max(1.,abs(upper),abs(combined)):
            raise ValueError("VALIDITY_COUNTEREXAMPLE: joint lower exceeds audited upper")
        gap=None if rmp_upper is None else max(0.,rmp_upper-best_lb)
        elapsed=time.monotonic()-started
        return JointResult((ctx.period,ctx.scenario),A,node_signature(self.data,node),
            "LP_CERTIFIED" if lp_closed else stop_reason,best_lb if eta else None,rmp_upper,gap,
            combined,upper,policy,tuple(eta),tuple(theta),tuple(dict.fromkeys(added)),
            dict(Counter(r.status for r in all_results)),tuple(sorted({r.facility_id for r in all_results
                if r.error_accounting.get("fallback") or
                   r.certificate_source=="nonnegative_cost_fallback"})),
            {"total":elapsed,"cg":elapsed,"pricing":sum(r.elapsed_solve for r in all_results)},
            stop_reason,tuple(trace))
