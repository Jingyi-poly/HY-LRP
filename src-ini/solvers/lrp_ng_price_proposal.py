"""Auxiliary NG master used ONLY to propose a signed customer price vector.

No scalar from this module is a physical lower bound or a cache certificate.
The existing original-domain facility pricing/eta construction must price the
returned lambda independently. Repeated walks stay in this local master;
only routes reaudited against original data can leave it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
import math
import time


@dataclass(frozen=True)
class PriceProposal:
    lambda_vector: tuple | None
    routes: tuple
    diagnostics: dict


def propose_ng_price(data, node, A, routes, *, deadline, pricing=None,
                     existing_lower=None, generation_id=0, ng_size=16,
                     call_cap=3., max_rounds=40, max_return_routes=64):
    """Use the caller's absolute deadline, including preparation and audits.

    Symmetric positive-demand instances only. Unsupported data returns no
    proposal; the ordinary original-domain CG path remains responsible for
    general directed/zero-demand inputs. This helper never mutates its pool.
    """
    begun=time.monotonic();ctx=node.context
    report=dict(status='NO_BUDGET',proposal_only=True,physical_bound=False,
                ng_size=ng_size,trace=[],pricing_calls=0)
    def finish(lam=None, output=()):
        report['seconds']=time.monotonic()-begun
        return PriceProposal(lam,tuple(output),report)
    if not math.isfinite(float(deadline)):
        raise ValueError('NG proposal requires a finite absolute deadline')
    if time.monotonic()>=deadline: return finish()
    proposal_stop=deadline-min(5.,max(.01,.15*(deadline-begun)))
    if len(A)!=ctx.m or any(v not in (0,1) for v in A):
        raise ValueError('Invalid physical availability mask')
    if (type(ng_size) is not int or ng_size<1 or type(max_rounds) is not int
            or max_rounds<1 or not math.isfinite(call_cap) or call_cap<=0
            or type(max_return_routes) is not int or max_return_routes<0):
        raise ValueError('Invalid bounded NG proposal options')
    if existing_lower is not None and not math.isfinite(float(existing_lower)):
        raise ValueError('Invalid screening floor')
    active=tuple(j for j in range(ctx.n) if ctx.active[j])
    opened=tuple(i for i in range(ctx.m) if A[i])
    if not active or not opened:
        report['status']='EMPTY_SCOPE';return finish()
    # No replacement of original costs/demands. These relaxed integer arrays
    # are local proposal inputs; the original arrays remain authoritative.
    scale=10**6;demand_scale=1000;vertices=(0,)+tuple(j+1 for j in active)
    demands=[math.floor(F(float(ctx.demand[j]))*demand_scale) for j in active]
    if any(d<=0 for d in demands):
        report['status']='UNSUPPORTED_RESOURCE';return finish()
    caps={i:math.ceil(F(float(ctx.capacity[i]))*demand_scale) for i in opened}
    if any(c<=0 for c in caps.values()):
        report['status']='UNSUPPORTED_RESOURCE';return finish()
    for i in opened:
        for v in vertices:
            if float(ctx.route_cost[i,v,v])!=0:
                report['status']='UNSUPPORTED_COSTS';return finish()
            for w in vertices:
                x=float(ctx.route_cost[i,v,w]);y=float(ctx.route_cost[i,w,v])
                if not math.isfinite(x) or x<0 or x!=y:
                    report['status']='UNSUPPORTED_COSTS';return finish()
        if time.monotonic()>=deadline:return finish()
    grids={i:[[math.floor(F(float(ctx.route_cost[i,v,w]))*scale)
               for w in vertices] for v in vertices] for i in opened}
    if time.monotonic()>=deadline:return finish()
    from solvers.lrp_physical_policy_pool import audit_route
    from solvers.lrp_facility_pricing_adapter import price_hash
    from solvers.lrp_capacity_price_bound import capacity_price_lower_bound
    physical={};initial=set();new={}
    for r in routes:
        if time.monotonic()>=deadline:return finish()
        if r.facility_id not in opened:continue
        checked=audit_route(data,node,r.facility_id,r.customers_in_order,
                            source=r.source,generation_id=r.generation_id)
        if checked!=r:raise ValueError('Proposal input is not an original audited route')
        physical[r.audit_signature]=r;initial.add(r.audit_signature)
    if pricing is None:
        # The caller must provide the reviewed, source-bound adapter. Never
        # silently choose a possibly stale optional binary from another path.
        report['status']='VERIFIED_PROVIDER_REQUIRED';return finish()
    if time.monotonic()>=deadline:return finish()
    import gurobipy as gp
    model=gp.Model('lrp_auxiliary_ng_price_proposal')
    options=[];seen=set();stop='ROUND_LIMIT'
    try:
        model.Params.OutputFlag=0;model.Params.Threads=1
        model.Params.Seed=42;model.Params.Method=1
        cover={j:model.addConstr(gp.LinExpr()==1.) for j in active}
        fleet={i:model.addConstr(gp.LinExpr()<=1.) for i in opened}
        for j in active:model.addVar(obj=float(ctx.outsourcing[j]),column=gp.Column([1.],[cover[j]]))
        columns={}
        def add(i,order,cost):
            counts=Counter(order);key=(i,tuple(sorted(counts.items())))
            if key in columns:
                var,old=columns[key]
                if cost>=old:return False
                var.Obj=float(cost);columns[key]=(var,cost);return True
            var=model.addVar(obj=float(cost),column=gp.Column(
                list(counts.values())+[1.],[cover[j] for j in counts]+[fleet[i]]))
            columns[key]=(var,cost);return True
        for r in physical.values():
            if time.monotonic()>=proposal_stop:
                report['status']='BUILD_DEADLINE';return finish()
            add(r.facility_id,r.customers_in_order,F(*r.cost_exact))
        for step in range(max_rounds):
            left=proposal_stop-time.monotonic()
            if left<=.01:stop='DEADLINE';break
            model.Params.TimeLimit=left;model.optimize()
            if model.Status!=gp.GRB.OPTIMAL:stop='RMP_UNRESOLVED';break
            lam=tuple(min(float(cover[j].Pi),float(ctx.outsourcing[j]))
                      if j in cover else 0. for j in range(ctx.n))
            if any(not math.isfinite(x) for x in lam):raise ValueError('Non-finite proposed price')
            key=price_hash(lam)
            if key in seen:stop='REPEATED_PRICE';break
            seen.add(key);prices=[math.ceil(F(lam[j])*scale) for j in active]
            score=sum((F(lam[j]) for j in active),F());changed=0;details=[]
            complete_batch=True
            for pos,i in enumerate(opened):
                left=proposal_stop-time.monotonic()
                if left<=.02:complete_batch=False;stop='DEADLINE';break
                allowance=min(call_cap,left/(len(opened)-pos))
                try:p=pricing(grids[i],prices,demands,caps[i],time_limit_s=allowance,
                              max_routes=100,ng_size=ng_size)
                except (RuntimeError,ValueError,OSError) as exc:
                    report['unavailable']=str(exc);stop='PROPOSAL_UNAVAILABLE'
                    complete_batch=False;break
                report['pricing_calls']+=1
                if time.monotonic()>=proposal_stop:
                    complete_batch=False;stop='DEADLINE';break
                h=F(*capacity_price_lower_bound(ctx,i,lam)['lower_exact'])
                # This number ranks proposals only; it is never exported as a
                # physical certificate, nor substituted for original pricing.
                if p.get('lb_certified') and p.get('min_rc') is not None:
                    h=max(h,F(int(p['min_rc']),scale))
                score+=min(F(),h);added=0;repeated=0
                for walk in p.get('routes',()):
                    if time.monotonic()>=proposal_stop:
                        complete_batch=False;stop='DEADLINE';break
                    if (not walk or any(type(v) is not int or not 1<=v<=len(active) for v in walk)
                            or sum(demands[v-1] for v in walk)>caps[i]):
                        raise ValueError('Invalid auxiliary pricing walk')
                    order=tuple(active[v-1] for v in walk);path=(0,*walk,0)
                    cost=F(sum(grids[i][v][w] for v,w in zip(path,path[1:])),scale)
                    added+=int(add(i,order,cost))
                    if len(order)!=len(set(order)):repeated+=1;continue
                    try:r=audit_route(data,node,i,order,source='aux_ng_price_proposal',generation_id=generation_id)
                    except ValueError:continue
                    physical[r.audit_signature]=r
                    if r.audit_signature not in initial:
                        cov=(i,frozenset(order));old=new.get(cov)
                        if old is None or F(*r.cost_exact)<F(*old.cost_exact):new[cov]=r
                changed+=added
                details.append(dict(facility=i,status=p.get('status'),seconds=p.get('seconds'),
                                    added_or_replaced=added,repeated_walks=repeated))
                if not complete_batch:break
            report['trace'].append(dict(round=step,lambda_hash=key,pricing=details,
                actual_column_changes=changed,complete_facility_batch=complete_batch,
                proposal_score_not_certificate=float(score)))
            if complete_batch:options.append((score,lam,key))
            if not complete_batch:break
            if not changed:stop='NO_NEW_COLUMNS';break
    finally:model.dispose()
    # A known original route gives an upper on this fixed dual point, enabling
    # safe rejection of unhelpful candidates without another native solve.
    floor=None if existing_lower is None else F(float(existing_lower))
    chosen=None;physical_groups={i:[] for i in opened}
    for r in physical.values():physical_groups[r.facility_id].append(r)
    for score,lam,key in sorted(options,key=lambda x:x[0],reverse=True):
        if time.monotonic()>=deadline:stop='SCREEN_DEADLINE';break
        upper=sum((F(lam[j]) for j in active),F())
        screened=True
        for i in opened:
            best=F()
            for pos,r in enumerate(physical_groups[i]):
                if pos%16==0 and time.monotonic()>=deadline-.005:
                    screened=False;stop='SCREEN_DEADLINE';break
                best=min(best,F(*r.cost_exact)-sum((F(lam[j]) for j in r.customers_in_order),F()))
            if not screened:break
            upper+=best
        if not screened:break
        if floor is None or upper>floor+F('0.000001'):
            chosen=lam;report['selected_price_hash']=key;break
    report.update(status=stop,options=len(options),physical_route_candidates=len(new),
                  selected=chosen is not None,original_native_calls=0)
    # Keep a compact deterministic physical-route prefix. The master walks and
    # any primal/dual/scalar objective remain private to this helper.
    byfacility={i:[] for i in opened}
    for r in new.values():
        if time.monotonic()>=deadline-.005:break
        cost=F(*r.cost_exact)
        if chosen is not None:cost-=sum((F(chosen[j]) for j in r.customers_in_order),F())
        group=byfacility[r.facility_id]
        group.append((cost,r.customers_in_order,r))
        # Each facility's retained prefix stays bounded during preparation.
        group.sort(key=lambda item:item[:2]);del group[max_return_routes:]
    returned=[]
    for rank in range(max(map(len,byfacility.values()),default=0)):
        for i in opened:
            if rank<len(byfacility[i]) and len(returned)<max_return_routes:
                returned.append(byfacility[i][rank][2])
        if len(returned)>=max_return_routes:break
    return finish(chosen,returned)
