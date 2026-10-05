"""Two-stage stochastic LRP extensive form, independent of the decomposition.

The reviewed Codex formulation/data/audit are copied from the frozen handoff
reference. A/o/h/b commit one complete common facility plan; all alpha/e/u/r/nu
are second-stage recourse. This file does not import the reference package or
any forward/backward model builder. See artifacts/ef_replacement for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations
from math import fsum
from typing import Any
import hashlib
import json
import textwrap
import time

import numpy as np
from scipy import sparse

@dataclass
class Instance:
    name: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        H, S, n = self.arrays['active'].shape
        m, L = self.arrays['opening_cost'].shape
        return m, n, H, L, S

    def validate(self) -> None:
        a = self.arrays
        required = ['active', 'demand', 'outsourcing_cost', 'capacity', 'opening_cost',
                    'continuation_cost', 'closing_cost', 'location_periods',
                    'period_to_interval', 'min_open', 'scenario_prob']
        for key in required:
            if key not in a:
                raise ValueError(f'Missing array {key}')
        if a['active'].ndim != 3 or a['opening_cost'].ndim != 2:
            raise ValueError('active must have shape [T,S,J], opening_cost [I,L]')
        m,n,H,L,S = self.shape
        if min(m,n,H,L,S) < 1:
            raise ValueError('All dimensions must be positive')
        shapes = {'demand':(H,S,n), 'outsourcing_cost':(H,S,n), 'capacity':(m,H),
                  'continuation_cost':(m,L), 'closing_cost':(m,L), 'location_periods':(L,),
                  'period_to_interval':(H,), 'min_open':(L,), 'scenario_prob':(S,)}
        for k,sh in shapes.items():
            if a[k].shape != sh:
                raise ValueError(f'{k}: expected {sh}, received {a[k].shape}')
        for k in required:
            if not np.isfinite(a[k]).all() or (a[k] < 0).any():
                raise ValueError(f'{k} must be finite/nonnegative')
        for k in ['active','location_periods','period_to_interval','min_open']:
            if not np.equal(a[k], np.floor(a[k])).all():
                raise ValueError(f'{k} must be integral')
        if not np.isin(a['active'],[0,1]).all():
            raise ValueError('Activity must be binary')
        if np.any(a['demand'][a['active']==0] != 0):
            raise ValueError('An inactive customer cannot have positive demand')
        tl = a['location_periods']
        if tl[0] != 1 or np.any(np.diff(tl)<=0) or tl[-1]>H:
            raise ValueError('location_periods must start at 1 and increase within 1..T')
        expected = np.array([max(k for k,ell in enumerate(tl) if ell <= t)
                             for t in range(1,H+1)],dtype=int)
        if not np.array_equal(a['period_to_interval'],expected):
            raise ValueError('Incorrect t -> facility interval mapping')
        if (a['min_open']>m).any():
            raise ValueError('Minimum available-facility count exceeds candidate count')
        p=a['scenario_prob']
        if (p<=0).any() or abs(float(p.sum())-1)>1e-12:
            raise ValueError('Positive scenario probabilities must sum to ONE; no silent normalization')
        if 'initial_state' in a:
            if a['initial_state'].shape!=(m,) or np.any(a['initial_state']!=0):
                raise ValueError('This model assumes all facilities initially closed')
        if np.any(a['continuation_cost'][:,0]!=0) or np.any(a['closing_cost'][:,0]!=0):
            raise ValueError('First-interval continuation/closure costs must be stored as zero')
        flags=self.metadata.get('model_flags',{})
        for key in ['allow_outsourcing','allow_idle','enforce_facility_capacity']:
            if key in flags and flags[key] is not True:
                raise ValueError(f'This is the agreed outsourcing+idle+capacity model, but {key} is not true. '
                                 'Refusing silent conversion of deterministic-reference data.')
        if 'stages' in flags and flags['stages']!=2:
            raise ValueError('Exactly two information stages are required')
        if 'route_cost' in a:
            c=a['route_cost']
            if c.shape!=(H,m,n+1,n+1):
                raise ValueError('route_cost must have shape [T,I,J+1,J+1]')
        else:
            if a.get('cost_fc',np.zeros(0)).shape!=(H,m,n) or a.get('cost_cc',np.zeros(0)).shape!=(H,n,n):
                raise ValueError('Expected cost_fc[T,I,J] and cost_cc[T,J,J]')
            c=self.route_costs()
        if not np.isfinite(c).all() or (c<0).any() or np.any(np.diagonal(c,axis1=-2,axis2=-1)!=0):
            raise ValueError('Route costs must be finite/nonnegative, with zero diagonal')
        for key,dim in [('facility_ids',m),('customer_ids',n)]:
            if key in a and (a[key].shape!=(dim,) or len(set(a[key].tolist()))!=dim):
                raise ValueError(f'Invalid physical IDs: {key}')
        if 'facility_ids' in a and 'customer_ids' in a:
            if set(a['facility_ids'].tolist()) & set(a['customer_ids'].tolist()):
                raise ValueError('Warehouse and customer physical IDs overlap')

    def route_costs(self) -> np.ndarray:
        a=self.arrays
        if 'route_cost' in a:
            return a['route_cost']
        m,n,H,L,S=self.shape
        out=np.zeros((H,m,n+1,n+1),dtype=float)
        out[:,:,0,1:]=a['cost_fc']
        out[:,:,1:,0]=a['cost_fc']
        out[:,:,1:,1:]=a['cost_cc'][:,None,:,:]
        return out

    def logical_hash(self) -> str:
        h=hashlib.sha256()
        for k,v in sorted(self.arrays.items()):
            v=np.ascontiguousarray(v)
            h.update(k.encode()); h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode()); h.update(v.tobytes())
        return h.hexdigest()

    def save(self,directory:Path) -> None:
        self.validate(); directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'arrays.npz',**self.arrays)
        meta=dict(self.metadata,name=self.name,logical_sha256=self.logical_hash(),
                  dimensions=dict(zip(['I','J','T','L','S'],self.shape)))
        (directory/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')


def load_instance(directory: str|Path) -> Instance:
    path=Path(directory)
    with np.load(path/'arrays.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    meta=json.loads((path/'metadata.json').read_text())
    d=Instance(meta.get('name',meta.get('config',{}).get('id',path.name)),arrays,meta)
    d.validate()
    return d


@dataclass
class LinearMILP:
    names:list[str]=field(default_factory=list)
    cost:list[float]=field(default_factory=list)
    lower:list[float]=field(default_factory=list)
    upper:list[float]=field(default_factory=list)
    integer:list[int]=field(default_factory=list)
    rows:list[dict[int,float]]=field(default_factory=list)
    row_names:list[str]=field(default_factory=list)
    row_lb:list[float]=field(default_factory=list)
    row_ub:list[float]=field(default_factory=list)
    groups:dict[str,dict[tuple[int,...],int]]=field(default_factory=dict)
    connectivity:str='mtz'

    def var(self,group:str,key:tuple[int,...],cost:float=0,ub:float=1,integer:bool=True)->int:
        ix=len(self.names)
        self.names.append(group+'_'+'_'.join(map(str,key)))
        self.cost.append(float(cost)); self.lower.append(0.); self.upper.append(float(ub))
        self.integer.append(int(integer)); self.groups.setdefault(group,{})[key]=ix
        return ix

    def row(self,name:str,terms:list[tuple[int,float]],lb:float=-np.inf,ub:float=np.inf)->None:
        d:dict[int,float]={}
        for col,val in terms: d[col]=d.get(col,0.)+float(val)
        d={c:v for c,v in d.items() if v!=0}
        self.rows.append(d); self.row_names.append(name)
        self.row_lb.append(float(lb)); self.row_ub.append(float(ub))

    def matrix(self)->sparse.csc_matrix:
        rr=[]; cc=[]; vv=[]
        for r,row in enumerate(self.rows):
            for c,v in row.items(): rr.append(r);cc.append(c);vv.append(v)
        return sparse.csc_matrix((vv,(rr,cc)),shape=(len(self.rows),len(self.names)))

    def save_matrix(self,path:Path)->None:
        mat=self.matrix()
        np.savez_compressed(path,objective=self.cost,lb=self.lower,ub=self.upper,
                            integrality=self.integer,data=mat.data,indices=mat.indices,
                            indptr=mat.indptr,shape=mat.shape,row_lb=self.row_lb,
                            row_ub=self.row_ub,var_names=np.array(self.names),
                            row_names=np.array(self.row_names))

    def write_lp(self,path:Path)->None:
        """Solver-readable plain LP export, without requiring a Gurobi installation."""
        def expr(row:dict[int,float])->str:
            parts=[]
            for c,v in row.items():
                parts.append(f'{"+" if v>=0 else "-"} {abs(v):.17g} {self.names[c]}')
            return ' '.join(parts) or f'0 {self.names[0]}'
        lines=['Minimize',' obj: '+expr({i:c for i,c in enumerate(self.cost) if c}), 'Subject To']
        for name,row,lo,hi in zip(self.row_names,self.rows,self.row_lb,self.row_ub):
            if lo==hi: lines.append(f' {name}: {expr(row)} = {lo:.17g}')
            else:
                if np.isfinite(lo): lines.append(f' {name}_lb: {expr(row)} >= {lo:.17g}')
                if np.isfinite(hi): lines.append(f' {name}_ub: {expr(row)} <= {hi:.17g}')
        lines.append('Bounds')
        for name,lo,hi in zip(self.names,self.lower,self.upper):
            lines.append(f' {lo:.17g} <= {name} <= {hi:.17g}')
        lines.append('Binary')
        lines += [' '+name for name,it in zip(self.names,self.integer) if it]
        lines.append('End')
        # Gurobi LP format limits a physical line to 999 characters.
        # Wrap only at whitespace, never inside a variable name or numeric token.
        wrapped=[]
        for line in lines:
            wrapped.extend(textwrap.wrap(line,width=120,subsequent_indent='   ',
                break_long_words=False,break_on_hyphens=False) or [''])
        path.write_text('\n'.join(wrapped)+'\n')


def build_ef(d:Instance,connectivity:str='mtz')->LinearMILP:
    d.validate(); m,n,H,L,S=d.shape; a=d.arrays; C=d.route_costs()
    if connectivity not in {'mtz','cutset'}: raise ValueError('Choose mtz or cutset')
    if connectivity=='cutset' and n>8: raise ValueError('Full cutset enumeration is restricted to J<=8')
    M=LinearMILP(connectivity=connectivity)
    for i in range(m):
        for k in range(L):
            M.var('A',(i,k)); M.var('o',(i,k),a['opening_cost'][i,k])
            M.var('h',(i,k),a['continuation_cost'][i,k] if k else 0.)
            M.var('b',(i,k),a['closing_cost'][i,k] if k else 0.)
    A,o,h,b=(M.groups[g] for g in ['A','o','h','b'])
    for i in range(m):
        for k in range(L):
            prev=[] if k==0 else [(A[i,k-1],-1)]
            M.row(f'F2_previous_{i}_{k}',[(h[i,k],1)]+prev,ub=0)
            M.row(f'F2_current_{i}_{k}',[(h[i,k],1),(A[i,k],-1)],ub=0)
            M.row(f'F2_lower_{i}_{k}',[(h[i,k],1),(A[i,k],-1)]+prev,lb=-1)
            M.row(f'F3_open_{i}_{k}',[(o[i,k],1),(A[i,k],-1),(h[i,k],1)],lb=0,ub=0)
            terms=[(b[i,k],1),(h[i,k],1)]+prev
            M.row(f'F3_close_{i}_{k}',terms,lb=0,ub=0)
    for k in range(L):
        M.row(f'F4_minimum_{k}',[(A[i,k],1) for i in range(m)],lb=a['min_open'][k])
    # p_s multiplies every operating cost exactly ONCE. No division by H or m.
    for t in range(H):
        for s in range(S):
            p=float(a['scenario_prob'][s])
            for j in range(n): M.var('e',(j,t,s),p*a['outsourcing_cost'][t,s,j])
            for i in range(m):
                M.var('u',(i,t,s))
                for j in range(n):
                    M.var('alpha',(i,j,t,s))
                    if connectivity=='mtz': M.var('nu',(i,j,t,s),ub=n,integer=False)
                for v in range(n+1):
                    for w in range(n+1):
                        if v!=w: M.var('r',(i,v,w,t,s),p*C[t,i,v,w])
    alpha,e,u,r=(M.groups[g] for g in ['alpha','e','u','r'])
    nu=M.groups.get('nu',{})
    for t in range(H):
        k=int(a['period_to_interval'][t])
        for s in range(S):
            for j in range(n):
                M.row(f'R1_service_{j}_{t}_{s}',[(alpha[i,j,t,s],1) for i in range(m)]+[(e[j,t,s],1)],
                      lb=a['active'][t,s,j],ub=a['active'][t,s,j])
            for i in range(m):
                tag=f'{i}_{t}_{s}'; ui=u[i,t,s]
                M.row('R2_available_'+tag,[(ui,1),(A[i,k],-1)],ub=0)
                M.row('R2_nonempty_'+tag,[(ui,1)]+[(alpha[i,j,t,s],-1) for j in range(n)],ub=0)
                M.row('R3_capacity_'+tag,[(alpha[i,j,t,s],a['demand'][t,s,j]) for j in range(n)]
                      +[(ui,-a['capacity'][i,t])],ub=0)
                M.row('R5_root_out_'+tag,[(r[i,0,j+1,t,s],1) for j in range(n)]+[(ui,-1)],lb=0,ub=0)
                M.row('R5_root_in_'+tag,[(r[i,j+1,0,t,s],1) for j in range(n)]+[(ui,-1)],lb=0,ub=0)
                for j in range(n):
                    aj=alpha[i,j,t,s]; v=j+1
                    M.row(f'R2_assign_{i}_{j}_{t}_{s}',[(aj,1),(ui,-1)],ub=0)
                    M.row(f'R4_out_{i}_{j}_{t}_{s}',[(r[i,v,w,t,s],1) for w in range(n+1) if w!=v]+[(aj,-1)],lb=0,ub=0)
                    M.row(f'R4_in_{i}_{j}_{t}_{s}',[(r[i,w,v,t,s],1) for w in range(n+1) if w!=v]+[(aj,-1)],lb=0,ub=0)
                    if connectivity=='mtz':
                        M.row(f'R6_lb_{i}_{j}_{t}_{s}',[(nu[i,j,t,s],1),(aj,-1)],lb=0)
                        M.row(f'R6_ub_{i}_{j}_{t}_{s}',[(nu[i,j,t,s],1),(aj,-n)],ub=0)
                if connectivity=='mtz':
                    for j in range(n):
                        for q in range(n):
                            if j!=q:
                                M.row(f'R7_mtz_{i}_{j}_{q}_{t}_{s}',[(nu[i,j,t,s],1),(nu[i,q,t,s],-1),
                                      (r[i,j+1,q+1,t,s],n+1)],ub=n)
                else:
                    for size in range(1,n+1):
                        for subset in combinations(range(n),size):
                            nodes={j+1 for j in subset}
                            cut=[(r[i,v,w,t,s],1) for v in nodes for w in range(n+1) if w not in nodes]
                            sid=sum(1<<j for j in subset)
                            for j in subset:
                                M.row(f'SEC_{i}_{sid}_{j}_{t}_{s}',cut+[(alpha[i,j,t,s],-1)],lb=0)
    return M


def build_gurobi(M:LinearMILP,*,env=None):
    """Build a real gurobipy.Model, with one-to-one coefficient mapping.

    This function raises ImportError if Gurobi is not installed. It NEVER falls
    back to HiGHS. env can be a user-configured, authorized gp.Env.
    """
    import gurobipy as gp
    model=gp.Model('two_stage_lrp',env=env)
    variables=[]
    for name,c,lo,hi,it in zip(M.names,M.cost,M.lower,M.upper,M.integer):
        variables.append(model.addVar(lb=lo,ub=hi,obj=c,vtype=gp.GRB.BINARY if it else gp.GRB.CONTINUOUS,name=name))
    model.ModelSense=gp.GRB.MINIMIZE
    for name,row,lo,hi in zip(M.row_names,M.rows,M.row_lb,M.row_ub):
        expr=gp.LinExpr(list(row.values()),[variables[j] for j in row])
        if lo==hi: model.addConstr(expr==lo,name=name)
        else:
            if np.isfinite(lo): model.addConstr(expr>=lo,name=name+'_lb')
            if np.isfinite(hi): model.addConstr(expr<=hi,name=name+'_ub')
    model.update()
    return model,variables


def audit_solution(d:Instance,M:LinearMILP,x:np.ndarray,solver_objective:float,
                   enumeration:dict[str,Any]|None=None,tol:float=2e-6)->dict[str,Any]:
    d.validate();m,n,H,L,S=d.shape;a=d.arrays
    errors=[]
    def require(ok:bool,message:str):
        if not ok: errors.append(message)
    x=np.asarray(x,float)
    if x.shape!=(len(M.names),) or not np.isfinite(x).all():
        return {'passed':False,'errors':['Missing, non-finite, or wrong-sized primal vector']}
    integrality=max([abs(x[k]-round(x[k])) for k,it in enumerate(M.integer) if it] or [0.])
    boundvio=max(float(np.max(np.asarray(M.lower)-x)),float(np.max(x-np.asarray(M.upper))),0.)
    lhs=M.matrix()@x
    rowvio=max(float(np.max(np.asarray(M.row_lb)-lhs)),float(np.max(lhs-np.asarray(M.row_ub))),0.)
    require(integrality<=tol,f'Integrality violation {integrality}')
    require(boundvio<=tol,f'Bound violation {boundvio}')
    require(rowvio<=tol,f'Constraint violation {rowvio}')
    def val(group,*key): return int(round(x[M.groups[group][tuple(key)]]))
    A=np.array([[val('A',i,k) for k in range(L)] for i in range(m)],int)
    charges={'opening':0.,'continuation':0.,'closing':0.}; events=[]
    for i in range(m):
        prev=0
        for k in range(L):
            curr=int(A[i,k]); expected_h=prev*curr
            require(val('h',i,k)==expected_h,f'h[{i},{k}] is not true continuation')
            require(val('o',i,k)==int(prev==0 and curr==1),f'o[{i},{k}] is not true opening')
            require(val('b',i,k)==int(prev==1 and curr==0),f'b[{i},{k}] is not true closure')
            event='closed';cost=0.
            if curr and not prev: event='opening';cost=float(a['opening_cost'][i,k])
            elif curr and prev: event='continuation';cost=float(a['continuation_cost'][i,k])
            elif prev and not curr: event='closing';cost=float(a['closing_cost'][i,k])
            if event in charges: charges[event]+=cost
            events.append({'facility_index':i,'interval':k,'date':int(a['location_periods'][k]),
                           'previous':prev,'current':curr,'event':event,'cost':cost})
            prev=curr
    for k in range(L): require(A[:,k].sum()>=a['min_open'][k],f'Min-open violation at k={k}')
    # Audit costs independently of the model's weighted coefficients.
    def arc_cost(i,t,v,w):
        if 'route_cost' in a: return float(a['route_cost'][t,i,v,w])
        if v==0: return float(a['cost_fc'][t,i,w-1])
        if w==0: return float(a['cost_fc'][t,i,v-1])
        return float(a['cost_cc'][t,v-1,w-1])
    nodes=[]; all_routes=0; singletons=0; positive_cap_tight=0
    for t in range(H):
        k=max(kk for kk,ell in enumerate(a['location_periods']) if int(ell)<=t+1)
        for s in range(S):
            assigned=[[j for j in range(n) if val('alpha',i,j,t,s)] for i in range(m)]
            outsourced=[j for j in range(n) if val('e',j,t,s)]
            for j in range(n):
                served=sum(j in g for g in assigned)+int(j in outsourced)
                require(served==int(a['active'][t,s,j]),f'Service error at {(j,t,s)}')
            routing=0.; routes=[]; used=[]; loads=[]
            for i,g in enumerate(assigned):
                ui=val('u',i,t,s);used.append(ui)
                require(ui==int(bool(g)),f'Dispatch/nonempty mismatch at {(i,t,s)}')
                require(not g or bool(A[i,k]),f'Using closed facility at {(i,t,s)}')
                load=fsum(float(a['demand'][t,s,j]) for j in g);loads.append(load)
                require(load<=float(a['capacity'][i,t])+tol,f'Warehouse capacity violation at {(i,t,s)}')
                if ui and a['capacity'][i,t]>0 and abs(load-a['capacity'][i,t])<tol: positive_cap_tight+=1
                arcs=[(v,w) for v in range(n+1) for w in range(n+1)
                      if v!=w and val('r',i,v,w,t,s)]
                # Degree audit does not use MTZ or cutset constraints.
                for v in range(n+1):
                    expected=ui if v==0 else int(v-1 in g)
                    require(sum(1 for z,w in arcs if z==v)==expected,f'Out-degree mismatch {(i,v,t,s)}')
                    require(sum(1 for z,w in arcs if w==v)==expected,f'In-degree mismatch {(i,v,t,s)}')
                if not g:
                    require(not arcs,f'An idle facility has arcs {(i,t,s)}');route=[]
                else:
                    route=[0]; current=0; seen=set(); traversed=[]
                    for _ in range(n+2):
                        successors=[w for v,w in arcs if v==current]
                        if len(successors)!=1: break
                        nxt=successors[0];traversed.append((current,nxt));route.append(nxt)
                        if nxt==0: break
                        if nxt in seen: break
                        seen.add(nxt);current=nxt
                    require(route[-1]==0 and set(route[1:-1])=={j+1 for j in g}
                            and len(route[1:-1])==len(g) and set(traversed)==set(arcs),
                            f'Disconnected/repeated/wrong-root route {(i,t,s)}')
                    all_routes+=1;singletons+=int(len(g)==1)
                routes.append(route)
                routing+=fsum(arc_cost(i,t,v,w) for v,w in arcs)
            outsourcing=fsum(float(a['outsourcing_cost'][t,s,j]) for j in outsourced)
            node={'period':t,'scenario':s,'probability':float(a['scenario_prob'][s]),
                  'available':A[:,k].tolist(),'active':[int(j) for j in range(n) if a['active'][t,s,j]],
                  'assigned':assigned,'outsourced':outsourced,'dispatch':used,'loads':loads,
                  'capacities':a['capacity'][:,t].tolist(),'routes':routes,
                  'routing_cost':routing,'outsourcing_cost':outsourcing,'total':routing+outsourcing}
            if enumeration is not None:
                mask=sum(int(A[i,k])<<i for i in range(m))
                reference=float(enumeration['node_value_tables'][f'{t},{s}'][mask])
                node['exact_recourse_for_returned_A']=reference
                node['node_optimality_gap']=node['total']-reference
                # Entire optimal EF with all positive pi has node-wise optimal recourse.
                require(abs(node['total']-reference)<=tol*max(1.,reference),f'Node recourse not optimal for returned A {(t,s)}')
            nodes.append(node)
    facility=fsum(charges.values())
    er=fsum(z['probability']*z['routing_cost'] for z in nodes)
    eo=fsum(z['probability']*z['outsourcing_cost'] for z in nodes)
    recalculated=facility+er+eo; algebraic=float(np.dot(M.cost,x))
    require(abs(recalculated-solver_objective)<=tol*max(1,abs(recalculated)),
            f'Original-cost objective {recalculated} differs from solver {solver_objective}')
    require(abs(algebraic-solver_objective)<=tol*max(1,abs(algebraic)),
            f'Coefficient objective {algebraic} differs from solver {solver_objective}')
    enum_error=None
    if enumeration is not None:
        enum_error=abs(recalculated-enumeration['objective'])
        require(enum_error<=tol*max(1,abs(recalculated)),f'Global enumeration disagreement {enum_error}')
        masks=[sum(int(A[i,k])<<i for i in range(m)) for k in range(L)]
        true_plan_cost=facility+fsum(float(a['scenario_prob'][s])*
                        enumeration['node_value_tables'][f'{t},{s}'][masks[max(kk for kk,ell in enumerate(a['location_periods']) if ell<=t+1)]]
                        for t in range(H) for s in range(S))
        require(abs(true_plan_cost-enumeration['objective'])<=tol*max(1,abs(recalculated)),
                'Returned facility plan is not an enumerated optimal plan')
    expected_active=fsum(z['probability']*len(z['active']) for z in nodes)
    expected_out=fsum(z['probability']*len(z['outsourced']) for z in nodes)
    return {'passed':not errors,'errors':errors,'A':A.tolist(),'facility_events':events,
            'facility_costs':charges,'facility_cost':facility,'expected_routing_cost':er,
            'expected_outsourcing_cost':eo,'recomputed_objective':recalculated,
            'matrix_objective':algebraic,'objective_difference':abs(recalculated-solver_objective),
            'enumeration_objective_difference':enum_error,'integrality_violation':integrality,
            'bound_violation':boundvio,'constraint_violation':rowvio,
            'nonempty_routes':all_routes,'singleton_routes':singletons,
            'positive_capacity_tight_routes':positive_cap_tight,
            'expected_active_orders':expected_active,'expected_outsourced_orders':expected_out,
            'outsourcing_fraction':expected_out/expected_active if expected_active else 0.,'nodes':nodes}


def check_gurobi_matrix(spec, model, variables):
    """Read the actual Gurobi model back before optimizing the copied EF."""
    import gurobipy as gp
    model.update()
    if model.NumVars != len(spec.names) or model.NumConstrs != len(spec.rows):
        raise AssertionError("Gurobi adapter changed EF dimensions")
    if model.getAttr("VarName", variables) != spec.names:
        raise AssertionError("Gurobi variable order differs from EF specification")
    for attr, expected in (("Obj", spec.cost), ("LB", spec.lower), ("UB", spec.upper)):
        if not np.allclose(model.getAttr(attr, variables), expected, rtol=0, atol=1e-12):
            raise AssertionError(f"Gurobi {attr} differs from EF specification")
    expected_types = [gp.GRB.BINARY if flag else gp.GRB.CONTINUOUS for flag in spec.integer]
    if model.getAttr("VType", variables) != expected_types or model.ModelSense != gp.GRB.MINIMIZE:
        raise AssertionError("Gurobi types or objective sense differ from EF specification")
    difference = (model.getA().tocsr() - spec.matrix().tocsr()).tocoo()
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise AssertionError("Gurobi row coefficients differ from EF specification")
    for row, lo, hi in zip(model.getConstrs(), spec.row_lb, spec.row_ub):
        if lo != hi and np.isfinite(lo) and np.isfinite(hi):
            raise ValueError("This EF adapter supports equality or one-sided rows only")
        sense = "=" if lo == hi else ">" if np.isfinite(lo) else "<"
        rhs = lo if np.isfinite(lo) else hi
        if row.Sense != sense or abs(row.RHS - rhs) > 1e-12:
            raise AssertionError("Gurobi row sense or RHS differs from EF specification")
    return True


class ExtensiveModelBuilder:
    """Independent two-stage LRP EF copied from the verified Codex reference.

    Supply an instance directory or an object exposing ``arrays`` and
    ``metadata``. The complete scenario trajectories are already in the input;
    a decomposition tree is neither needed nor accepted. ``build()`` returns
    a real Gurobi model. The caller owns and must dispose the returned model.
    """

    information_stages = 2

    def __init__(self, prob_data, *, connectivity="mtz", env=None):
        import copy
        if isinstance(prob_data, (str, Path)):
            prob_data = load_instance(prob_data)
        if not hasattr(prob_data, "arrays") or not hasattr(prob_data, "metadata"):
            raise TypeError(
                "ExtensiveModelBuilder now requires two-stage LRP arrays/metadata "
                "or an instance directory. Investment VRP data and its scenario "
                "tree are not LRP inputs; migrate the comparison caller separately."
            )
        self.prob_data = Instance(
            getattr(prob_data, "name", "lrp"),
            {key: np.array(value, copy=True) for key, value in prob_data.arrays.items()},
            copy.deepcopy(prob_data.metadata),
        )
        self.prob_data.validate()
        self.connectivity = connectivity
        self.env = env
        self.spec = build_ef(self.prob_data, connectivity)

    @staticmethod
    def _check_no_tree(second_stage_nodes):
        if second_stage_nodes is not None:
            raise TypeError(
                "The two-stage LRP EF reads full scenarios from arrays. "
                "Call build()/solve() without decomposition stage nodes."
            )

    def build(self, second_stage_nodes=None):
        self._check_no_tree(second_stage_nodes)
        model, variables = build_gurobi(self.spec, env=self.env)
        try:
            model.Params.OutputFlag = 0
            check_gurobi_matrix(self.spec, model, variables)
            model._lrp_spec = self.spec
            model._lrp_variables = variables
            return model
        except BaseException:
            model.dispose()
            raise

    def variables(self, model, group):
        """Expose e.g. ``builder.variables(model, 'A')[i,k]`` without renaming."""
        return {key: model._lrp_variables[index]
                for key, index in self.spec.groups[group].items()}

    def certify_rounded_incumbent(self, model, second_stage_nodes=None, *, enumeration=None):
        """Audit the returned plan/routes directly; never reoptimize a policy."""
        self._check_no_tree(second_stage_nodes)
        if model.SolCount < 1:
            raise ValueError("EF model has no incumbent to audit")
        variables = model.getVars()
        if model.getAttr("VarName", variables) != self.spec.names:
            raise ValueError("EF model variable mapping differs from this builder")
        primal = np.array(model.getAttr("X", variables), dtype=float)
        audit = audit_solution(self.prob_data, self.spec, primal, float(model.ObjVal), enumeration)
        if not audit["passed"]:
            raise ValueError(f"EF raw solution audit failed: {audit['errors']}")
        # Keep the result name used by existing EF reporting, but the LRP
        # policy is certified by the independent raw-data audit above.
        from fractions import Fraction
        import math
        # Compute a directed upper endpoint from the actual discrete policy.
        # The semantic p*cost products and the stored matrix coefficients can
        # round differently, so bound both without using the solver's LB.
        a = self.prob_data.arrays
        semantic = Fraction(0)
        for event in audit["facility_events"]:
            semantic += Fraction.from_float(event["cost"])
        costs = self.prob_data.route_costs()
        for node in audit["nodes"]:
            t, s = node["period"], node["scenario"]
            local = sum((Fraction.from_float(float(a["outsourcing_cost"][t, s, j]))
                         for j in node["outsourced"]), Fraction(0))
            for i, route in enumerate(node["routes"]):
                local += sum((Fraction.from_float(float(costs[t, i, v, w]))
                              for v, w in zip(route, route[1:])), Fraction(0))
            semantic += Fraction.from_float(float(a["scenario_prob"][s])) * local
        matrix = sum((Fraction.from_float(cost) * int(round(primal[index]))
                      for index, cost in enumerate(self.spec.cost) if cost), Fraction(0))
        exact = max(semantic, matrix)
        upper = float(exact)
        if Fraction.from_float(upper) < exact:
            upper = math.nextafter(upper, math.inf)
        audit["objective_ub"] = max(upper, float(model.ObjVal))
        audit["certified_nodes"] = len(audit["nodes"])
        return audit

    def rounded_capacity_violation(self, model, second_stage_nodes=None):
        audit = self.certify_rounded_incumbent(model, second_stage_nodes)
        return max([0.0] + [
            load - capacity * dispatch
            for node in audit["nodes"]
            for load, capacity, dispatch in zip(node["loads"], node["capacities"], node["dispatch"])
        ])

    def solve(self, second_stage_nodes=None, time_limit=120.0, mip_gap=0.0, *,
              threads=1, seed=0, output=False, log_path=None, model_path=None):
        import math
        import gurobipy as gp
        from types import SimpleNamespace
        from core.solver_bounds import certified_gurobi_minimization_lower_bound

        self._check_no_tree(second_stage_nodes)
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive and finite, or None")
        if not math.isfinite(mip_gap) or mip_gap < 0:
            raise ValueError("mip_gap must be finite and nonnegative")
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        model = self.build()
        started = time.perf_counter()
        try:
            model.Params.OutputFlag = int(output or log_path is not None)
            model.Params.LogToConsole = int(output)
            if log_path is not None:
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                model.Params.LogFile = str(log_path)
            if time_limit is not None:
                model.Params.TimeLimit = float(time_limit)
            model.Params.MIPGap = float(mip_gap)
            model.Params.MIPGapAbs = 1e-8
            model.Params.FeasibilityTol = 1e-8
            model.Params.IntFeasTol = 1e-8
            model.Params.OptimalityTol = 1e-8
            model.Params.Threads = threads
            model.Params.Seed = seed
            if model_path is not None:
                model_path = Path(model_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.write(str(model_path))
            model.optimize()
            status = int(model.Status)
            has = model.SolCount > 0
            audit = self.certify_rounded_incumbent(model) if has else None
            raw_objective = float(model.ObjVal) if has else None
            objective = audit["objective_ub"] if audit else None
            bound = certified_gurobi_minimization_lower_bound(model)
            if bound is not None and objective is not None and bound > objective:
                bound = None
            bound_source = "ObjBound" if bound is not None else "analytic_nonnegative_cost"
            if bound is None and status not in (gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED,
                                               gp.GRB.INF_OR_UNBD, gp.GRB.NUMERIC):
                try:
                    candidate = certified_gurobi_minimization_lower_bound(SimpleNamespace(
                        Status=status, ObjBound=float(model.ObjBoundC),
                        SolCount=model.SolCount, ObjVal=raw_objective))
                except (AttributeError, gp.GurobiError, TypeError):
                    candidate = None
                if candidate is not None and (objective is None or candidate <= objective):
                    bound, bound_source = candidate, "ObjBoundC"
            # All declared costs are nonnegative; 0 remains a valid lower
            # bound when the solver cannot provide a usable certificate.
            bound = max(0.0, bound) if bound is not None else 0.0
            if objective is not None and bound > objective:
                raise ValueError("EF solver lower bound exceeds audited policy cost")
            gap = (objective - bound) / max(1.0, abs(objective)) if has else None
            # Gurobi OPTIMAL can merely mean the requested relative gap was
            # met. Independent benchmark certification always stays strict.
            optimal = status == gp.GRB.OPTIMAL and has and gap <= 1e-7
            names = {gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.TIME_LIMIT: "TIME_LIMIT",
                     gp.GRB.INFEASIBLE: "INFEASIBLE", gp.GRB.UNBOUNDED: "UNBOUNDED",
                     gp.GRB.INF_OR_UNBD: "INF_OR_UNBD", gp.GRB.NUMERIC: "NUMERIC"}
            primal = np.array(model.getAttr("X", model._lrp_variables)) if has else None
            def finite_attr(name):
                try:
                    value = float(getattr(model, name))
                    return value if math.isfinite(value) and abs(value) < 1e99 else None
                except (AttributeError, gp.GurobiError):
                    return None
            return {
                "solver": "Gurobi", "gurobi_version": ".".join(map(str, gp.gurobi.version())),
                "information_stages": 2, "instance": self.prob_data.name,
                "instance_sha256": self.prob_data.logical_hash(), "connectivity": self.connectivity,
                "status": names.get(status, str(status)), "status_code": status,
                "objective": objective, "objective_bound": bound, "bound_source": bound_source,
                "recomputed_objective": audit["recomputed_objective"] if audit else None,
                "requested_mip_gap": mip_gap, "certification_relative_tolerance": 1e-7,
                "mip_gap": finite_attr("MIPGap") if has else None,
                "solver_runtime": float(model.Runtime), "wall_seconds": time.perf_counter() - started,
                "variables": model.NumVars, "constraints": model.NumConstrs,
                "matrix_roundtrip_passed": True, "audit_passed": bool(audit and audit["passed"]),
                "optimality_certified": bool(optimal), "automatic_solver_fallback": False,
                "primal": primal, "solution": audit, "model": model,
                "ObjVal": objective, "RawObjVal": raw_objective,
                "ObjBound": bound, "RawObjBound": finite_attr("ObjBound"),
                "RawObjBoundC": finite_attr("ObjBoundC"), "Gap": None if gap is None else 100 * gap,
                "Status": status, "Runtime": float(model.Runtime),
                "RoundedPolicyCertified": bool(audit), "OptimalityCertified": bool(optimal),
                "CertifiedNodeCount": len(audit["nodes"]) if audit else 0,
            }
        except BaseException:
            model.dispose()
            raise
