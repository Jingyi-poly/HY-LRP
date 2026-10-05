"""Shared LRP data, matrix, domain and certification contracts.

Transcribed from the reviewed lrp_forward_backward_gurobi bundle. The existing
stage_builder/subproblem_builder files own the layer models. No debug algorithm,
old VRP constraints, or runtime reference-package imports are used here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
import time
import textwrap
import numpy as np
from scipy import sparse

SCHEMA = 'lrp_two_information_stages_forward_backward_v1'

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
        if not isinstance(flags, dict):
            raise ValueError('model_flags must be an object')
        for key in ['allow_outsourcing','allow_idle','enforce_facility_capacity']:
            if key in flags and flags[key] is not True:
                raise ValueError(f'This is the agreed outsourcing+idle+capacity model, but {key} is not true. '
                                 'Refusing silent conversion of deterministic-reference data.')
        if 'stages' in flags and (type(flags['stages']) is not int or flags['stages'] != 2):
            raise ValueError('Exactly two information stages are required')
        # Fail closed for known declarations which the reference EF does NOT implement.
        for key in ['cross_period_recourse_coupling', 'cross_scenario_assignment_commitment',
                    'cross_period_assignment_commitment', 'additional_vehicle_capacity',
                    'outsourcing_uses_own_capacity', 'limited_outsourcing', 'route_duration_limit',
                    'force_terminal_closure', 'allow_split_orders']:
            if key in flags and flags[key] is not False:
                raise ValueError(f'Unsupported model semantics: {key} must be false')
        if 'vehicles_per_facility_period' in flags and (
                type(flags['vehicles_per_facility_period']) is not int or
                flags['vehicles_per_facility_period'] != 1):
            raise ValueError('At most one route per facility-period is required')
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
        if key in self.groups.get(group, {}):
            raise ValueError(f'Duplicate variable key {group}{key}')
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

    def validate(self)->None:
        """Validate the shared binary/continuous matrix contract before any backend.

        'integer=1' means binary here, NOT an arbitrary general integer. Refuse
        to let HiGHS and Gurobi interpret the same object differently.
        """
        n=len(self.names); nr=len(self.rows)
        if not n or any(len(v)!=n for v in [self.cost,self.lower,self.upper,self.integer]):
            raise ValueError('Invalid variable-vector dimensions')
        if any(len(v)!=nr for v in [self.row_names,self.row_lb,self.row_ub]):
            raise ValueError('Invalid row-vector dimensions')
        if len(set(self.names))!=n or len(set(self.row_names))!=nr:
            raise ValueError('Variable and row names must be unique')
        if not np.isfinite(self.cost).all():
            raise ValueError('Objective coefficients must be finite')
        for lo,hi,it in zip(self.lower,self.upper,self.integer):
            if np.isnan(lo) or np.isnan(hi) or lo>hi or lo==np.inf or hi==-np.inf:
                raise ValueError('Invalid variable bounds')
            if it not in (0,1) or (it and not (0<=lo<=hi<=1)):
                raise ValueError('This matrix contract supports binary or continuous variables only')
        for row,lo,hi in zip(self.rows,self.row_lb,self.row_ub):
            if np.isnan(lo) or np.isnan(hi) or lo>hi or lo==np.inf or hi==-np.inf:
                raise ValueError('Invalid row bounds')
            if not np.isfinite(lo) and not np.isfinite(hi):
                raise ValueError('A row must have at least one finite bound')
            for col,val in row.items():
                if not isinstance(col,(int,np.integer)) or not 0<=col<n or not np.isfinite(val):
                    raise ValueError('Invalid matrix column or coefficient')

    def expanded_rows(self):
        """Canonical row -> native equality/one-sided rows, including ranged rows.

        Returns (source_row_index, native_name, sense, rhs), in native order.
        No implicit Gurobi range slack variables are introduced.
        """
        self.validate(); result=[]
        for idx,(name,lo,hi) in enumerate(zip(self.row_names,self.row_lb,self.row_ub)):
            if lo==hi:
                result.append((idx,name,'=',float(lo)))
            else:
                if np.isfinite(lo): result.append((idx,name+'_lb','>',float(lo)))
                if np.isfinite(hi): result.append((idx,name+'_ub','<',float(hi)))
        if len({item[1] for item in result})!=len(result):
            raise ValueError('Native expanded-row names collide')
        return result

    def matrix(self)->sparse.csc_matrix:
        rr=[]; cc=[]; vv=[]
        for r,row in enumerate(self.rows):
            for c,v in row.items(): rr.append(r);cc.append(c);vv.append(v)
        return sparse.csc_matrix((vv,(rr,cc)),shape=(len(self.rows),len(self.names)))

    def save_matrix(self,path:Path)->None:
        self.validate(); mat=self.matrix()
        np.savez_compressed(path,objective=self.cost,lb=self.lower,ub=self.upper,
                            integrality=self.integer,data=mat.data,indices=mat.indices,
                            indptr=mat.indptr,shape=mat.shape,row_lb=self.row_lb,
                            row_ub=self.row_ub,var_names=np.array(self.names),
                            row_names=np.array(self.row_names))

    def write_lp(self,path:Path)->None:
        """Solver-readable plain LP export, without requiring a Gurobi installation."""
        self.validate()
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



def _validate_solve_options(time_limit, mip_gap, threads=None):
    if np.isnan(time_limit) or time_limit < 0:
        raise ValueError('time_limit must be nonnegative; +inf means no per-solve cap')
    if not np.isfinite(mip_gap) or mip_gap < 0:
        raise ValueError('mip_gap must be finite and nonnegative')
    if threads is not None and (not isinstance(threads, (int,np.integer)) or isinstance(threads,bool) or threads < 1):
        raise ValueError('threads must be a positive integer')



def build_gurobi(M:LinearMILP,*,env=None):
    """Build a genuine gurobipy.Model; NEVER fall back to another backend."""
    M.validate()
    import gurobipy as gp
    model=gp.Model('two_stage_lrp',env=env)
    # Suppress construction messages before addConstr, not after model.update().
    model.Params.OutputFlag = 0
    # Default once at construction; explicit caller tolerances survive optimize.
    model.Params.FeasibilityTol = 1e-8
    model.Params.IntFeasTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    try:
        variables=[]
        for name,c,lo,hi,it in zip(M.names,M.cost,M.lower,M.upper,M.integer):
            variables.append(model.addVar(lb=lo,ub=hi,obj=c,
                vtype=gp.GRB.BINARY if it else gp.GRB.CONTINUOUS,name=name))
        model.ModelSense=gp.GRB.MINIMIZE
        for idx,name,sense,rhs in M.expanded_rows():
            row=M.rows[idx]
            expr=gp.LinExpr(list(row.values()),[variables[j] for j in row])
            if sense=='=': model.addConstr(expr==rhs,name=name)
            elif sense=='>': model.addConstr(expr>=rhs,name=name)
            else: model.addConstr(expr<=rhs,name=name)
        model.update()
        return model,variables
    except Exception:
        model.dispose()
        raise



def audit_gurobi_matrix(M:LinearMILP,model,variables)->dict[str,Any]:
    """Read native coefficients BEFORE optimization; fail on any changed semantics."""
    import gurobipy as gp
    M.validate(); expanded=M.expanded_rows()
    if model.NumVars!=len(M.names) or model.NumConstrs!=len(expanded):
        raise AssertionError('Gurobi adapter changed model dimensions')
    def normalized_bounds(values):
        return np.asarray([np.inf if v>=gp.GRB.INFINITY else
                           -np.inf if v<=-gp.GRB.INFINITY else v for v in values],float)
    for attr,expected in [('Obj',M.cost),('LB',M.lower),('UB',M.upper)]:
        actual=model.getAttr(attr,variables)
        if attr in ('LB','UB'): actual=normalized_bounds(actual)
        if not np.allclose(actual,expected,rtol=0,atol=1e-12):
            raise AssertionError(f'Gurobi {attr} differs from canonical matrix')
    if model.getAttr('VType',variables)!=[gp.GRB.BINARY if flag else gp.GRB.CONTINUOUS for flag in M.integer]:
        raise AssertionError('Gurobi variable types differ from canonical matrix')
    if model.getAttr('VarName',variables)!=M.names:
        raise AssertionError('Gurobi variable order or names changed')
    if model.ModelSense!=gp.GRB.MINIMIZE or abs(model.ObjCon)>1e-12:
        raise AssertionError('Gurobi objective sense or constant changed')
    native=model.getA().tocsr()
    desired=M.matrix().tocsr()[[item[0] for item in expanded],:]
    difference=(native-desired).tocoo()
    residual=float(np.max(np.abs(difference.data))) if difference.nnz else 0.
    if residual>1e-12:
        raise AssertionError('Gurobi constraint coefficients differ from canonical matrix')
    for rr,(_,name,sense,rhs) in zip(model.getConstrs(),expanded):
        if rr.ConstrName!=name or rr.Sense!=sense or abs(rr.RHS-rhs)>1e-12:
            raise AssertionError('Gurobi row name, RHS, or sense mismatch')
    return {'matrix_roundtrip_passed':True,'canonical_constraints':len(M.rows),
            'native_constraints':len(expanded),'matrix_max_abs_difference':residual}



def solve_gurobi(M:LinearMILP,*,time_limit:float=120,mip_gap:float=0.,threads:int=1,
                 log_path:Path|None=None,output:bool=False,model_path:Path|None=None,env=None):
    _validate_solve_options(time_limit,mip_gap,threads)
    import gurobipy as gp
    start=time.perf_counter()
    model,variables=build_gurobi(M,env=env)
    try:
        model.Params.OutputFlag=1 if output or log_path is not None else 0
        model.Params.LogToConsole=int(output)
        if log_path: model.Params.LogFile=str(log_path)
        model.Params.TimeLimit=float(time_limit); model.Params.MIPGap=float(mip_gap)
        model.Params.MIPGapAbs=1e-8; model.Params.FeasibilityTol=1e-8
        model.Params.IntFeasTol=1e-8; model.Params.OptimalityTol=1e-8
        model.Params.Threads=int(threads); model.Params.Seed=0
        roundtrip=audit_gurobi_matrix(M,model,variables)
        if model_path: model.write(str(model_path))
        model.optimize()
        status=int(model.Status); sol_count=int(model.SolCount); has=sol_count>0
        def safe_number(attr):
            try: value=float(model.getAttr(attr))
            except (AttributeError,gp.GurobiError,TypeError,ValueError): return None
            return value if np.isfinite(value) and abs(value)<gp.GRB.INFINITY else None
        status_names=['LOADED','OPTIMAL','INFEASIBLE','INF_OR_UNBD','UNBOUNDED','CUTOFF',
            'ITERATION_LIMIT','NODE_LIMIT','TIME_LIMIT','SOLUTION_LIMIT','INTERRUPTED',
            'NUMERIC','SUBOPTIMAL','INPROGRESS','USER_OBJ_LIMIT','WORK_LIMIT','MEM_LIMIT',
            'LOCALLY_OPTIMAL','LOCALLY_INFEASIBLE']
        mapping={getattr(gp.GRB,k):k for k in status_names if hasattr(gp.GRB,k)}
        no_bound=status in [gp.GRB.INFEASIBLE,gp.GRB.UNBOUNDED,gp.GRB.INF_OR_UNBD]
        x=np.asarray(model.getAttr('X',variables),float) if has else None
        report={'solver':'Gurobi','gurobi_version':'.'.join(map(str,gp.gurobi.version())),
                'status_code':status,'status':mapping.get(status,f'UNKNOWN_{status}'),
                'objective':safe_number('ObjVal') if has else None,
                'objective_bound':None if no_bound else safe_number('ObjBound'),
                'objective_bound_continuous':None if no_bound else safe_number('ObjBoundC'),
                'mip_gap':safe_number('MIPGap') if has else None,
                'solution_count':sol_count,'node_count':safe_number('NodeCount'),
                'wall_seconds':time.perf_counter()-start,'solver_runtime':safe_number('Runtime'),
                'variables':model.NumVars,'constraints':model.NumConstrs,'connectivity':M.connectivity,
                'requested_mip_rel_gap':mip_gap,'requested_mip_abs_gap':1e-8,'threads':threads,
                'IntVio':safe_number('IntVio') if has else None,
                'ConstrVio':safe_number('ConstrVio') if has else None,
                'optimization_completed':True,'gurobi_executed':True,**roundtrip}
        return report,x
    finally:
        model.dispose()



def binary_vector(value: Sequence[float], size: int, label: str) -> tuple[int, ...]:
    v = np.asarray(value)
    if v.shape != (size,) or not np.isfinite(v).all() or not np.isin(v, [0, 1]).all():
        raise ValueError(f'{label} must have shape ({size},) and exact binary values')
    return tuple(int(x) for x in v)



def real_vector(value: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f'{label} must have shape ({size},) and finite values')
    return tuple(float(x) for x in v)



def index(value: int, size: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < size:
        raise ValueError(f'{label} must be an integer in [0,{size})')
    return int(value)



def exact_capacity(demand: Sequence[float], chosen: Sequence[int], capacity: float) -> bool:
    """Compare exact binary representations of input doubles; do not add capacity."""
    return sum((Fraction.from_float(float(d)) for d, b in zip(demand, chosen) if b), Fraction()) <= Fraction.from_float(float(capacity))



def _immutable_array(a: np.ndarray) -> np.ndarray:
    # from bytes prevents a caller from re-enabling WRITEABLE on an owned array.
    b = np.ascontiguousarray(a, dtype=np.float64)
    return np.frombuffer(b.tobytes(), dtype=np.float64).reshape(b.shape)



@dataclass(frozen=True)
class NodeContext:
    period: int
    scenario: int
    interval: int
    m: int
    n: int
    active: np.ndarray = field(repr=False, compare=False)
    demand: np.ndarray = field(repr=False, compare=False)
    capacity: np.ndarray = field(repr=False, compare=False)
    outsourcing: np.ndarray = field(repr=False, compare=False)
    route_cost: np.ndarray = field(repr=False, compare=False)
    key: str
    # Values belong to this immutable context, never to a filename or a mutable
    # instance. A separate cache per context avoids stale cross-instance reuse.
    _route_degree_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_instance(cls, data: Instance, t: int, s: int) -> 'NodeContext':
        data.validate()
        m, n, H, L, S = data.shape
        t, s = index(t, H, 'period'), index(s, S, 'scenario')
        a = data.arrays
        arrays = tuple(_immutable_array(v) for v in [a['active'][t,s], a['demand'][t,s],
            a['capacity'][:,t], a['outsourcing_cost'][t,s], data.route_costs()[t]])
        k = int(a['period_to_interval'][t])
        h = hashlib.sha256(json.dumps([SCHEMA,t,s,k,m,n]).encode())
        for v in arrays: h.update(str(v.shape).encode()); h.update(v.tobytes())
        return cls(t,s,k,m,n,*arrays,h.hexdigest())

    def route_key(self, i: int) -> str:
        i = index(i, self.m, 'facility')
        return hashlib.sha256(f'{self.key}:facility:{i}'.encode()).hexdigest()

    def check_route_state(self, i: int, alpha, u) -> tuple[tuple[int,...], int]:
        i = index(i,self.m,'facility')
        av = binary_vector(alpha,self.n,'alpha')
        uv = binary_vector([u],1,'u')[0]
        if uv != int(any(av)):
            raise ValueError('INVALID_PARENT_ASSIGNMENT: u must equal any(alpha)')
        if any(b > a for b,a in zip(av,self.active)):
            raise ValueError('INVALID_PARENT_ASSIGNMENT: inactive customer assigned')
        if not exact_capacity(self.demand, av, float(self.capacity[i])*uv):
            raise ValueError('INVALID_PARENT_ASSIGNMENT: warehouse capacity exceeded')
        return av,uv


@dataclass(frozen=True)
class RouteDegreeBounds:
    eligible: tuple[int, ...]
    incoming: tuple[float, ...]
    outgoing: tuple[float, ...]
    return_cost: float
    depart_cost: float


def route_degree_bounds(ctx: NodeContext, facility: int) -> RouteDegreeBounds:
    """The common physical incoming/outgoing minima used by cuts and S2.

    Compare demands as exact input doubles. A predecessor is eligible only if
    the two customers can share this physical facility's capacity. Inactive or
    individually infeasible customers get a zero coefficient. This function
    does not replace the route costs or authorize transit through extra nodes.
    """
    i = index(facility, ctx.m, 'facility')
    if i in ctx._route_degree_cache:
        return ctx._route_degree_cache[i]
    demands = tuple(Fraction.from_float(float(d)) for d in ctx.demand)
    cap = Fraction.from_float(float(ctx.capacity[i]))
    eligible = tuple(j for j in range(ctx.n) if ctx.active[j] and demands[j] <= cap)
    incoming, outgoing = [0.] * ctx.n, [0.] * ctx.n
    for j in eligible:
        others = [k+1 for k in eligible if k != j and demands[k]+demands[j] <= cap]
        incoming[j] = min(float(ctx.route_cost[i, k, j+1]) for k in [0]+others)
        outgoing[j] = min(float(ctx.route_cost[i, j+1, k]) for k in [0]+others)
    result = RouteDegreeBounds(
        eligible, tuple(incoming), tuple(outgoing),
        min((float(ctx.route_cost[i, j+1, 0]) for j in eligible), default=0.),
        min((float(ctx.route_cost[i, 0, j+1]) for j in eligible), default=0.),
    )
    ctx._route_degree_cache[i] = result
    return result



@dataclass(frozen=True)
class AffineCut:
    """eta/theta >= intercept + dot(coefficients, parent state).

Route parent ordering: alpha[0:n],u. Node parent ordering: A[0:m,k(t)].
This object records origin, not a proof for arbitrary externally supplied data.
Only consume cuts from a verified generator; exhaustive checks are for tiny data.
"""
    level: str
    scope: str
    intercept: float
    coefficients: tuple[float,...]
    domain: str
    certificate: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.level not in {'route','node'}:
            raise ValueError('Cut level must be route or node')
        if not math.isfinite(self.intercept) or not all(math.isfinite(float(v)) for v in self.coefficients):
            raise ValueError('Cut coefficients/intercept must be finite')
        if not self.scope or not self.coefficients:
            raise ValueError('Missing cut scope or coefficients')
        object.__setattr__(self, 'coefficients', tuple(float(v) for v in self.coefficients))

    def value(self, state) -> float:
        v=real_vector(state,len(self.coefficients),'cut state')
        return math.fsum([self.intercept,*[a*b for a,b in zip(self.coefficients,v)]])

    def to_dict(self) -> dict:
        return {'level':self.level,'scope':self.scope,'intercept':self.intercept,
                'coefficients':list(self.coefficients),'domain':self.domain,
                'certificate':dict(self.certificate)}



def check_route_cut(cut: AffineCut, ctx: NodeContext, i: int):
    if cut.level!='route' or cut.scope!=ctx.route_key(i) or len(cut.coefficients)!=ctx.n+1:
        raise ValueError('Route cut has wrong context, facility, level, or coefficient order')
    if cut.domain not in {'parent','active','full'}:
        raise ValueError('Unsupported route cut domain')



def check_node_cut(cut: AffineCut, ctx: NodeContext):
    if cut.level!='node' or cut.scope!=ctx.key or len(cut.coefficients)!=ctx.m or cut.domain!='availability_box':
        raise ValueError('Node cut has wrong context/level/coefficient order/domain')



@dataclass
class NativeModel:
    """Owns a REAL gurobipy.Model and typed variable handles. Call close()."""
    model: Any
    variables: dict[str, dict[tuple, Any]]
    ordered_variables: list[Any]
    specification: 'Subproblem'
    matrix_audit: dict

    def close(self): self.model.dispose()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()



@dataclass
class Subproblem:
    linear: LinearMILP
    layer: str
    direction: str
    objective_space: str
    context: NodeContext | None = None
    fixed_state: tuple[int,...] | None = None
    facility: int | None = None
    domain: str | None = None
    multipliers: tuple[float,...] | None = None
    uses_exact_routes: bool = False
    name: str = ''

    @property
    def information_stage(self) -> int:
        return 1 if self.layer=='facility' else 2

    def to_gurobi(self, *, env=None) -> NativeModel:
        model, variables = build_gurobi(self.linear, env=env)
        try:
            model.ModelName = self.name or f'{self.layer}_{self.direction}'
            model.update()
            audit = audit_gurobi_matrix(self.linear,model,variables)
            groups = {g:{key:variables[col] for key,col in mapping.items()}
                      for g,mapping in self.linear.groups.items()}
            return NativeModel(model,groups,variables,self,audit)
        except Exception:
            model.dispose()
            raise

    def export(self, directory: str | Path):
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        self.linear.write_lp(p/'model.lp'); self.linear.save_matrix(p/'matrix.npz')
        (p/'model_contract.json').write_text(json.dumps({
            'schema':SCHEMA,'information_stage':self.information_stage,
            'algorithm_layer':self.layer,'direction':self.direction,
            'objective_space':self.objective_space,'context':None if self.context is None else self.context.key,
            'period':None if self.context is None else self.context.period,
            'scenario':None if self.context is None else self.context.scenario,
            'facility':self.facility,'fixed_state':self.fixed_state,'domain':self.domain,
            'multipliers':self.multipliers,'uses_exact_routes':self.uses_exact_routes,
            'variables':len(self.linear.names),'constraints':len(self.linear.rows)},indent=2))



def facility_block(M: LinearMILP, data):
    m,n,H,L,S = data.shape; a=data.arrays
    for i in range(m):
        for k in range(L):
            M.var('A',(i,k)); M.var('o',(i,k),a['opening_cost'][i,k])
            M.var('h',(i,k),a['continuation_cost'][i,k] if k else 0.)
            M.var('b',(i,k),a['closing_cost'][i,k] if k else 0.)
    A,o,h,b=(M.groups[g] for g in ('A','o','h','b'))
    for i in range(m):
        for k in range(L):
            prev=[] if k==0 else [(A[i,k-1],-1.)]
            M.row(f'F2_prev_{i}_{k}',[(h[i,k],1)]+prev,ub=0)
            M.row(f'F2_curr_{i}_{k}',[(h[i,k],1),(A[i,k],-1)],ub=0)
            M.row(f'F2_both_{i}_{k}',[(h[i,k],1),(A[i,k],-1)]+prev,lb=-1)
            M.row(f'F3_open_{i}_{k}',[(o[i,k],1),(A[i,k],-1),(h[i,k],1)],lb=0,ub=0)
            M.row(f'F3_close_{i}_{k}',[(b[i,k],1),(h[i,k],1)]+prev,lb=0,ub=0)
    for k in range(L):
        M.row(f'F4_minimum_{k}',[(A[i,k],1) for i in range(m)],lb=float(a['min_open'][k]))



def service_block(M:LinearMILP, ctx:NodeContext, fixed_A=None):
    """z is a local facility-state COPY; no min-open/history/setup cost here."""
    for i in range(ctx.m):
        z=M.var('z',(i,)); M.var('u',(i,))
        if fixed_A is not None: M.lower[z]=M.upper[z]=float(fixed_A[i])
        for j in range(ctx.n): M.var('alpha',(i,j))
    for j in range(ctx.n): M.var('e',(j,),float(ctx.outsourcing[j]))
    z,u,alpha,e=(M.groups[g] for g in ('z','u','alpha','e'))
    for j in range(ctx.n):
        M.row(f'R1_service_{j}',[(alpha[i,j],1) for i in range(ctx.m)]+[(e[j,],1)],
              lb=float(ctx.active[j]),ub=float(ctx.active[j]))
    for i in range(ctx.m):
        M.row(f'R2_available_{i}',[(u[i,],1),(z[i,],-1)],ub=0)
        M.row(f'R2_nonempty_{i}',[(u[i,],1)]+[(alpha[i,j],-1) for j in range(ctx.n)],ub=0)
        M.row(f'R3_capacity_{i}',[(alpha[i,j],float(ctx.demand[j])) for j in range(ctx.n)]
              +[(u[i,],-float(ctx.capacity[i]))],ub=0)
        for j in range(ctx.n):
            M.row(f'R2_dispatch_{i}_{j}',[(alpha[i,j],1),(u[i,],-1)],ub=0)



def route_block(M:LinearMILP, ctx:NodeContext, i:int, alpha:list[int], u:int,
                connectivity='mtz', *, customers=None):
    """Own-root route, optionally projected onto customers known able to be used.

    Customers retain their original local indices j+1. The caller must prove
    every omitted assignment is zero (fixed forward assignment or backward
    domain activity). All assignment copies and degree equalities remain, so
    omitted variables still have explicit zero-degree constraints. Keeping the
    original MTZ constant n preserves the original formulation's projection.
    """
    n=ctx.n
    if connectivity not in ('mtz','cutset'):
        raise ValueError('connectivity must be mtz or cutset')
    if customers is None:
        customers=tuple(range(n))
    else:
        customers=tuple(index(j,n,'route customer') for j in customers)
        if len(set(customers))!=len(customers):
            raise ValueError('Route customer indices must be distinct')
        customers=tuple(sorted(customers))
    if connectivity=='cutset' and len(customers)>8:
        raise ValueError('Complete cutset limited to 8 eligible customers')
    local=(0,)+tuple(j+1 for j in customers)
    eligible=set(customers)
    r=M.groups.setdefault('r',{})
    for v in local:
        for w in local:
            if v!=w: M.var('r',(i,v,w),float(ctx.route_cost[i,v,w]))
    for j,aj in enumerate(alpha):
        v=j+1
        outgoing=[(r[i,v,w],1) for w in local if w!=v] if j in eligible else []
        incoming=[(r[i,w,v],1) for w in local if w!=v] if j in eligible else []
        M.row(f'R4_out_{i}_{j}',outgoing+[(aj,-1)],lb=0,ub=0)
        M.row(f'R4_in_{i}_{j}',incoming+[(aj,-1)],lb=0,ub=0)
    M.row(f'R5_out_{i}',[(r[i,0,j+1],1) for j in customers]+[(u,-1)],lb=0,ub=0)
    M.row(f'R5_in_{i}',[(r[i,j+1,0],1) for j in customers]+[(u,-1)],lb=0,ub=0)
    if connectivity=='mtz':
        nu={}
        M.groups.setdefault('nu',{})
        for j in customers:
            aj=alpha[j]
            v=M.var('nu',(i,j),ub=n,integer=False); nu[j]=v
            M.row(f'R6_lb_{i}_{j}',[(v,1),(aj,-1)],lb=0)
            M.row(f'R6_ub_{i}_{j}',[(v,1),(aj,-n)],ub=0)
        for j in customers:
            for q in customers:
                if j!=q:
                    M.row(f'R7_mtz_{i}_{j}_{q}',[(nu[j],1),(nu[q],-1),(r[i,j+1,q+1],n+1)],ub=n)
    else:
        for size in range(1,len(customers)+1):
            for subset in combinations(customers,size):
                selected={j+1 for j in subset}; tag='_'.join(map(str,subset))
                lhs=[(r[i,v,w],1) for v in selected for w in local if w not in selected]
                for j in subset: M.row(f'R7_cutset_{i}_{tag}_{j}',lhs+[(alpha[j],-1)],lb=0)



_BOUND_STATUSES = {'OPTIMAL', 'LIMIT', 'TIME_LIMIT', 'NODE_LIMIT', 'ITERATION_LIMIT',
                   'SOLUTION_LIMIT', 'WORK_LIMIT', 'MEM_LIMIT', 'INTERRUPTED', 'USER_OBJ_LIMIT'}


def finite_number(x):
    return isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(float(x)) and abs(x)<1e100



class InvalidSolverPrimal(ValueError):
    """A finite solver incumbent violates the unchanged primal tolerances."""


def matrix_primal_check(problem:Subproblem,x,atol=2e-6)->dict:
    M=problem.linear; x=np.asarray(x,float)
    if x.shape!=(len(M.names),) or not np.isfinite(x).all():
        raise ValueError('Malformed solver primal')
    ints=np.asarray(M.integer,bool)
    iv=float(np.max(np.abs(x[ints]-np.rint(x[ints])))) if ints.any() else 0.
    y=M.matrix()@x
    bv=max(0.,float(np.max(np.asarray(M.lower)-x)),float(np.max(x-np.asarray(M.upper))))
    rv=max(0.,float(np.max(np.asarray(M.row_lb)-y)),float(np.max(y-np.asarray(M.row_ub)))) if len(y) else 0.
    if max(iv,bv,rv)>atol: raise InvalidSolverPrimal(f'Invalid solver primal: integrality={iv},bounds={bv},rows={rv}')
    return {'passed':True,'integrality_violation':iv,'bound_violation':bv,'row_violation':rv}



@dataclass
class Evaluation:
    problem: Subproblem
    report: dict[str,Any]
    x: np.ndarray|None
    certified_lower_bound: float|None
    raw_certified_lower_bound: float|None
    lower_bound_guard: float

    @property
    def optimal(self):
        ub=self.report.get('objective'); lb=self.raw_certified_lower_bound
        return (self.report.get('status')=='OPTIMAL' and self.x is not None
            and finite_number(ub) and finite_number(lb)
            and -2e-6 <= ub-lb <= 2e-6+1e-10*max(1.,abs(ub),abs(lb)))

    def values(self,group:str)->dict[tuple,Any]:
        if self.x is None: raise RuntimeError('No feasible incumbent is available')
        M=self.problem.linear
        return {k:(int(round(float(self.x[col]))) if M.integer[col] else float(self.x[col]))
                for k,col in M.groups.get(group,{}).items()}

    def summary(self)->dict:
        return {**self.report,'layer':self.problem.layer,'direction':self.problem.direction,
            'information_stage':self.problem.information_stage,'objective_space':self.problem.objective_space,
            'certified_lower_bound':self.certified_lower_bound,'raw_certified_lower_bound':self.raw_certified_lower_bound,
            'lower_bound_guard':self.lower_bound_guard,'closed_numerical_optimality_certificate':self.optimal}



def cut_from_backward(result:Evaluation)->AffineCut:
    """NEVER uses ObjVal. An unavailable certified bound means no cut."""
    p=result.problem
    if p.direction!='backward' or p.context is None or p.multipliers is None:
        raise ValueError('Only a free-state backward oracle produces a Lagrangian cut')
    if result.certified_lower_bound is None:
        raise RuntimeError('NO_VALID_CUT: oracle has no usable certified global lower bound')
    if result.report.get('status') not in _BOUND_STATUSES:
        raise RuntimeError('NO_VALID_CUT: unsafe oracle status')
    cert={'solver':result.report.get('solver'),'solver_version':result.report.get('gurobi_version',result.report.get('scipy_version')),
          'status':result.report.get('status'),'raw_global_bound':result.raw_certified_lower_bound,
          'downward_guard':result.lower_bound_guard,'oracle_incumbent_NOT_used_as_intercept':result.report.get('objective'),
          'objective_space':p.objective_space,'exact_route_model':p.uses_exact_routes,
          'gurobi_executed':result.report.get('gurobi_executed',False)}
    if p.layer=='tsp':
        return AffineCut('route',p.context.route_key(p.facility),result.certified_lower_bound,
                         p.multipliers,p.domain,cert)
    if p.layer=='assignment':
        return AffineCut('node',p.context.key,result.certified_lower_bound,p.multipliers,'availability_box',cert)
    raise ValueError('The root has no parent and no backward oracle')



def tour_from_evaluation(result:Evaluation,i:int)->dict:
    p=result.problem; ctx=p.context
    if result.x is None or ctx is None or not p.uses_exact_routes:
        raise ValueError('A true route incumbent is required')
    if p.layer=='tsp':
        if p.facility!=i: raise ValueError('Wrong facility')
        alpha=tuple(result.values('a_copy')[j,] for j in range(ctx.n)); u=result.values('u_copy')[()]
    elif p.layer=='assignment':
        alpha=tuple(result.values('alpha')[i,j] for j in range(ctx.n)); u=result.values('u')[i,]
    else: raise ValueError('Root surrogate does not contain routes')
    arcs=[(v,w) for (fi,v,w),selected in result.values('r').items() if fi==i and selected]
    return audit_tour(ctx,i,alpha,u,arcs)



def audit_tour(ctx:NodeContext,i:int,alpha,u,arcs)->dict:
    alpha=binary_vector(alpha,ctx.n,'alpha'); u=binary_vector([u],1,'u')[0]
    if u!=int(any(alpha)): raise ValueError('Dispatch and assignment disagree')
    chosen={j+1 for j,v in enumerate(alpha) if v}
    parsed=[]
    for v,w in arcs:
        if type(v) not in (int,np.int64,np.int32) or type(w) not in (int,np.int64,np.int32):
            raise ValueError('Arc endpoints must be integer local indices')
        if not 0<=v<=ctx.n or not 0<=w<=ctx.n or v==w:
            raise ValueError('Invalid/self-loop arc')
        parsed.append((int(v),int(w)))
    if len(set(parsed))!=len(parsed): raise ValueError('Duplicate directed arc')
    if not chosen:
        if parsed: raise ValueError('Idle facility has a nonempty route')
        return {'facility':i,'customers':[],'local_route':[],'arcs':[],'cost':0.}
    selected=chosen|{0}; succ={}; indeg={v:0 for v in selected}
    for v,w in parsed:
        if v not in selected or w not in selected or v in succ:
            raise ValueError('Wrong customer/root or multiple outgoing arcs')
        succ[v]=w; indeg[w]+=1
    if set(succ)!=selected or any(v!=1 for v in indeg.values()):
        raise ValueError('Route degrees are incorrect')
    path=[0]; seen={0}; now=0
    for _ in range(len(selected)):
        now=succ[now]; path.append(now)
        if now==0: break
        if now in seen: raise ValueError('Repeated customer')
        seen.add(now)
    if path[-1]!=0 or seen!=selected or len(path)!=len(selected)+1:
        raise ValueError('Disconnected subtour or route missing the own root')
    cost=math.fsum(float(ctx.route_cost[i,v,w]) for v,w in parsed)
    return {'facility':i,'customers':[j for j,b in enumerate(alpha) if b],
            'local_route':path,'arcs':parsed,'cost':cost}



def certify_node(ctx:NodeContext,A,alpha,e,u,tours)->dict:
    A=binary_vector(A,ctx.m,'availability')
    alpha=np.asarray(alpha)
    if alpha.shape!=(ctx.m,ctx.n): raise ValueError('Wrong assignment dimensions')
    alpha=np.array([binary_vector(row,ctx.n,'alpha') for row in alpha],int)
    e=np.array(binary_vector(e,ctx.n,'outsourcing'),int)
    u=np.array(binary_vector(u,ctx.m,'dispatch'),int)
    if not np.array_equal(alpha.sum(axis=0)+e,ctx.active):
        raise ValueError('Orders must be fulfilled exactly once')
    audited=[]
    for i in range(ctx.m):
        if u[i]>A[i] or u[i]!=int(any(alpha[i])): raise ValueError('Unavailable/empty dispatch')
        if not exact_capacity(ctx.demand,alpha[i],float(ctx.capacity[i])*int(u[i])):
            raise ValueError('Primal violates the actual warehouse capacity (no slack added)')
        arcs=tours[i]['arcs'] if i in tours else []
        audited.append(audit_tour(ctx,i,alpha[i],u[i],arcs))
    outsource=math.fsum(float(ctx.outsourcing[j])*int(e[j]) for j in range(ctx.n))
    routing=math.fsum(t['cost'] for t in audited)
    return {'context':ctx.key,'period':ctx.period,'scenario':ctx.scenario,'availability':list(A),
            'alpha':alpha.tolist(),'e':e.tolist(),'u':u.tolist(),'tours':audited,
            'routing_cost':routing,'outsourcing_cost':outsource,'true_feasible_cost':math.fsum([routing,outsource])}



def certify_exact_node(result:Evaluation)->dict:
    p=result.problem; ctx=p.context
    if p.layer!='assignment' or not p.uses_exact_routes:
        raise ValueError('An assignment model with actual routes is required')
    alpha=result.values('alpha');e=result.values('e');u=result.values('u');z=result.values('z')
    return certify_node(ctx,[z[i,] for i in range(ctx.m)],
        [[alpha[i,j] for j in range(ctx.n)] for i in range(ctx.m)],
        [e[j,] for j in range(ctx.n)],[u[i,] for i in range(ctx.m)],
        {i:tour_from_evaluation(result,i) for i in range(ctx.m)})



def all_outsourcing(ctx:NodeContext,A)->dict:
    return certify_node(ctx,A,np.zeros((ctx.m,ctx.n),int),ctx.active,np.zeros(ctx.m,int),{})



def facility_cost(data:Instance,A)->float:
    m,n,H,L,S=data.shape; A=np.asarray(A)
    if A.shape!=(m,L): raise ValueError('Wrong shared facility plan shape')
    A=np.array([binary_vector(row,L,'facility plan row') for row in A],int)
    if np.any(A.sum(axis=0)<data.arrays['min_open']): raise ValueError('Insufficient available facilities')
    charges=[]
    for i in range(m):
        prev=0
        for k in range(L):
            cur=A[i,k]
            if cur and not prev: charges.append(float(data.arrays['opening_cost'][i,k]))
            elif cur and prev: charges.append(float(data.arrays['continuation_cost'][i,k]))
            elif prev and not cur: charges.append(float(data.arrays['closing_cost'][i,k]))
            prev=cur
    return math.fsum(charges)



def certify_policy(data:Instance,A,nodes:dict[tuple[int,int],dict])->dict:
    m,n,H,L,S=data.shape; total_fac=facility_cost(data,A); costs=[]; routing=[];outsourcing=[]
    if set(nodes)!=set((t,s) for t in range(H) for s in range(S)):
        raise ValueError('A global UB requires every original period/scenario node')
    for t in range(H):
        for s in range(S):
            ctx=NodeContext.from_instance(data,t,s); rec=nodes[t,s]
            expected=tuple(int(v) for v in np.asarray(A)[:,ctx.interval])
            if rec['context']!=ctx.key or tuple(rec['availability'])!=expected:
                raise ValueError('Cannot combine node policies from different data/facility states')
            check=certify_node(ctx,expected,rec['alpha'],rec['e'],rec['u'],{i:r for i,r in enumerate(rec['tours'])})
            p=float(data.arrays['scenario_prob'][s])
            costs.append(p*check['true_feasible_cost']);routing.append(p*check['routing_cost']);outsourcing.append(p*check['outsourcing_cost'])
    return {'A':np.asarray(A,int).tolist(),'facility_cost':total_fac,'expected_routing':math.fsum(routing),
            'expected_outsourcing':math.fsum(outsourcing),'feasible_upper_bound':math.fsum([total_fac,*costs])}


def evaluate_model(model, *, problem=None, time_limit=120.0, mip_gap=0.0,
                   threads=1, output=False, out=None, guard_abs=1e-9, guard_rel=1e-12,
                   deadline=None, optimize_context=None, mip_abs_gap=1e-8):
    """Optimize an existing layer model and return its numerical certificate.

    The caller retains ownership of ``model``. The specification must describe
    its actual coefficients, including explicit state-copy equalities. An
    assignment/master incumbent is a surrogate score, never a policy UB.
    """
    import gurobipy as gp

    _validate_solve_options(time_limit, mip_gap, threads)
    if not finite_number(mip_abs_gap) or mip_abs_gap < 0:
        raise ValueError('mip_abs_gap must be finite and nonnegative')
    if not finite_number(guard_abs) or not finite_number(guard_rel) or min(guard_abs, guard_rel) < 0:
        raise ValueError('Lower-bound guards must be finite and nonnegative')
    if problem is None:
        problem = getattr(model, '_lrp_spec', None)
    if not isinstance(problem, Subproblem):
        raise TypeError('A current LRP Subproblem specification is required')
    model.update()
    variables = model.getVars()
    roundtrip = audit_gurobi_matrix(problem.linear, model, variables)
    directory = None if out is None else Path(out)
    if directory is not None:
        problem.export(directory)
        model.write(str(directory / 'native.lp'))
    model.Params.OutputFlag = int(output or directory is not None)
    model.Params.LogToConsole = int(output)
    if directory is not None:
        model.Params.LogFile = str(directory / 'gurobi.log')
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.MIPGapAbs = float(mip_abs_gap)
    model.Params.Threads = int(threads)
    model.Params.Seed = 0
    from contextlib import nullcontext
    from core.solve_deadline import bounded_solve_time
    # Matrix construction/audit can consume the allowance. Check again at the
    # actual optimize boundary; no old model status is used after expiry.
    model.Params.TimeLimit = bounded_solve_time(model.Params.TimeLimit, deadline)
    start = time.perf_counter()
    with nullcontext() if optimize_context is None else optimize_context():
        model.optimize()
    status = int(model.Status)
    has = model.SolCount > 0

    def number(name):
        try:
            value = float(getattr(model, name))
        except (AttributeError, gp.GurobiError, TypeError, ValueError):
            return None
        return value if finite_number(value) else None

    status_names = ('LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED',
                    'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT',
                    'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL',
                    'INPROGRESS', 'USER_OBJ_LIMIT', 'WORK_LIMIT', 'MEM_LIMIT')
    mapping = {getattr(gp.GRB, name): name for name in status_names if hasattr(gp.GRB, name)}
    x = np.array(model.getAttr('X', variables), dtype=float) if has else None
    report = {
        'solver': 'Gurobi', 'gurobi_version': '.'.join(map(str, gp.gurobi.version())),
        'status': mapping.get(status, f'UNKNOWN_{status}'), 'status_code': status,
        'objective': number('ObjVal') if has else None,
        'objective_bound': number('ObjBound'),
        'objective_bound_continuous': number('ObjBoundC'),
        'mip_gap': number('MIPGap') if has else None,
        'solution_count': int(model.SolCount), 'solver_runtime': float(model.Runtime),
        'wall_seconds': time.perf_counter() - start,
        'variables': model.NumVars, 'constraints': model.NumConstrs,
        'requested_mip_rel_gap': mip_gap, 'requested_mip_abs_gap': float(mip_abs_gap),
        'optimization_completed': True,
        'gurobi_executed': True, **roundtrip,
    }
    if x is not None:
        report['matrix_primal_audit'] = matrix_primal_check(problem, x)
        rebuilt = math.fsum(float(c) * float(v) for c, v in zip(problem.linear.cost, x))
        if not finite_number(report['objective']) or abs(rebuilt - report['objective']) > 2e-6 + 1e-10 * max(1, abs(rebuilt)):
            raise ValueError('Returned objective does not match the actual model')
    raw_bound, bound, guard = None, None, 0.0
    rejected = []
    if report['status'] in _BOUND_STATUSES:
        for source in ('objective_bound_continuous', 'objective_bound'):
            candidate = report[source]
            if not finite_number(candidate):
                continue
            candidate_guard = float(guard_abs + guard_rel * max(1.0, abs(candidate)))
            guarded = math.nextafter(float(candidate) - candidate_guard, -math.inf)
            if has and guarded > report['objective']:
                # Check the other actual solver bound; never replace a bound
                # with ObjVal, widen the guard, or clip negative intercepts.
                rejected.append(source)
                continue
            raw_bound, bound, guard = float(candidate), guarded, candidate_guard
            report['lower_bound_source'] = source
            break
    if rejected:
        report['rejected_bound_sources'] = rejected
    if bound is None:
        report['lower_bound_source'] = None
    result = Evaluation(problem, report, x, bound, raw_bound, guard)
    if directory is not None:
        (directory / 'result.json').write_text(json.dumps(result.summary(), indent=2, allow_nan=False) + '\n')
        if x is not None:
            np.save(directory / 'solution.npy', x)
    return result


def solve_problem(problem, **options):
    """Solve one specification with Gurobi, closing the owned native model."""
    env = options.pop('env', None)
    with problem.to_gurobi(env=env) as native:
        return evaluate_model(native.model, problem=problem, **options)
