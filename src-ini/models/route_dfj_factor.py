"""Share physical DFJ outflow expressions without changing their LP projection.

For a root-free physical set U, let f_U = delta+(U). Summing the customer
outdegree equations gives the equivalent internal-arc expression used below.
Both incoming and outgoing degrees are at most one (including the root), so
0 <= f_U <= min(|U|, |V\\U|) is already implied in the original LP. Each
f_U >= a_copy[j] therefore has exactly the original DFJ projection, including
asymmetric and nonmetric route costs. Factor only when it reduces nonzeros.

The canonical installer only appends; its caller owns copy-on-write isolation.
The native wrapper mirrors column and row suffixes. Physical (U, j) identities
remain the sole pool/checkpoint representation; auxiliary columns are local to
one model and are never persisted or shared between physical facilities.
"""
from collections import defaultdict
from models.stage_model_core import index


def grouped(pairs,vertices):
    result=defaultdict(set)
    for original,j in pairs:
        U=tuple(sorted(set(original)&vertices))
        if 0 in U or j+1 not in U:raise ValueError('Invalid physical DFJ set')
        if len(U)>1:result[U].add(j)
    return result


def projected_extra(M,vertices,pairs):
    total=0;flowmap=getattr(M,'dfj_flow_map',{})
    for U,js in grouped(pairs,vertices).items():
        if U in flowmap:total+=2*len(js);continue
        size=len(U);inside=size*(size-1);cross=size*(len(vertices)-size)
        flow=min(inside+size,cross);dense=min(inside+size-1,cross+1)*len(js)
        total+=min(dense,flow+1+2*len(js))
    return total


def append_canonical_rows(M,ctx,facility,assignment_columns,pairs):
    """Owner must use a fresh matrix or copy-on-write all appendable fields."""
    i=index(facility,ctx.m,'facility')
    if len(assignment_columns)!=ctx.n:raise ValueError('Incomplete assignment copies')
    arcs={(v,w):col for (fi,v,w),col in M.groups['r'].items() if fi==i}
    vertices={0}|{v for arc in arcs for v in arc}
    if not hasattr(M,'dfj_flow_map'):M.dfj_flow_map={}
    owner=getattr(M,'dfj_flow_facility',i)
    if owner!=i:raise ValueError('Flow auxiliaries belong to one physical facility')
    M.dfj_flow_facility=i
    flowmap=M.dfj_flow_map;added=[]
    for ordered,customers in grouped(pairs,vertices).items():
        U=set(ordered);outside=vertices-U
        crossing=[(arcs[v,w],1.) for v in ordered for w in sorted(outside) if (v,w) in arcs]
        inside=[(arcs[v,w],-1.) for v in ordered for w in ordered if (v,w) in arcs]
        internal_flow=inside+[(assignment_columns[v-1],1.) for v in ordered]
        flow_terms=min((crossing,internal_flow),key=len)
        dense={j:min((inside+[(assignment_columns[v-1],1.) for v in ordered if v!=j+1],
                      crossing+[(assignment_columns[j],-1.)]),key=len) for j in customers}
        f=flowmap.get(ordered);definition_nnz=0
        if f is None and len(flow_terms)+1+2*len(customers)<sum(map(len,dense.values())):
            serial=len(M.names);f=M.var('dfj_flow',(i,serial),ub=min(len(U),len(outside)),integer=False)
            M.names[f]=f'dfj_flow[{i},{serial}]'
            M.row(f'dfj_flow_definition_{i}_{serial}',[(f,1.)]+[(col,-value) for col,value in flow_terms],lb=0.,ub=0.)
            flowmap[ordered]=f;definition_nnz=len(flow_terms)+1
        for position,j in enumerate(sorted(customers)):
            index(j,ctx.n,'DFJ customer')
            terms=dense[j] if f is None else [(f,1.),(assignment_columns[j],-1.)]
            M.row(f'factor_free_dfj_{i}_{len(M.rows)}',terms,lb=0.)
            added.append(dict(U=list(ordered),j=j,form='dense' if f is None else 'shared_flow',aux_column=f,
                nonzeros=len(terms)+(definition_nnz if position==0 else 0),
                aux_created=bool(definition_nnz and position==0)))
    return added


def append_factor_rows(model,lp,pairs,serial):
    """Append canonical columns/rows to the owned MIP and private LP together."""
    import gurobipy as gp
    from models.route_dfj_pool import remember_route_dfj_row
    problem=model._lrp_spec;M=problem.linear;ctx=problem.context;i=problem.facility
    old_columns,old_rows=len(M.names),len(M.rows)
    columns=[M.groups['a_copy'][j,] for j in range(ctx.n)]
    added=append_canonical_rows(M,ctx,i,columns,pairs)
    registry=dict(model._lrp_variables);registry['dfj_flow']=dict(registry.get('dfj_flow',{}))
    for native,is_mip in ((model,True),(lp,False)):
        variables=native.getVars()
        for column in range(old_columns,len(M.names)):
            var=native.addVar(lb=M.lower[column],ub=M.upper[column],obj=M.cost[column],
                vtype=gp.GRB.BINARY if is_mip and M.integer[column] else gp.GRB.CONTINUOUS,name=M.names[column])
            variables.append(var)
        if is_mip:
            for key,column in M.groups.get('dfj_flow',{}).items():registry['dfj_flow'][key]=variables[column]
        for r in range(old_rows,len(M.rows)):
            terms=M.rows[r];expression=gp.LinExpr(list(terms.values()),[variables[c] for c in terms])
            if M.row_lb[r]==M.row_ub[r]:native.addConstr(expression==M.row_lb[r],name=M.row_names[r])
            else:
                assert M.row_lb[r]==0. and M.row_ub[r]==float('inf')
                native.addConstr(expression>=0.,name=M.row_names[r]+'_lb')
        native.update()
    model._lrp_variables=registry;model._lrp_native.variables=registry
    model._lrp_native.ordered_variables=model.getVars()
    for row in added:remember_route_dfj_row(ctx,i,row['U'],row['j'])
    return added
