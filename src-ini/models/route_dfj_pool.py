"""Reuse physical directed connectivity rows across route-oracle builds.

Each entry is the globally valid inequality delta+(U) >= a_copy[j], with the
physical customer vertex j+1 in a root-free U. No fixed parent values, duals,
solver bounds, or substituted route costs enter this pool. Entries are scoped
by the immutable node/facility fingerprint, not by the trial assignment.
"""
from __future__ import annotations

from collections import OrderedDict

from .stage_model_core import index
from .route_dfj_factor import append_canonical_rows


# Eviction only omits redundant integer-valid rows from future relaxations.
# It cannot invalidate an existing cut or alter the integer route domain.
MAX_CONTEXTS = 512
MAX_ROWS_PER_ROUTE = 2048
_POOLS: OrderedDict[str, OrderedDict[tuple, None]] = OrderedDict()


def remember_route_dfj_row(ctx, facility, vertices, customer):
    """Record a validated physical (U, zero-based j) row; return whether new."""
    i = index(facility, ctx.m, 'facility')
    j = index(customer, ctx.n, 'DFJ customer')
    U = tuple(sorted({index(v, ctx.n+1, 'DFJ physical vertex') for v in vertices}))
    if not U or 0 in U or j+1 not in U:
        raise ValueError('DFJ requires a root-free set containing the selected customer')
    scope = ctx.route_key(i)
    if scope not in _POOLS:
        if len(_POOLS) >= MAX_CONTEXTS:
            _POOLS.popitem(last=False)
        _POOLS[scope] = OrderedDict()
    _POOLS.move_to_end(scope)
    pool = _POOLS[scope]
    key = (U, j)
    if key in pool:
        return False
    if len(pool) >= MAX_ROWS_PER_ROUTE:
        pool.popitem(last=False)
    pool[key] = None
    return True


def route_dfj_rows(ctx, facility):
    """Return an immutable snapshot of this physical route's reusable rows."""
    scope = ctx.route_key(facility)
    pool = _POOLS.get(scope)
    if pool is None:
        return ()
    _POOLS.move_to_end(scope)
    return tuple(pool)


def clear_route_dfj_pool():
    """Discard optional connectivity rows, for isolated tests/experiments."""
    _POOLS.clear()


def route_dfj_checkpoint(routes):
    """Save pure physical row identities for the supplied (context, facility)s."""
    scopes = {ctx.route_key(i) for ctx, i in routes}
    return {'version': 1, 'routes': {
        scope: tuple(pool) for scope, pool in _POOLS.items() if scope in scopes}}


def restore_route_dfj_checkpoint(routes, state):
    """Validate an entire checkpoint, then merge rows with matching physics.

    These rows only strengthen relaxations. Foreign physical fingerprints are
    ignored; malformed rows for a matching fingerprint fail before mutation.
    Bounds, incumbent tours, and solver models are never restored here.
    """
    if not isinstance(state, dict) or state.get('version') != 1:
        raise ValueError('Unsupported route DFJ checkpoint version')
    saved = state.get('routes')
    if not isinstance(saved, dict):
        raise ValueError('Route DFJ checkpoint requires a route mapping')
    current = {ctx.route_key(i): (ctx, i) for ctx, i in routes}
    pending = []
    report = {'matching_routes': 0, 'foreign_routes': 0, 'restored_rows': 0}
    for scope, rows in saved.items():
        if scope not in current:
            report['foreign_routes'] += 1
            continue
        ctx, i = current[scope]
        if not isinstance(rows, (list, tuple)) or len(rows) > MAX_ROWS_PER_ROUTE:
            raise ValueError('Invalid route DFJ checkpoint row collection')
        report['matching_routes'] += 1
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError('Invalid route DFJ checkpoint row')
            vertices, customer = row
            j = index(customer, ctx.n, 'DFJ checkpoint customer')
            if not isinstance(vertices, (list, tuple)):
                raise ValueError('Invalid route DFJ checkpoint vertex set')
            U = tuple(sorted({index(v, ctx.n+1, 'DFJ checkpoint vertex') for v in vertices}))
            if not U or 0 in U or j+1 not in U:
                raise ValueError('DFJ checkpoint requires root-free U containing j')
            pending.append((ctx, i, U, j))
    for ctx, i, U, j in pending:
        report['restored_rows'] += int(remember_route_dfj_row(ctx, i, U, j))
    return report


def add_route_dfj_rows(M,ctx,facility,assignment_columns):
    """Install the physical pool using shared outflow expressions where smaller.

    Project only vertices absent from this domain's physical graph; omitted
    customers and singleton sets remain redundant. The physical pool and its
    checkpoint format are unchanged by this purely algebraic compression.
    """
    i=index(facility,ctx.m,'facility')
    arcs={(v,w):c for (fi,v,w),c in M.groups.get('r',{}).items() if fi==i}
    vertices={0}|{v for arc in arcs for v in arc};source=route_dfj_rows(ctx,i);installed=set();pairs=[]
    complete=len(arcs)==len(vertices)*(len(vertices)-1) and all(v!=w for v,w in arcs)
    report=dict(available_rows=len(source),installed_rows=0,omitted_customer_rows=0,projected_rows=0,
        duplicate_projections=0,internal_arc_rows=0,crossing_arc_rows=0,redundant_singletons=0)
    for original,j in source:
        if j+1 not in vertices:report['omitted_customer_rows']+=1;continue
        U=frozenset(original)&vertices;key=(U,j)
        if key in installed:report['duplicate_projections']+=1;continue
        installed.add(key);report['projected_rows']+=int(len(U)!=len(original))
        if len(U)==1:report['redundant_singletons']+=1;continue
        inside=(len(U)*(len(U)-1) if complete else sum(v in U and w in U for v,w in arcs))+len(U)-1
        crossing=(len(U)*(len(vertices)-len(U)) if complete else sum(v in U and w not in U for v,w in arcs))+1
        report['internal_arc_rows' if inside<crossing else 'crossing_arc_rows']+=1
        pairs.append((tuple(sorted(U)),j))
    rows=append_canonical_rows(M,ctx,i,assignment_columns,pairs)
    report.update(installed_rows=len(rows),factored_sets=sum(r['aux_created'] for r in rows),
                  factor_nonzeros=sum(r['nonzeros'] for r in rows))
    return report
