"""The Investment fixed-state SCF relaxation, rooted at a physical facility.

Only SBC's multiplier LP uses this builder. Forward route policies and the
independent EF retain their exact builders. The complete active arc domain and
flow capacity stay fixed when the linking RHS changes; no trial-based node
reduction can invalidate a resulting cut at another assignment.
"""
from __future__ import annotations

import numpy as np

from .stage_model_core import LinearMILP, Subproblem
from .route_dfj_pool import add_route_dfj_rows
from .stage_builder import (_facility, _finish, _native, _node_context,
                            _stage_cost, _state_keys, _state_values)


def build_stage3_dual_spec(ctx, facility, alpha, dispatch, *, reuse_route_dfj=False):
    """Return an exact integer route formulation with a single SCF LP.

    As in Investment Final, parent copies are nonnegative continuous columns
    fixed by equalities, rather than binary columns with artificial upper bounds.
    Actual binary route states are checked before fixing the RHS. Customer-count
    flow, rather than demand flow, connects zero-demand customers as well.
    """
    if not isinstance(reuse_route_dfj, bool):
        raise TypeError("reuse_route_dfj must be boolean")
    alpha, dispatch = ctx.check_route_state(facility, alpha, dispatch)
    M = LinearMILP(connectivity='scf')
    copies = [M.var('a_copy', (j,), ub=np.inf, integer=False) for j in range(ctx.n)]
    used = M.var('u_copy', (), ub=np.inf, integer=False)
    parent_copy, fixed_rows = {}, {}
    for j, value in enumerate(alpha):
        name, row = f'alpha[{facility},{j}]', f'alpha_prev_eq[{j}]'
        parent_copy[name], fixed_rows[name] = f'a_copy[{j}]', row
        M.row(row, [(copies[j], 1.)], lb=value, ub=value)
    name = f'u[{facility}]'
    parent_copy[name], fixed_rows[name] = 'u_copy', 'u_prev_eq'
    M.row('u_prev_eq', [(used, 1.)], lb=dispatch, ub=dispatch)
    active = tuple(j for j in range(ctx.n) if ctx.active[j])
    vertices = (0,) + tuple(j + 1 for j in active)
    arcs = {}
    for v in vertices:
        for w in vertices:
            if v != w:
                arcs[v, w] = M.var('r', (facility, v, w), float(ctx.route_cost[facility, v, w]))
    for j in range(ctx.n):
        vertex = j + 1
        incoming = [(column, 1.) for (v, w), column in arcs.items() if w == vertex]
        outgoing = [(column, 1.) for (v, w), column in arcs.items() if v == vertex]
        M.row(f'R4_in_{facility}_{j}', incoming + [(copies[j], -1.)], lb=0., ub=0.)
        M.row(f'R4_out_{facility}_{j}', outgoing + [(copies[j], -1.)], lb=0., ub=0.)
        M.row(f'domain_dispatch_{j}', [(copies[j], 1.), (used, -1.)], ub=0.)
    M.row(f'R5_out_{facility}', [(column, 1.) for (v, _), column in arcs.items() if v == 0]
          + [(used, -1.)], lb=0., ub=0.)
    M.row(f'R5_in_{facility}', [(column, 1.) for (_, w), column in arcs.items() if w == 0]
          + [(used, -1.)], lb=0., ub=0.)
    # Same fixed-state capacity row as the original LP. At a legal trial this
    # is redundant, but keeping it preserves the original multiplier problem.
    M.row('domain_warehouse_capacity', [(copies[j], float(ctx.demand[j]))
          for j in active], ub=float(ctx.capacity[facility]))
    if active:
        size = len(active)
        flow = {}
        for (v, w), arc in arcs.items():
            if w == 0:
                continue
            f = M.var('connectivity_flow', (facility, v, w), ub=size, integer=False)
            flow[v, w] = f
            M.row(f'connectivity_capacity_{facility}_{v}_{w}',
                  [(f, 1.), (arc, -float(size))], ub=0.)
        for j in active:
            v = j + 1
            M.row(f'connectivity_balance_{facility}_{j}',
                  [(f, 1.) for (_, w), f in flow.items() if w == v]
                  + [(f, -1.) for (w, _), f in flow.items() if w == v]
                  + [(copies[j], -1.)], lb=0., ub=0.)
        M.row(f'connectivity_supply_{facility}',
              [(f, 1.) for (v, _), f in flow.items() if v == 0]
              + [(copies[j], -1.) for j in active], lb=0., ub=0.)
    _stage_cost(M, ('r',))
    spec = _finish(Subproblem(M, 'tsp', 'forward', 'true_route', ctx,
        fixed_state=(*alpha, dispatch), facility=facility,
        domain='fixed_parent_assignment', uses_exact_routes=True,
        name=f'S3_fixed_rhs_scf_i{facility}_t{ctx.period}_s{ctx.scenario}'),
        copies=parent_copy, fixed_rows=fixed_rows)
    if reuse_route_dfj:
        # Reuse only this full physical domain, with variable a_copy RHS.
        # No separation solve or trial-based graph reduction is introduced.
        M.route_dfj_reuse = add_route_dfj_rows(M, ctx, facility, copies)
        M.validate()
    return spec


def build_stage3_dual_problem(builder, node, cut_lag, x_prev, *, reuse_route_dfj=False):
    """Use the same builder environment and expose ordinary fixed RHS duals."""
    ctx = _node_context(builder.instance, node, stage=3)
    facility = _facility(node, ctx, None)
    values = _state_values(x_prev, _state_keys(ctx, facility), 'assignment state')
    options = {} if reuse_route_dfj is False else {"reuse_route_dfj": reuse_route_dfj}
    spec = build_stage3_dual_spec(ctx, facility, values[:-1], values[-1], **options)
    model = _native(spec, env=builder.env, mip_gap=builder.mip_gap,
                    lazy_threshold=builder.lazy_threshold)
    model._lrp_dual_formulation = 'investment_scf'
    return model
