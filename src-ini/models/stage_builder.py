"""LRP forward models in the original three computational-layer interface.

There are TWO information stages: the common facility plan, then recourse.
The third computational layer prices a facility's route without new information.
Connectivity remains explicit; learned route cuts use the original per-successor
explicit/lazy threshold, with complete native/canonical rows retained for audit.
This module builds models; it does not replace the production iteration or
multiplier search. Tuple cuts are accepted only as caller-trusted LRP cuts in
the current node's domain. Archived investment z/y cuts are rejected.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path

import numpy as np

from cuts.lrp_static_bounds import basic_node_cuts, basic_route_cuts

from .learned_cut_dispatch import (
    MIP_PURPOSE, DUAL_LP_PURPOSE, install_as_explicit_rows,
    normalize_learned_cut_purpose,
)
from .stage_model_core import (
    AffineCut, Instance, LinearMILP, NodeContext, Subproblem,
    binary_vector, check_node_cut, check_route_cut, facility_block, index,
    load_instance, real_vector, route_block, route_degree_bounds, service_block,
)


DEFAULT_LAZY_THRESHOLD = 0


# Original Investment S2 search controls; the physical model is unchanged.
STAGE2_GRB_PARAM_ENV = {
    "forward": {
        "Presolve": ("VRP_S2_GRB_PRESOLVE", 2),
        "MIPFocus": ("VRP_S2_GRB_MIPFOCUS", -1),
    },
    "oracle": {
        "Presolve": ("VRP_S2_ORACLE_GRB_PRESOLVE", -1),
        "MIPFocus": ("VRP_S2_ORACLE_GRB_MIPFOCUS", -1),
    },
}


def _purpose(value):
    return normalize_learned_cut_purpose(value)


def _threshold(value):
    if value is None:
        return DEFAULT_LAZY_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("lazy_threshold must be a nonnegative integer")
    return int(value)


def _finite(value, label, *, minimum=None):
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return value


def _name(group, key):
    return f"{group}[{','.join(map(str, key))}]" if key else group


def _state_keys(ctx, facility=None):
    if facility is None:
        return tuple(f"A[{i},{ctx.interval}]" for i in range(ctx.m))
    return tuple(f"alpha[{facility},{j}]" for j in range(ctx.n)) + (f"u[{facility}]",)


def _as_cut(value, ctx, facility=None):
    """Validate current-domain names, including zero-slope and constant cuts.

    A two-tuple carries no provenance: its mathematical validity is the
    caller's responsibility. AffineCut additionally checks scope and domain.
    This adapter never relabels an existing AffineCut to another context.
    """
    if isinstance(value, AffineCut):
        cut = value
    else:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("cut must be AffineCut or (LRP coefficient dict, intercept)")
        coefficients, intercept = value
        if not isinstance(coefficients, Mapping):
            raise TypeError("tuple cut coefficients must be a mapping of LRP variable names")
        keys = _state_keys(ctx, facility)
        unknown = set(coefficients) - set(keys)
        if unknown:
            raise ValueError(f"unknown or investment cut state keys: {sorted(map(str, unknown))}")
        values = tuple(_finite(coefficients.get(key, 0.0), key) for key in keys)
        cut = AffineCut(
            "node" if facility is None else "route",
            ctx.key if facility is None else ctx.route_key(facility),
            _finite(intercept, "cut intercept"), values,
            "availability_box" if facility is None else "parent",
            {"source": "caller_trusted_lrp_tuple", "domain_validation": "keys_only"},
        )
    if facility is None:
        check_node_cut(cut, ctx)
    else:
        check_route_cut(cut, ctx, facility)
    return cut


def _stage_cost(M, groups):
    """Expose only this layer's physical cost; leave theta/pi terms separate."""
    terms = []
    for group in groups:
        for col in M.groups.get(group, {}).values():
            if M.cost[col]:
                terms.append((col, -M.cost[col]))
            M.cost[col] = 0.0
    cost = M.var("stage_cost", (), 1.0, ub=np.inf, integer=False)
    M.row("stage_cost_def", [(cost, 1.0)] + terms, lb=0.0, ub=0.0)


def _finish(spec, *, copies=None, fixed_rows=None):
    for group, entries in spec.linear.groups.items():
        for key, column in entries.items():
            spec.linear.names[column] = _name(group, key)
    # The map separates predecessor state names from local model column names.
    spec.parent_copy = dict(copies or {})
    spec.fixed_rows = dict(fixed_rows or {})
    spec.linear.validate()
    return spec


def build_stage1_forward(data, cuts=None):
    """Common A/o/h/b master; eta[t,s] is weighted by pi[s] exactly once."""
    data.validate()
    m, n, H, L, S = data.shape
    M = LinearMILP()
    facility_block(M, data)
    for t in range(H):
        for s in range(S):
            M.var("eta", (t, s), float(data.arrays["scenario_prob"][s]),
                  ub=np.inf, integer=False)
            ctx = NodeContext.from_instance(data, t, s)
            for q, cut in enumerate(basic_node_cuts(ctx)):
                terms = [(M.groups["eta"][t, s], 1.0)]
                terms += [(M.groups["A"][i, ctx.interval], -value)
                          for i, value in enumerate(cut.coefficients) if value]
                M.row(f"CapCut_{t}_{s}_{q}", terms, lb=cut.intercept)
    for (t, s), items in (cuts or {}).items():
        t, s = index(t, H, "period"), index(s, S, "scenario")
        ctx = NodeContext.from_instance(data, t, s)
        for q, item in enumerate(items):
            cut = _as_cut(item, ctx)
            terms = [(M.groups["eta"][t, s], 1.0)]
            terms += [(M.groups["A"][i, ctx.interval], -v)
                      for i, v in enumerate(cut.coefficients)]
            M.row(f"Qcut_{t}_{s}_{q}", terms, lb=cut.intercept)
    _stage_cost(M, ("o", "h", "b"))
    return _finish(Subproblem(M, "facility", "forward", "root_underestimator",
                             name="S1_facility_master"))


def _route_base_block(M, ctx):
    """Store a repeated incoming-degree affine expression once per facility.

    The auxiliary is an equality with zero objective coefficient. It neither
    changes theta's meaning nor imposes a new lower bound on route cost.
    """
    for i in range(ctx.m):
        degree = route_degree_bounds(ctx, i)
        coefficients = (*degree.incoming, degree.return_cost)
        if not any(coefficients):
            continue
        # This auxiliary is an equality in bounded alpha/u variables, so the
        # exact box maximum is redundant even in the continuous relaxation.
        # A finite bound also lets strict dual certification charge its tiny
        # residual instead of treating it as an unbounded direction. Round
        # upward once; a rounded-down sum could cut off a feasible LP point.
        exact_upper = sum((Fraction.from_float(value) for value in coefficients), Fraction())
        upper = float(exact_upper)
        if Fraction.from_float(upper) < exact_upper:
            upper = math.nextafter(upper, math.inf)
        base = M.var("route_base", (i,), ub=upper, integer=False)
        terms = [(base, 1.)]
        terms += [(M.groups["alpha"][i, j], -value)
                  for j, value in enumerate(degree.incoming) if value]
        if degree.return_cost:
            terms.append((M.groups["u"][i,], -degree.return_cost))
        M.row(f"route_base_def_{i}", terms, lb=0., ub=0.)


def _difference_down(coefficient, base):
    """Subtract exact binary64 values and round downward, never strengthen."""
    if coefficient == base:
        return 0.
    if base == 0.:
        return coefficient
    exact = Fraction.from_float(coefficient) - Fraction.from_float(base)
    value = float(exact)
    return math.nextafter(value, -math.inf) if Fraction.from_float(value) > exact else value


def _route_cut_terms(M, ctx, i, cut):
    """Use the shared base only when it strictly reduces row nonzeros.

    alpha and u are nonnegative, so flooring each residual coefficient makes
    the transformed affine cut no stronger anywhere in the parent LP box.
    Sparse general learned cuts retain their original, exact coefficients.
    """
    coefficients = cut.coefficients
    terms = [(M.groups["theta"][i,], 1.)]
    base_column = M.groups.get("route_base", {}).get((i,))
    if base_column is not None:
        degree = route_degree_bounds(ctx, i)
        base = (*degree.incoming, degree.return_cost)
        direct_nnz = sum(value != 0. for value in coefficients)
        residual_nnz = 1 + sum(value != common for value, common in zip(coefficients, base))
        if residual_nnz < direct_nnz:
            terms.append((base_column, -1.))
            coefficients = tuple(_difference_down(value, common)
                                 for value, common in zip(coefficients, base))
    terms += [(M.groups["alpha"][i, j], -value)
              for j, value in enumerate(coefficients[:-1]) if value]
    if coefficients[-1]:
        terms.append((M.groups["u"][i,], -coefficients[-1]))
    return terms


def _build_stage2(ctx, *, fixed_A, lam, route_cuts, mode, connectivity):
    if mode not in {"cuts", "exact"}:
        raise ValueError("mode must be 'cuts' or 'exact'")
    if connectivity not in {"mtz", "cutset"}:
        raise ValueError("connectivity must be 'mtz' or 'cutset'")
    if mode == "exact" and route_cuts and any(route_cuts.values()):
        raise ValueError("exact mode uses explicit routes, not additional theta cuts")
    M = LinearMILP(connectivity=connectivity)
    # The forward availability is inherited through named equality rows.
    # Match Investment's nonnegative continuous z_prev with no redundant UB;
    # free backward z retains its physical binary [0,1] domain.
    service_block(M, ctx, fixed_A=None)
    copies = {f"A[{i},{ctx.interval}]": f"z[{i}]" for i in range(ctx.m)}
    fixed_rows = {}
    if fixed_A is not None:
        for i, value in enumerate(fixed_A):
            column = M.groups["z"][i,]
            M.integer[column] = 0
            M.upper[column] = np.inf
            key, row = f"A[{i},{ctx.interval}]", f"z_prev_eq[{i}]"
            M.row(row, [(M.groups["z"][i,], 1.0)], lb=value, ub=value)
            fixed_rows[key] = row
    if lam is not None:
        for i, value in enumerate(lam):
            M.cost[M.groups["z"][i,]] = -value
    if mode == "exact":
        for i in range(ctx.m):
            route_block(M, ctx, i, [M.groups["alpha"][i, j] for j in range(ctx.n)],
                        M.groups["u"][i,], connectivity)
    else:
        for i in range(ctx.m):
            M.var("theta", (i,), 1.0, ub=np.inf, integer=False)
        _route_base_block(M, ctx)
        # The original incoming-only RouteCut is always an ordinary model
        # row, including free-state backward and dual LP models. Its name is
        # separate from Tcut so learned-cut lazy dispatch never removes it.
        for i in range(ctx.m):
            for q, cut in enumerate(basic_route_cuts(ctx, i)):
                M.row(f"RouteCut_{i}_{q}", _route_cut_terms(M, ctx, i, cut),
                      lb=cut.intercept)
        for i, items in (route_cuts or {}).items():
            i = index(i, ctx.m, "facility")
            for q, item in enumerate(items):
                cut = _as_cut(item, ctx, i)
                terms = _route_cut_terms(M, ctx, i, cut)
                M.row(f"Tcut_{i}_{q}", terms, lb=cut.intercept)
    _stage_cost(M, ("e",))
    backward = lam is not None
    space = (("lagrangian_true_node" if mode == "exact" else "lagrangian_node_underestimator")
             if backward else ("true_node" if mode == "exact" else "node_underestimator"))
    spec = Subproblem(
        M, "assignment", "backward" if backward else "forward", space, ctx,
        fixed_state=fixed_A,
        domain="availability_box" if backward else "fixed_availability",
        multipliers=lam, uses_exact_routes=(mode == "exact"),
        name=f"S2_{'backward' if backward else 'forward'}_{mode}_t{ctx.period}_s{ctx.scenario}",
    )
    return _finish(spec, copies=copies, fixed_rows=fixed_rows)


def build_stage2_forward(ctx, Abar, route_cuts=None, *, mode="cuts", connectivity="mtz"):
    """Fixed common availability; unweighted outsourcing + route envelope."""
    return _build_stage2(ctx, fixed_A=binary_vector(Abar, ctx.m, "Abar"), lam=None,
                         route_cuts=route_cuts, mode=mode, connectivity=connectivity)


def build_stage3_forward(ctx, i, alpha, u, *, connectivity="mtz", reduce_nodes=True):
    """Fixed parent assignment; optionally eliminate fixed-unassigned nodes."""
    if not isinstance(reduce_nodes, bool):
        raise TypeError("reduce_nodes must be boolean")
    i = index(i, ctx.m, "facility")
    alpha, u = ctx.check_route_state(i, alpha, u)
    M = LinearMILP(connectivity=connectivity)
    cols = [M.var("a_copy", (j,)) for j in range(ctx.n)]
    uc = M.var("u_copy", ())
    copies, fixed_rows = {}, {}
    for j, value in enumerate(alpha):
        key, row = f"alpha[{i},{j}]", f"alpha_prev_eq[{j}]"
        copies[key], fixed_rows[key] = f"a_copy[{j}]", row
        M.row(row, [(cols[j], 1.0)], lb=value, ub=value)
    copies[f"u[{i}]"], fixed_rows[f"u[{i}]"] = "u_copy", "u_prev_eq"
    M.row("u_prev_eq", [(uc, 1.0)], lb=u, ub=u)
    customers = tuple(j for j, value in enumerate(alpha) if value) if reduce_nodes else None
    route_block(M, ctx, i, cols, uc, connectivity, customers=customers)
    _stage_cost(M, ("r",))
    return _finish(Subproblem(
        M, "tsp", "forward", "true_route", ctx, fixed_state=(*alpha, u),
        facility=i, domain="fixed_parent_assignment", uses_exact_routes=True,
        name=f"S3_forward_i{i}_t{ctx.period}_s{ctx.scenario}",
    ), copies=copies, fixed_rows=fixed_rows)


def _instance(data):
    """Accept an LRP instance/path or a wrapper with instance/lrp_instance."""
    if isinstance(data, (str, Path)):
        return load_instance(Path(data))
    for attribute in ("lrp_instance", "instance"):
        candidate = getattr(data, attribute, None)
        if candidate is not None:
            data = candidate
            break
    if not hasattr(data, "arrays") or not hasattr(data, "metadata"):
        raise TypeError("LRP builders require an LRP Instance; investment VRP data is unsupported")
    copied = Instance(str(getattr(data, "name", "lrp")),
                      {key: np.array(value, copy=True) for key, value in data.arrays.items()},
                      deepcopy(data.metadata))
    copied.validate()
    for array in copied.arrays.values():
        array.setflags(write=False)
    return copied


def _successors(node, count, *, optional=False):
    values = list(getattr(node, "successor", ()))
    if optional and node is None:
        return list(range(count))
    if len(values) != count or len(set(values)) != count:
        raise ValueError(f"node must have exactly {count} distinct successors")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
           for value in values):
        raise ValueError("successor IDs must be nonnegative integers")
    return [int(value) for value in values]


def _node_context(data, node, *, stage, context=None):
    if context is None:
        context = node if isinstance(node, NodeContext) else getattr(node, "context", None)
    if context is not None:
        if not isinstance(context, NodeContext):
            raise TypeError("context must be NodeContext")
        expected = NodeContext.from_instance(data, context.period, context.scenario)
        if context.key != expected.key:
            raise ValueError("NodeContext does not match builder instance")
        return expected
    parent = node if stage == 2 else getattr(node, "predecessor", None)
    if isinstance(parent, (int, np.integer)) and not isinstance(parent, bool):
        m, n, H, L, S = data.shape
        p = index(parent, H * S, "Stage-2 predecessor")
        s, t = divmod(p, H)
    else:
        info = getattr(parent, "info", None)
        if not isinstance(info, (tuple, list)) or len(info) != 2:
            raise ValueError("S2 info must be (s,t); S3 requires its S2 predecessor or context")
        s, t = info
    return NodeContext.from_instance(data, t, s)


def _facility(node, context, facility=None):
    return index(getattr(node, "info", None) if facility is None else facility,
                 context.m, "facility")


def _state_values(state, keys, label):
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must map LRP state names to exact binary values")
    # A complete S2 solution legitimately includes its local z[i] copies.
    # Keep them available in the dictionary passed to S3; only the required
    # alpha/u entries become the fixed route state. Investment root z[i,t]
    # and vehicle y[i] have no meaning in this LRP interface.
    legacy = [key for key in state if str(key).startswith("y[") or (
        str(key).startswith("z[") and (label == "facility state" or "," in str(key))
    )]
    if legacy:
        raise ValueError(f"investment state keys are unsupported: {legacy}")
    missing = set(keys) - set(state)
    if missing:
        raise KeyError(f"missing {label} keys: {sorted(missing)}")
    return binary_vector([state[key] for key in keys], len(keys), label)


def _pool(cut_lag, stage, node_id):
    return (cut_lag or {}).get(stage, {}).get(node_id, ())


def _root_spec(data, node, cut_lag):
    m, n, H, L, S = data.shape
    successors = _successors(node, H * S, optional=True)
    if set(successors) != set(range(H * S)):
        raise ValueError("root successor IDs must encode s*H+t; no implicit scenario reordering")
    mapping = {(t, s): s * H + t for t in range(H) for s in range(S)}
    cuts = {key: _pool(cut_lag, 2, successor) for key, successor in mapping.items()}
    spec = build_stage1_forward(data, cuts)
    for key, successor in mapping.items():
        spec.linear.names[spec.linear.groups["eta"][key]] = f"eta[{successor}]"
    spec.epigraph_ids = {key: value for key, value in mapping.items()}
    return spec


def _route_pools(ctx, node, cut_lag, *, explicit=None):
    if explicit is not None:
        return explicit, {i: i for i in range(ctx.m)}
    successors = _successors(node, ctx.m)
    # Existing tree construction creates successors in facility order. A
    # caller with a different ordering must supply its explicit bijection.
    by_successor = getattr(node, "successor_facilities", None)
    if by_successor is None:
        mapping = dict(enumerate(successors))
    else:
        if set(by_successor) != set(successors) or set(by_successor.values()) != set(range(ctx.m)):
            raise ValueError("successor_facilities must be a bijection successor ID -> facility")
        mapping = {int(i): int(successor) for successor, i in by_successor.items()}
    return {i: _pool(cut_lag, 3, successor) for i, successor in mapping.items()}, mapping


def _rename_theta(spec, mapping):
    for (i,), column in spec.linear.groups.get("theta", {}).items():
        spec.linear.names[column] = f"theta[{mapping[i]}]"
    spec.epigraph_ids = dict(mapping)
    spec.linear.validate()
    return spec



def _dispatch_s3_cut_rows(model, lazy_threshold, learned_cut_purpose, *, successors=None):
    """Restore per-successor dispatch without removing any canonical row.

    Gurobi owns the complete pre-enumerated lazy pool. This keeps native matrix
    audits and raw-incumbent checks over every cut, including constant cuts.
    The Lazy attribute affects MIPs only; dual_lp additionally forces zero.
    """
    threshold, purpose = _threshold(lazy_threshold), _purpose(learned_cut_purpose)
    selected = set(model._lrp_s3_cut_rows if successors is None else successors)
    if purpose == DUAL_LP_PURPOSE:
        selected = set(model._lrp_s3_cut_rows)
    for successor, rows in model._lrp_s3_cut_rows.items():
        if successor not in selected:
            continue
        ordinary = install_as_explicit_rows(len(rows), threshold, purpose)
        for row in rows:
            row.Lazy = 0 if ordinary else 1
    model.update()
    flags = [row.Lazy for rows in model._lrp_s3_cut_rows.values() for row in rows]
    lazy = sum(flag != 0 for flag in flags)
    explicit = len(flags) - lazy
    model._lrp_lazy_threshold = threshold
    model._lrp_learned_cut_purpose = purpose
    model._lrp_learned_cut_dispatch = 'native_attribute'
    model._s3_learned_cut_count = explicit + lazy
    model._s3_explicit_cut_count = explicit
    model._s3_lazy_cut_count = lazy
    model._lrp_all_cuts_explicit = lazy == 0
    model.update()


def _native(spec, *, env=None, mip_gap=0.0,
            lazy_threshold=DEFAULT_LAZY_THRESHOLD, learned_cut_purpose=MIP_PURPOSE):
    native = spec.to_gurobi(env=env)
    model = native.model
    try:
        model.Params.OutputFlag = 0
        # Original investment master contract: a coarse outer tolerance must
        # not hide progress in the S1 lower bound or change its trial cheaply.
        model.Params.MIPGap = min(mip_gap, 1e-4) if spec.layer == 'facility' else mip_gap
        model.Params.MIPGapAbs = 0.0
        model.Params.IntFeasTol = 1e-9 if spec.layer == 'facility' else 1e-8
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        if spec.layer == 'assignment':
            apply_stage2_search_params(model, role=(
                'forward' if spec.direction == 'forward' else 'oracle'))
        model._lrp_native = native
        model._lrp_spec = spec
        model._lrp_variables = native.variables
        model._lrp_parent_copy = dict(spec.parent_copy)
        model._lrp_fixed_rows = dict(spec.fixed_rows)
        model._lrp_dual_bindings = tuple(spec.fixed_rows.items())
        model._lrp_stage_cost = native.variables["stage_cost"][()]
        model._lrp_epigraph_ids = dict(getattr(spec, "epigraph_ids", {}))
        model._lrp_all_cuts_explicit = True
        model._lrp_information_stages = 2
        model._lrp_route_domain = spec.domain
        count = sum(name.startswith("Tcut_") for name in spec.linear.row_names)
        model._s3_learned_cut_count = count
        model._s3_explicit_cut_count = count
        model._s3_lazy_cut_count = 0
        model._lrp_s3_cut_rows = {successor: [] for successor in model._lrp_epigraph_ids.values()}
        if spec.layer == 'assignment' and 'theta' in spec.linear.groups:
            for name in spec.linear.row_names:
                if name.startswith('Tcut_'):
                    facility = int(name.split('_')[1])
                    successor = model._lrp_epigraph_ids[facility]
                    model._lrp_s3_cut_rows[successor].append(model.getConstrByName(name + '_lb'))
        _dispatch_s3_cut_rows(model, lazy_threshold, learned_cut_purpose)
        return model
    except Exception:
        model.dispose()
        raise


class StageModelBuilder:
    """Drop-in construction entry point with explicit LRP state contracts.

    A caller owns/disposes each returned bare Gurobi model. Fixed RHS bindings
    are available as model._lrp_dual_bindings for the existing SBC LP routine.
    reduce_nodes projects fixed Stage-3 routes onto assigned customers; false
    preserves the full graph required by fixed-RHS LP dual extraction.
    Each successor's archive is explicit up to lazy_threshold and native lazy
    above it. dual_lp always installs ordinary rows for fixed-RHS duals.
    """

    def __init__(self, prob_data, mip_gap=1e-4, lazy_threshold=DEFAULT_LAZY_THRESHOLD,
                 *, connectivity="mtz", env=None):
        self.prob_data = self.instance = _instance(prob_data)
        self.mip_gap = _finite(mip_gap, "mip_gap", minimum=0.0)
        self.lazy_threshold = _threshold(lazy_threshold)
        if connectivity not in {"mtz", "cutset"}:
            raise ValueError("connectivity must be 'mtz' or 'cutset'")
        self.connectivity, self.env = connectivity, env

    def worker_options(self):
        if self.env is not None:
            raise ValueError("Gurobi Env cannot be copied to workers; create one per worker")
        return {"mip_gap": self.mip_gap, "lazy_threshold": self.lazy_threshold,
                "connectivity": self.connectivity}

    def build_stage_problem(self, stage_no, node, cut_lag, x_prev, reduce_nodes=True,
                            *, learned_cut_purpose=MIP_PURPOSE, context=None,
                            facility=None, mode="cuts"):
        purpose = _purpose(learned_cut_purpose)
        if stage_no not in {1, 2, 3}:
            raise ValueError("stage_no must be 1, 2, or 3")
        if purpose != "mip" and stage_no != 2:
            raise ValueError("dual_lp learned-cut purpose applies only to Stage 2")
        if not isinstance(reduce_nodes, bool):
            raise TypeError("reduce_nodes must be boolean")
        if mode != "cuts" and stage_no != 2:
            raise ValueError("mode is an assignment-layer option only")
        if stage_no == 1:
            spec = _root_spec(self.instance, node, cut_lag)
        elif stage_no == 2:
            ctx = _node_context(self.instance, node, stage=2, context=context)
            Abar = _state_values(x_prev, _state_keys(ctx), "facility state")
            pools, mapping = _route_pools(ctx, node, cut_lag)
            spec = build_stage2_forward(ctx, Abar, pools, mode=mode, connectivity=self.connectivity)
            _rename_theta(spec, mapping)
        else:
            ctx = _node_context(self.instance, node, stage=3, context=context)
            i = _facility(node, ctx, facility)
            values = _state_values(x_prev, _state_keys(ctx, i), "assignment state")
            spec = build_stage3_forward(ctx, i, values[:-1], values[-1],
                                        connectivity=self.connectivity, reduce_nodes=reduce_nodes)
        return _native(spec, env=self.env, mip_gap=self.mip_gap,
                       lazy_threshold=self.lazy_threshold, learned_cut_purpose=purpose)

    def _build_stage_1(self, node):
        return self.build_stage_problem(1, node, {}, {})

    def _build_stage_2(self, node, x_prev):
        return self.build_stage_problem(2, node, {}, x_prev)

    def _build_stage_3(self, node, x_prev, reduce_nodes=True):
        return self.build_stage_problem(3, node, {}, x_prev, reduce_nodes)

    def build_stage3_dual_problem(self, node, cut_lag, x_prev, *, reuse_route_dfj=False):
        """Build the original full SCF route LP used to obtain Phase-1 Pi."""
        from .route_dual_builder import build_stage3_dual_problem
        options = {} if reuse_route_dfj is False else {"reuse_route_dfj": reuse_route_dfj}
        return build_stage3_dual_problem(self, node, cut_lag, x_prev, **options)

    # A legacy/custom override without this marker keeps its original call.
    build_stage3_dual_problem._lrp_accepts_route_dfj_reuse = True


def apply_stage2_search_params(model, role="forward"):
    """Restore role-specific controls, with LRP aliases before VRP names."""
    import os
    if role not in {"forward", "oracle"}:
        raise ValueError("role must be forward or oracle")
    model.setParam("OptimalityTol", 1e-9)
    for param, (env_name, default) in STAGE2_GRB_PARAM_ENV[role].items():
        names = ("LRP_" + env_name.removeprefix("VRP_"), env_name)
        name, raw = next(((name, os.environ[name].strip()) for name in names
                         if os.environ.get(name, "").strip()), (env_name, ""))
        try:
            value = int(raw) if raw else int(default)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if value >= 0:
            model.setParam(param, value)


def add_explicit_s3_cut(model, succ_ind, pi_dict, v_val, *, name):
    """Append one validated current-context cut and update its matrix contract."""
    import gurobipy as gp
    spec = getattr(model, "_lrp_spec", None)
    if spec is None or spec.layer != "assignment" or "theta" not in spec.linear.groups:
        raise ValueError("explicit route cuts require an LRP assignment cuts-mode model")
    mapping = model._lrp_epigraph_ids
    matches = [i for i, successor in mapping.items() if successor == succ_ind]
    if len(matches) != 1:
        raise KeyError(f"unknown theta successor {succ_ind}")
    i = matches[0]
    cut = _as_cut((pi_dict, v_val), spec.context, i)
    M = spec.linear
    if name in M.row_names:
        raise ValueError(f"duplicate learned-cut row name {name}")
    terms = _route_cut_terms(M, spec.context, i, cut)
    constraint = model.addConstr(
        gp.quicksum(coefficient * model._lrp_native.ordered_variables[column]
                    for column, coefficient in terms) >= cut.intercept,
        name=str(name) + "_lb",
    )
    M.row(str(name), terms, lb=cut.intercept)
    model.update()
    model._lrp_s3_cut_rows[succ_ind].append(constraint)
    model._s3_learned_cut_count += 1
    model._s3_explicit_cut_count += 1
    return constraint


def add_s3_to_s2_cuts(model, succ_ind, cuts, lazy_threshold=DEFAULT_LAZY_THRESHOLD,
                      learned_cut_purpose=MIP_PURPOSE):
    threshold = _threshold(lazy_threshold)
    purpose = _purpose(learned_cut_purpose)
    # A dual model stays explicit even when an append caller uses default MIP.
    if getattr(model, '_lrp_learned_cut_purpose', None) == DUAL_LP_PURPOSE:
        purpose = DUAL_LP_PURPOSE
    spec = getattr(model, "_lrp_spec", None)
    if spec is None:
        raise TypeError("LRP cut installation requires an LRP builder model")
    matches = [i for i, value in model._lrp_epigraph_ids.items() if value == succ_ind]
    if len(matches) != 1:
        raise KeyError(f"unknown theta successor {succ_ind}")
    i = matches[0]
    for item in cuts:
        cut = _as_cut(item, spec.context, i)
        coefficients = dict(zip(_state_keys(spec.context, i), cut.coefficients))
        add_explicit_s3_cut(model, succ_ind, coefficients, cut.intercept,
                            name=f"Tcut_added_{succ_ind}_{model._s3_learned_cut_count}")
    _dispatch_s3_cut_rows(model, threshold, purpose, successors=(succ_ind,))


def add_s3_to_s2_cut_pools(model, successors, cut_lag, *,
                          lazy_threshold=DEFAULT_LAZY_THRESHOLD,
                          learned_cut_purpose=MIP_PURPOSE):
    successors = list(successors)
    if len(successors) != len(set(successors)):
        raise ValueError("duplicate successor IDs")
    if set(successors) != set(model._lrp_epigraph_ids.values()):
        raise ValueError("successors disagree with the model's facility mapping")
    for successor in successors:
        add_s3_to_s2_cuts(model, successor, _pool(cut_lag, 3, successor),
                          lazy_threshold, learned_cut_purpose)


def lazy_cut_callback(model, where):
    """The native Lazy attribute enforces pre-enumerated cuts without cbLazy."""
    if not (getattr(model, '_lrp_all_cuts_explicit', False)
            or getattr(model, '_lrp_learned_cut_dispatch', None) == 'native_attribute'):
        raise ValueError('LRP callback cannot operate an unaudited investment lazy archive')


def _setup_lazy_cuts(model, succ_ind, cuts):
    return add_s3_to_s2_cuts(model, succ_ind, cuts, lazy_threshold=0)


def _add_explicit_s3_to_s2_cuts(model, succ_ind, cuts):
    matches = [i for i, successor in model._lrp_epigraph_ids.items() if successor == succ_ind]
    if len(matches) != 1:
        raise KeyError(f'unknown theta successor {succ_ind}')
    for item in cuts:
        cut = _as_cut(item, model._lrp_spec.context, matches[0])
        coefficients = dict(zip(_state_keys(model._lrp_spec.context, matches[0]), cut.coefficients))
        add_explicit_s3_cut(model, succ_ind, coefficients, cut.intercept,
                            name=f'Tcut_added_{succ_ind}_{model._s3_learned_cut_count}')
