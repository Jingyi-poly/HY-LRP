"""Free-state LRP backward oracles for the existing SBC / Level Set callers."""
from __future__ import annotations

from collections.abc import Mapping
import math
import os

from .stage3_solver_settings import _apply_stage3_solve_params

from .stage_model_core import LinearMILP, Subproblem, index, real_vector, route_block
from .route_dfj_pool import add_route_dfj_rows
from .stage_builder import (
    _build_stage2, _facility, _finite, _finish, _instance, _native,
    _node_context, _rename_theta, _route_pools, _stage_cost, _state_keys, _threshold,
    _purpose, MIP_PURPOSE,
)


def build_stage2_backward(ctx, lam, route_cuts=None, *, mode="cuts", connectivity="mtz"):
    """min outsourcing + route envelope - lambda*z over the binary z box."""
    return _build_stage2(ctx, fixed_A=None, lam=real_vector(lam, ctx.m, "lambda"),
                         route_cuts=route_cuts, mode=mode, connectivity=connectivity)


def build_stage3_backward(ctx, i, rho, sigma=0.0, *, domain="parent", connectivity="mtz",
                          reduce_nodes=True):
    """min route-rho*a_copy-sigma*u_copy, without any fixed forward state.

    Reduction only removes domain-inactive customers; the full domain keeps
    every customer regardless of activity or the current forward assignment.
    """
    if not isinstance(reduce_nodes, bool):
        raise TypeError("reduce_nodes must be boolean")
    i = index(i, ctx.m, "facility")
    rho = real_vector(rho, ctx.n, "rho")
    sigma = _finite(sigma, "sigma")
    if domain not in {"parent", "active", "full"}:
        raise ValueError("domain must be 'parent', 'active', or 'full'")
    M = LinearMILP(connectivity=connectivity)
    cols = [M.var("a_copy", (j,), -rho[j]) for j in range(ctx.n)]
    uc = M.var("u_copy", (), -sigma)
    for j, aj in enumerate(cols):
        M.row(f"domain_dispatch_{j}", [(aj, 1), (uc, -1)], ub=0)
        if domain != "full":
            M.row(f"domain_active_{j}", [(aj, 1)], ub=float(ctx.active[j]))
    M.row("domain_nonempty", [(uc, 1)] + [(v, -1) for v in cols], ub=0)
    if domain == "parent":
        M.row("domain_warehouse_capacity",
              [(v, float(ctx.demand[j])) for j, v in enumerate(cols)]
              + [(uc, -float(ctx.capacity[i]))], ub=0)
    customers = tuple(j for j in range(ctx.n) if ctx.active[j]) if reduce_nodes and domain != "full" else None
    route_block(M, ctx, i, cols, uc, connectivity, customers=customers)
    M.route_dfj_reuse = add_route_dfj_rows(M, ctx, i, cols)
    _stage_cost(M, ("r",))
    copies = {f"alpha[{i},{j}]": f"a_copy[{j}]" for j in range(ctx.n)}
    copies[f"u[{i}]"] = "u_copy"
    return _finish(Subproblem(
        M, "tsp", "backward", "lagrangian_route", ctx, facility=i,
        domain=domain, multipliers=(*rho, sigma), uses_exact_routes=True,
        name=f"S3_backward_i{i}_t{ctx.period}_s{ctx.scenario}",
    ), copies=copies)


def _multipliers(pi_value, keys):
    if not isinstance(pi_value, Mapping):
        raise TypeError("pi_value must map LRP predecessor state names to multipliers")
    unknown = set(pi_value) - set(keys)
    if unknown:
        raise ValueError(f"unknown or investment multiplier keys: {sorted(map(str, unknown))}")
    return tuple(_finite(pi_value.get(key, 0.0), key) for key in keys)


class SubproblemBuilder:
    """Preserves the original oracle builder API; returns a bare Gurobi Model.

    Model._lrp_parent_copy maps full predecessor keys to local free variables.
    The objective and its bound are unweighted node units. Only a certified
    global lower bound can be used as a cut intercept; ObjVal is an incumbent.
    Each build creates fresh variables, preventing forward bound/RHS leakage.
    """

    phase = 2

    def __init__(self, prob_data, lazy_threshold=None, oracle_int_feas_tol=None,
                 oracle_feas_tol=None, *, connectivity="mtz", env=None, phase=None):
        self.phase = type(self).phase if phase is None else phase
        if isinstance(self.phase, bool) or self.phase not in (1, 2):
            raise ValueError("phase must be 1 or 2")
        self.prob_data = self.instance = _instance(prob_data)
        self.lazy_threshold = _threshold(lazy_threshold)
        self.oracle_int_feas_tol = self._tolerance(oracle_int_feas_tol, "IntFeasTol",
            self._oracle_setting("INT_FEAS_TOL"))
        self.oracle_feas_tol = self._tolerance(oracle_feas_tol, "FeasibilityTol",
            self._oracle_setting("FEAS_TOL"))
        if connectivity not in {"mtz", "cutset"}:
            raise ValueError("connectivity must be 'mtz' or 'cutset'")
        self.connectivity, self.env = connectivity, env

    def _oracle_setting(self, suffix):
        if self.phase != 2:
            return None
        return os.environ.get("LRP_PHASE2_ORACLE_" + suffix,
                              os.environ.get("VRP_PHASE2_ORACLE_" + suffix))

    @staticmethod
    def _tolerance(value, label, default):
        value = default if value is None else value
        if value is None:
            return None
        value = _finite(float(value), label)
        upper = 0.1 if label == "IntFeasTol" else 0.01
        if not 1e-9 <= value <= upper:
            raise ValueError(f"{label} must be within Gurobi's [1e-9,{upper}] range")
        return value

    def worker_options(self):
        if self.env is not None:
            raise ValueError("Gurobi Env cannot be copied to workers; create one per worker")
        return {"phase": self.phase, "lazy_threshold": self.lazy_threshold,
                "oracle_int_feas_tol": self.oracle_int_feas_tol,
                "oracle_feas_tol": self.oracle_feas_tol,
                "connectivity": self.connectivity}

    def build_subproblem(self, stage_no, node, cut_lag, pi_value, *,
                         context=None, facility=None, mode="cuts", domain="parent",
                         reduce_nodes=True, learned_cut_purpose=MIP_PURPOSE,
                         node_ind=None, phase=None):
        purpose = _purpose(learned_cut_purpose)
        if purpose != MIP_PURPOSE and stage_no != 2:
            raise ValueError('dual_lp learned-cut purpose applies only to Stage 2')
        if stage_no not in {2, 3}:
            raise ValueError("only Stage 2 and Stage 3 have backward oracles")
        ctx = _node_context(self.instance, node, stage=stage_no, context=context)
        if stage_no == 2:
            if domain != "parent":
                raise ValueError("route domain is a Stage-3 option only")
            lam = _multipliers(pi_value, _state_keys(ctx))
            pools, mapping = _route_pools(ctx, node, cut_lag)
            spec = build_stage2_backward(ctx, lam, pools, mode=mode, connectivity=self.connectivity)
            _rename_theta(spec, mapping)
        else:
            if mode != "cuts":
                raise ValueError("mode is a Stage-2 option only")
            i = _facility(node, ctx, facility)
            values = _multipliers(pi_value, _state_keys(ctx, i))
            spec = build_stage3_backward(ctx, i, values[:-1], values[-1],
                                         domain=domain, connectivity=self.connectivity, reduce_nodes=reduce_nodes)
        model = _native(spec, env=self.env, mip_gap=0.0,
                        lazy_threshold=self.lazy_threshold, learned_cut_purpose=purpose)
        if stage_no == 3:
            _apply_stage3_solve_params(model, node_ind=node_ind,
                                      phase=self.phase if phase is None else phase)
        if self.oracle_int_feas_tol is not None:
            model.Params.IntFeasTol = self.oracle_int_feas_tol
        if self.oracle_feas_tol is not None:
            model.Params.FeasibilityTol = self.oracle_feas_tol
        return model

    def _build_stage_2_subproblem(self, node, cut_lag, pi_value):
        return self.build_subproblem(2, node, cut_lag, pi_value)

    def _build_stage_3_subproblem(self, node, pi_value):
        return self.build_subproblem(3, node, {}, pi_value)
