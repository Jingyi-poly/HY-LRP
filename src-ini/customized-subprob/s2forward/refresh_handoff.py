"""One-pass transfer of solved refresh policies to the next Phase-2 forward.

This is a feasible-policy handoff, not an exact-optimum cache.  Entries must
come from an actual refresh solve, retain their original certificate status,
and are consumed even when the next model does not match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from models.stage2_symmetry import assignment_order_holds
from solvers.forward_policy_certification import certify_stage2_forward_policy
from solvers.forward_period_dedup import (
    canonical_float,
    canonical_options,
    forward_semantic_fingerprint,
    stage2_period_key,
)

def score_forward_stage2_policy_from_archive(*args, **kwargs):
    # Legacy-only adapter: importing LRP handoff must not initialize fleet backends.
    from solvers.forward_stage2_policy import score_forward_stage2_policy_from_archive as score
    return score(*args, **kwargs)


def _semantics(prob_data, scen_tree):
    if hasattr(prob_data, "arrays"):
        from models.stage_builder import _instance
        nodes = tuple((stage, tuple((n.time, n.index, n.info, n.predecessor, tuple(n.successor),
            tuple(sorted(n.probability.items())), n.multi_coeff, n.context.key if n.context else None)
            for n in layer)) for stage, layer in sorted(scen_tree.items()))
        return ("lrp", _instance(prob_data).logical_hash(), nodes)
    return (
        forward_semantic_fingerprint(prob_data, scen_tree),
        tuple(tuple(getattr(prob_data, name)) for name in ("J", "V", "N")),
    )


def _node_key(prob_data, scen_tree, node, x_prev, cut_lag,
              model_options, solve_policy):
    if hasattr(prob_data, "arrays"):
        from models.stage_builder import _instance
        data = _instance(prob_data)
        m, _, _, intervals, _ = data.shape
        root = tuple(canonical_float(x_prev[f'A[{i},{k}]'])
                     for i in range(m) for k in range(intervals))
        import hashlib
        physical = stage2_period_key(node, x_prev, cut_lag, prob_data, scen_tree,
                                    model_options=model_options, solve_policy=solve_policy)
        return ("lrp", root, hashlib.sha256(repr(physical).encode()).hexdigest())
    period = node.info[1]
    # Do not round the incoming fleet when deciding whether to transfer a
    # result: require the same binary64 state, in addition to model identity.
    fleet = tuple(canonical_float(x_prev[f"z[{v},{period}]"])
                  for v in prob_data.V)
    return (
        fleet,
        stage2_period_key(
            node, x_prev, cut_lag, prob_data, scen_tree,
            model_options=model_options, solve_policy=solve_policy,
        ),
    )


@dataclass(frozen=True)
class RefreshPolicy:
    x_dict: dict
    cost_star_value: float
    stage_cost_value: float
    theta_by_succ: dict
    objective_lower_bound: float
    exact_optimal: bool
    backend: str
    status: object = None
    closed_time_key: object = None
    source_time_limit: float | None = None


def _score_matches(record, score):
    return (
        canonical_float(record.cost_star_value)
        == canonical_float(score["cost_star_value"])
        and canonical_float(record.stage_cost_value)
        == canonical_float(score["stage_cost_value"])
        and canonical_options(record.theta_by_succ)
        == canonical_options(score["theta_by_succ"])
    )


@dataclass
class RefreshHandoffPass:
    entries: dict = field(default_factory=dict)
    hits: int = 0
    rejected: int = 0

    def take(self, prob_data, scen_tree, node_ind, x_prev, cut_lag, *,
             model_options, solve_policy=None):
        entry = self.entries.pop(node_ind, None)
        if entry is None:
            return None
        key, record = entry
        node = scen_tree[2][node_ind]
        requested_key = _node_key(prob_data, scen_tree, node, x_prev, cut_lag,
                                  model_options, solve_policy)
        time_compatible = (hasattr(prob_data, "arrays")
            and record.closed_time_key is not None and record.exact_optimal
            and record.status == 'OPTIMAL'
            and _lrp_closed_interval(record.cost_star_value, record.objective_lower_bound)
            and record.closed_time_key == _node_key(prob_data, scen_tree, node, x_prev,
                cut_lag, model_options, _without_time_limit(solve_policy)))
        if key != requested_key and not time_compatible:
            self.rejected += 1
            return None
        if hasattr(prob_data, "arrays"):
            try:
                score = _lrp_score(prob_data, node, x_prev, record.x_dict, cut_lag, model_options)
            except (ValueError, RuntimeError, KeyError, TypeError):
                self.rejected += 1
                return None
            # Re-scoring may round differently from the original native
            # policy audit. Reject an inverted pair instead of clipping its
            # independently valid lower certificate or replaying the mismatch.
            if (math.isnan(record.objective_lower_bound)
                    or record.objective_lower_bound > score['cost_star_value']):
                self.rejected += 1
                return None
        else:
            score = score_forward_stage2_policy_from_archive(
                prob_data, node, record.x_dict, cut_lag
            )
        if not _score_matches(record, score):
            self.rejected += 1
            return None
        self.hits += 1
        # Consumers own their dictionaries; neither this entry nor a saved
        # backward trial is mutated by the subsequent forward/backward pass.
        return RefreshPolicy(
            dict(record.x_dict), record.cost_star_value,
            record.stage_cost_value, dict(record.theta_by_succ),
            record.objective_lower_bound, record.exact_optimal,
            record.backend, record.status, record.closed_time_key, record.source_time_limit,
        )


@dataclass
class RefreshHandoff:
    semantic_key: tuple
    entries: dict

    def begin_pass(self, prob_data, scen_tree):
        entries, self.entries = self.entries, {}
        if self.semantic_key != _semantics(prob_data, scen_tree):
            return RefreshHandoffPass(rejected=len(entries))
        return RefreshHandoffPass(entries=entries)


def build_refresh_handoff(prob_data, scen_tree, x_prev, x_stage2, scores,
                          cut_lag, *, model_options, solve_policy=None):
    """Freeze the final refresh trial; re-scoring alone cannot mint entries.

    ``scores[node]['fresh_solve']`` is set by the backward caller only for a
    real Gurobi/DP/BPC solve (or its exact period copy) in this backward pass.
    Policies obtained solely by re-scoring or consuming an earlier handoff
    must not carry this flag.  ``objective_lower_bound`` remains independent
    of the feasible objective; a missing bound is represented by ``-inf``.
    """
    if hasattr(prob_data, "arrays"):
        return _build_lrp_refresh_handoff(prob_data, scen_tree, x_prev, x_stage2,
                                         scores, cut_lag, model_options, solve_policy)
    entries = {}
    for node_ind, raw in scores.items():
        if raw.get("fresh_solve") is not True:
            continue
        node = scen_tree[2][node_ind]
        policy = dict(x_stage2[node_ind])
        period = node.info[1]
        for vehicle in prob_data.V:
            if float(x_prev[f"z[{vehicle},{period}]"]) not in (0.0, 1.0):
                raise ValueError("refresh fleet must contain certified binary decisions")
            if policy[f"y[{vehicle}]"] > round(x_prev[f"z[{vehicle},{period}]"]):
                raise ValueError("refresh policy uses an unpurchased vehicle")
        certified, _ = certify_stage2_forward_policy(prob_data, node, x_prev, policy)
        if any(float(policy[name]) != value for name, value in certified.items()):
            raise ValueError("refresh policy must contain certified binary decisions")
        policy = certified
        if not assignment_order_holds(
            prob_data, prob_data.V,
            [[policy[f"alpha[{j},{v}]"] for j in prob_data.J] for v in prob_data.V],
        ):
            # A physical policy can fail an exact canonical row through solver
            # feasibility tolerances.  It must not seed a same-model handoff.
            continue
        score = score_forward_stage2_policy_from_archive(
            prob_data, node, policy, cut_lag
        )
        lower = float(raw.get("objective_lower_bound", float("-inf")))
        if math.isnan(lower) or lower == math.inf:
            raise ValueError("invalid refresh lower-bound certificate")
        if lower > score["cost_star_value"] + 1e-6:
            raise ValueError("refresh lower bound exceeds its feasible score")
        record = RefreshPolicy(
            policy, float(raw.get("cost_star_value", score["cost_star_value"])),
            float(raw["stage_cost"]), dict(raw["theta_by_succ"]), lower,
            bool(raw.get("exact_optimal", False)), str(raw.get("backend", "")),
            raw.get("status"),
        )
        if not _score_matches(record, score):
            raise ValueError("refresh scores do not match the final trial/archive")
        entries[node_ind] = (
            _node_key(prob_data, scen_tree, node, x_prev, cut_lag,
                      model_options, solve_policy),
            record,
        )
    return RefreshHandoff(_semantics(prob_data, scen_tree), entries)


def _lrp_score(prob_data, node, x_prev, policy, cuts, model_options):
    """Reaudit the physical state and every original assignment row, no solve."""
    import numpy as np
    from models.stage_builder import (_instance, _node_context, _route_pools,
                                      _rename_theta, build_stage2_forward)
    ctx = _node_context(_instance(prob_data), node, stage=2)
    normalized, physical = certify_stage2_forward_policy(prob_data, node, x_prev, policy)
    if any(float(policy[k]) != v for k, v in normalized.items()):
        raise ValueError("refresh state is not exactly binary")
    pools, mapping = _route_pools(ctx, node, cuts)
    spec = build_stage2_forward(ctx, [x_prev[f'A[{i},{ctx.interval}]'] for i in range(ctx.m)],
        pools, connectivity=model_options.get('connectivity', 'mtz'))
    _rename_theta(spec, mapping)
    M = spec.linear
    x = np.array([policy[name] for name in M.names], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("refresh primal is nonfinite")
    lhs = M.matrix() @ x
    violation = max(0., float(np.max(np.asarray(M.lower)-x)),
                    float(np.max(x-np.asarray(M.upper))),
                    float(np.max(np.asarray(M.row_lb)-lhs)),
                    float(np.max(lhs-np.asarray(M.row_ub))))
    if violation > 2e-6:
        raise ValueError("refresh primal violates current assignment model")
    objective = float(np.dot(M.cost, x))
    if not math.isfinite(objective):
        raise ValueError("refresh objective is nonfinite")
    return {'cost_star_value': objective, 'stage_cost_value': physical,
            'theta_by_succ': {q: float(policy[f'theta[{q}]']) for q in node.successor}}


def _without_time_limit(policy):
    return {key: value for key, value in (policy or {}).items() if key != 'sub_time_limit'}


def _lrp_closed_interval(upper, lower):
    # Same numerical certificate band as Evaluation.optimal, additionally
    # checking the guarded lower endpoint against the independently rescored
    # assignment. Never infer optimality from Gurobi status alone.
    return (math.isfinite(upper) and math.isfinite(lower)
            and 0. <= upper-lower <= 2e-6 + 1e-10*max(1., abs(upper), abs(lower)))


def _build_lrp_refresh_handoff(prob_data, tree, root, policies, scores, cuts,
                                model_options, solve_policy):
    entries = {}
    for index, raw in scores.items():
        if raw.get('fresh_solve') is not True:
            continue
        node = tree[2][index]
        policy = dict(policies[index])
        try:
            score = _lrp_score(prob_data, node, root, policy, cuts, model_options)
        except (ValueError, RuntimeError, KeyError, TypeError):
            continue
        lower = raw.get('envelope_lower')
        lower = -math.inf if lower is None else float(lower)
        if math.isnan(lower) or lower == math.inf or lower > score['cost_star_value']:
            continue
        # Preserve the actual optimization allowance, never the nominal one.
        allowance = raw.get('solve_time_limit')
        if allowance is None or not math.isfinite(float(allowance)) or allowance <= 0:
            continue
        actual_policy = dict(solve_policy, sub_time_limit=float(allowance))
        key = _node_key(prob_data, tree, node, root, cuts, model_options, actual_policy)
        if raw.get('refresh_key') != key:
            continue
        diagnostic = raw.get('diagnostic', {})
        closed = (raw.get('exact_optimal') is True
                  and diagnostic.get('optimization_completed') is True
                  and diagnostic.get('gurobi_executed') is True
                  and diagnostic.get('status') == 'OPTIMAL'
                  and diagnostic.get('closed_numerical_optimality_certificate') is True
                  and _lrp_closed_interval(score['cost_star_value'], lower))
        # A closed numerical optimality certificate no longer depends on how
        # long the already completed solve was allowed to run. Every other
        # solve setting, physical state and learned cut must still match.
        closed_key = (_node_key(prob_data, tree, node, root, cuts, model_options,
                               _without_time_limit(actual_policy)) if closed else None)
        backend = 'bpc_lrp' if diagnostic.get('backend') == 'bpc' else 'gurobi_lrp'
        record = RefreshPolicy(policy, score['cost_star_value'], score['stage_cost_value'],
            score['theta_by_succ'], lower, closed, backend, diagnostic.get('status'),
            closed_key, float(allowance))
        entries[index] = (key, record)
    return RefreshHandoff(_semantics(prob_data, tree), entries)
