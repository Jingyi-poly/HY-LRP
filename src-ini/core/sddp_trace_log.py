"""Structured SDDP forward / backward trace lines for experiment logs."""
from __future__ import annotations

_MAX_TERMS = 16


def _fmt_terms(coeffs: dict[str, float], max_terms: int = _MAX_TERMS) -> str:
    items = [(k, float(c)) for k, c in coeffs.items() if abs(float(c)) >= 1e-10]
    items.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    if not items:
        return "0"
    shown = items[:max_terms]
    tail = len(items) - len(shown)
    expr = " + ".join(f"{c:+.6g}·{k}" for k, c in shown)
    if tail > 0:
        expr += f" + ...({tail} more)"
    return expr


def format_benders_cut(pi_dict: dict, v_value: float, rhs_var: str) -> str:
    """Cut stored as Σ π_k·x_k + v ≤ rhs."""
    lin = _fmt_terms(pi_dict)
    return f"{lin} + {float(v_value):.6g} ≤ {rhs_var}"


def log_forward_stage1(investment_cost: float, eta_by_succ: dict[int, float]) -> None:
    eta_str = " ".join(f"eta[{k}]={v:.2f}" for k, v in sorted(eta_by_succ.items()))
    print(f"    [Forward] stage1 | investment_cost={investment_cost:.2f} | {eta_str}")


def log_forward_stage2(
    second_ind: int,
    omega,
    t,
    stage_cost: float,
    theta_by_succ: dict[int, float],
    obj: float,
) -> None:
    th_str = " ".join(f"theta[{k}]={v:.2f}" for k, v in sorted(theta_by_succ.items()))
    print(
        f"    [Forward] stage2 ω={omega},t={t} node={second_ind} | "
        f"stage_cost={stage_cost:.2f} | {th_str} | obj={obj:.2f}"
    )


def log_forward_stage3(third_ind: int, v: int, omega, tsp_cost: float) -> None:
    print(f"    [Forward] stage3 node={third_ind} v={v} ω={omega} | TSP_cost={tsp_cost:.2f}")


def log_backward_cut_s3_s2(third_ind: int, v: int, pi_dict: dict, v_value: float, cut_idx: int = 0) -> None:
    rhs = f"theta[{third_ind}]"
    print(
        f"    [Backward 3→2] node={third_ind}(v={v}) cut#{cut_idx}: "
        f"{format_benders_cut(pi_dict, v_value, rhs)}"
    )


def log_backward_cut_s2_s1(second_ind: int, omega, t, pi_dict: dict, v_value: float) -> None:
    rhs = f"eta[{second_ind}]"
    print(
        f"    [Backward 2→1] node={second_ind}(ω={omega},t={t}): "
        f"{format_benders_cut(pi_dict, v_value, rhs)}"
    )
