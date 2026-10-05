"""Cut archive helpers shared by Phase-1 SBC (and Phase-2 cut insertion).

``clean_pi`` / ``add_unique_cut`` are the production entry points.  Phase-1 uses
``clean_pi`` before inserting; both phases call ``add_unique_cut`` to keep the
strongest intercept per exact slope.
"""

import math
from fractions import Fraction


# ---------------------------------------------------------------------------
# Cut coefficient cleaning (Phase1 SBC 专用, Phase2 不调用)
# ---------------------------------------------------------------------------
EPS_CUT_COEF = 1e-6


def _fraction_to_finite_float_down(value, *, label):
    """Round an exact rational toward ``-inf`` into finite binary64."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        if value > 0:
            return math.nextafter(math.inf, 0.0)
        raise ValueError(f"{label} is below the finite binary64 range") from exc
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    if not math.isfinite(rounded):
        raise ValueError(f"{label} is non-finite after directed rounding")
    return rounded


def sum_binary64_lower_bounds(*values):
    """Sum finite binary64 lower bounds without rounding the result upward."""
    exact = Fraction(0)
    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite lower-bound summand: {raw_value!r}")
        exact += Fraction.from_float(value)
    return _fraction_to_finite_float_down(exact, label="lower-bound sum")


def clean_pi(pi_dict, v_value, var_ub=1.0):
    """Treat ``|coefficient| <= 1e-6`` as zero without invalidating a cut.

    Every Phase-1 slope variable is bounded by ``[0, var_ub]``.  Removing a
    positive coefficient only weakens the lower-bounding cut.  For a removed
    negative coefficient ``c``, lower the intercept by ``-c * var_ub`` using
    exact binary64 rationals and directed rounding toward ``-inf``.  Thus the
    cleaned right-hand side never exceeds the original one at any feasible
    binary state.

    返回 (pi_clean, v_clean, n_dropped, slack).
    """
    intercept = float(v_value)
    if not math.isfinite(intercept):
        raise ValueError(f"non-finite cut intercept: {v_value!r}")

    finite_pi = {}
    for key, raw_value in pi_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"cut variable name must be a string: {key!r}")
        coefficient = float(raw_value)
        if not math.isfinite(coefficient):
            raise ValueError(
                f"non-finite cut coefficient for {key!r}: {raw_value!r}"
            )
        finite_pi[key] = coefficient

    if not finite_pi:
        return {}, intercept, 0, 0.0

    upper_bound = float(var_ub)
    if not math.isfinite(upper_bound) or upper_bound < 0.0:
        raise ValueError(f"invalid cut variable upper bound: {var_ub!r}")

    pi_clean = {}
    slack_exact = Fraction(0)
    n_dropped = 0
    for k, c in finite_pi.items():
        if abs(c) <= EPS_CUT_COEF:
            if c < 0.0:
                slack_exact += (
                    -Fraction.from_float(c)
                    * Fraction.from_float(upper_bound)
                )
            n_dropped += 1
        else:
            pi_clean[k] = c
    clean_intercept_exact = Fraction.from_float(intercept) - slack_exact
    clean_intercept = _fraction_to_finite_float_down(
        clean_intercept_exact, label="cleaned cut intercept"
    )
    return pi_clean, clean_intercept, n_dropped, float(slack_exact)


def _canonical_exact_slope(pi_dict):
    """Canonical exact slope; missing entries and signed/exact zero agree."""
    canonical = []
    for key, value in pi_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"cut variable name must be a string: {key!r}")
        coefficient = float(value)
        if not math.isfinite(coefficient):
            raise ValueError(
                f"non-finite cut coefficient for {key!r}: {value!r}"
            )
        if coefficient != 0.0:
            canonical.append((key, coefficient.hex()))
    return tuple(sorted(canonical))


def add_unique_cut(
    cut_list, pi_dict, v_value, *_legacy_positional_tolerances,
    **_legacy_tolerances,
):
    """Add a cut without approximate deletion.

    Different slopes are always retained, even when numerically close.  For
    an exactly identical canonical slope, the larger intercept globally
    dominates the smaller one, so the archive keeps exactly the strongest
    intercept.  ``True`` means the archive was added to or strengthened.

    Legacy tol kwargs ignored.
    """
    intercept = float(v_value)
    if not math.isfinite(intercept):
        raise ValueError(f"non-finite cut intercept: {v_value!r}")
    canonical = _canonical_exact_slope(pi_dict)
    stored_pi = {
        key: value for key, value in pi_dict.items() if float(value) != 0.0
    }
    for index, (pi_old, v_old) in enumerate(cut_list):
        if _canonical_exact_slope(pi_old) != canonical:
            continue
        old_intercept = float(v_old)
        if not math.isfinite(old_intercept):
            raise ValueError(f"non-finite archived cut intercept: {v_old!r}")
        if intercept > old_intercept:
            cut_list[index] = [stored_pi, v_value]
            return True
        return False
    cut_list.append([stored_pi, v_value])
    return True
