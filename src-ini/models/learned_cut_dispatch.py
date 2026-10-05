"""Shared policy for learned Stage-3→Stage-2 cuts (static RouteCut is separate)."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from fractions import Fraction


MIP_PURPOSE = "mip"
DUAL_LP_PURPOSE = "dual_lp"
VALID_LEARNED_CUT_PURPOSES = (MIP_PURPOSE, DUAL_LP_PURPOSE)


def resolve_learned_cut(model, target_name: str, pi_dict, intercept):
    """Resolve one archived learned cut; missing vars with nonzero coeff error."""
    target = model.getVarByName(str(target_name))
    if target is None:
        raise KeyError(f"model has no {target_name} learned-cut target")

    constant = float(intercept)
    if not math.isfinite(constant):
        raise ValueError(f"non-finite learned-cut intercept: {intercept!r}")

    terms = []
    for variable_name, raw_coefficient in pi_dict.items():
        coefficient = float(raw_coefficient)
        if not math.isfinite(coefficient):
            raise ValueError(
                "non-finite learned-cut coefficient for "
                f"{variable_name!r}: {raw_coefficient!r}"
            )
        if coefficient == 0.0:
            continue
        variable = model.getVarByName(str(variable_name))
        if variable is None:
            raise KeyError(
                "model has no variable for nonzero learned-cut term "
                f"{variable_name!r}"
            )
        terms.append((variable, coefficient))
    return terms, constant, target


def normalize_learned_cut_purpose(value) -> str:
    """Validate Stage-2 model purpose: ``mip`` or ``dual_lp`` (Python arg only)."""
    if value is None:
        value = MIP_PURPOSE
    purpose = str(value).strip().lower().replace("-", "_")
    if purpose not in VALID_LEARNED_CUT_PURPOSES:
        allowed = ", ".join(VALID_LEARNED_CUT_PURPOSES)
        raise ValueError(
            f"invalid learned S3 cut model purpose: {value!r}; "
            f"expected one of {allowed}"
        )
    return purpose


def install_as_explicit_rows(
    cut_count: int,
    lazy_threshold: int,
    purpose: str,
) -> bool:
    """Choose the single production representation for learned cuts.

    Ordinary Stage-2 MIPs use the accepted threshold-based explicit/lazy
    policy.  A model that will be relaxed for dual multipliers must contain
    the complete archive as ordinary rows because ``Model.relax()`` does not
    copy a Python callback archive.
    """
    resolved_purpose = normalize_learned_cut_purpose(purpose)
    return (
        resolved_purpose == DUAL_LP_PURPOSE
        or int(cut_count) <= int(lazy_threshold)
    )


def record_dispatch(model, *, explicit: int = 0, lazy: int = 0) -> None:
    """Accumulate learned-cut counts on a Gurobi model."""
    model._s3_learned_cut_count = int(
        getattr(model, "_s3_learned_cut_count", 0)
    ) + int(explicit) + int(lazy)
    model._s3_explicit_cut_count = int(
        getattr(model, "_s3_explicit_cut_count", 0)
    ) + int(explicit)
    model._s3_lazy_cut_count = int(
        getattr(model, "_s3_lazy_cut_count", 0)
    ) + int(lazy)


def replace_lazy_cut_archive(model, rows):
    """Install immutable lazy-cut archive (replace generation, no in-place mutate)."""
    frozen_rows_list = []
    for terms, intercept, theta_var in rows:
        frozen_terms = []
        for variable, coefficient in terms:
            frozen_coefficient = float(coefficient)
            if not math.isfinite(frozen_coefficient):
                raise ValueError("lazy-cut callback values must be finite")
            frozen_terms.append((variable, frozen_coefficient))
        frozen_intercept = float(intercept)
        if not math.isfinite(frozen_intercept):
            raise ValueError("lazy-cut callback values must be finite")
        frozen_rows_list.append(
            (tuple(frozen_terms), frozen_intercept, theta_var)
        )
    frozen_rows = tuple(frozen_rows_list)
    model._lazy_cuts = frozen_rows
    model._s3_lazy_archive_generation = int(
        getattr(model, "_s3_lazy_archive_generation", 0)
    ) + 1
    model._s3_lazy_callback_cache_v1 = None
    return frozen_rows


def initialize_callback_stats(model) -> None:
    """Initialize callback counters without overwriting an active solve."""
    for name in (
        "_s3_lazy_callback_calls",
        "_s3_lazy_callback_checks",
        "_s3_lazy_callback_added",
        "_s3_lazy_callback_add_calls",
    ):
        if not hasattr(model, name):
            setattr(model, name, 0)


def begin_lazy_cut_optimize(model) -> None:
    """Start one optimize epoch for an archived lazy-cut model.

    ``cbLazy`` rows are guaranteed for the active optimize call.  Level Set
    can re-optimize one model, so indices added in an earlier call must become
    eligible again.  Counters stay cumulative; uniqueness is per call.
    """
    initialize_callback_stats(model)
    model._s3_optimize_epochs = int(
        getattr(model, "_s3_optimize_epochs", 0)
    ) + 1
    model._lazy_cut_added = set()


def record_callback(model, *, checked: int, added: int, add_calls: int = 0) -> None:
    initialize_callback_stats(model)
    model._s3_lazy_callback_calls += 1
    model._s3_lazy_callback_checks += int(checked)
    model._s3_lazy_callback_added += int(added)
    model._s3_lazy_callback_add_calls += int(add_calls)


def _lazy_cut_violation(
    terms,
    intercept: float,
    theta_solution: float,
    solution,
) -> bool:
    """Return whether one archived cut is violated in exact binary64 math.

    The common path uses ``math.fsum``.  Only cancellation cases whose sign
    cannot be certified by a conservative floating-point error envelope are
    recomputed as exact rational arithmetic over the supplied binary64
    values.  This avoids a numerical dead zone in which a genuinely violated
    archived cut could otherwise be skipped.
    """
    constant = float(intercept)
    theta = float(theta_solution)
    if not math.isfinite(constant) or not math.isfinite(theta):
        raise ValueError("lazy-cut callback values must be finite")

    products = []
    product_abs = 0.0
    exact_required = False
    exact_inputs = []
    for variable, coefficient in terms:
        coeff = float(coefficient)
        value = float(solution[variable])
        if not math.isfinite(coeff) or not math.isfinite(value):
            raise ValueError("lazy-cut callback values must be finite")
        product = coeff * value
        if not math.isfinite(product):
            exact_required = True
        else:
            products.append(product)
            product_abs += abs(product)
        exact_inputs.append((coeff, value))

    if not exact_required:
        try:
            approximate = math.fsum((constant, -theta, *products))
        except OverflowError:
            exact_required = True
            approximate = math.inf
        magnitude = abs(constant) + abs(theta) + product_abs
        unit_roundoff = 2.0 ** -53
        term_count = len(products) + 2
        gamma = (
            term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)
            if term_count * unit_roundoff < 1.0
            else math.inf
        )
        # Conservative FP error bound for product sum
        error_bound = (
            (unit_roundoff / (1.0 - unit_roundoff) + gamma)
            * magnitude
            * 4.0
        )
        error_bound += len(products) * math.ulp(0.0)
        if not exact_required:
            if approximate > error_bound:
                return True
            if approximate < -error_bound:
                return False
    exact = Fraction.from_float(constant) - Fraction.from_float(theta)
    for coeff, value in exact_inputs:
        exact += Fraction.from_float(coeff) * Fraction.from_float(value)
    return exact > 0


def _compile_lazy_cut_callback_cache(model, quicksum):
    """Compile one immutable archive generation for the MIPSOL hot path."""
    raw_rows = tuple(getattr(model, "_lazy_cuts", ()))

    # Compile archive (also freezes legacy list attributes)
    rows = replace_lazy_cut_archive(model, raw_rows)
    cut_ids = tuple(range(len(rows)))

    variables = set()
    for terms, _intercept, theta_var in rows:
        variables.add(theta_var)
        variables.update(variable for variable, _coefficient in terms)
    variable_list = tuple(variables)
    variable_index = {
        variable: position
        for position, variable in enumerate(variable_list)
    }

    # Integer indices let the unchanged exact violation routine read the
    # fresh cbGetSolution vector directly.  Coefficient order, fsum/error
    # envelope, and Fraction fallback remain byte-for-byte in one routine.
    violation_terms = tuple(
        tuple(
            (variable_index[variable], coefficient)
            for variable, coefficient in terms
        )
        for terms, _intercept, _theta_var in rows
    )
    theta_indices = tuple(
        variable_index[theta_var]
        for _terms, _intercept, theta_var in rows
    )
    lhs = tuple(
        quicksum(
            coefficient * variable
            for variable, coefficient in terms
        ) + intercept
        for terms, intercept, _theta_var in rows
    )
    cache = {
        "generation": int(model._s3_lazy_archive_generation),
        "rows": rows,
        "cut_ids": cut_ids,
        "quicksum": quicksum,
        "variables": variable_list,
        "violation_terms": violation_terms,
        "theta_indices": theta_indices,
        "lhs": lhs,
    }
    model._s3_lazy_callback_cache_v1 = cache
    return cache


def _lazy_cut_callback_cache_for(model, quicksum):
    cache = getattr(model, "_s3_lazy_callback_cache_v1", None)
    if cache is not None and (
        cache["generation"]
        == int(getattr(model, "_s3_lazy_archive_generation", 0))
        and cache["rows"] is getattr(model, "_lazy_cuts", ())
        and cache["quicksum"] is quicksum
    ):
        return cache
    return _compile_lazy_cut_callback_cache(model, quicksum)


def enforce_lazy_cut_archive(model, where, *, mipsol_code, quicksum,
                             violation_tolerance: float = 0.0) -> None:
    """Submit every violated archived cut on each MIPSOL."""
    if where != mipsol_code:
        return
    initialize_callback_stats(model)
    if not hasattr(model, "_lazy_cut_added"):
        model._lazy_cut_added = set()

    tolerance = float(violation_tolerance)
    if not math.isfinite(tolerance) or tolerance != 0.0:
        raise ValueError(
            "exact lazy-cut enforcement requires zero violation tolerance"
        )

    cache = _lazy_cut_callback_cache_for(model, quicksum)
    rows = cache["rows"]
    if not rows:
        record_callback(model, checked=0, added=0, add_calls=0)
        return

    # Always query the current incumbent.  Only immutable structure is cached.
    solution = model.cbGetSolution(cache["variables"])

    added = 0
    add_calls = 0
    for position, (_terms, intercept, theta_var) in enumerate(rows):
        cut_id = cache["cut_ids"][position]
        violated = _lazy_cut_violation(
            cache["violation_terms"][position],
            intercept,
            solution[cache["theta_indices"][position]],
            solution,
        )
        if violated:
            # A fresh TempConstr/cbLazy call is retained for every violated row
            # at every MIPSOL; only the immutable LinExpr is reused.
            model.cbLazy(cache["lhs"][position] <= theta_var)
            add_calls += 1
            if cut_id not in model._lazy_cut_added:
                model._lazy_cut_added.add(cut_id)
                added += 1
    record_callback(
        model,
        checked=len(rows),
        added=added,
        add_calls=add_calls,
    )


def optimize_with_learned_cut_lifecycle(
    model,
    callback=None,
):
    """Run one optimize call with the required lazy-cut epoch lifecycle."""
    if getattr(model, "_lazy_cuts", ()) and callback is None:
        raise ValueError("a callback is required while learned lazy cuts remain")
    begin_lazy_cut_optimize(model)
    if callback is None:
        return model.optimize()
    return model.optimize(callback)


def mark_s2_build_started() -> float:
    return time.perf_counter()


def mark_s2_build_finished(model, started: float) -> None:
    model._s2_build_seconds = max(0.0, time.perf_counter() - float(started))


def collect_s2_model_stats(model, *, solve_seconds: float | None = None) -> dict:
    """Return a process-safe statistics payload for one Stage-2 model."""
    if solve_seconds is None:
        try:
            solve_seconds = float(model.Runtime)
        except Exception:
            solve_seconds = 0.0
    cuts = int(getattr(model, "_s3_learned_cut_count", 0))
    explicit = int(getattr(model, "_s3_explicit_cut_count", 0))
    lazy = int(getattr(model, "_s3_lazy_cut_count", 0))
    return {
        "models": 1,
        "backend_gurobi": 1,
        "backend_dp": 0,
        "backend_bp": 0,
        "backend_probe": 0,
        "cuts": cuts,
        "explicit": explicit,
        "lazy": lazy,
        "callback_calls": int(getattr(model, "_s3_lazy_callback_calls", 0)),
        "callback_checks": int(getattr(model, "_s3_lazy_callback_checks", 0)),
        "callback_added": int(getattr(model, "_s3_lazy_callback_added", 0)),
        "callback_add_calls": int(
            getattr(model, "_s3_lazy_callback_add_calls", 0)
        ),
        "build_seconds": float(getattr(model, "_s2_build_seconds", 0.0)),
        "solve_seconds": max(0.0, float(solve_seconds)),
        "epochs": int(getattr(model, "_s3_optimize_epochs", 0)),
    }


def merge_s2_model_stats(stats: Iterable[Mapping]) -> dict:
    """Aggregate process-safe Stage-2 model statistics."""
    out = {
        "models": 0,
        "backend_gurobi": 0,
        "backend_dp": 0,
        "backend_bp": 0,
        "backend_probe": 0,
        "cuts": 0,
        "explicit": 0,
        "lazy": 0,
        "callback_calls": 0,
        "callback_checks": 0,
        "callback_added": 0,
        "callback_add_calls": 0,
        "epochs": 0,
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
    }
    for item in stats:
        if not item:
            continue
        for name in (
            "models",
            "backend_gurobi",
            "backend_dp",
            "backend_bp",
            "backend_probe",
            "cuts",
            "explicit",
            "lazy",
            "callback_calls",
            "callback_checks",
            "callback_added",
            "callback_add_calls",
            "epochs",
        ):
            out[name] += int(item.get(name, 0))
        for name in ("build_seconds", "solve_seconds"):
            value = float(item.get(name, 0.0))
            if math.isfinite(value) and value > 0.0:
                out[name] += value
    return out


def format_s2_model_stats(stats: Mapping) -> str:
    """Stable, grep-friendly A/B summary."""
    return (
        f"models={int(stats.get('models', 0))} "
        f"gurobi={int(stats.get('backend_gurobi', 0))} "
        f"dp={int(stats.get('backend_dp', 0))} "
        f"bp={int(stats.get('backend_bp', 0))} "
        f"probes={int(stats.get('backend_probe', 0))} "
        f"cuts={int(stats.get('cuts', 0))} "
        f"explicit={int(stats.get('explicit', 0))} "
        f"lazy={int(stats.get('lazy', 0))} "
        f"cb_calls={int(stats.get('callback_calls', 0))} "
        f"cb_checks={int(stats.get('callback_checks', 0))} "
        f"cb_added={int(stats.get('callback_added', 0))} "
        f"cb_add_calls={int(stats.get('callback_add_calls', 0))} "
        f"epochs={int(stats.get('epochs', 0))} "
        f"build={float(stats.get('build_seconds', 0.0)):.6f}s "
        f"solve={float(stats.get('solve_seconds', 0.0)):.6f}s"
    )
