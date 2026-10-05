"""LRP SBC/Level Set configuration and explicit physical backend selection."""
from __future__ import annotations

from core.solver_settings import configured_gurobi_threads
import os

# Only paths without an LRP adapter remain disabled. Forward S2 now uses the
# original BPC switches through a physical-facility adapter.
_DISABLED = {
    "VRP_PHASE1_S3_LIFT_Y_BACKWARD": "0", "VRP_FORWARD_S3_USE_CONCORDE": "0",
    "VRP_FORWARD_S3_SOLVER": "gurobi", "VRP_FORWARD_S3_CONCORDE_THRESHOLD": "1000000000",
    "VRP_PHASE15_BACKEND": "none",
    "VRP_PHASE15_SCHEDULE": "off", "VRP_PHASE15_TOTAL_TIME_LIMIT": "0",
    "VRP_PHASE15_LATE_TOTAL_TIME_LIMIT": "0",
}
MAIN_ALGORITHM_PROFILE = dict(_DISABLED)
COMPARE_ALGORITHM_PROFILE = dict(_DISABLED)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    return default if raw in (None, "") else raw.lower() not in ("0", "false", "no", "off")


def _as_bool(value):
    return str(value).lower() not in ('0', 'false', 'no', 'off', '')


def _setting(name, default, kind=float):
    raw = os.environ.get("LRP_" + name, os.environ.get("VRP_" + name, default))
    return kind(raw)


def _first_setting(names, default, kind=float):
    """Read explicit aliases in order, without changing their old meaning."""
    for name in names:
        if name in os.environ:
            return kind(os.environ[name])
    return kind(default)


def configure_lrp_backends():
    # Phase-specific S3 ESP flags are read by the physical LRP selector.
    # Never inject zero defaults: absence and an explicit Gurobi request differ.
    os.environ.update(_DISABLED)


def configure_thread_defaults(default_cpu_budget=12):
    threads = configured_gurobi_threads()
    cpu_budget = max(1, _setting("CPU_BUDGET", default_cpu_budget, int))
    os.environ.setdefault("VRP_CPU_BUDGET", str(cpu_budget))
    processes = _setting("NUM_PROCESSES", cpu_budget, int)
    if min(threads, processes) < 1:
        raise ValueError("Thread and process counts must be positive")
    os.environ["VRP_GRB_THREADS"] = str(threads)
    # Do not turn the default into an apparent user override.  In particular,
    # Compare's editable parameter block must still work after bootstrap.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("VRP_EF_GRB_THREADS", str(cpu_budget))
    configure_lrp_backends()
    return cpu_budget


def configure_gurobi_threads(entrypoint="runtime"):
    import gurobipy as gp
    threads = configured_gurobi_threads()
    gp.setParam("Threads", threads)
    return threads


def bootstrap_solver_environment(entrypoint):
    cpu_budget = configure_thread_defaults()
    from multiprocessing import current_process
    # Spawn reimports main before the isolated worker creates its Env.
    if current_process().name != "MainProcess":
        return cpu_budget, configured_gurobi_threads()
    return cpu_budget, configure_gurobi_threads(entrypoint)


def load_algorithm_config(*, entry_defaults=None, **overrides):
    """Load shared options; explicit Python overrides take precedence.

    The historical VRP_PHASE2_TIME_LIMIT is the *outer* Phase-2 budget.
    LRP_PHASE2_TIME_LIMIT and phase2_time_limit retain their current LRP
    per-solve meaning, including when replaying an existing LRP snapshot.
    *_SUB_TIME_LIMIT accepts the original per-solve spelling as well.
    """
    from core.solution import AlgorithmConfig, PHYSICAL_FORWARD_DEFAULTS, PHYSICAL_CG_DEFAULTS
    configure_lrp_backends()
    entry_defaults = {} if entry_defaults is None else dict(entry_defaults)

    def setting(name, default, kind=float):
        return _setting(name, entry_defaults.get(name, default), kind)

    warmup_requested = any(prefix + suffix in os.environ
        for prefix in ('LRP_', 'VRP_')
        for suffix in ('S2_MIPGAP_EARLY', 'S2_MIPGAP_LATE', 'S2_MIPGAP_SWITCH'))
    options = dict(
        physical_cg_profile=setting('PHYSICAL_CG_PROFILE','off',str),
        physical_cg_options={key:setting('PHYSICAL_CG_'+key.upper(),default,type(default))
                             for key,default in PHYSICAL_CG_DEFAULTS.items()},
        physical_forward_profile=setting('PHYSICAL_FORWARD_PROFILE', 'off', str),
        physical_forward_options={key: setting('PHYS_' + key.upper(), default, type(default))
                                  for key, default in PHYSICAL_FORWARD_DEFAULTS.items()},
        lrp_static_bounds=setting('STATIC_BOUNDS', False, _as_bool),
        phase1_preseed_time_limit=setting('PHASE1_PRESEED_TIME_LIMIT', 0.),
        phase1_preseed_rounds=setting('PHASE1_PRESEED_ROUNDS', 2, int),
        phase1_tol=setting("PHASE1_TOL", 1e-2),
        phase1_mip_gap=setting("PHASE1_MIP_GAP", 1e-4),
        phase1_s2_mipgap_schedule=_setting('PHASE1_S2_MIPGAP_SCHEDULE',
            warmup_requested or entry_defaults.get('PHASE1_S2_MIPGAP_SCHEDULE', False), _as_bool),
        phase1_s2_mipgap_early=setting('S2_MIPGAP_EARLY', 1e-3),
        phase1_s2_mipgap_late=setting('S2_MIPGAP_LATE', 1e-4),
        phase1_s2_mipgap_switch=setting('S2_MIPGAP_SWITCH', 30, int),
        phase2_tol=setting("PHASE2_TOL", 0.01),
        phase1_iter_limit=setting("PHASE1_ITER_LIMIT", 100, int),
        phase2_iter_limit=setting("PHASE2_ITER_LIMIT", 0, int),
        phase1_time_limit=_first_setting((
            'LRP_PHASE1_TIME_LIMIT', 'LRP_PHASE1_SUB_TIME_LIMIT',
            'VRP_PHASE1_SUB_TIME_LIMIT', 'VRP_PHASE1_TIME_LIMIT'),
            entry_defaults.get('PHASE1_TIME_LIMIT', entry_defaults.get('PHASE1_SUB_TIME_LIMIT', 1800.0))),
        phase1_total_time_limit=setting("PHASE1_TOTAL_TIME_LIMIT", 0.0),
        phase1_forward_s2_time_limit=setting('PHASE1_FORWARD_S2_TIME_LIMIT', None,
                                             lambda value: None if value is None else float(value)),
        phase2_time_limit=_first_setting((
            'LRP_PHASE2_TIME_LIMIT', 'LRP_PHASE2_SUB_TIME_LIMIT',
            'VRP_PHASE2_SUB_TIME_LIMIT'),
            entry_defaults.get('PHASE2_TIME_LIMIT', entry_defaults.get('PHASE2_SUB_TIME_LIMIT', 1800.0))),
        phase2_total_time_limit=_first_setting((
            'LRP_PHASE2_TOTAL_TIME_LIMIT', 'VRP_PHASE2_TOTAL_TIME_LIMIT',
            'VRP_PHASE2_TIME_LIMIT'), entry_defaults.get('PHASE2_TOTAL_TIME_LIMIT', 10 * 3600.)),
        num_processes=setting("NUM_PROCESSES", max(1, setting('CPU_BUDGET', 12, int)), int),
        phase1_no_improve_limit=setting("PHASE1_NO_IMPROVE_LIMIT", 20, int),
        phase2_no_improve_limit=setting("PHASE2_NO_IMPROVE_LIMIT", 0, int),
        lambda_level=setting("LAMBDA_LEVEL", 0.3),
        mu_level=setting("MU_LEVEL", 0.5),
        norm_option=setting("NORM_OPTION", 1, int),
        levelset_iter_limit=setting("LEVELSET_ITER_LIMIT", 100, int),
        levelset_tol=setting("LEVELSET_TOL", 1.0),
        reset_levelset=True, adaptive_alpha=True,
        phase1_forward_period_dedup=setting("PHASE1_FORWARD_PERIOD_DEDUP", True, _as_bool),
        phase2_forward_period_dedup=setting("PHASE2_FORWARD_PERIOD_DEDUP", True, _as_bool),
        phase_forward_reuse=setting("PHASE_FORWARD_REUSE", True, _as_bool),
        phase15_backend=setting('PHASE15_BACKEND', 'none', str),
        phase15_total_time_limit=setting('PHASE15_TOTAL_TIME_LIMIT', 0.),
        phase15_schedule=setting('PHASE15_SCHEDULE', 'off', str),
        phase15_per_call_time_limit=setting('PHASE15_PER_CALL_TIME_LIMIT', 15.),
        phase15_on_demand_streak=setting('PHASE15_ON_DEMAND_STREAK', 2, int),
        phase15_late_total_time_limit=setting('PHASE15_LATE_TOTAL_TIME_LIMIT', 0.),
        phase15_late_per_call_time_limit=setting('PHASE15_LATE_PER_CALL_TIME_LIMIT', 180.),
        phase15_late_window=setting('PHASE15_LATE_WINDOW', 3, int),
        phase15_late_relative_gain=setting('PHASE15_LATE_RELATIVE_GAIN', 1e-4),
        phase2_inner_s3_rounds=setting("PHASE2_INNER_S3_ROUNDS", 30, int),
        phase2_inner_s3_time_limit=setting("PHASE2_INNER_S3_TIME_LIMIT", 300.0),
        phase2_s2_cut_time_limit=setting("PHASE2_S2_CUT_TIME_LIMIT", 0.0),
        phase2_s2_abs_tol_kappa=setting('PHASE2_S2_ABS_TOL_KAPPA', 0.5),
        phase2_s2_abs_tol_floor_share=setting('PHASE2_S2_ABS_TOL_FLOOR_SHARE', 0.5),
        phase2_s2_abs_tol_rel_cap=setting('PHASE2_S2_ABS_TOL_REL_CAP', 0.02),
        phase1_lazy_threshold=setting('S2_LAZY_THRESHOLD', 8, int),
        phase2_lazy_threshold=setting('PHASE2_S2_LAZY_THRESHOLD', 256, int),
        phase1_stability_handoff=setting('PHASE1_STABILITY_HANDOFF', True, _as_bool),
        phase1_stability_window=setting('PHASE1_STABILITY_WINDOW', 3, int),
        phase1_stability_min_iteration=setting('PHASE1_STABILITY_MIN_ITERATION', 6, int),
        phase1_stagnation_window=setting('PHASE1_STAGNATION_WINDOW', 5, int),
        phase1_stagnation_rel_tol=setting('PHASE1_STAGNATION_REL_TOL', 0.0),
        phase1_stagnation_min_iteration=setting('PHASE1_STAGNATION_MIN_ITERATION', 8, int),
        phase2_no_new_cut_limit=setting('PHASE2_NO_NEW_CUT_LIMIT', 3, int),
        phase15_late_strategy=setting('PHASE15_LATE_STRATEGY', 'a_then_b', str),
    )
    overrides = {k: v for k, v in overrides.items() if v is not None}
    # Legacy Python keyword overrides must beat environment/default values,
    # but an explicitly supplied canonical keyword remains authoritative.
    for phase in (1, 2):
        canonical = f'phase{phase}_time_limit'
        legacy = f'phase{phase}_sub_time_limit'
        if canonical not in overrides:
            if legacy in overrides:
                options[canonical] = overrides[legacy]
            elif 'sub_time_limit' in overrides:
                options[canonical] = overrides['sub_time_limit']
    options.update(overrides)
    if ('phase1_mip_gap' in overrides or 'PHASE1_MIP_GAP' in entry_defaults
            or any(name in os.environ for name in ('LRP_PHASE1_MIP_GAP', 'VRP_PHASE1_MIP_GAP'))):
        # An existing explicit constant request must not silently turn into a
        # warmup schedule just because the entry point now enables that mode.
        options['phase1_s2_mipgap_schedule'] = False
    return AlgorithmConfig(**options)


def configure_main_solver(_cfg, _inst):
    configure_lrp_backends()


def configure_exact_backend(_num_customers=None):
    configure_lrp_backends()


def configure_forward_stage3_solver():
    configure_lrp_backends()


def configure_compare_exact_backend(_num_customers=None):
    configure_lrp_backends()
    applied = dict(_DISABLED)
    if _setting("COMPARE_FORCE_GUROBI", False, _as_bool):
        os.environ["LRP_S3_BACKEND"] = "gurobi"
        os.environ["LRP_S2_FORWARD_SOLVER"] = "gurobi"
        os.environ["LRP_PHASE1_S2_ORACLE"] = "gurobi"
        os.environ["LRP_PHASE2_S2_ORACLE"] = "gurobi"
        applied["LRP_S3_BACKEND"] = "gurobi"
        applied["LRP_S2_FORWARD_SOLVER"] = "gurobi"
        applied["LRP_PHASE1_S2_ORACLE"] = "gurobi"
        applied["LRP_PHASE2_S2_ORACLE"] = "gurobi"
        print("[solver-config] compare backend policy: FORCE all-Gurobi")
    return applied


def apply_algorithm_profile(profile, *, override_env=True):
    # A foreign profile cannot reopen paths without a physical LRP adapter.
    if profile != MAIN_ALGORITHM_PROFILE and profile != COMPARE_ALGORITHM_PROFILE:
        raise ValueError("LRP entry points support only the verified LRP profiles")
    configure_lrp_backends()
