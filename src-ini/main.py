"""Run stochastic LRP through the original SBC -> Level Set algorithm."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_ROOT))

from setup.solver_config import (bootstrap_solver_environment,
    load_algorithm_config as _load_algorithm_config,
    configure_exact_backend as _configure_exact_backend,
    configure_forward_stage3_solver as _configure_forward_stage3_solver)

# Keep the original CPU-budget editing point before importing solver libraries.
_CPU_BUDGET = max(1, int(os.environ.get("LRP_CPU_BUDGET", os.environ.get("VRP_CPU_BUDGET", "12"))))
os.environ.setdefault("VRP_CPU_BUDGET", str(_CPU_BUDGET))
os.environ.setdefault("VRP_LEVELSET_TRACE", "0")  # Phase2: outer bounds only; set 1 for LS detail
# Continue unfinished S3 work within this run; no saved run is loaded.
os.environ.setdefault("LRP_S3_RESUME_INCOMPLETE_CHECKPOINT", "1")
_, _GRB_THREADS = bootstrap_solver_environment("main")

from core.instance import DEFAULT_LRP_INSTANCE, LRPConfig
from core.run_logging import console_log_path, setup_run_logging
from setup.reporting import (prepare_output_directory, write_json, print_algorithm_config,
    gap_pct as _gap_pct, print_phase_result as _print_phase_result,
    print_instance_summary as _print_instance_summary, merge_results as _merge_results,
    save_outputs as _save_outputs, print_summary as _print_summary)
from setup.runner import run_main_solver

# 数据规模集中在这里改；命令行 / LRP_* / VRP_* 仍可覆盖。None 保留源数据设置。
LRP_INSTANCE = Path(DEFAULT_LRP_INSTANCE).with_name("kroA100_I10_J90_T3_S3_scale")
DEFAULT_NUM_CUSTOMERS = 90
DEFAULT_NUM_FACILITIES = 10
DEFAULT_T = 5                    # delivery periods
DEFAULT_NUM_SCENARIOS = 3        # scenarios
DEFAULT_SCENARIO_SEED = 42
DEFAULT_LOCATION_PERIODS = (1, 3, 5)  # 须含 1、严格递增、≤T；None 保留源日期


def _instance_int(name, explicit, default=None):
    """Explicit argument > LRP_* > original VRP_* > entry-point setting."""
    if explicit is not None:
        return explicit
    value = os.environ.get("LRP_" + name.upper(),
                           os.environ.get("VRP_" + name.upper(), default))
    return int(value) if isinstance(value, str) else value


def _validate_lrp_data_settings():
    """Reject obsolete input requests instead of silently loading different data."""
    source = os.environ.get("LRP_DATA_SOURCE", os.environ.get("VRP_DATA_SOURCE", "lrp"))
    if source.strip().lower() != "lrp":
        raise ValueError("DATA_SOURCE must be lrp; choose LRP_INSTANCE / --instance for physical LRP data")
    retired = ("HFVRP_FILE", "NUM_PER_TYPE", "NUM_TYPES", "DEMAND_DIST", "DEMAND_CV",
               "DEMAND_ACTIVE_PROB", "C_OUT", "C_PER_KM", "B_T0", "OPS_PER_YEAR",
               "PURCHASE_COST_SCALE", "CAPACITY_SCALE")
    requested = [prefix + name for name in retired for prefix in ("LRP_", "VRP_")
                 if os.environ.get(prefix + name, "").strip()]
    if requested:
        raise ValueError("Unsupported investment data settings for LRP: " + ", ".join(requested)
                         + ". Costs, capacities and demand laws come from the selected LRP data recipe; "
                         "use num_facilities / --facilities for distinct physical facilities.")


def default_instance(*, instance=None, out=None,
                     num_customers=None, num_facilities=None,
                     T=None, num_scenarios=None, scenario_seed=None,
                     location_periods=None):
    _validate_lrp_data_settings()
    if location_periods is None:
        location_periods = DEFAULT_LOCATION_PERIODS
    elif location_periods is not None:
        location_periods = tuple(location_periods)
    return LRPConfig(LRP_INSTANCE if instance is None else instance, out,
                     num_customers=_instance_int("num_customers", num_customers, DEFAULT_NUM_CUSTOMERS),
                     num_facilities=_instance_int("num_facilities", num_facilities, DEFAULT_NUM_FACILITIES),
                     T=_instance_int("T", T, DEFAULT_T),
                     num_scenarios=_instance_int("num_scenarios", num_scenarios, DEFAULT_NUM_SCENARIOS),
                     scenario_seed=_instance_int("scenario_seed", scenario_seed, DEFAULT_SCENARIO_SEED),
                     location_periods=location_periods)


def load_algorithm_config(**overrides):
    """Keep the original main.py editing location; environment overrides defaults.

    Shared loading/advanced LRP options live in setup.solver_config. These are
    the original main-entry defaults, separate from compare's tighter settings.
    """
    # An explicit legacy forward profile still selects that profile by itself.
    # If both profiles are explicitly enabled, shared validation rejects it.
    forward_profile = overrides.get("physical_forward_profile")
    if forward_profile is None:
        forward_profile = os.environ.get("LRP_PHYSICAL_FORWARD_PROFILE",
                                        os.environ.get("VRP_PHYSICAL_FORWARD_PROFILE", "off"))
    return _load_algorithm_config(entry_defaults={
        # Cold start: original Phase 1 -> Phase 2, with the verified CG path.
        # Change settings here as before; CLI / LRP_* / VRP_* take precedence.
        "PHYSICAL_CG_PROFILE": "cg" if forward_profile == "off" else "off",
        "PHYSICAL_CG_EPOCH_WALL": 250.,
        "PHYSICAL_CG_GLOBAL_POLICY_POOL_CAP": 10.,
        "PHYSICAL_CG_JOINT_SHARED_CAP": 220.,
        "PHYSICAL_CG_JOINT_NODE_CAP": 120.,
        "PHYSICAL_CG_MAX_JOINT_NODES": 2,
        "PHYSICAL_CG_PRICING_CALL_CAP": 15.,
        "PHYSICAL_CG_CG_MAX_ROUNDS": 50,
        "PHYSICAL_CG_COMMIT_AUDIT_REFRESH_RESERVE": 15.,
        "PHYSICAL_CG_PACED_BACKWARD_WALL": 300.,
        "PHYS_BACKWARD_EPOCH_BUDGET_S": 300.,
        "PHYS_POOL_MAX_ROUTES_PER_NODE_FACILITY": 1000,
        "PHYS_POOL_MIP_GAP": 0.,
        "PHASE1_TOL": 0.01,
        "PHASE2_TOL": 0.01,
        "PHASE1_ITER_LIMIT": 10000,
        "PHASE1_TOTAL_TIME_LIMIT": 0.,  # no time cutoff; use gap / consecutive no improvement
        "PHASE2_TOTAL_TIME_LIMIT": 10 * 3600.,  # old VRP_PHASE2_TIME_LIMIT
        "PHASE1_SUB_TIME_LIMIT": 30.,
        "PHASE1_FORWARD_S2_TIME_LIMIT": 600.,
        "PHASE1_S2_MIPGAP_SCHEDULE": True,  # S2: early 1e-3, late 1e-4 (iteration 30)
        "PHASE2_SUB_TIME_LIMIT": float("inf"),
        "NUM_PROCESSES": _CPU_BUDGET,
        "PHASE1_NO_IMPROVE_LIMIT": 20,
        "PHASE1_STAGNATION_WINDOW": 0,  # avoid a second, shorter stagnation threshold
        "PHASE1_STABILITY_HANDOFF": False,
        "PHASE2_NO_IMPROVE_LIMIT": 20,
        "LEVELSET_ITER_LIMIT": 100,
        "LEVELSET_TOL": 1.,
        "LAMBDA_LEVEL": 0.3,
        "MU_LEVEL": 0.5,
        "NORM_OPTION": 1,
    }, **overrides)


def run_solver(cfg, log_path, inst=None, *, config=None):
    """Original Python entry point, including an already built LRP instance."""
    kwargs = {} if inst is None else {"inst": inst}
    return run_main_solver(cfg, log_path,
                           config=load_algorithm_config() if config is None else config,
                           **kwargs)


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--instance", type=Path, default=LRP_INSTANCE,
                        help="LRP directory containing arrays.npz and metadata.json")
    parser.add_argument("--out", type=Path, help="Result directory; an existing directory is allowed if run output files are absent")
    parser.add_argument("--customers", type=int,
                        help="Retain the first N customers from the loaded instance")
    parser.add_argument("--facilities", type=int,
                        help="Retain the first N distinct facilities from the loaded instance")
    parser.add_argument("--periods", dest="T", type=int, help="Delivery periods (also VRP_T)")
    parser.add_argument("--scenarios", dest="num_scenarios", type=int,
                        help="Scenario trajectories (also VRP_NUM_SCENARIOS)")
    parser.add_argument("--scenario-seed", type=int,
                        help="Activity/demand sampling seed (also VRP_SCENARIO_SEED)")
    parser.add_argument("--location-periods", type=int, nargs="+",
                        help="Explicit facility decision dates, starting at 1; default retains source dates")
    parser.add_argument("--tol", type=float, help="Phase 2 relative bound-gap tolerance")
    parser.add_argument("--phase1-tol", type=float)
    parser.add_argument("--time-limit", type=float, help="Phase 2 total time budget in seconds")
    parser.add_argument("--phase1-time-limit", type=float, help="Phase 1 total time budget in seconds")
    parser.add_argument("--subproblem-time-limit", type=float, help="Gurobi time limit per subproblem")
    parser.add_argument("--phase1-iterations", type=int)
    parser.add_argument("--phase2-iterations", type=int, help="Phase 2 iteration limit, zero disables it")
    parser.add_argument("--levelset-iterations", type=int)
    parser.add_argument("--levelset-tol", type=float)
    parser.add_argument("--norm", type=int, choices=(1, 2), help="Level Set proximal norm")
    parser.add_argument("--lambda-level", type=float)
    parser.add_argument("--mu-level", type=float)
    parser.add_argument("--processes", type=int, help="Number of node worker processes")
    return parser


def config_from_args(args):
    positive = ("subproblem_time_limit",)
    nonnegative = ("time_limit", "phase1_time_limit", "tol", "phase1_tol", "levelset_tol", "phase1_iterations", "phase2_iterations")
    for key in positive:
        value = getattr(args, key, None)
        if value is not None and not 0 < value < float("inf"):
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    for key in nonnegative:
        value = getattr(args, key, None)
        if value is not None and not 0 <= value < float("inf"):
            raise ValueError(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.processes is not None and args.processes < 1:
        raise ValueError("--processes must be positive")
    return load_algorithm_config(
        phase1_tol=args.phase1_tol, phase2_tol=args.tol,
        phase1_total_time_limit=args.phase1_time_limit,
        phase2_total_time_limit=args.time_limit,
        phase1_time_limit=args.subproblem_time_limit,
        phase1_forward_s2_time_limit=args.subproblem_time_limit,
        phase2_time_limit=args.subproblem_time_limit,
        phase1_iter_limit=args.phase1_iterations, phase2_iter_limit=args.phase2_iterations,
        levelset_iter_limit=args.levelset_iterations, levelset_tol=args.levelset_tol,
        norm_option=args.norm, lambda_level=args.lambda_level, mu_level=args.mu_level,
        num_processes=args.processes,
    )


def _parse_cli_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = _parse_cli_args(argv)
    cfg = default_instance(instance=args.instance, out=args.out,
                           num_customers=args.customers, num_facilities=args.facilities,
                           T=args.T, num_scenarios=args.num_scenarios,
                           scenario_seed=args.scenario_seed, location_periods=args.location_periods)
    config = config_from_args(args)
    out = prepare_output_directory(cfg)
    try:
        with setup_run_logging(console_log_path(cfg)) as log_path:
            print(f"[console log] {log_path}")
            result = run_solver(cfg, log_path, config=config)
    except Exception as exc:
        write_json(out / "error.json", {"status": "ERROR", "type": type(exc).__name__, "message": str(exc)})
        raise
    # A completed budget-limited experiment is a successful command.
    # Model/audit failures still propagate as exceptions.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
