"""Shared two-phase execution and Main orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from setup.reporting import (
    merge_results,
    print_algorithm_config,
    print_instance_summary,
    print_phase_result,
    prepare_output_directory,
    print_summary,
    save_outputs,
)
from setup.solver_config import configure_main_solver, load_algorithm_config


def _physical_runtime(prob_data, scen_tree, config, supplied):
    cg=config.get('physical_cg_profile','off')
    enabled=(config.get('physical_forward_profile','off')!='off' or cg in ('paced','pool','cg'))
    if supplied is not None or not enabled:
        return supplied
    from solvers.lrp_physical_forward import PhysicalForwardRuntime
    return PhysicalForwardRuntime(prob_data, scen_tree, config)


def solve_phase1(prob_data, scen_tree, config, *, physical_forward_runtime=None):
    """Run Phase 1 and return ``(result, elapsed_seconds)``."""
    # Backend env is finalized by the caller before these imports.
    from algorithms.sddp_sbc import SDDP_SBC

    print("\n" + "=" * 80)
    print("Phase 1: SDDP with Strengthened Benders Cuts")
    print("=" * 80)
    print("\n开始求解 Phase 1...")
    started = time.time()
    solver = SDDP_SBC(prob_data, scen_tree, config)
    if physical_forward_runtime is not None:
        solver.physical_forward_runtime = physical_forward_runtime
    result_p1 = solver.solve()
    phase1_time = time.time() - started
    print_phase_result("Phase 1", result_p1, phase1_time)
    return result_p1, phase1_time


def solve_phase2(prob_data, scen_tree, config, result_p1, *, physical_forward_runtime=None):
    """Warm-start and run Phase 2 from a completed Phase 1 result."""
    # Backend env is finalized by the caller before this import.
    from algorithms.sddlp import SDDLP

    print("\n" + "=" * 80)
    print("Phase 2: SDDP with Level Set Method")
    print("=" * 80)
    sddlp = SDDLP(
        prob_data,
        scen_tree,
        config,
        cut_lag_init=result_p1["cut_lag"],
        ub_init=result_p1["UB_list"][-1] if result_p1["UB_list"] else None,
        lb_init=max(result_p1["LB_list"]) if result_p1["LB_list"] else None,
        x_best_init=result_p1["x_best"],
        last_forward_init=(
            result_p1.get("last_forward")
            if config.get("phase_forward_reuse", True)
            else None
        ),
    )
    if physical_forward_runtime is not None:
        sddlp.physical_forward_runtime = physical_forward_runtime
    print("\n开始求解 Phase 2...")
    started = time.time()
    result_p2 = sddlp.solve()
    phase2_time = time.time() - started
    print_phase_result("Phase 2", result_p2, phase2_time)
    return result_p2, phase2_time


def solve_two_phase(prob_data, scen_tree, config, *, physical_forward_runtime=None):
    """Run Phase 1 and warm-started Phase 2 for Main and Compare."""
    runtime = _physical_runtime(prob_data, scen_tree, config, physical_forward_runtime)
    options = {} if runtime is None else {'physical_forward_runtime': runtime}
    result_p1, phase1_time = solve_phase1(prob_data, scen_tree, config, **options)
    result_p2, phase2_time = solve_phase2(
        prob_data,
        scen_tree,
        config,
        result_p1,
        **options,
    )
    return result_p1, result_p2, phase1_time, phase2_time


def run_main_solver(cfg, log_path: Path, config=None, *, inst=None):
    """Read LRP data, run the original two algorithm phases and persist results."""
    from core.instance import LRPInstance

    if not hasattr(cfg, '_legacy_output'):
        prepare_output_directory(cfg)
    if inst is None:
        inst = LRPInstance(cfg).build()
    else:
        if getattr(inst, 'prob_data', None) is None or getattr(inst, 'scen_tree', None) is None:
            raise ValueError('The supplied LRP instance must be built before run_main_solver')
        from core.scenario_tree import validate_operating_weights
        inst.prob_data.validate()
        validate_operating_weights(inst.prob_data, inst.scen_tree)
    configure_main_solver(cfg, inst)
    print_instance_summary(cfg, inst)
    config = load_algorithm_config() if config is None else config
    print_algorithm_config(config, num_customers=inst.prob_data.shape[1])
    result_p1, result_p2, phase1_time, phase2_time = solve_two_phase(
        inst.prob_data, inst.scen_tree, config,
    )
    final_result = merge_results(
        cfg, result_p1, result_p2, phase1_time, phase2_time, config=config,
    )
    final_result["instance_sha256"] = inst.prob_data.logical_hash()
    final_result["dimensions"] = dict(zip(("I", "J", "T", "L", "S"), inst.prob_data.shape))
    final_result["data_selection"] = inst.prob_data.metadata.get("subset")
    final_result['report_data'] = {k: inst.prob_data.arrays[k].tolist() for k in (
        'facility_ids', 'customer_ids', 'capacity', 'opening_cost', 'continuation_cost',
        'closing_cost', 'active', 'demand', 'scenario_prob', 'location_periods', 'period_to_interval',
        'cost_fc', 'cost_cc', 'route_cost') if k in inst.prob_data.arrays}

    pkl_path, report_path = save_outputs(cfg, final_result)
    print_summary(cfg, config, log_path, pkl_path, report_path,
                  result_p1, result_p2, phase1_time, phase2_time)
    return final_result
