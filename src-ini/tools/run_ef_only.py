"""Run the independent two-stage LRP extensive form with Gurobi."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))
REPO = SRC_ROOT.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, help="Default: COMPARE_PARAMS['instance']")
    parser.add_argument("--out", "--output", type=Path)
    parser.add_argument("--customers", type=int)
    parser.add_argument("--facilities", "--vehicles", dest="facilities", type=int,
                        help="Distinct physical facility count (--vehicles is the old resource-count spelling)")
    parser.add_argument("--periods", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--scenario-seed", type=int)
    parser.add_argument("--location-periods", type=int, nargs="+")
    parser.add_argument("--time-limit", "--ef-time-limit", type=float)
    parser.add_argument("--mip-gap", "--ef-mip-gap", type=float)
    parser.add_argument("--threads", "--ef-threads", type=int)
    parser.add_argument("--connectivity", choices=("mtz", "cutset"), default="mtz")
    parser.add_argument("--enumerate", action="store_true", help="Validate small cases against the frozen independent enumerator")
    parser.add_argument("--log", action="store_true", help="Also display the Gurobi solver log")
    args = parser.parse_args(argv)
    from compare_ef_sddp import COMPARE_PARAMS, comparison_instance
    from core.instance import LRPInstance
    cfg = comparison_instance(args.customers, args.facilities, instance=args.instance,
                              T=args.periods, num_scenarios=args.scenarios,
                              scenario_seed=args.scenario_seed, location_periods=args.location_periods)
    time_limit = COMPARE_PARAMS['ef_time_limit_s'] if args.time_limit is None else args.time_limit
    mip_gap = COMPARE_PARAMS['ef_mip_gap'] if args.mip_gap is None else args.mip_gap
    threads = (int(os.environ.get('LRP_EF_GRB_THREADS', os.environ.get('VRP_EF_GRB_THREADS', '1')))
               if args.threads is None else args.threads)
    out = args.out or REPO / "artifacts/ef_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if out.exists():
        parser.error(f"output already exists; choose a new directory: {out}")
    out.mkdir(parents=True)
    result = None
    try:
        import numpy as np
        from models.extensive_model_builder import ExtensiveModelBuilder
        inst = LRPInstance(cfg).build()
        print(f'[ef-only] instance={cfg.instance_tag()}')
        print(f'[ef-only] time_limit={time_limit}s  mip_gap={mip_gap}')
        builder = ExtensiveModelBuilder(inst.prob_data, connectivity=args.connectivity)
        builder.spec.write_lp(out / "model.lp")
        builder.spec.save_matrix(out / "coefficient_matrix.npz")
        result = builder.solve(time_limit=time_limit, mip_gap=mip_gap,
                               threads=threads, output=args.log,
                               log_path=out / "gurobi.log", model_path=out / "gurobi_native.lp")
        report = {key: value for key, value in result.items() if key not in ("model", "primal", "solution")}
        report['dimensions'] = dict(zip(('I', 'J', 'T', 'L', 'S'), inst.prob_data.shape))
        if result["primal"] is not None:
            np.savez_compressed(out / "primal.npz", x=result["primal"], names=np.array(builder.spec.names))
        if result["solution"] is not None:
            (out / "solution_audit.json").write_text(json.dumps(result["solution"], indent=2, allow_nan=False) + "\n")
        if args.enumerate:
            reference = REPO / "codex_lrp_handoff/reference/lrp_gurobi_verified"
            sys.path.insert(0, str(reference))
            try:
                from enumerate_exact import enumerate_exact
                enumeration = enumerate_exact(builder.prob_data)
            finally:
                sys.path.remove(str(reference))
            (out / "enumeration.json").write_text(json.dumps(enumeration, indent=2, allow_nan=False) + "\n")
            report["enumeration_objective"] = enumeration["objective"]
            # Enumeration never supplies a start, cut or cutoff to the solver.
            if result["optimality_certified"]:
                audit = builder.certify_rounded_incumbent(result["model"], enumeration=enumeration)
                (out / "enumeration_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
                report["enumeration_audit_passed"] = audit["passed"]
        (out / "solve.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in
                         ("instance", "status", "objective", "objective_bound", "audit_passed")}, ensure_ascii=False))
        print(f"Results: {out.resolve()}")
        return 0 if result["optimality_certified"] and result["audit_passed"] else 1
    except Exception as exc:
        (out / "error.json").write_text(json.dumps({
            "error_type": type(exc).__name__, "error": str(exc),
            "automatic_solver_fallback": False, "traceback": traceback.format_exc(),
        }, indent=2) + "\n")
        print(f"EF failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if result is not None:
            result["model"].dispose()


if __name__ == "__main__":
    raise SystemExit(main())
