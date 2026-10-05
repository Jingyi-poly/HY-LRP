#!/usr/bin/env python3
"""Batch Compare EF vs SDDP over a stochastic LRP instance grid.

Writes the same per-run logs into experiments/compare_ef_result/, then builds
experiments/compare_ef_result/batch_summary.csv (+ .md) from completed logs.

Defaults follow COMPARE_PARAMS['instance']; set grids below or use the CLI.
Facilities are distinct physical locations. The old --vehicles spelling is
an alias for their count, not an investment vehicle-type configuration.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent
_RESULT_DIR = _REPO_ROOT / "experiments" / "compare_ef_result"
_COMPARE_PY = _SRC_ROOT / "compare_ef_sddp.py"

CUSTOMERS = None  # e.g. (5, 8, 10); None uses the selected comparison instance.
FACILITIES = None
VEHICLES = None  # Legacy grid setting: physical facility counts in LRP.
DEMAND_DISTS = None  # Setting an Investment distribution now raises an error.
SCENARIOS = None
PERIODS = None

_SUMMARY_DONE = re.compile(r"^={10,}\s*$")
_METRIC_PATTERNS = {
    "sddp_phase1_s": re.compile(
        r"(?:SDDP Phase1:|Phase1 用时:)\s*([0-9,]+\.?[0-9]*)s"
    ),
    "sddp_phase2_s": re.compile(
        r"(?:SDDP Phase2:|Phase2 用时:)\s*([0-9,]+\.?[0-9]*)s"
    ),
    "sddp_total_s": re.compile(
        r"(?:SDDP 合计:|合计用时:)\s*([0-9,]+\.?[0-9]*)s"
    ),
    "ef_runtime_s": re.compile(
        r"(?:EF \(Gurobi\):|EF 用时:)\s*([0-9,]+\.?[0-9]*)s"
    ),
    "ef_certified_ub": re.compile(
        r"EF certified(?: feasible-policy)? UB\s*[:=]\s*([-+0-9,]+\.?[0-9]*)"
    ),
    "ef_lb": re.compile(
        r"EF LB \(ObjBound\)\s*[:=]\s*([-+0-9,]+\.?[0-9]*)"
    ),
    "ef_gap": re.compile(r"EF Gap \(LB-UB\):\s*(\S+)"),
    "sddp_lb": re.compile(r"SDDP LB\s*[:=]\s*([-+0-9,]+\.?[0-9]*)"),
    "sddp_ub": re.compile(r"SDDP UB\s*[:=]\s*([-+0-9,]+\.?[0-9]*)"),
    "phase2_gap": re.compile(r"Phase2 Gap:\s*(\S+)"),
}


def _parse_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def expected_log_stem(
    c: int, v: int, s: int, t: int, dist: str | None = None,
    pscale: str | None = None, *, instance=None, scenario_seed=None
) -> str:
    if dist is not None or pscale is not None:
        raise ValueError('Investment demand distributions and purchase scaling do not apply to LRP')
    from compare_ef_sddp import comparison_instance, _compare_log_path
    cfg = comparison_instance(c, v, instance=instance, T=t, num_scenarios=s,
                              scenario_seed=scenario_seed)
    suffix = os.environ.get('LRP_COMPARE_LOG_SUFFIX', os.environ.get('VRP_COMPARE_LOG_SUFFIX', '')).strip()
    if suffix:
        cfg._output_tag = cfg.instance_tag() + '_' + suffix
    return _compare_log_path(_RESULT_DIR, cfg).stem


def log_is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 200:
        return False
    row = parse_log(path)
    return (
        row.get("sddp_lb") is not None
        and row.get("sddp_ub") is not None
        and row.get("ef_certified_ub") is not None
        and row.get("sddp_phase1_s") is not None
        and row.get("ef_runtime_s") is not None
    )


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {"log": path.name}
    # Recover C/V/S/T/D from filename when possible.
    m = re.match(
        r"C(\d+)_V(\d+)_S(\d+)_T(\d+)_D([^_]+)_PScale([0-9pm.]+)",
        path.stem,
    )
    if m:
        row.update(
            {
                "C": int(m.group(1)),
                "V": int(m.group(2)),
                "S": int(m.group(3)),
                "T": int(m.group(4)),
                "demand_dist": m.group(5),
                "PScale": m.group(6),
            }
        )
    for key, pattern in _METRIC_PATTERNS.items():
        hit = pattern.search(text)
        if not hit:
            row[key] = None
            continue
        raw = hit.group(1)
        if key in ("phase2_gap", "ef_gap"):
            row[key] = raw
        else:
            row[key] = _parse_float(raw)
    # Older logs: fall back to Accepted/Raw ObjBound or status line ObjBound=.
    # Prefer a finite bound: Accepted can be -inf even when Raw ObjBound is finite.
    if row.get("ef_lb") is None or (
        isinstance(row.get("ef_lb"), float) and not math.isfinite(row["ef_lb"])
    ):
        for pat in (
            re.compile(r"Raw ObjBound\s*=\s*([-+0-9,]+\.?[0-9]*)"),
            re.compile(r"Accepted ObjBound\s*=\s*([-+0-9,]+\.?[0-9]*)"),
            re.compile(r"ObjBound\s*=\s*([-+0-9,]+\.?[0-9]*)"),
        ):
            hit = pat.search(text)
            if not hit:
                continue
            val = _parse_float(hit.group(1))
            if val is not None and math.isfinite(val):
                row["ef_lb"] = val
                break
            if row.get("ef_lb") is None and val is not None:
                row["ef_lb"] = val
    lb = row.get("sddp_lb")
    ub = row.get("sddp_ub")
    ef_ub = row.get("ef_certified_ub")
    ef_lb = row.get("ef_lb")
    if isinstance(ef_lb, float) and not math.isfinite(ef_lb):
        ef_lb = None
        row["ef_lb"] = None
    if isinstance(lb, float) and isinstance(ef_ub, float) and abs(ef_ub) > 1e-12:
        row["GAP"] = round(100.0 * (ef_ub - lb) / abs(ef_ub), 2)
    else:
        row["GAP"] = None
    if not row.get("phase2_gap") and isinstance(lb, float) and isinstance(ub, float):
        denom = max(abs(ub), 1e-12)
        row["phase2_gap"] = 100.0 * abs(ub - lb) / denom
    if not row.get("ef_gap") and isinstance(ef_lb, float) and isinstance(ef_ub, float):
        if ef_lb > ef_ub + 1e-9:
            row["ef_gap"] = "LB>UB"
        else:
            denom = max(abs(ef_ub), 1e-12)
            row["ef_gap"] = 100.0 * (ef_ub - ef_lb) / denom
    # Prefer finite Certified gap line when present.
    if not row.get("ef_gap") or row.get("ef_gap") in ("inf%", "inf"):
        hit = re.search(r"Certified gap \(UB denom\)=\s*(\S+)", text)
        if hit and hit.group(1) not in ("inf%", "inf"):
            row["ef_gap"] = hit.group(1)
    row["phase2_gap"] = _pct2(row.get("phase2_gap"))
    row["ef_gap"] = _pct2(row.get("ef_gap"))
    row["GAP"] = _pct2(row.get("GAP"))
    row["status"] = (
        "complete"
        if (
            row.get("sddp_lb") is not None
            and row.get("sddp_ub") is not None
            and row.get("ef_certified_ub") is not None
            and row.get("sddp_phase1_s") is not None
            and row.get("ef_runtime_s") is not None
        )
        else "incomplete"
    )
    return row


def _pct2(value) -> str | None:
    """Format a percentage string or number as two decimals plus %."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text in ("LB>UB", "inf", "inf%", "--", "  --"):
        return text
    text = text.rstrip("%").replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.2f}%"


def _row_sort_key(row: dict) -> tuple:
    def _as_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return (
        _as_int(row.get("C")),
        _as_int(row.get("I", row.get("V"))),
        _as_int(row.get("S")),
        _as_int(row.get("T")),
        str(row.get("demand_dist") or ""),
        str(row.get("log") or ""),
    )


def _read_result(path):
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def result_is_complete(result):
    """A processed comparison requires actual certified data, not a log footer."""
    if not result:
        return False
    comp, ef = result.get('comparison') or {}, result.get('ef') or {}
    identity = result.get('instance_sha256')
    values = (result.get('lower_bound'), result.get('upper_bound'),
              ef.get('ObjBound'), ef.get('objective'))
    return bool(identity and ef.get('instance_sha256') == identity
        and comp.get('status') in {'PASS', 'INCONCLUSIVE'}
        and comp.get('bounds_valid') is True
        and comp.get('policy_matches_recorded_upper_bound') is True
        and ef.get('audit_passed') is True
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values))


def _result_row(path, result):
    dims, cfg, ef = result.get('dimensions') or {}, result.get('cfg') or {}, result.get('ef') or {}
    lb, ub = result.get('lower_bound'), result.get('upper_bound')
    ef_lb, ef_ub = ef.get('ObjBound'), ef.get('objective')
    def gap(lower, upper):
        if lower is None or upper is None:
            return None
        return 'LB>UB' if lower > upper else _pct2(100 * (upper - lower) / max(1., abs(upper)))
    p1, p2 = result.get('phase1_time'), result.get('phase2_time')
    stem = path.stem.removeprefix('result_')
    from types import SimpleNamespace
    from compare_ef_sddp import _compare_log_path
    log_cfg = SimpleNamespace(_output_tag=stem, _legacy_output=True,
                              out=path.parent, instance_tag=lambda: stem)
    log = _compare_log_path(path.parent, log_cfg)
    old_log = path.with_name(stem + '.txt')
    if not log.exists() and old_log.exists():
        log = old_log
    return dict(C=dims.get('J'), I=dims.get('I'), S=dims.get('S'), T=dims.get('T'),
        scenario_seed=cfg.get('scenario_seed'), instance=result.get('instance_name'),
        instance_sha256=result.get('instance_sha256'),
        status='complete' if result_is_complete(result) else (result.get('comparison') or {}).get('status', 'incomplete'),
        sddp_lb=lb, sddp_ub=ub, ef_lb=ef_lb, ef_certified_ub=ef_ub,
        ef_gap=gap(ef_lb, ef_ub), phase2_gap=gap(lb, ub), GAP=gap(lb, ef_ub),
        sddp_phase1_s=p1, sddp_phase2_s=p2,
        sddp_total_s=sum(v for v in (p1,p2) if v is not None),
        ef_runtime_s=ef.get('Runtime'), log=log.name, result=path.name)


def matching_results(result_dir, instance_sha256):
    found = []
    for path in result_dir.glob('result_*.json'):
        result = _read_result(path)
        if result and result.get('instance_sha256') == instance_sha256:
            found.append((path, result))
    return found


def _lrp_log_header(path):
    try:
        with path.open(errors='replace') as stream:
            header = stream.read(16384)
    except OSError:
        return None
    dimensions = re.search(r'LRP (.+?): facilities=(\d+), customers=(\d+), '
                           r'delivery_periods=(\d+), facility_intervals=(\d+), scenarios=(\d+)', header)
    identity = re.search(r'SHA256=([0-9a-f]{64})\b', header)
    if not dimensions or not identity:
        return None
    return dict(instance=dimensions[1], I=int(dimensions[2]), C=int(dimensions[3]),
                T=int(dimensions[4]), S=int(dimensions[6]), instance_sha256=identity[1])


def write_summary(result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in result_dir.glob('result_*.json'):
        result = _read_result(path)
        if result and result.get('information_stages') == 2 and result.get('dimensions'):
            rows.append(_result_row(path, result))
    # Retain summary support for already-written Investment logs. They cannot
    # certify an LRP resume because they carry no matching physical-data hash.
    for path in result_dir.glob("C*_V*_S*_T*_D*_PScale*.txt"):
        rows.append(parse_log(path))
    recorded_logs = {row['log'] for row in rows}
    for path in result_dir.glob('*.txt'):
        if path.name in recorded_logs:
            continue
        header = _lrp_log_header(path)
        if header:
            row = parse_log(path)
            row.update(header, status='incomplete')
            rows.append(row)
    rows.sort(key=_row_sort_key)
    csv_path = result_dir / "batch_summary.csv"
    md_path = result_dir / "batch_summary.md"
    fields = [
        "C",
        "I",
        "V",
        "S",
        "T",
        "demand_dist",
        "PScale",
        "scenario_seed",
        "instance",
        "instance_sha256",
        "status",
        "sddp_lb",
        "sddp_ub",
        "ef_lb",
        "ef_certified_ub",
        "ef_gap",
        "phase2_gap",
        "GAP",
        "sddp_total_s",
        "ef_runtime_s",
        "sddp_phase1_s",
        "sddp_phase2_s",
        "log",
        "result",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    complete = [r for r in rows if r.get("status") == "complete"]
    lines = [
        "# Compare EF batch summary",
        "",
        f"- logs scanned: {len(rows)}",
        f"- complete: {len(complete)}",
        "",
        "| C | I (legacy V) | S | T | seed / legacy dist | SDDP LB | SDDP UB | EF LB | EF UB | EF gap | P2 gap | GAP | SDDP s | EF s |",
        "|---|---|---|---|------|---------|---------|-------|-------|--------|--------|-----|--------|------|",
    ]
    for r in complete:
        lines.append(
            "| {C} | {V} | {S} | {T} | {demand_dist} | {sddp_lb} | {sddp_ub} | "
            "{ef_lb} | {ef_certified_ub} | {ef_gap} | {phase2_gap} | "
            "{GAP} | {sddp_total_s} | {ef_runtime_s} |".format(
                C=r.get("C"),
                V=r.get("I", r.get("V")),
                S=r.get("S"),
                T=r.get("T"),
                demand_dist=r.get("scenario_seed", r.get("demand_dist")),
                sddp_lb=_fmt(r.get("sddp_lb")),
                sddp_ub=_fmt(r.get("sddp_ub")),
                ef_lb=_fmt(r.get("ef_lb")),
                ef_certified_ub=_fmt(r.get("ef_certified_ub")),
                ef_gap=r.get("ef_gap") or "",
                phase2_gap=r.get("phase2_gap") or "",
                GAP=r.get("GAP") or "",
                sddp_total_s=_fmt(r.get("sddp_total_s"), 1),
                ef_runtime_s=_fmt(r.get("ef_runtime_s"), 1),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def iter_jobs(
    customers=None, vehicles=None, dists=None, scenarios=None, periods=None,
):
    if dists or DEMAND_DISTS:
        raise ValueError('Investment --dists/--demand-dist have no LRP equivalent; use the source sampling recipe and --scenario-seed')
    if any(value is None for value in (customers, vehicles, scenarios, periods)):
        from compare_ef_sddp import comparison_instance
        from core.instance import LRPInstance
        m, n, H, _, S = LRPInstance(comparison_instance()).build().prob_data.shape
        customers = customers if customers is not None else CUSTOMERS or (n,)
        vehicles = vehicles if vehicles is not None else FACILITIES or VEHICLES or (m,)
        scenarios = scenarios if scenarios is not None else SCENARIOS or (S,)
        periods = periods if periods is not None else PERIODS or (H,)
    # Smallest first so early results appear sooner.
    for c, v, s, t in itertools.product(
        sorted(customers),
        sorted(vehicles),
        sorted(scenarios),
        sorted(periods),
    ):
        yield c, v, s, t, None


def run_one(c: int, v: int, s: int, t: int, dist=None, python=None, *,
            instance=None, scenario_seed=None, location_periods=None) -> int:
    if dist is not None:
        raise ValueError('Investment demand distributions do not apply to LRP')
    cmd = [
        python or sys.executable,
        str(_COMPARE_PY),
        str(c),
        str(v),
        "--scenarios",
        str(s),
        "--periods",
        str(t),
    ]
    if instance is not None:
        cmd += ['--instance', str(instance)]
    if scenario_seed is not None:
        cmd += ['--scenario-seed', str(scenario_seed)]
    if location_periods is not None:
        cmd += ['--location-periods', *map(str, location_periods)]
    env = os.environ.copy()
    # The data hash, not a cosmetic log suffix, identifies completed jobs.
    env.pop("VRP_COMPARE_LOG_SUFFIX", None)
    env.pop("LRP_COMPARE_LOG_SUFFIX", None)
    print(f"\n[batch] START C={c} I={v} S={s} T={t}", flush=True)
    print(f"[batch] cmd: {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(_SRC_ROOT), env=env)
    elapsed = time.time() - started
    print(
        f"[batch] DONE C={c} I={v} S={s} T={t} "
        f"exit={proc.returncode} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild batch_summary.csv/md from existing logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List jobs without running",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with gurobipy (default: current)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N new runs (0 = all)",
    )
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Only re-run jobs whose existing logs are incomplete",
    )
    parser.add_argument(
        "--customers",
        type=int,
        nargs="+",
        default=None,
        help="Customer counts (default: selected comparison instance)",
    )
    parser.add_argument(
        "--facilities", "--vehicles", dest="facilities",
        type=int,
        nargs="+",
        default=None,
        help="Physical facility counts; --vehicles retains the old count spelling",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=None,
        help="Scenario counts (default: selected comparison instance)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="+",
        default=None,
        help="Delivery-period counts (default: selected comparison instance)",
    )
    parser.add_argument(
        "--dists", "--demand-dist",
        nargs="+",
        default=None,
        help="Unsupported Investment option; LRP uses the source sampling recipe",
    )
    parser.add_argument('--instance', type=Path, help='Override the comparison instance directory')
    parser.add_argument('--scenario-seed', type=int, help='Override activity/demand sampling seed')
    args = parser.parse_args(argv)
    if args.dists is not None or DEMAND_DISTS:
        parser.error('--dists/--demand-dist are Investment distributions, not LRP parameters; use the source recipe and --scenario-seed')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    override = os.environ.get('LRP_EXPERIMENTS_ROOT', os.environ.get('VRP_EXPERIMENTS_ROOT', '')).strip()
    result_dir = Path(override).expanduser().resolve() / 'compare_ef_result' if override else _RESULT_DIR
    if args.summary_only:
        if args.dry_run:
            print(f'[batch] WOULD WRITE summary in {result_dir}')
            return 0
        path = write_summary(result_dir)
        print(f"[batch] wrote {path}")
        return 0
    from compare_ef_sddp import comparison_instance
    from core.instance import LRPInstance
    base = comparison_instance(instance=args.instance, scenario_seed=args.scenario_seed)
    resolved = LRPInstance(base).build()
    m, n, H, _, S = resolved.prob_data.shape
    jobs = list(
        iter_jobs(
            customers=args.customers or CUSTOMERS or (n,),
            vehicles=args.facilities or FACILITIES or VEHICLES or (m,),
            scenarios=args.scenarios or SCENARIOS or (S,),
            periods=args.periods or PERIODS or (H,),
        )
    )
    print(f"[batch] total grid size = {len(jobs)}", flush=True)
    started_runs = 0
    failed_runs = 0
    for c, v, s, t, dist in jobs:
        try:
            cfg = comparison_instance(c, v, instance=resolved.path, T=t, num_scenarios=s,
                                      scenario_seed=base.scenario_seed)
            identity = LRPInstance(cfg).build().prob_data.logical_hash()
        except ValueError as exc:
            parser.error(f'C={c} I={v} S={s} T={t}: {exc}')
        matches = matching_results(result_dir, identity)
        if any(result_is_complete(result) for _, result in matches):
            print(
                f"[batch] SKIP complete C={c} I={v} S={s} T={t} hash={identity[:12]}",
                flush=True,
            )
            continue
        if args.retry_incomplete:
            # A child can fail before writing result JSON. The real data hash
            # printed at startup identifies that failed log for retry only.
            failed_log = any((_lrp_log_header(path) or {}).get('instance_sha256') == identity
                             for path in result_dir.glob('*.txt'))
            if not matches and not failed_log:
                continue
        if args.dry_run:
            print(f"[batch] WOULD RUN C={c} I={v} S={s} T={t} hash={identity[:12]}")
            continue
        code = run_one(c, v, s, t, None, args.python, instance=resolved.path,
                       scenario_seed=cfg.scenario_seed, location_periods=cfg.location_periods)
        write_summary(result_dir)
        started_runs += 1
        if code != 0:
            failed_runs += 1
            print(
                f"[batch] WARNING non-zero exit={code}; continuing",
                flush=True,
            )
        if args.limit and started_runs >= args.limit:
            print(f"[batch] hit --limit={args.limit}; stopping", flush=True)
            break

    if args.dry_run:
        return 0
    path = write_summary(result_dir)
    print(f"[batch] finished; summary -> {path}", flush=True)
    return int(failed_runs > 0)


if __name__ == "__main__":
    raise SystemExit(main())
