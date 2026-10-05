"""Small exact cache for repeated forward Stage-2 solves across outer rounds."""
from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .mip_start import certify_unchanged_policy_reuse


_ENABLE_ENV = "VRP_S2_FORWARD_OUTER_REUSE"
_MIN_ACTIVE_CUSTOMERS = 20
_AUTO_MAX_PROCESSES = 6


def _finite_hex(value) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite Stage-2 node data")
    return (0.0 if number == 0.0 else number).hex()


def _canonical(value):
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _canonical(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, float):
        return ("float", _finite_hex(value))
    if isinstance(value, (str, int, bool)):
        return (type(value).__name__, value)
    return (type(value).__name__, repr(value))


def stage2_outer_reuse_key(
    node,
    *,
    period_dedup: bool,
    model_options=None,
    solve_policy=None,
) -> tuple:
    """Exact outer-round key with fleet and learned cuts intentionally omitted.

    The period remains part of this key even when same-pass period dedup is
    enabled.  An outer-reuse record owns a cut snapshot and ``theta`` values
    keyed by the concrete Stage-3 successor ids, and its saved Stage-1 fleet
    was read at one concrete period.  Collapsing periods here would compare
    both saved and current fleets at the *target* period and could therefore
    mis-certify reuse when purchases changed between periods.  Cross-period
    reuse belongs to ``stage2_period_key``, which explicitly canonicalizes
    successor pools by vehicle inside one forward pass.
    """
    scenario, period = node.info
    identity = (
        (int(scenario), int(period))
        if period_dedup
        else (int(getattr(node, "index", -1)), int(scenario), int(period))
    )
    return (
        "s2-outer-reuse-v1",
        identity,
        tuple(int(value) for value in node.active),
        tuple(_finite_hex(value) for value in node.volume),
        tuple(_finite_hex(value) for value in node.c_out),
        len(node.successor),
        _canonical(model_options),
        _canonical(solve_policy),
    )


@dataclass(frozen=True)
class CachedForwardStage2:
    x_dict: Mapping[str, float]
    cost_star_value: float
    stage_cost_value: float
    theta_by_succ: Mapping[Any, float]


@dataclass
class _CacheRecord:
    x_prev: dict
    cut_snapshot: dict
    result: CachedForwardStage2


def _relevant_cut_snapshot(node, cut_lag) -> dict:
    """Deep copy only pools entering this node; later in-place changes are isolated."""
    stage3 = cut_lag.get(3, {}) or {}
    return {
        3: {
            successor: copy.deepcopy(stage3.get(successor, ()))
            for successor in node.successor
        }
    }


def _env_mode() -> str:
    # The original six-worker speed result relied on a loose DP "optimal"
    # flag whose certified lower endpoint did not actually meet its feasible
    # upper endpoint.  The repeated strict six-worker A/B produced zero cache
    # hits, so there is no demonstrated work reduction and production remains
    # off; explicit modes exist only for controlled experiments.
    raw = os.environ.get(_ENABLE_ENV, "off").strip().lower() or "off"
    aliases = {
        "1": "on", "true": "on", "yes": "on", "on": "on",
        "0": "off", "false": "off", "no": "off", "off": "off",
        "auto": "auto",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(f"{_ENABLE_ENV}={raw!r}; expected auto/on/off") from exc


class ForwardStage2OuterReuseCache:
    """Retain only exact-optimal policies and reuse them after harmless appends.

    The production default is off.  ``auto`` enables the six-worker
    experiment and larger pools may explicitly request it with
    ``VRP_S2_FORWARD_OUTER_REUSE=1``.  Only results carrying a strict closed-
    interval optimality certificate are stored.
    """

    def __init__(self, *, min_active_customers: int = _MIN_ACTIVE_CUSTOMERS):
        self.min_active_customers = int(min_active_customers)
        self._records: dict[tuple, _CacheRecord] = {}
        self._enabled = False
        self._pass = {}

    def begin_pass(self, *, num_processes: int) -> None:
        mode = _env_mode()
        self._enabled = mode == "on" or (
            mode == "auto" and int(num_processes) <= _AUTO_MAX_PROCESSES
        )
        self._pass = {
            "enabled": int(self._enabled),
            "checks": 0,
            "hits": 0,
            "misses": 0,
            "ineligible": 0,
            "gate_seconds": 0.0,
            "snapshot_seconds": 0.0,
        }

    @staticmethod
    def _n_active(node) -> int:
        return sum(int(value) == 1 for value in node.active)

    def lookup(self, key, prob_data, node, cut_lag, x_prev):
        if not self._enabled:
            return None
        if self._n_active(node) < self.min_active_customers:
            self._pass["ineligible"] += 1
            return None
        record = self._records.get(key)
        if record is None:
            self._pass["misses"] += 1
            return None
        self._pass["checks"] += 1
        started = time.perf_counter()
        decision = certify_unchanged_policy_reuse(
            prob_data,
            node,
            policy=record.result.x_dict,
            previous_x_prev=record.x_prev,
            current_x_prev=x_prev,
            previous_cut_lag=record.cut_snapshot,
            current_cut_lag=cut_lag,
            previous_exact_optimal=True,
        )
        self._pass["gate_seconds"] += time.perf_counter() - started
        if not decision.reused:
            self._pass["misses"] += 1
            return None

        # The unchanged policy is exact-optimal for the new archive too.  Move
        # the certificate baseline forward so consecutive harmless appends can
        # also skip their solve.
        started = time.perf_counter()
        record.x_prev = dict(x_prev)
        record.cut_snapshot = _relevant_cut_snapshot(node, cut_lag)
        self._pass["snapshot_seconds"] += time.perf_counter() - started
        self._pass["hits"] += 1
        result = record.result
        return CachedForwardStage2(
            dict(result.x_dict),
            float(result.cost_star_value),
            float(result.stage_cost_value),
            dict(result.theta_by_succ),
        )

    def store(
        self,
        key,
        node,
        cut_lag,
        x_prev,
        *,
        x_dict,
        cost_star_value,
        stage_cost_value,
        theta_by_succ=None,
        exact_optimal: bool,
    ) -> None:
        if (
            not self._enabled
            or not exact_optimal
            or self._n_active(node) < self.min_active_customers
        ):
            return
        started = time.perf_counter()
        snapshot = _relevant_cut_snapshot(node, cut_lag)
        self._pass["snapshot_seconds"] += time.perf_counter() - started
        self._records[key] = _CacheRecord(
            dict(x_prev),
            snapshot,
            CachedForwardStage2(
                dict(x_dict),
                float(cost_star_value),
                float(stage_cost_value),
                dict(theta_by_succ or {}),
            ),
        )

    def pass_stats(self) -> dict:
        return dict(self._pass)


def format_outer_reuse_stats(stats) -> str:
    return (
        f"enabled={int(stats.get('enabled', 0))} "
        f"checks={int(stats.get('checks', 0))} "
        f"skip={int(stats.get('hits', 0))} "
        f"miss={int(stats.get('misses', 0))} "
        f"ineligible={int(stats.get('ineligible', 0))} "
        f"gate={float(stats.get('gate_seconds', 0.0)):.6f}s "
        f"snapshot={float(stats.get('snapshot_seconds', 0.0)):.6f}s"
    )
