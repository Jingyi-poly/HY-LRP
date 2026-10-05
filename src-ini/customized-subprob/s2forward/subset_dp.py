"""Exact subset dynamic programme for one fixed-fleet piece ``C(n)``.

The Stage-2 model with the fleet fixed to the leading run ``n`` is

    min  sum_j c_out_j s_j + sum_v theta_v
    s.t. every active customer is assigned to at most one vehicle (else
         outsourced), inactive customers are never assigned,
         load(S_v) <= Q_v,   y_v = [S_v non-empty],
         theta_v = max(0, max_c  beta_c + piY_c y_v + sum_{j in S_v} piAlpha_c[j])
         assignment_order:  score(S_{v_i}) >= score(S_{v_{i+1}})  within a type,
         score(S) = sum_{j in S} round(log(j + 2), 4).

``theta_v`` only depends on the set ``S_v`` served by vehicle ``v`` (every
Stage-3 cut of successor ``v`` references that vehicle's variables only;
checked), so the objective separates over vehicles and the problem is a
set-partitioning DP over the ``2^n`` subsets of active customers:

    f[i](R) = cheapest way for the first ``i`` vehicles of the current type
              to serve exactly the customers ``R`` (given the earlier types).

The ``assignment_order`` rows are handled without extra state: within a type
the feasible sets are scanned in *descending exact score* order and every set
is offered to slots ``1..p`` in ascending order, so slot ``i+1`` can only take
a set whose score is <= the score of the set in slot ``i`` (sets of equal
score are processed as one group in which any slot order is reachable).
Vehicles left without a set (``y = 0``) are the trailing ones of a type and
contribute ``theta_v(empty) = max(0, max_c beta_c)``.

All comparisons that define feasibility (capacity, score order) are exact:
scores are dyadic rationals summed in ``int64``; borderline loads are
re-checked in ``Fraction``.  The optimal value is accumulated in binary64 and
returned as a lower bound with a tiny safety margin, while the optimal
assignment is re-costed exactly by the shared policy certifier.  Work is
``O(m * sum_S 2^(n - |S|))`` over capacity-feasible sets ``S``; the
``max_customers`` guard is memory-aware (up to 22 when the aggregate memory
budget permits it).
"""

from __future__ import annotations

from core.backend_telemetry import backend_call, record_backend_event

import math
import os
import subprocess
import sys
import time
from collections import OrderedDict
from functools import lru_cache
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .fleet_pieces import Counts, FleetLayout

STATUS = "subset_dp"


def _load_kernel():
    """The optional C++ kernel (``cpp/build.sh``); None -> numpy fallback."""
    if os.environ.get("VRP_S2_FLEET_DP_KERNEL", "auto").strip().lower() in ("0", "off", "numpy"):
        return None
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
    if os.path.isdir(kernel_dir) and kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    try:
        import subset_dp_kernel  # type: ignore
    except ImportError:
        return None
    return subset_dp_kernel


_KERNEL = _load_kernel()


def kernel_available() -> bool:
    return _KERNEL is not None


# Applicability ceilings on the number of active customers.  C21/C22 require
# larger tables, with conservative process-peak estimates of about 460/750
# MiB respectively; simultaneous DP slots multiply that memory.
DEFAULT_MAX_CUSTOMERS = 16
DEFAULT_MAX_CUSTOMERS_KERNEL = 20
DEFAULT_MAX_CUSTOMERS_KERNEL_SERIAL = 22
DEFAULT_DP_MEMORY_BUDGET_MB = 2048
DEFAULT_DP_AUTO_MEMORY_FRACTION = 0.25
DEFAULT_DP_AUTO_MEMORY_CAP_MB = 8192
DEFAULT_DP_PEAK_SAFETY_FACTOR = 1.05
_KERNEL_PEAK_MB = {18: 160, 19: 220, 20: 300, 21: 460, 22: 750}
# Wall time per min-plus step of the C++ kernel (calibrated on an M-series
# core: 1.4e10 steps in 14.3 s); used to predict a solve's duration.
_KERNEL_SECONDS_PER_STEP = 1.2e-9
_PREFIX_LAYER_CACHE_ENV = "VRP_S2_DP_PREFIX_LAYER_CACHE"
_PREFIX_LAYER_CACHE_MAX_MB = 64.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r}; expected on/off")


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(default if raw is None or raw.strip() == "" else raw)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _sysconf_physical_memory_bytes() -> Optional[int]:
    """Physical RAM reported by POSIX ``sysconf`` (Linux and modern macOS)."""
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if pages > 0 and page_size > 0 and total > 0 else None


def _proc_physical_memory_bytes() -> Optional[int]:
    """Linux fallback using ``/proc/meminfo``."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as stream:
            for line in stream:
                if not line.startswith("MemTotal:"):
                    continue
                fields = line.split()
                if len(fields) >= 2:
                    total = int(fields[1]) * 1024
                    return total if total > 0 else None
    except (OSError, TypeError, ValueError):
        return None
    return None


def _darwin_physical_memory_bytes() -> Optional[int]:
    """macOS fallback using the absolute ``sysctl`` executable."""
    if sys.platform != "darwin":
        return None
    for executable in ("/usr/sbin/sysctl", "/sbin/sysctl"):
        if not os.path.isfile(executable):
            continue
        try:
            result = subprocess.run(
                [executable, "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            total = int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, TypeError, ValueError):
            continue
        if total > 0:
            return total
    return None


def _cgroup_memory_limit_bytes() -> Optional[int]:
    """Return a finite Linux cgroup limit, if the process has one."""
    paths = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    limits = []
    for path in paths:
        try:
            with open(path, "r", encoding="ascii") as stream:
                raw = stream.read().strip()
            if raw.lower() == "max":
                continue
            value = int(raw)
        except (OSError, TypeError, ValueError):
            continue
        # v1 commonly reports a huge sentinel rather than the word ``max``.
        if 0 < value < (1 << 60):
            limits.append(value)
    return min(limits) if limits else None


@lru_cache(maxsize=1)
def physical_memory_mb() -> Optional[float]:
    """Usable physical-memory ceiling in MiB, including a cgroup limit.

    ``None`` means detection failed.  Callers then use the conservative
    2-GiB aggregate DP budget rather than guessing from address space or free
    memory, both of which are unreliable predictors of a concurrent peak.
    """
    host = (
        _sysconf_physical_memory_bytes()
        or _proc_physical_memory_bytes()
        or _darwin_physical_memory_bytes()
    )
    cgroup = _cgroup_memory_limit_bytes() if sys.platform.startswith("linux") else None
    candidates = [value for value in (host, cgroup) if value is not None and value > 0]
    if not candidates:
        return None
    return min(candidates) / float(1 << 20)


def dp_memory_budget_mb() -> tuple[float, str, Optional[float]]:
    """Return ``(aggregate_budget_MiB, source, physical_MiB)``.

    An explicit ``VRP_S2_DP_MEMORY_BUDGET_MB`` always wins.  Otherwise DP may
    consume at most 25% of detected physical/cgroup memory, capped at 8 GiB.
    Both auto-policy constants are configurable for controlled benchmarks.
    """
    explicit = os.environ.get("VRP_S2_DP_MEMORY_BUDGET_MB")
    if explicit is not None and explicit.strip() != "":
        return (
            _positive_float_env("VRP_S2_DP_MEMORY_BUDGET_MB", DEFAULT_DP_MEMORY_BUDGET_MB),
            "explicit",
            None,
        )
    physical = physical_memory_mb()
    if physical is None:
        return float(DEFAULT_DP_MEMORY_BUDGET_MB), "fallback", None
    fraction = _positive_float_env(
        "VRP_S2_DP_AUTO_MEMORY_FRACTION", DEFAULT_DP_AUTO_MEMORY_FRACTION
    )
    if fraction > 1.0:
        raise ValueError("VRP_S2_DP_AUTO_MEMORY_FRACTION must be <= 1")
    cap = _positive_float_env(
        "VRP_S2_DP_AUTO_MEMORY_CAP_MB", DEFAULT_DP_AUTO_MEMORY_CAP_MB
    )
    return min(physical * fraction, cap), "physical", physical


def dp_concurrent_slots(concurrent_slots: Optional[int] = None) -> tuple[int, str]:
    """Actual simultaneous DP slots, conservatively falling back to workers."""
    if concurrent_slots is not None:
        return _positive_int(concurrent_slots, name="concurrent_slots"), "argument"
    explicit = os.environ.get("VRP_S2_DP_CONCURRENT_SLOTS")
    if explicit is not None and explicit.strip() != "":
        return (
            _positive_int(explicit, name="VRP_S2_DP_CONCURRENT_SLOTS"),
            "explicit",
        )
    return (
        _positive_int(os.environ.get("VRP_NUM_PROCESSES", "1"), name="VRP_NUM_PROCESSES"),
        "workers",
    )


def dp_memory_policy(concurrent_slots: Optional[int] = None) -> dict:
    """Describe the deterministic memory gate used by ``default_max_customers``."""
    slots, slots_source = dp_concurrent_slots(concurrent_slots)
    budget, budget_source, physical = dp_memory_budget_mb()
    ceiling = DEFAULT_MAX_CUSTOMERS
    required = None
    if kernel_available():
        for customers in range(DEFAULT_MAX_CUSTOMERS_KERNEL_SERIAL, 17, -1):
            candidate = (
                slots * _KERNEL_PEAK_MB[customers] * DEFAULT_DP_PEAK_SAFETY_FACTOR
            )
            if candidate <= budget:
                ceiling = customers
                required = candidate
                break
    return {
        "max_customers": ceiling,
        "concurrent_slots": slots,
        "slots_source": slots_source,
        "memory_budget_mb": budget,
        "budget_source": budget_source,
        "physical_memory_mb": physical,
        "peak_safety_factor": DEFAULT_DP_PEAK_SAFETY_FACTOR,
        "estimated_aggregate_peak_mb": required,
    }


def default_max_customers(concurrent_slots: Optional[int] = None) -> int:
    """Memory-aware exact-DP ceiling for simultaneous DP work.

    Schedulers that know fewer DP tasks can run concurrently than the process
    pool size should pass ``concurrent_slots`` (or set
    ``VRP_S2_DP_CONCURRENT_SLOTS``).  With neither, ``VRP_NUM_PROCESSES`` is
    deliberately used as a conservative upper bound.  C22 remains the hard
    automatic kernel ceiling; C23 requires an explicit caller override.
    """
    if not kernel_available():
        return DEFAULT_MAX_CUSTOMERS
    return int(dp_memory_policy(concurrent_slots)["max_customers"])


def _prefix_layer_cache_budget_bytes(active_customers: int) -> int:
    """Bound one solver's exact-prefix cache inside the aggregate DP budget."""
    n = int(active_customers)
    # The automatic DP domain ends at C22.  An explicitly forced larger DP
    # remains available, but it does not receive uncalibrated cache memory.
    if n > DEFAULT_MAX_CUSTOMERS_KERNEL_SERIAL:
        return 0
    budget_mb, _source, _physical = dp_memory_budget_mb()
    slots, _slots_source = dp_concurrent_slots()
    # Below C18, extrapolation would become smaller than ordinary interpreter
    # and theta-table overhead.  A 64-MiB floor keeps the headroom estimate
    # conservative; the cache itself is still capped at 64 MiB.
    if n >= 18:
        base_peak_mb = float(_KERNEL_PEAK_MB[n])
    else:
        base_peak_mb = max(64.0, float(_KERNEL_PEAK_MB[18]) / (2 ** (18 - n)))
    headroom_mb = max(
        0.0,
        float(budget_mb) / float(slots)
        - base_peak_mb * DEFAULT_DP_PEAK_SAFETY_FACTOR,
    )
    return int(min(_PREFIX_LAYER_CACHE_MAX_MB, headroom_mb) * (1 << 20))
# Minimum slack used by the scale-aware binary64 lower-bound margin.
_LB_ABS_SLACK = 1e-10
_LOAD_BORDER = 1e-6


class SubsetDPNotApplicable(ValueError):
    """The node/cuts do not fit the DP's assumptions (caller falls back)."""


def _subset_sums(values: np.ndarray) -> np.ndarray:
    """``out[mask] = sum_{i in mask} values[i]`` for all ``2^n`` masks."""
    n = len(values)
    out = np.zeros(1 << n, dtype=values.dtype)
    for i in range(n):
        step = 1 << i
        view = out.reshape(-1, 2, step)
        view[:, 1, :] = view[:, 0, :] + values[i]
    return out


def exact_capacity_feasible_mask(
    load: np.ndarray,
    item_values: Sequence[float],
    capacity: float,
    *,
    include_empty: bool = True,
) -> np.ndarray:
    """Classify subset loads without excluding a truly feasible mask.

    ``load`` is the recursively accumulated binary64 subset-sum table.  Its
    comparison is relaxed by a standard forward-error bound; every ambiguous
    mask is then decided exactly over the represented input doubles.
    """
    values = np.asarray(item_values, dtype=np.float64)
    table = np.asarray(load, dtype=np.float64)
    if table.ndim != 1 or table.size != 1 << len(values):
        raise ValueError("load table shape does not match item_values")
    additions = max(1, len(values))
    unit = 2.0 ** -53
    gamma = (additions * unit) / (1.0 - additions * unit)
    load_error = math.nextafter(
        gamma * math.fsum(abs(float(value)) for value in values)
        + additions * math.ulp(0.0),
        math.inf,
    )
    feasible = table <= math.nextafter(float(capacity) + load_error, math.inf)
    border = np.flatnonzero(
        np.abs(table - float(capacity)) <= max(_LOAD_BORDER, load_error)
    )
    capacity_exact = Fraction.from_float(float(capacity))
    for mask in border:
        exact_load = sum(
            (
                Fraction.from_float(float(values[bit]))
                for bit in range(len(values))
                if int(mask) >> bit & 1
            ),
            Fraction(0),
        )
        feasible[mask] = exact_load <= capacity_exact
    feasible[0] = bool(include_empty)
    return feasible


def _dyadic_int64(values: Sequence[float]) -> np.ndarray:
    """Exact ``value * 2^k`` as int64 (common ``k``) so subset sums are exact."""
    fractions = [Fraction(float(value)) for value in values]
    scale_bits = 0
    for value in fractions:
        denominator = value.denominator  # a power of two for binary64 inputs
        if denominator & (denominator - 1):
            raise SubsetDPNotApplicable(f"weight {value!r} is not dyadic")
        scale_bits = max(scale_bits, denominator.bit_length() - 1)
    out = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(fractions):
        scaled = value * (1 << scale_bits)
        assert scaled.denominator == 1
        out[i] = int(scaled.numerator)
    if len(values) and int(np.abs(out).sum()) >= (1 << 62):
        raise SubsetDPNotApplicable("weights too large for exact int64 subset sums")
    return out


class SubsetDPPieceSolver:
    """Per-node precomputation shared by all pieces of one Level Set call."""

    def __init__(self, prob_data, node, cuts_payload, layout: FleetLayout, *,
                 max_customers: int = DEFAULT_MAX_CUSTOMERS,
                 use_kernel: Optional[bool] = None,
                 prefix_layer_cache: bool = False,
                 prefix_layer_cache_budget_bytes: Optional[int] = None):
        self.kernel = _KERNEL if (use_kernel or use_kernel is None) else None
        if use_kernel and self.kernel is None:
            raise RuntimeError("subset_dp_kernel requested but not built (customized-subprob/s2forward/cpp/build.sh)")
        self.prob_data = prob_data
        self.node = node
        self.layout = layout
        self.cuts_payload = list(cuts_payload)
        self.customers: List = list(prob_data.J)
        self.vehicles: List = list(prob_data.V)
        self.vehicle_pos: Dict = {v: pos for pos, v in enumerate(self.vehicles)}
        self.active_idx: List[int] = [
            pos for pos, j in enumerate(self.customers) if int(node.active[j]) == 1
        ]
        self.n = len(self.active_idx)
        if self.n > int(max_customers):
            raise SubsetDPNotApplicable(
                f"{self.n} active customers > max_customers={max_customers}"
            )
        if len(node.successor) != len(self.vehicles):
            raise SubsetDPNotApplicable("one Stage-3 successor per vehicle required")
        self._check_cut_locality()

        active_customers = [self.customers[pos] for pos in self.active_idx]
        active_set = set(self.active_idx)
        self.inactive_outsourcing = sum(
            float(node.c_out[j]) for pos, j in enumerate(self.customers)
            if pos not in active_set
        )
        c_out = np.asarray([float(node.c_out[j]) for j in active_customers], dtype=np.float64)
        self.outsource_sum = _subset_sums(c_out)
        volumes = [float(node.volume[j]) for j in active_customers]
        self.load = _subset_sums(np.asarray(volumes, dtype=np.float64))
        self._volumes = volumes
        weights = [float(np.round(np.log(j + 2), 4)) for j in active_customers]
        self.score = _subset_sums(_dyadic_int64(weights))
        self.full_mask = (1 << self.n) - 1
        self._theta_cache: Dict[int, np.ndarray] = {}
        self._empty_theta_cache: Dict[int, float] = {}
        # Theta tables are keyed by (capacity, private cut set).  Canonical
        # ranks normally have different learned pools; exact signature matches
        # can still reuse a table without assuming same-type sharing.
        self._theta_by_signature: Dict[tuple, np.ndarray] = {}
        self._feasible_cache: Dict[int, np.ndarray] = {}
        # Capacity feasibility and the assignment-order scan are identical
        # for every vehicle with the same represented capacity.  A Level-Set
        # session solves several fleet pieces, so preparing these arrays in
        # every type layer used to repeat a large stable argsort (and, for the
        # C++ path, create then immediately concatenate up to 2^n tiny score
        # groups).  Cache the compact ``sorted masks + group starts`` payload
        # once per capacity instead.  This changes no comparison or ordering.
        self._feasible_by_capacity: Dict[float, np.ndarray] = {}
        self._score_order_by_capacity: Dict[float, tuple] = {}
        self._score_groups_by_capacity: Dict[float, List[np.ndarray]] = {}
        self._work_cache: Dict[int, int] = {}
        self._idx_cache: Dict[int, tuple] = {}
        self._fixed_lb_margin_cache: Optional[float] = None
        self.prefix_layer_cache = bool(prefix_layer_cache)
        if prefix_layer_cache_budget_bytes is None:
            cache_budget = (
                _prefix_layer_cache_budget_bytes(self.n)
                if self.prefix_layer_cache
                else 0
            )
        else:
            cache_budget = int(prefix_layer_cache_budget_bytes)
            if cache_budget < 0:
                raise ValueError("prefix_layer_cache_budget_bytes must be nonnegative")
        self.prefix_layer_cache_budget_bytes = (
            cache_budget if self.prefix_layer_cache else 0
        )
        # A key is the canonical count prefix through one vehicle type.  The
        # node, cut archive and theta tables are fixed for this solver object,
        # so cached DP layers never cross an archive boundary.
        self._prefix_layer_cache = OrderedDict()
        self.stats = {
            "solves": 0,
            "seconds": 0.0,
            "theta_seconds": 0.0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_stores": 0,
            "prefix_cache_evictions": 0,
            "prefix_cache_bytes": 0,
            "prefix_cache_budget_bytes": self.prefix_layer_cache_budget_bytes,
        }

    def _prefix_cache_get(self, key: tuple):
        cached = self._prefix_layer_cache.pop(key, None)
        if cached is None:
            self.stats["prefix_cache_misses"] += 1
            return None
        self._prefix_layer_cache[key] = cached
        self.stats["prefix_cache_hits"] += 1
        return cached[0], cached[1]

    def _prefix_cache_put(self, key: tuple, g: np.ndarray, slot_sets) -> None:
        if self.prefix_layer_cache_budget_bytes <= 0:
            return
        cached_sets = None if slot_sets is None else tuple(slot_sets)
        entry_bytes = int(g.nbytes) + sum(
            int(values.nbytes) for values in (cached_sets or ())
        )
        if entry_bytes > self.prefix_layer_cache_budget_bytes:
            return
        previous = self._prefix_layer_cache.pop(key, None)
        if previous is not None:
            self.stats["prefix_cache_bytes"] -= int(previous[2])
        while (
            self._prefix_layer_cache
            and self.stats["prefix_cache_bytes"] + entry_bytes
            > self.prefix_layer_cache_budget_bytes
        ):
            _old_key, old = self._prefix_layer_cache.popitem(last=False)
            self.stats["prefix_cache_bytes"] -= int(old[2])
            self.stats["prefix_cache_evictions"] += 1
        self._prefix_layer_cache[key] = (g, cached_sets, entry_bytes)
        self.stats["prefix_cache_bytes"] += entry_bytes
        self.stats["prefix_cache_stores"] += 1

    # ----------------------------------------------------------- validation
    def _check_cut_locality(self):
        m = len(self.vehicles)
        n = len(self.customers)
        for index, cut in enumerate(self.cuts_payload):
            succ = int(cut["succ"])
            if succ < 0 or succ >= m:
                raise SubsetDPNotApplicable(f"cut {index} has successor {succ} out of range")
            pi_y = list(cut["piY"])
            pi_alpha = list(cut["piAlpha"])
            if len(pi_y) != m or len(pi_alpha) != m:
                raise SubsetDPNotApplicable(f"cut {index} has a bad shape")
            for pos in range(m):
                if pos == succ:
                    continue
                if float(pi_y[pos]) != 0.0 or any(float(c) != 0.0 for c in pi_alpha[pos]):
                    raise SubsetDPNotApplicable(
                        f"cut {index} of successor {succ} references vehicle position {pos}"
                    )
            if len(pi_alpha[succ]) != n:
                raise SubsetDPNotApplicable(f"cut {index} has a bad shape")

    # -------------------------------------------------------------- tables
    def _feasible(self, vehicle) -> np.ndarray:
        pos = self.vehicle_pos[vehicle]
        cached = self._feasible_cache.get(pos)
        if cached is not None:
            return cached
        capacity = float(self.prob_data.Qv[vehicle])
        feasible = self._feasible_by_capacity.get(capacity)
        if feasible is None:
            feasible = exact_capacity_feasible_mask(
                self.load,
                self._volumes,
                capacity,
                include_empty=False,  # empty is the separate unused-vehicle case
            )
            self._feasible_by_capacity[capacity] = feasible
        self._feasible_cache[pos] = feasible
        return feasible

    def theta_empty(self, vehicle) -> float:
        """Return ``theta_v`` at ``y=alpha=0`` without building its full table."""
        pos = self.vehicle_pos[vehicle]
        cached = self._empty_theta_cache.get(pos)
        if cached is not None:
            return cached
        value = max(
            0.0,
            max(
                (
                    float(cut["beta"])
                    for cut in self.cuts_payload
                    if int(cut["succ"]) == pos
                ),
                default=0.0,
            ),
        )
        self._empty_theta_cache[pos] = value
        return value

    def theta(self, vehicle) -> np.ndarray:
        """``theta_v(S)`` for every mask (``+inf`` when ``S`` does not fit)."""
        pos = self.vehicle_pos[vehicle]
        cached = self._theta_cache.get(pos)
        if cached is not None:
            return cached
        started = time.time()
        cuts = [
            cut for cut in self.cuts_payload
            if int(cut["succ"]) == pos
        ]
        signature = (
            float(self.prob_data.Qv[vehicle]),
            tuple(sorted(
                (float(cut["beta"]), float(cut["piY"][pos]),
                 tuple(float(cut["piAlpha"][pos][j]) for j in self.active_idx))
                for cut in cuts
            )),
        )
        shared = self._theta_by_signature.get(signature)
        if shared is not None:
            record_backend_event("subset_dp", "cache", "theta_signature", stage=2)
            self._theta_cache[pos] = shared
            self._empty_theta_cache[pos] = float(shared[0])
            self.stats["theta_seconds"] += time.time() - started
            return shared
        feasible = self._feasible(vehicle)
        if self.kernel is not None:
            coeffs = np.asarray(
                [[float(cut["piAlpha"][pos][j]) for j in self.active_idx] for cut in cuts],
                dtype=np.float64,
            ).reshape(len(cuts), self.n)
            used = np.asarray([float(cut["beta"]) + float(cut["piY"][pos]) for cut in cuts])
            empty = np.asarray([float(cut["beta"]) for cut in cuts])
            with backend_call("subset_dp", "theta_table", stage=2,
                              active_customers=self.n):
                theta = np.asarray(self.kernel.theta_table(coeffs, used, empty, feasible))
        else:
            theta = np.zeros(1 << self.n, dtype=np.float64)
            for cut in cuts:
                beta = float(cut["beta"])
                pi_y = float(cut["piY"][pos])
                coeffs = np.asarray(
                    [float(cut["piAlpha"][pos][j]) for j in self.active_idx], dtype=np.float64
                )
                rhs = _subset_sums(coeffs)
                rhs += beta + pi_y
                rhs[0] = beta  # y = 0 for the empty set
                np.maximum(theta, rhs, out=theta)
            theta[~feasible] = np.inf
            theta[0] = max(0.0, max((float(cut["beta"]) for cut in cuts), default=0.0))
        self._theta_cache[pos] = theta
        self._theta_by_signature[signature] = theta
        self._empty_theta_cache[pos] = float(theta[0])
        self.stats["theta_seconds"] += time.time() - started
        return theta

    def work_steps(self, counts: Counts) -> int:
        """Exact number of min-plus steps ``solve(counts)`` performs."""
        counts = self.layout.check_counts(counts)
        total = 0
        for count, group in zip(counts, self.layout.groups):
            if count == 0:
                continue
            total += count * self._type_work(group[0])
        return total

    def _type_work(self, vehicle) -> int:
        pos = self.vehicle_pos[vehicle]
        cached = self._work_cache.get(pos)
        if cached is None:
            feasible = self._feasible(vehicle)
            if not hasattr(self, "_popcount"):
                masks = np.arange(1 << self.n, dtype=np.int64)
                pop = np.zeros(1 << self.n, dtype=np.int64)
                for bit in range(self.n):
                    pop += (masks >> bit) & 1
                self._popcount = pop
            cached = int((2 ** (self.n - self._popcount[feasible])).sum())
            self._work_cache[pos] = cached
        return cached

    def predicted_seconds(self, counts: Counts) -> Optional[float]:
        """Predicted wall time of the C++ kernel solve (None for the numpy path)."""
        if self.kernel is None:
            return None
        return _KERNEL_SECONDS_PER_STEP * self.work_steps(counts)

    @staticmethod
    def _rounding_gamma(operations: int) -> float:
        operations = max(1, int(operations))
        unit = 2.0 ** -53
        return (operations * unit) / (1.0 - operations * unit)

    def _lagrangian_lb_margin(self, pi) -> float:
        """Uniform error allowance for every feasible free-fleet objective."""
        outsourcing_scale = math.fsum(
            abs(float(self.node.c_out[customer])) for customer in self.customers
        )
        theta_scale = 0.0
        for position in range(len(self.vehicles)):
            largest = 0.0
            for cut in self.cuts_payload:
                if int(cut["succ"]) != position:
                    continue
                scale = (
                    abs(float(cut["beta"]))
                    + abs(float(cut["piY"][position]))
                    + math.fsum(
                        abs(float(cut["piAlpha"][position][index]))
                        for index in self.active_idx
                    )
                )
                largest = max(largest, scale)
            theta_scale += largest
        multiplier_scale = math.fsum(abs(float(pi[v])) for v in self.vehicles)
        total_scale = outsourcing_scale + theta_scale + multiplier_scale
        theta_error = self._rounding_gamma(self.n + 3) * theta_scale
        outsourcing_error = self._rounding_gamma(self.n + 1) * outsourcing_scale
        accumulation_ops = (
            4 * len(self.vehicles)
            + 2 * self.n
            + 4 * len(self.layout.groups)
            + 16
        )
        accumulation_error = self._rounding_gamma(accumulation_ops) * total_scale
        conversion_error = self._rounding_gamma(2) * multiplier_scale
        margin = 8.0 * (
            theta_error
            + outsourcing_error
            + accumulation_error
            + conversion_error
        )
        margin += accumulation_ops * math.ulp(0.0)
        return math.nextafter(max(_LB_ABS_SLACK, margin), math.inf)

    def _fixed_lb_margin(self) -> float:
        if self._fixed_lb_margin_cache is None:
            zero_pi = {vehicle: Fraction(0) for vehicle in self.vehicles}
            self._fixed_lb_margin_cache = self._lagrangian_lb_margin(zero_pi)
        return self._fixed_lb_margin_cache

    def _score_order(self, vehicle) -> tuple[np.ndarray, np.ndarray]:
        """Compact assignment-order scan, cached for one exact capacity.

        Returns feasible masks in descending exact-score order and offsets of
        equal-score groups.  The C++ kernel consumes this representation
        directly, avoiding the costly ``np.split``/``np.concatenate`` pair.
        """
        capacity = float(self.prob_data.Qv[vehicle])
        cached = self._score_order_by_capacity.get(capacity)
        if cached is not None:
            return cached
        feasible_masks = np.flatnonzero(self._feasible(vehicle))
        scores = self.score[feasible_masks]
        order = np.argsort(-scores, kind="stable")
        masks = np.ascontiguousarray(feasible_masks[order], dtype=np.int64)
        sorted_scores = scores[order]
        cut_points = np.flatnonzero(np.diff(sorted_scores)) + 1
        starts = np.empty(len(cut_points) + 2, dtype=np.int64)
        starts[0] = 0
        starts[-1] = len(masks)
        starts[1:-1] = cut_points
        cached = (masks, starts)
        self._score_order_by_capacity[capacity] = cached
        return cached

    def _score_groups_numpy(self, vehicle) -> List[np.ndarray]:
        """Array views of score groups for the numpy fallback."""
        capacity = float(self.prob_data.Qv[vehicle])
        cached = self._score_groups_by_capacity.get(capacity)
        if cached is not None:
            return cached
        masks, starts = self._score_order(vehicle)
        cached = [
            masks[int(starts[index]):int(starts[index + 1])]
            for index in range(len(starts) - 1)
        ]
        self._score_groups_by_capacity[capacity] = cached
        return cached

    def _index_pair(self, mask: int) -> tuple:
        cached = self._idx_cache.get(mask)
        if cached is None:
            without = []
            with_ = []
            for bit in range(self.n - 1, -1, -1):  # axis 0 is the highest bit
                if mask >> bit & 1:
                    without.append(0)
                    with_.append(1)
                else:
                    without.append(slice(None))
                    with_.append(slice(None))
            # Trailing length-1 axis keeps every slice a (writable) array view
            # even when ``mask`` covers all customers.
            without.append(slice(None))
            with_.append(slice(None))
            cached = (tuple(without), tuple(with_))
            self._idx_cache[mask] = cached
        return cached

    # ----------------------------------------------------------------- DP
    def _type_layer_kernel(self, g, thetas, masks, starts, tail_cost):
        with backend_call("subset_dp", "type_layer", stage=2,
                          active_customers=self.n):
            g_new, sets = self.kernel.type_layer(
                np.ascontiguousarray(g, dtype=np.float64),
                np.ascontiguousarray(np.stack(thetas), dtype=np.float64),
                masks, starts, np.ascontiguousarray(tail_cost, dtype=np.float64),
            )
        return np.asarray(g_new), [np.asarray(sets[k]) for k in range(len(thetas))]

    def _terminal_single_kernel(self, g, theta, tail_cost):
        """Close a final one-vehicle type by exact subset-min zeta.

        With outsourcing reward ``r(S)``, the used-vehicle case is
        ``r(J) + min_S(theta(S)-r(S) + min_{T subset J\\S}(g(T)-r(T)))``;
        the unused case is the same inner subset minimum at ``J``.  The zeta
        pass evaluates every eligible ``T`` and returns its argmin for trace
        reconstruction, replacing the terminal ``O(3^n)`` layer by
        ``O(n 2^n)`` without relaxing the assignment domain.
        """
        with backend_call("subset_dp", "terminal_single_layer", stage=2,
                          active_customers=self.n):
            value, previous_mask, assigned_mask = self.kernel.terminal_single_layer(
                np.ascontiguousarray(g, dtype=np.float64),
                np.ascontiguousarray(theta, dtype=np.float64),
                np.ascontiguousarray(self.outsource_sum, dtype=np.float64),
                np.ascontiguousarray(tail_cost, dtype=np.float64),
            )
        return float(value), int(previous_mask), int(assigned_mask)

    @backend_call("subset_dp_numpy", "type_layer", stage=2)
    def _type_layer_numpy(self, g, thetas, groups, tail_cost, shape):
        p = len(thetas)
        size = 1 << self.n
        f = [g.reshape(shape)] + [np.full(size, np.inf).reshape(shape) for _ in range(p)]
        # chain[i][k]: set in slot k (1..i) of the assignment behind f[i].
        # Chains are copied by value on every improvement: a plain
        # back-pointer would be wrong because f[i-1] keeps improving (with
        # lower-score sets) after f[i] has consumed it, and slot i may not
        # follow those later sets.
        chain = [None] + [
            [None] + [np.zeros(size, dtype=np.int64).reshape(shape) for _ in range(i)]
            for i in range(1, p + 1)
        ]
        for group_masks in groups:
            for slot in range(p):
                src = f[slot]
                dst = f[slot + 1]
                theta = thetas[slot]
                src_chain = chain[slot]
                dst_chain = chain[slot + 1]
                for mask in group_masks.tolist():
                    cost = float(theta[mask])
                    if not math.isfinite(cost):
                        continue
                    idx0, idx1 = self._index_pair(mask)
                    cand = src[idx0] + cost
                    target = dst[idx1]
                    better = cand < target
                    if not better.any():
                        continue
                    target[better] = cand[better]
                    for k in range(1, slot + 1):
                        dst_chain[k][idx1][better] = src_chain[k][idx0][better]
                    dst_chain[slot + 1][idx1][better] = mask
        # Collapse over the number of used slots; trailing vehicles stay unused.
        stacked = np.stack([f[i].reshape(size) + tail_cost[p - i] for i in range(p + 1)])
        collapse = np.argmin(stacked, axis=0)
        g_new = stacked[collapse, np.arange(size)]
        # Sets per slot for the chosen slot count of every state R (0 = slot
        # unused); consistent because both f and chain are final here.
        slot_sets = []
        positions = np.arange(size)
        for k in range(1, p + 1):
            sets_k = np.zeros(size, dtype=np.int64)
            for i in range(k, p + 1):
                use = collapse == i
                sets_k[use] = chain[i][k].reshape(size)[positions[use]]
            slot_sets.append(sets_k)
        return g_new, slot_sets

    def solve(self, counts: Counts) -> dict:
        """Exact ``C(counts)`` with its optimal assignment (``alpha``/``y`` dict)."""
        started = time.time()
        counts = self.layout.check_counts(counts)
        shape = (2,) * self.n + (1,)
        size = 1 << self.n
        g = np.full(size, np.inf)
        g[0] = 0.0
        trace = []  # per type: (vehicles, per-slot set masks for the chosen chain)
        nonempty_types = [index for index, count in enumerate(counts) if count]
        terminal_type = nonempty_types[-1] if nonempty_types else None
        terminal = None
        for type_index, (count, group) in enumerate(zip(counts, self.layout.groups)):
            vehicles = group[:count]
            p = len(vehicles)
            cache_key = tuple(counts[:type_index + 1])
            cacheable_layer = (
                self.prefix_layer_cache_budget_bytes > 0
                and type_index + 1 < len(self.layout.groups)
            )
            cached_layer = (
                self._prefix_cache_get(cache_key)
                if cacheable_layer
                else None
            )
            if cached_layer is not None:
                record_backend_event("subset_dp", "cache", "prefix_layer", stage=2)
                g, cached_sets = cached_layer
                trace.append((vehicles, cached_sets))
                continue
            if p == 0:
                g = g + sum(self.theta_empty(v) for v in group)
                trace.append((vehicles, None))
                if cacheable_layer:
                    self._prefix_cache_put(cache_key, g, None)
                continue
            for other in vehicles[1:]:
                if float(self.prob_data.Qv[other]) != float(self.prob_data.Qv[vehicles[0]]):
                    raise SubsetDPNotApplicable("vehicles of one type must share capacity")
            thetas = [self.theta(v) for v in vehicles]
            # Every used vehicle in the type shares the capacity, so the
            # feasible sets of the first one are the candidate sets.
            score_order = self._score_order(vehicles[0])
            # tail_cost[k]: theta(empty) of the last k vehicles of the prefix.
            tail_cost = np.cumsum(
                [0.0] + [self.theta_empty(v) for v in reversed(vehicles)]
            )
            if (
                self.kernel is not None
                and type_index == terminal_type
                and p == 1
            ):
                terminal_value, previous_mask, assigned_mask = (
                    self._terminal_single_kernel(g, thetas[0], tail_cost)
                )
                suffix_empty = sum(
                    self.theta_empty(vehicle)
                    for later_index in range(type_index, len(self.layout.groups))
                    for vehicle in self.layout.groups[later_index][
                        count if later_index == type_index else 0:
                    ]
                )
                terminal = (
                    terminal_value + suffix_empty,
                    previous_mask,
                    assigned_mask,
                    vehicles[0],
                )
                break
            if self.kernel is not None:
                g, slot_sets = self._type_layer_kernel(
                    g, thetas, *score_order, tail_cost
                )
            else:
                g, slot_sets = self._type_layer_numpy(
                    g, thetas, self._score_groups_numpy(vehicles[0]), tail_cost, shape
                )
            g = g + sum(self.theta_empty(v) for v in group[count:])
            trace.append((vehicles, slot_sets))
            if cacheable_layer:
                self._prefix_cache_put(cache_key, g, slot_sets)

        if terminal is None:
            total = g + self.outsource_sum[
                self.full_mask ^ np.arange(1 << self.n)
            ]
            best_mask = int(np.argmin(total))
            value = float(total[best_mask]) + self.inactive_outsourcing
        else:
            terminal_value, previous_mask, assigned_mask, terminal_vehicle = terminal
            value = terminal_value + self.inactive_outsourcing
        if not math.isfinite(value):
            raise SubsetDPNotApplicable("no feasible assignment (unexpected)")

        # Reconstruct the assignment backwards through the types.
        alpha: Dict[str, int] = {}
        y: Dict[str, int] = {}
        if terminal is None:
            R = best_mask
        else:
            R = previous_mask
            if assigned_mask:
                y[f"y[{terminal_vehicle}]"] = 1
                for i in range(self.n):
                    if assigned_mask >> i & 1:
                        customer = self.customers[self.active_idx[i]]
                        alpha[f"alpha[{customer},{terminal_vehicle}]"] = 1
        for vehicles, slot_sets in reversed(trace):
            if not vehicles:
                continue
            served = 0
            for slot, v in enumerate(vehicles, start=1):
                mask = int(slot_sets[slot - 1][R])
                if mask == 0:
                    continue
                if mask & ~R or mask & served:
                    raise RuntimeError("subset DP trace is inconsistent")
                served |= mask
                y[f"y[{v}]"] = 1
                for i in range(self.n):
                    if mask >> i & 1:
                        alpha[f"alpha[{self.customers[self.active_idx[i]]},{v}]"] = 1
            R ^= served
        if R != 0:
            raise RuntimeError("subset DP trace did not return to the empty set")

        seconds = time.time() - started
        self.stats["solves"] += 1
        self.stats["seconds"] += seconds
        lb = math.nextafter(value - self._fixed_lb_margin(), -math.inf)
        return {
            "counts": counts,
            "value": value,
            "lb": lb,
            "x_dict": {**alpha, **y},
            "status": STATUS + ("_cpp" if self.kernel is not None else ""),
            "optimal": True,
            "seconds": seconds,
        }

    def solve_lagrangian(self, pi_value: Mapping[str, float]) -> dict:
        """Solve the free-fleet Stage-2 Lagrangian problem in one DP pass.

        For a type with ``i`` leading vehicles used by the assignment,
        ``purchase_order`` permits every purchased prefix ``q >= i``.  Its
        best contribution is therefore the largest exact prefix sum of the
        multipliers among those ``q``.  Folding that choice into the type
        layer's unused-vehicle tail cost optimizes assignment, activation and
        purchase decisions together without enumerating fixed-fleet pieces.
        """
        started = time.time()
        period = int(self.node.info[1])
        pi = self.layout.pi_by_vehicle(pi_value, period)
        shape = (2,) * self.n + (1,)
        size = 1 << self.n
        g = np.full(size, np.inf)
        g[0] = 0.0
        trace = []
        terminal = None

        for type_index, group in enumerate(self.layout.groups):
            vehicles = list(group)
            count = len(vehicles)
            if count == 0:
                trace.append((vehicles, None, (0,)))
                continue
            for other in vehicles[1:]:
                if float(self.prob_data.Qv[other]) != float(self.prob_data.Qv[vehicles[0]]):
                    raise SubsetDPNotApplicable("vehicles of one type must share capacity")

            prefix_credit = [Fraction(0)]
            for vehicle in vehicles:
                prefix_credit.append(prefix_credit[-1] + pi[vehicle])
            best_purchase = []
            for used in range(count + 1):
                chosen = used
                for purchased in range(used + 1, count + 1):
                    if prefix_credit[purchased] > prefix_credit[chosen]:
                        chosen = purchased
                best_purchase.append(chosen)

            thetas = [self.theta(vehicle) for vehicle in vehicles]
            score_order = self._score_order(vehicles[0])
            empty_suffix = np.cumsum(
                [0.0]
                + [self.theta_empty(vehicle) for vehicle in reversed(vehicles)]
            )
            tail_cost = np.empty(count + 1, dtype=np.float64)
            for used, purchased in enumerate(best_purchase):
                tail_cost[count - used] = (
                    empty_suffix[count - used] - float(prefix_credit[purchased])
                )
            if (
                self.kernel is not None
                and type_index == len(self.layout.groups) - 1
                and count == 1
            ):
                terminal_value, previous_mask, assigned_mask = (
                    self._terminal_single_kernel(g, thetas[0], tail_cost)
                )
                terminal = (
                    terminal_value,
                    previous_mask,
                    assigned_mask,
                    vehicles[0],
                    tuple(best_purchase),
                )
                break
            if self.kernel is not None:
                g, slot_sets = self._type_layer_kernel(
                    g, thetas, *score_order, tail_cost
                )
            else:
                g, slot_sets = self._type_layer_numpy(
                    g, thetas, self._score_groups_numpy(vehicles[0]), tail_cost, shape
                )
            trace.append((vehicles, slot_sets, tuple(best_purchase)))

        if terminal is None:
            total = g + self.outsource_sum[
                self.full_mask ^ np.arange(1 << self.n)
            ]
            best_mask = int(np.argmin(total))
            value = float(total[best_mask]) + self.inactive_outsourcing
        else:
            terminal_value, previous_mask, assigned_mask, terminal_vehicle, _ = terminal
            value = terminal_value + self.inactive_outsourcing
        if not math.isfinite(value):
            raise SubsetDPNotApplicable("no feasible assignment (unexpected)")

        alpha: Dict[str, int] = {}
        y: Dict[str, int] = {}
        z: Dict[str, int] = {}
        if terminal is None:
            remaining = best_mask
        else:
            remaining = previous_mask
            used = int(bool(assigned_mask))
            purchased = terminal[-1][used]
            if assigned_mask:
                y[f"y[{terminal_vehicle}]"] = 1
                for active_pos in range(self.n):
                    if assigned_mask >> active_pos & 1:
                        customer = self.customers[self.active_idx[active_pos]]
                        alpha[f"alpha[{customer},{terminal_vehicle}]"] = 1
            if purchased:
                z[f"z[{terminal_vehicle},{period}]"] = 1
        for vehicles, slot_sets, best_purchase in reversed(trace):
            if not vehicles:
                continue
            served = 0
            used = 0
            for slot, vehicle in enumerate(vehicles, start=1):
                mask = int(slot_sets[slot - 1][remaining])
                if mask == 0:
                    continue
                if mask & ~remaining or mask & served:
                    raise RuntimeError("subset DP trace is inconsistent")
                served |= mask
                used += 1
                y[f"y[{vehicle}]"] = 1
                for active_pos in range(self.n):
                    if mask >> active_pos & 1:
                        customer = self.customers[self.active_idx[active_pos]]
                        alpha[f"alpha[{customer},{vehicle}]"] = 1
            purchased = best_purchase[used]
            for vehicle in vehicles[:purchased]:
                z[f"z[{vehicle},{period}]"] = 1
            remaining ^= served
        if remaining != 0:
            raise RuntimeError("subset DP trace did not return to the empty set")

        seconds = time.time() - started
        self.stats["solves"] += 1
        self.stats["seconds"] += seconds
        lb = math.nextafter(
            value - self._lagrangian_lb_margin(pi), -math.inf
        )
        return {
            "value": value,
            "lb": lb,
            "x_dict": {**alpha, **y, **z},
            "status": STATUS + "_lagrangian" + ("_cpp" if self.kernel is not None else ""),
            "optimal": True,
            "seconds": seconds,
        }


def subset_dp_solver_or_none(prob_data, node, cuts_payload, layout: FleetLayout, *,
                             max_customers: Optional[int] = None,
                             log=None) -> Optional[SubsetDPPieceSolver]:
    """Use the memory-aware C++ ceiling, or 16 with the numpy fallback."""
    if max_customers is None:
        max_customers = default_max_customers()
    try:
        return SubsetDPPieceSolver(
            prob_data,
            node,
            cuts_payload,
            layout,
            max_customers=max_customers,
            prefix_layer_cache=_env_flag(
                _PREFIX_LAYER_CACHE_ENV, default=True
            ),
        )
    except SubsetDPNotApplicable as exc:
        if log is not None:
            log(f"subset DP not applicable ({exc}); Gurobi piece solver")
        return None
