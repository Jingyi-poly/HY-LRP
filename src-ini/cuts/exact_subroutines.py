"""ESPPRC / Stage-2 BPC 接口：分别传递可行 incumbent 与认证下界。"""
import glob
import importlib
import importlib.util
import math
import operator
import os
import subprocess
import sys
import sysconfig
from fractions import Fraction

import numpy as np
from core.backend_telemetry import backend_call, record_backend_event

from cuts.static_valid_inequalities import (
    STATIC_CUT_ZERO_TOL,
    compute_min_incoming_costs,
)


class InvalidS2LagrangianPolicy(ValueError):
    """A solver point cannot certify an original-model Stage-2 policy."""


class InvalidS3LagrangianPolicy(ValueError):
    """A solver point cannot certify an original-model Stage-3 policy."""


# Private compatibility name used by the BPC adapter below.
_InvalidS2BpcIncumbent = InvalidS2LagrangianPolicy


def _run_native_backend(backend, function, *args, **kwargs):
    """Count native invocations, separately from adapter skips and validation."""
    with backend_call(backend) as outcome:
        result = function(*args, **kwargs)
        if isinstance(result, dict):
            for key in ("status", "status_reason", "timed_out", "interrupted",
                        "label_budget_exhausted", "optimality_proven"):
                if key in result:
                    outcome[key] = result[key]
            outcome["native_lb_certified"] = result.get("lb_certified")
        return result


def _finite_binary64_fraction(value, *, label):
    """Exact rational represented by one finite binary64 input."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidS2BpcIncumbent(
            f"{label}_not_finite_binary64"
        ) from exc
    if not math.isfinite(scalar):
        raise _InvalidS2BpcIncumbent(f"{label}_not_finite_binary64")
    return Fraction.from_float(scalar)


def _fraction_to_finite_float_up(value, *, label):
    """Finite binary64 upper endpoint for an exact rational value."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise _InvalidS2BpcIncumbent(f"{label}_overflow") from exc
    if not math.isfinite(rounded):
        raise _InvalidS2BpcIncumbent(f"{label}_overflow")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise _InvalidS2BpcIncumbent(f"{label}_overflow")
    return rounded


class CompiledCutPayload:
    """Dense binary64 view of one Stage-3 cut payload for fast exact scoring.

    Scoring a 0/1 Stage-2 policy needs ``theta[h] = max(floor, max over cuts of
    beta + piY.y + piAlpha.alpha)`` exactly.  Doing that in ``Fraction`` over
    every coefficient costs ``O(#cuts * m * n)`` rational operations per policy
    (seconds once the pools hold a few hundred cuts).  Here every cut is first
    evaluated in float64 by vectorised dot products, a rigorous rounding-error
    envelope isolates the cuts that can still attain the true maximum, and only
    those are re-evaluated as exact rationals.  Multipliers are exact 0/1, so
    each product is exact and the standard bound
    ``|fl(sum) - sum| <= gamma_k * sum|terms|`` (``gamma_k = k*u/(1-k*u)``,
    ``u = 2^-53``) holds for any summation order and with or without FMA.
    The returned maxima are therefore bit-identical to the all-rational loop.
    """

    _UNIT_ROUNDOFF = 2.0 ** -53

    def __init__(self, cuts_payload, m, n, *, error=None):
        error = error or _InvalidS2BpcIncumbent
        cuts = list(cuts_payload)
        count = len(cuts)
        succ = np.empty(count, dtype=np.int64)
        beta = np.empty(count, dtype=np.float64)
        pi_y = np.empty((count, m), dtype=np.float64)
        pi_alpha = np.empty((count, m, n), dtype=np.float64)
        for index, cut in enumerate(cuts):
            try:
                position = int(cut["succ"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise error(f"cut_{index}_bad_successor") from exc
            if position != cut["succ"]:
                raise error(f"cut_{index}_bad_successor")
            succ[index] = position
            try:
                beta[index] = float(cut.get("beta"))
                row_y = np.asarray(cut["piY"], dtype=np.float64)
                row_alpha = np.asarray(cut["piAlpha"], dtype=np.float64)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise error(f"cut_{index}_bad_shape") from exc
            if row_y.shape != (m,):
                raise error(f"cut_{index}_bad_shape")
            if row_alpha.shape != (m, n):
                # An empty vehicle/customer set is a 1-D empty list in the payload.
                if m * n == 0 and row_alpha.size == 0:
                    row_alpha = row_alpha.reshape(m, n)
                else:
                    raise error(f"cut_{index}_bad_shape")
            pi_y[index] = row_y
            pi_alpha[index] = row_alpha
        if not (
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(pi_y))
            and np.all(np.isfinite(pi_alpha))
        ):
            bad = [
                index for index in range(count)
                if not (
                    math.isfinite(beta[index])
                    and np.all(np.isfinite(pi_y[index]))
                    and np.all(np.isfinite(pi_alpha[index]))
                )
            ]
            raise error(f"cut[{bad[0]}] coefficient is not finite binary64")
        self.count = count
        self.m = m
        self.n = n
        self.succ = succ
        self.beta = beta
        self.pi_y = pi_y.reshape(count, m)
        self.pi_alpha = pi_alpha.reshape(count, m * n)
        self._abs_beta = np.abs(beta)
        self._abs_pi_y = np.abs(self.pi_y)
        self._abs_pi_alpha = np.abs(self.pi_alpha)
        # gamma_k for the longest possible dot product (all coefficients),
        # doubled for slack; zero multipliers only make the true error smaller.
        k = 1 + m + m * n
        ku = k * self._UNIT_ROUNDOFF
        self._gamma = 2.0 * ku / (1.0 - ku)
        self._by_successor = {}
        for position in np.unique(succ):
            self._by_successor[int(position)] = np.flatnonzero(succ == position)

    def successor_positions(self):
        return self._by_successor.keys()

    def _exact_max(self, indices, y_positions, alpha_positions):
        """Exact ``max`` of the candidate cuts' right-hand sides as a Fraction.

        Every binary64 value is ``num / 2^k`` exactly, so the terms of all
        candidates are put over one common power-of-two denominator and summed
        as Python integers; only the winner is converted to a ``Fraction``.
        """
        blocks = [self.beta[indices][:, None]]
        if y_positions:
            blocks.append(self.pi_y[indices][:, y_positions])
        if len(alpha_positions):
            blocks.append(self.pi_alpha[indices][:, alpha_positions])
        rows = np.concatenate(blocks, axis=1).tolist()
        ratios = [[value.as_integer_ratio() for value in row] for row in rows]
        shift = 0
        for row in ratios:
            for _num, den in row:
                bits = den.bit_length() - 1
                if bits > shift:
                    shift = bits
        best = None
        for row in ratios:
            total = 0
            for num, den in row:
                total += num << (shift - (den.bit_length() - 1))
            if best is None or total > best:
                best = total
        return Fraction(best, 1 << shift)

    def exact_theta(self, y_bits, alpha_bits, floor, n_successors):
        """``[max(floor, max_c rhs_c)]`` per successor position, exact.

        ``y_bits`` (``m``) and ``alpha_bits`` (``m x n``) are exact 0/1 ints;
        ``floor`` is a ``Fraction``.  Successor positions without cuts get
        ``floor``.
        """
        y_vec = np.asarray(y_bits, dtype=np.float64).reshape(self.m)
        alpha_mat = np.asarray(alpha_bits, dtype=np.float64).reshape(self.m * self.n)
        if not (
            np.all((y_vec == 0.0) | (y_vec == 1.0))
            and np.all((alpha_mat == 0.0) | (alpha_mat == 1.0))
        ):
            raise _InvalidS2BpcIncumbent("policy_multipliers_not_binary")
        theta = [floor] * n_successors
        if self.count == 0:
            return theta
        values = self.beta + self.pi_y @ y_vec + self.pi_alpha @ alpha_mat
        magnitude = self._abs_beta + self._abs_pi_y @ y_vec + self._abs_pi_alpha @ alpha_mat
        envelope = self._gamma * magnitude
        y_positions = [v_pos for v_pos in range(self.m) if y_bits[v_pos]]
        alpha_positions = np.flatnonzero(alpha_mat)
        for position, indices in self._by_successor.items():
            if position < 0 or position >= n_successors:
                raise _InvalidS2BpcIncumbent(f"cut_{int(indices[0])}_bad_successor")
            local_values = values[indices]
            best = int(np.argmax(local_values))
            threshold = local_values[best] - envelope[indices][best] - envelope[indices]
            candidates = indices[local_values >= threshold]
            exact = self._exact_max(candidates, y_positions, alpha_positions)
            if exact > floor:
                theta[position] = exact
        return theta


_COMPILED_PAYLOAD_CACHE = {}
# One payload per oracle session / forward node is live at a time; the cache
# holds the payload lists themselves, so keep it small (tens of MB each at C20).
_COMPILED_PAYLOAD_CACHE_LIMIT = 8


def compiled_cut_payload(cuts_payload, m, n, *, error=None):
    """``CompiledCutPayload`` for ``cuts_payload``, memoised on the list object.

    Payload lists are built once per oracle session / forward node and then
    only read, so identity plus length is a sound cache key; the cache keeps a
    strong reference to the list so its ``id`` cannot be recycled while cached.
    """
    if not isinstance(cuts_payload, list):
        return CompiledCutPayload(cuts_payload, m, n, error=error)
    key = id(cuts_payload)
    entry = _COMPILED_PAYLOAD_CACHE.get(key)
    if entry is not None:
        held, first, last, compiled = entry
        if (
            held is cuts_payload
            and compiled.count == len(cuts_payload)
            and compiled.m == m
            and compiled.n == n
            and (compiled.count == 0
                 or (first is cuts_payload[0] and last is cuts_payload[-1]))
        ):
            return compiled
        del _COMPILED_PAYLOAD_CACHE[key]
    compiled = CompiledCutPayload(cuts_payload, m, n, error=error)
    if len(_COMPILED_PAYLOAD_CACHE) >= _COMPILED_PAYLOAD_CACHE_LIMIT:
        _COMPILED_PAYLOAD_CACHE.pop(next(iter(_COMPILED_PAYLOAD_CACHE)))
    first = cuts_payload[0] if cuts_payload else None
    last = cuts_payload[-1] if cuts_payload else None
    _COMPILED_PAYLOAD_CACHE[key] = (cuts_payload, first, last, compiled)
    return compiled


def _strict_native_bit(value, *, label):
    """Decode the pybind integer contract without an integrality tolerance.

    The current extension returns ``int`` arrays.  Accepting a near-binary
    float from an old/corrupt ABI would silently turn a relaxed point into an
    original-model incumbent.  Such a point may still accompany a valid native
    relaxation LB, but it cannot provide an incumbent or bundle subgradient.
    """
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidS2BpcIncumbent(f"{label}_not_binary") from exc
    if not math.isfinite(scalar) or scalar not in (0.0, 1.0):
        raise _InvalidS2BpcIncumbent(f"{label}_not_binary")
    return int(scalar)


def _policy_bit(value, *, label, tolerance):
    """Decode one binary decision, optionally snapping solver noise.

    Snapping does not certify the raw point.  It constructs an exact binary
    candidate which is subsequently checked against every physical Stage-2
    row and re-scored from scratch.  ``tolerance=0`` enforces the stricter
    native pybind integer contract.
    """
    try:
        tol = float(tolerance)
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidS2BpcIncumbent(f"{label}_not_binary") from exc
    if (
        not math.isfinite(tol)
        or tol < 0.0
        or tol >= 0.5
        or not math.isfinite(scalar)
    ):
        raise _InvalidS2BpcIncumbent(f"{label}_not_binary")
    if tol == 0.0:
        return _strict_native_bit(scalar, label=label)
    candidate = 1 if scalar > 0.5 else 0
    if abs(scalar - candidate) > tol:
        raise _InvalidS2BpcIncumbent(f"{label}_not_binary")
    return candidate


def _normalize_certified_min_lb(lb, claimed_certified, incumbent=None):
    """Validate a backend-provided lower bound for a minimization problem.

    A contradictory ``lb > incumbent`` pair is not repaired with ``min``:
    the incumbent is an upper bound, so using it as a lower bound would make
    backward cuts invalid. Instead, discard the claimed certificate.
    """
    try:
        lb_value = float(lb)
    except (TypeError, ValueError, OverflowError):
        return False, float("-inf")
    if not claimed_certified or not np.isfinite(lb_value):
        return False, float("-inf")

    if incumbent is not None:
        try:
            incumbent_value = float(incumbent)
        except (TypeError, ValueError, OverflowError):
            return False, float("-inf")
        if not np.isfinite(incumbent_value) or lb_value > incumbent_value:
            return False, float("-inf")
    return True, lb_value


# Env: VRP_USE_ESP_BP, VRP_PHASE{1,2}_S3_USE_ESP, VRP_PHASE1_S2_USE_BP
def _bool_env(name: str, default_val: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default_val)
    return raw not in ("0", "false", "False")


def is_enabled() -> bool:
    """Master switch (caller 一般不直接用这个, 用下面 phase-specific 的)."""
    return _bool_env("VRP_USE_ESP_BP", False)


def is_phase1_s3_esp_enabled() -> bool:
    return _bool_env("VRP_PHASE1_S3_USE_ESP", False)


def is_phase1_s2_bp_enabled() -> bool:
    return _bool_env("VRP_PHASE1_S2_USE_BP", False)


def is_phase2_s3_esp_enabled() -> bool:
    return _bool_env("VRP_PHASE2_S3_USE_ESP", is_enabled())


# C++ 扩展加载 (s3backward / s2backward/bpc, 含可选 auto-build)
def _customized_subprob_root():
    from core import customized_subprob
    return customized_subprob.ROOT


def _import_espprc_cpp():
    """Import espprc_cpp from src-ini/customized-subprob/s3backward."""
    try:
        import espprc_cpp  # type: ignore
        return espprc_cpp
    except Exception:
        pass
    from core import customized_subprob
    esp_dir = os.environ.get("VRP_S3_ESP_DIR", customized_subprob.S3_BACKWARD_DIR)
    if os.path.isdir(esp_dir) and esp_dir not in sys.path:
        sys.path.insert(0, esp_dir)
    try:
        import espprc_cpp  # type: ignore
        return espprc_cpp
    except Exception:
        return None


def _build_stage2_bpc_cpp_if_needed(bpc_dir):
    """Auto-build s2backward/bpc/stage2_bp_cpp when source is newer than the .so."""
    if int(os.environ.get("VRP_S2_BPC_AUTO_BUILD", "1")) == 0:
        return False

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = os.path.join(bpc_dir, f"stage2_bp_cpp{ext_suffix}")
    candidates = glob.glob(os.path.join(bpc_dir, "stage2_bp_cpp*.so"))
    if candidates and not os.path.exists(so_path):
        so_path = max(candidates, key=os.path.getmtime)

    src_paths = [
        os.path.join(bpc_dir, "stage2_branch_price.cpp"),
        os.path.join(bpc_dir, "stage2_bp_pybind.cpp"),
        os.path.join(bpc_dir, "build.sh"),
    ]
    if os.path.exists(so_path):
        so_mtime = os.path.getmtime(so_path)
        if all((not os.path.exists(p)) or os.path.getmtime(p) <= so_mtime for p in src_paths):
            return False

    build_sh = os.path.join(bpc_dir, "build.sh")
    if not os.path.exists(build_sh):
        return False
    try:
        subprocess.run(
            ["bash", build_sh],
            cwd=bpc_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(os.environ.get("VRP_S2_BPC_BUILD_TIMEOUT_S", "120")),
            # The extension is imported by *this* interpreter, so it has to be
            # compiled against it; build.sh would otherwise take ``python`` from
            # PATH, which need not even be the env holding pybind11.
            env={**os.environ, "PYTHON_BIN": sys.executable},
        )
        return True
    except Exception as exc:
        print(f"[S2-BPC] auto-build failed ({type(exc).__name__}: {exc}); fallback",
              flush=True)
        return False


def _import_stage2_bpc_cpp():
    """Import stage2_bp_cpp from src-ini/customized-subprob/s2backward/bpc."""
    from core import customized_subprob
    bpc_dir = os.environ.get("VRP_S2_BPC_DIR", customized_subprob.S2_BPC_DIR)
    if not os.path.isdir(bpc_dir):
        return None
    _build_stage2_bpc_cpp_if_needed(bpc_dir)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    preferred = os.path.join(bpc_dir, f"stage2_bp_cpp{ext_suffix}")
    candidates = glob.glob(os.path.join(bpc_dir, "stage2_bp_cpp*.so"))
    so_path = preferred if os.path.exists(preferred) else (max(candidates, key=os.path.getmtime) if candidates else None)
    if so_path is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("stage2_bp_cpp", so_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"[S2-BPC] import failed ({type(exc).__name__}: {exc}); fallback", flush=True)
        return None


espprc_cpp = _import_espprc_cpp()
stage2_bpc_cpp = _import_stage2_bpc_cpp()


# Backward exact backends (env 默认值; phase 开关见文件头)
_S3_USE_ESP            = int(os.environ.get("VRP_S3_USE_ESP", "1"))
_S3_ESP_TOP_K          = int(os.environ.get("VRP_S3_ESP_TOP_K", "5"))
_S3_ESP_LABEL_BUDGET   = int(os.environ.get("VRP_S3_ESP_LABEL_BUDGET", "0"))
# ng-route relaxation neighbourhood of the certifying PCTSP core (8 is the
# usual sweet spot; the DSSR loop restores elementarity where it matters).
_S3_ESP_NG_SIZE        = int(os.environ.get("VRP_S3_ESP_NG_SIZE", "8"))

_S2_USE_BP             = int(os.environ.get("VRP_S2_USE_BP", "1"))
_S2_BP_TOPK            = int(os.environ.get("VRP_S2_BP_TOPK", "5"))
_S2_BP_MAX_NODES       = int(os.environ.get("VRP_S2_BP_MAX_NODES", "1000000"))
_S2_BP_MAX_DEPTH       = int(os.environ.get("VRP_S2_BP_MAX_DEPTH", "100000"))
_S2_BP_MAX_CG          = int(os.environ.get("VRP_S2_BP_MAX_CG", "1000"))
_S2_BP_TIME_LIMIT_S    = float(os.environ.get("VRP_S2_BP_TIME_LIMIT_S", "7200.0"))
_S2_BP_RC_TOL          = float(os.environ.get("VRP_S2_BP_RC_TOL", "1e-7"))
_S2_BP_INT_TOL         = float(os.environ.get("VRP_S2_BP_INT_TOL", "1e-6"))
_S2_BPC_USE_HEURISTIC  = _bool_env("VRP_S2_BPC_USE_HEURISTIC_PRICING", True)
_S2_BPC_USE_CLUSTERING = _bool_env("VRP_S2_BPC_USE_VEHICLE_CLUSTERING", True)
_S2_BPC_USE_DIVING     = _bool_env("VRP_S2_BPC_USE_DIVING", True)
_S2_BPC_USE_RYAN_FOSTER = _bool_env("VRP_S2_BPC_USE_RYAN_FOSTER", True)
_S2_BPC_USE_DUAL_STAB  = _bool_env("VRP_S2_BPC_USE_DUAL_STABILIZATION", True)
_S2_BPC_CULL_RC        = float(os.environ.get("VRP_S2_BPC_CULL_RC_THRESHOLD", "10.0"))
_S2_BPC_NUM_THREADS    = int(os.environ.get("VRP_S2_BPC_NUM_THREADS", "1"))
_S2_BPC_TOPK_ROOT      = int(os.environ.get("VRP_S2_BPC_PRICING_TOPK_ROOT", "30"))
_S2_BPC_TOPK_SHALLOW   = int(os.environ.get("VRP_S2_BPC_PRICING_TOPK_SHALLOW", "20"))
_S2_BPC_TOPK_DEEP      = int(os.environ.get("VRP_S2_BPC_PRICING_TOPK_DEEP", "5"))
_S2_BPC_USE_RESTRICTED_MIP = _bool_env("VRP_S2_BPC_USE_RESTRICTED_MIP", True)
_S2_BPC_RESTRICTED_MIP_TIME_LIMIT = float(os.environ.get("VRP_S2_BPC_RESTRICTED_MIP_TIME_LIMIT", "0.5"))
_S2_BPC_USE_CUT_AGING  = _bool_env("VRP_S2_BPC_USE_CUT_AGING", True)
# Theta LB fixed at 0 (unused vehicle / zero Stage-3 is valid)
_S2_BPC_THETA_LOWER_BOUND = 0.0
# Generic defaults; Phase 1/2 wrappers override root-only separately
_S2_BPC_ROOT_BOUND_ONLY = _bool_env("VRP_S2_BPC_ROOT_BOUND_ONLY", False)
_S2_BPC_CUT_ROUNDS = int(os.environ.get("VRP_S2_BPC_CUT_ROUNDS", "-1"))
_S2_BPC_USE_SR3_CUTS = _bool_env("VRP_S2_BPC_USE_SR3_CUTS", True)
_S2_BPC_USE_COVER_CUTS = _bool_env("VRP_S2_BPC_USE_COVER_CUTS", True)
_S2_BPC_USE_CLIQUE_CUTS = _bool_env("VRP_S2_BPC_USE_CLIQUE_CUTS", True)
_PHASE1_S2_BPC_ROOT_BOUND_ONLY = _bool_env(
    "VRP_PHASE1_S2_BPC_ROOT_BOUND_ONLY", True
)
_PHASE1_S2_BPC_ROOT_CUT_ROUNDS = int(
    os.environ.get("VRP_PHASE1_S2_BPC_ROOT_CUT_ROUNDS", "0")
)
_PHASE1_S2_BPC_MIN_ACTIVE = int(
    os.environ.get("VRP_PHASE1_S2_BPC_MIN_ACTIVE", "21")
)

def _s3_fraction(value, *, label):
    """Stage-3 wrapper around the shared exact binary64 decoder."""
    try:
        return _finite_binary64_fraction(value, label=label)
    except InvalidS2LagrangianPolicy as exc:
        raise InvalidS3LagrangianPolicy(str(exc)) from exc


def _s3_float_up(value, *, label):
    """Stage-3 wrapper around the shared directed-up conversion."""
    try:
        return _fraction_to_finite_float_up(value, label=label)
    except InvalidS2LagrangianPolicy as exc:
        raise InvalidS3LagrangianPolicy(str(exc)) from exc


def _s3_policy_bit(value, *, label, tolerance=0.0):
    """Decode one Stage-3 bit, then validate the snapped policy exactly."""
    if isinstance(value, (str, bytes)):
        raise InvalidS3LagrangianPolicy(f"{label}_not_binary")
    try:
        return _policy_bit(value, label=label, tolerance=tolerance)
    except InvalidS2LagrangianPolicy as exc:
        raise InvalidS3LagrangianPolicy(str(exc)) from exc


def _strict_s3_index(value, *, label):
    """Decode a native/global node identifier without lossy ``int(...)``."""
    if isinstance(value, (bool, np.bool_)):
        raise InvalidS3LagrangianPolicy(f"{label}_not_integer")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidS3LagrangianPolicy(f"{label}_not_integer") from exc


def certify_s3_lagrangian_policy(
    probData,
    node,
    pi_value,
    raw_policy,
    *,
    binary_tolerance=0.0,
):
    """Validate and independently score one Stage-3 Lagrangian policy.

    The canonical payload contains ``y`` and either a complete global-customer
    ``alpha`` mapping or a dense alpha vector in ``probData.J`` order.  It must
    then contain exactly one route representation:

    - global ``path``: empty for no active customer, otherwise the open route
      ``[depot_start, active customers..., depot_end]``;
    - global ``selected_arcs``: a mapping ``(i,j)->raw_bit``, a sequence of
      ``(i,j,raw_bit)``, or an already-binary sequence of selected ``(i,j)``.

    The arc form is snapped strictly and reconstructed into one unique
    depot-to-depot path; branches, cycles, missing visits, and foreign arcs
    are rejected. Backend objective values are not inputs: feasibility, route
    cost, and

        route_cost - pi_y*y - sum_j(pi_alpha[j]*alpha[j])

    are rebuilt from exact rationals in the binary64 model data.
    Only the final public endpoints are rounded toward ``+inf``.  The default
    native boundary requires literal 0/1 values; a Gurobi adapter may provide a
    small ``binary_tolerance`` and receives a snapped policy only after every
    original-model row below has been checked exactly.
    """
    if not hasattr(pi_value, "get") or not hasattr(raw_policy, "__getitem__"):
        raise InvalidS3LagrangianPolicy("bad_policy_shape")
    try:
        raw_customers = list(probData.J)
        vehicle = node.info
        num_all_nodes = _strict_s3_index(
            probData.numAllnodes, label="numAllnodes"
        )
    except (AttributeError, TypeError) as exc:
        raise InvalidS3LagrangianPolicy("stage3_metadata_missing") from exc

    customers = [
        _strict_s3_index(customer, label=f"customer[{position}]")
        for position, customer in enumerate(raw_customers)
    ]
    if len(set(customers)) != len(customers):
        raise InvalidS3LagrangianPolicy("duplicate_customer_identifier")
    customer_set = set(customers)
    depot_start = num_all_nodes - 2
    depot_end = num_all_nodes - 1
    if (
        num_all_nodes < 2
        or any(customer < 0 or customer >= depot_start for customer in customers)
        or depot_start in customer_set
        or depot_end in customer_set
    ):
        raise InvalidS3LagrangianPolicy("invalid_depot_metadata")

    active = {}
    volumes = {}
    dual_alpha = {}
    for customer in customers:
        try:
            raw_active = node.active[customer]
            raw_volume = node.volume[customer]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise InvalidS3LagrangianPolicy(
                f"customer_{customer}_metadata_missing"
            ) from exc
        active[customer] = _s3_policy_bit(
            raw_active, label=f"active[{customer}]", tolerance=0.0
        )
        volumes[customer] = _s3_fraction(
            raw_volume, label=f"volume[{customer}]"
        )
        if volumes[customer] < 0:
            raise InvalidS3LagrangianPolicy(
                f"volume[{customer}]_negative"
            )
        dual_alpha[customer] = _s3_fraction(
            pi_value.get(f"alpha[{customer},{vehicle}]", 0.0),
            label=f"pi_alpha[{customer},{vehicle}]",
        )

    try:
        raw_y = raw_policy["y"]
        raw_alpha = raw_policy["alpha"]
        has_path = "path" in raw_policy
        has_selected_arcs = "selected_arcs" in raw_policy
    except (KeyError, TypeError) as exc:
        raise InvalidS3LagrangianPolicy("bad_policy_shape") from exc
    if has_path == has_selected_arcs:
        raise InvalidS3LagrangianPolicy(
            "policy_requires_exactly_one_route_representation"
        )

    y = _s3_policy_bit(
        raw_y, label=f"y[{vehicle}]", tolerance=binary_tolerance
    )
    if hasattr(raw_alpha, "items"):
        try:
            alpha_items = list(raw_alpha.items())
        except TypeError as exc:
            raise InvalidS3LagrangianPolicy("alpha_bad_shape") from exc
    else:
        try:
            dense_alpha = list(raw_alpha)
        except TypeError as exc:
            raise InvalidS3LagrangianPolicy("alpha_bad_shape") from exc
        if len(dense_alpha) != len(customers):
            raise InvalidS3LagrangianPolicy("alpha_bad_shape")
        alpha_items = list(zip(customers, dense_alpha))

    alpha = {}
    for raw_customer, raw_value in alpha_items:
        customer = _strict_s3_index(
            raw_customer, label="alpha_customer"
        )
        if customer in alpha:
            raise InvalidS3LagrangianPolicy(
                f"alpha[{customer}]_duplicate"
            )
        alpha[customer] = _s3_policy_bit(
            raw_value,
            label=f"alpha[{customer},{vehicle}]",
            tolerance=binary_tolerance,
        )
    if set(alpha) != customer_set:
        raise InvalidS3LagrangianPolicy("alpha_customer_domain_mismatch")

    selected = {customer for customer in customers if alpha[customer]}
    selected_active = {
        customer for customer in selected if active[customer]
    }
    selected_arc_set = set()
    if has_path:
        try:
            raw_path = list(raw_policy["path"])
            path = tuple(
                _strict_s3_index(value, label=f"path[{position}]")
                for position, value in enumerate(raw_path)
            )
        except TypeError as exc:
            raise InvalidS3LagrangianPolicy("path_bad_shape") from exc
        selected_arc_set = set(zip(path, path[1:])) if path else set()
    else:
        raw_selected_arcs = raw_policy["selected_arcs"]
        arc_values = []
        if hasattr(raw_selected_arcs, "items"):
            try:
                raw_arc_items = list(raw_selected_arcs.items())
            except TypeError as exc:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_bad_shape"
                ) from exc
            for position, (raw_arc, raw_value) in enumerate(raw_arc_items):
                try:
                    arc_pair = list(raw_arc)
                except TypeError as exc:
                    raise InvalidS3LagrangianPolicy(
                        f"selected_arc[{position}]_bad_shape"
                    ) from exc
                if len(arc_pair) != 2:
                    raise InvalidS3LagrangianPolicy(
                        f"selected_arc[{position}]_bad_shape"
                    )
                arc_values.append((arc_pair[0], arc_pair[1], raw_value))
        else:
            try:
                raw_arc_items = list(raw_selected_arcs)
            except TypeError as exc:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_bad_shape"
                ) from exc
            for position, raw_arc in enumerate(raw_arc_items):
                try:
                    arc_parts = list(raw_arc)
                except TypeError as exc:
                    raise InvalidS3LagrangianPolicy(
                        f"selected_arc[{position}]_bad_shape"
                    ) from exc
                if len(arc_parts) == 2:
                    arc_values.append((arc_parts[0], arc_parts[1], 1))
                elif len(arc_parts) == 3:
                    arc_values.append(tuple(arc_parts))
                else:
                    raise InvalidS3LagrangianPolicy(
                        f"selected_arc[{position}]_bad_shape"
                    )

        active_customer_set = {
            customer for customer in customers if active[customer]
        }
        legal_route_nodes = active_customer_set | {depot_start, depot_end}
        decoded_arcs = {}
        for position, (raw_tail, raw_head, raw_value) in enumerate(arc_values):
            tail = _strict_s3_index(
                raw_tail, label=f"selected_arc[{position}].tail"
            )
            head = _strict_s3_index(
                raw_head, label=f"selected_arc[{position}].head"
            )
            arc = (tail, head)
            if arc in decoded_arcs:
                raise InvalidS3LagrangianPolicy("selected_arc_duplicate")
            if (
                tail not in legal_route_nodes
                or head not in legal_route_nodes
                or tail == head
                or tail == depot_end
                or head == depot_start
                or (tail == depot_start and head == depot_end)
            ):
                raise InvalidS3LagrangianPolicy(
                    f"selected_arc_{tail}_{head}_outside_stage3_domain"
                )
            decoded_arcs[arc] = _s3_policy_bit(
                raw_value,
                label=f"x[{tail},{head}]",
                tolerance=binary_tolerance,
            )
        selected_arc_set = {
            arc for arc, value in decoded_arcs.items() if value
        }

        incoming = {}
        outgoing = {}
        for tail, head in selected_arc_set:
            if tail in outgoing:
                raise InvalidS3LagrangianPolicy(
                    f"selected_arcs_branch_out_{tail}"
                )
            if head in incoming:
                raise InvalidS3LagrangianPolicy(
                    f"selected_arcs_branch_in_{head}"
                )
            outgoing[tail] = head
            incoming[head] = tail

        if not selected_active:
            if selected_arc_set:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_without_active_alpha"
                )
            path = ()
        else:
            if depot_start in incoming or depot_end in outgoing:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_bad_depot_degree"
                )
            if depot_start not in outgoing or depot_end not in incoming:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_missing_depot_connection"
                )
            for customer in active_customer_set:
                expected = customer in selected_active
                if ((customer in incoming) != expected
                        or (customer in outgoing) != expected):
                    raise InvalidS3LagrangianPolicy(
                        f"selected_arcs_alpha_mismatch_{customer}"
                    )

            reconstructed = [depot_start]
            traversed_arcs = set()
            visited_nodes = {depot_start}
            current = depot_start
            while current != depot_end:
                if current not in outgoing:
                    raise InvalidS3LagrangianPolicy(
                        "selected_arcs_broken_route"
                    )
                next_node = outgoing[current]
                arc = (current, next_node)
                if arc in traversed_arcs or next_node in visited_nodes:
                    raise InvalidS3LagrangianPolicy(
                        "selected_arcs_cycle"
                    )
                traversed_arcs.add(arc)
                reconstructed.append(next_node)
                visited_nodes.add(next_node)
                current = next_node
            if traversed_arcs != selected_arc_set:
                raise InvalidS3LagrangianPolicy(
                    "selected_arcs_disconnected_component"
                )
            path = tuple(reconstructed)

    if path:
        if len(path) < 3:
            raise InvalidS3LagrangianPolicy("path_too_short")
        if path[0] != depot_start:
            raise InvalidS3LagrangianPolicy("path_wrong_start_depot")
        if path[-1] != depot_end:
            raise InvalidS3LagrangianPolicy("path_wrong_end_depot")
    interior = path[1:-1] if path else ()
    if depot_start in interior or depot_end in interior:
        raise InvalidS3LagrangianPolicy("path_depot_interior")
    if len(set(interior)) != len(interior):
        raise InvalidS3LagrangianPolicy("path_duplicate_customer")
    for customer in interior:
        if customer not in customer_set:
            raise InvalidS3LagrangianPolicy(
                f"path_unknown_customer_{customer}"
            )
        if not active[customer]:
            raise InvalidS3LagrangianPolicy(
                f"path_inactive_customer_{customer}"
            )

    if set(interior) != selected_active:
        raise InvalidS3LagrangianPolicy("path_active_alpha_mismatch")
    if bool(selected_active) != bool(path):
        raise InvalidS3LagrangianPolicy("path_presence_mismatch")
    # Stage-3 inherits both alpha[j] <= y and y <= sum(alpha): y is exactly
    # the indicator that at least one active or inactive customer is assigned.
    if y != int(bool(selected)):
        raise InvalidS3LagrangianPolicy("vehicle_activation_mismatch")

    try:
        capacity = _s3_fraction(
            probData.Qv[vehicle], label=f"capacity[{vehicle}]"
        )
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise InvalidS3LagrangianPolicy(
            f"capacity[{vehicle}]_missing"
        ) from exc
    if capacity < 0:
        raise InvalidS3LagrangianPolicy(f"capacity[{vehicle}]_negative")
    load = sum((volumes[customer] for customer in selected), Fraction(0))
    if load > capacity:
        raise InvalidS3LagrangianPolicy(
            f"vehicle_{vehicle}_capacity_violation"
        )

    route_cost = Fraction(0)
    if path:
        try:
            routing_costs = probData.c_routing[vehicle]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise InvalidS3LagrangianPolicy(
                f"routing_cost[{vehicle}]_missing"
            ) from exc
        for tail, head in zip(path, path[1:]):
            try:
                arc_cost = routing_costs[tail, head]
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidS3LagrangianPolicy(
                    f"routing_cost[{vehicle}][{tail},{head}]_missing"
                ) from exc
            route_cost += _s3_fraction(
                arc_cost, label=f"routing_cost[{vehicle}][{tail},{head}]"
            )

    dual_y = _s3_fraction(
        pi_value.get(f"y[{vehicle}]", 0.0), label=f"pi_y[{vehicle}]"
    )
    objective = route_cost - dual_y * y - sum(
        (dual_alpha[customer] * alpha[customer] for customer in customers),
        Fraction(0),
    )
    objective_up = _s3_float_up(
        objective, label="stage3_lagrangian_incumbent_objective"
    )
    route_cost_up = _s3_float_up(
        route_cost, label="stage3_route_cost"
    )
    xcp = {f"y[{vehicle}]": float(y)}
    xcp.update({
        f"alpha[{customer},{vehicle}]": float(alpha[customer])
        for customer in customers
    })
    return {
        "V": objective_up,
        "objective_exact": objective,
        "route_cost": route_cost_up,
        "route_cost_exact": route_cost,
        "load_exact": load,
        "xcp": xcp,
        "y": y,
        "alpha_dict": dict(alpha),
        "path": list(path),
        "selected_arcs": sorted(selected_arc_set),
    }


def _build_s3_esp_inputs(probData, node, pi_value):
    """Build the exact reachable-domain ``solve_pctsp`` inputs.

    Stage 2 enforces ``alpha[j,v] <= active[j]``.  Hence every inactive
    coordinate is identically zero on the entire predecessor domain and must
    not become a free knapsack item after Lagrangian relaxation.  We retain
    ``inactive_J`` only to reconstruct a complete zero-filled policy at the
    Python certification boundary; the native oracle receives no inactive
    items.
    """
    v = node.info
    depot_start = probData.numAllnodes - 2
    depot_end = probData.numAllnodes - 1

    active_J = [j for j in probData.J if node.active[j] == 1]
    inactive_J = [j for j in probData.J if node.active[j] == 0]
    n_active = len(active_J)

    use_N = active_J + [depot_start, depot_end]
    N = n_active + 2
    local_depot_start = N - 2
    local_depot_end = N - 1

    # Advanced indexing performs the same submatrix copy in C instead of an
    # O(N^2) Python loop.  Preserve the historical zero diagonal even if a
    # caller supplies a routing matrix with nonzero self arcs.
    use_idx = np.asarray(use_N, dtype=np.intp)
    routing_cost = np.asarray(probData.c_routing[v], dtype=np.float64)
    cost = np.ascontiguousarray(
        routing_cost[np.ix_(use_idx, use_idx)], dtype=np.float64
    )
    np.fill_diagonal(cost, 0.0)

    pi_alpha = np.zeros(N, dtype=np.float64)
    vol = np.zeros(N, dtype=np.float64)
    if n_active:
        pi_alpha[:n_active] = np.fromiter(
            (float(pi_value.get(f"alpha[{j},{v}]", 0.0)) for j in active_J),
            dtype=np.float64,
            count=n_active,
        )
        vol[:n_active] = np.asarray(node.volume, dtype=np.float64)[use_idx[:n_active]]

    vol_inactive = np.empty(0, dtype=np.float64)
    pi_alpha_inactive = np.empty(0, dtype=np.float64)
    pi_y = float(pi_value.get(f"y[{v}]", 0.0))
    capacity = float(probData.Qv[v])

    return {
        "vehicle": v,
        "active_J": active_J,
        "inactive_J": inactive_J,
        "n_active": n_active,
        "cost": cost,
        "pi_alpha": pi_alpha,
        "vol": vol,
        "depot_start": local_depot_start,
        "depot_end": local_depot_end,
        "capacity": capacity,
        "pi_y": pi_y,
        "vol_inactive": vol_inactive,
        "pi_alpha_inactive": pi_alpha_inactive,
    }


def solve_s3_with_esp(probData, node, pi_value, time_limit_s=None):
    """Backward stage-3 Lagrangian 子问题 (ESPPRC).

    ``V`` 只有在 Python 独立认证 native policy 后才是可行 incumbent
    （因而是最小化子问题的 UB）。Malformed incumbent 会被丢弃，但与其
    独立的 native ``lb`` 证书仍可保留。
    ``lb`` 只在 ``lb_certified=True`` 时才可用于 backward cut。ESP
    超时时不能仅凭“数值有限”就信任一个 bound；未认证的原始值
    保留在 ``lb_raw`` 中供诊断，对外 ``lb=-inf`` 强制 caller 回退。
    """
    if not _S3_USE_ESP:
        record_backend_event("esp", "skip", "disabled")
        return {"ok": False, "reason": "esp_disabled"}
    if espprc_cpp is None:
        record_backend_event("esp", "skip", "module_unavailable")
        return {"ok": False, "reason": "esp_module_unavailable"}

    inps = _build_s3_esp_inputs(probData, node, pi_value)
    if inps["n_active"] > 256:
        record_backend_event("esp", "skip", "active_limit", active=inps["n_active"])
        return {"ok": False, "reason": f"n_active_gt_256({inps['n_active']})"}

    # Exact zero-dual shortcut.  The all-zero policy gives objective zero.
    # With no multiplier reward and nonnegative route costs, every y=1 route
    # has objective >= 0 as well, so both endpoints are proved without native
    # preprocessing.  Use exact equality, not the project's numerical zero
    # band: this path is shared by Phase 2's exact checkpoints.
    active_multipliers_zero = (
        inps["pi_y"] == 0.0
        and all(
            float(pi_value.get(f"alpha[{j},{inps['vehicle']}]", 0.0)) == 0.0
            for j in inps["active_J"]
        )
    )
    nonnegative_route_costs = bool(np.all(inps["cost"] >= 0.0))
    # Preserve native tie-breaking when a zero-cost y=1 route can exist: its
    # nonzero assignment slope is a more informative Phase-2 bundle support.
    # Every nonempty route uses one start arc and one end arc, so positivity
    # of either complete boundary set proves y=0 is the unique optimum.
    zero_policy_is_unique = (
        inps["n_active"] == 0
        or np.all(
            inps["cost"][inps["depot_start"], :inps["n_active"]] > 0.0
        )
        or np.all(
            inps["cost"][:inps["n_active"], inps["depot_end"]] > 0.0
        )
    )
    if (active_multipliers_zero and nonnegative_route_costs
            and zero_policy_is_unique):
        record_backend_event("esp", "analytic_result", "zero_multipliers_exact",
                             lb_certified=True, exact=True)
        alpha_dict = {j: 0 for j in probData.J}
        xcp = {f"y[{inps['vehicle']}]": 0.0}
        xcp.update({
            f"alpha[{j},{inps['vehicle']}]": 0.0 for j in probData.J
        })
        zero_diagnostics = {
            "n_labels": 0,
            "n_pruned_dom": 0,
            "n_pruned_lb": 0,
            "preprocess_time_s": 0.0,
            "label_time_s": 0.0,
            "cpp_total_time_s": 0.0,
            "completion_lb_time_s": 0.0,
            "knapsack_setup_time_s": 0.0,
            "arc_elimination_time_s": 0.0,
            "greedy_time_s": 0.0,
            "frontier_lb_time_s": 0.0,
            "postprocess_time_s": 0.0,
            "n_labels_popped": 0,
            "n_arcs_considered": 0,
            "n_pruned_arc": 0,
            "n_pruned_elementary": 0,
            "n_pruned_capacity": 0,
            "n_dom_checks_forward": 0,
            "n_dom_checks_reverse": 0,
            "max_bucket_size": 0,
            "n_final_labels": 0,
            "n_pending_labels": 0,
            "dssr_iterations": 0,
            "n_customers_dropped": 0,
            "bound_level": 0,
            "n_escalations": 0,
            "root_lb": 0.0,
            "relaxation_lb": 0.0,
            "t_solve": 0.0,
        }
        return {
            "ok": True,
            "reason": None,
            "V": 0.0,
            "lb": 0.0,
            "lb_raw": 0.0,
            "lb_certified": True,
            "ub_certified": True,
            "incumbent_policy_certified": True,
            "incumbent_certification_reason": None,
            "obj_incumbent_raw": 0.0,
            "xcp": xcp,
            "route_cost": 0.0,
            "y": 0,
            "alpha_dict": alpha_dict,
            "path": [],
            "esp_status": 0,
            "status_reason": "zero_multipliers_exact",
            "interrupted": False,
            "label_budget_exhausted": False,
            "timed_out": False,
            "exact": True,
            **zero_diagnostics,
        }

    import time as _time
    _t0 = _time.time()
    eff_time_limit = 0.0 if time_limit_s is None else float(time_limit_s)
    try:
        res = _run_native_backend("esp", espprc_cpp.solve_pctsp,
            inps["cost"],
            inps["pi_alpha"],
            inps["vol"],
            int(inps["n_active"]),
            int(inps["depot_start"]),
            int(inps["depot_end"]),
            float(inps["capacity"]),
            float(inps["pi_y"]),
            inps["vol_inactive"],
            inps["pi_alpha_inactive"],
            cutoff=float("inf"),
            top_k=int(_S3_ESP_TOP_K),
            label_budget=int(_S3_ESP_LABEL_BUDGET),
            time_limit_s=eff_time_limit,
            ng_size=int(_S3_ESP_NG_SIZE),
        )
    except Exception as exc:
        return {"ok": False, "reason": f"esp_exception:{type(exc).__name__}"}
    _t_solve = _time.time() - _t0

    def _diag_int(name, default=0):
        try:
            if isinstance(res.get(name, default), (str, bytes)):
                return int(default)
            return int(res.get(name, default))
        except (TypeError, ValueError, OverflowError):
            return int(default)

    def _diag_float(name, default=0.0):
        try:
            if isinstance(res.get(name, default), (str, bytes)):
                return float(default)
            value = float(res.get(name, default))
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _diag_bool(name, default=False):
        try:
            return bool(_s3_policy_bit(
                res.get(name, default), label=name, tolerance=0.0
            ))
        except InvalidS3LagrangianPolicy:
            return bool(default)

    try:
        status = _strict_s3_index(res.get("status", 1), label="status")
        status_protocol_error = None
    except InvalidS3LagrangianPolicy as exc:
        status = 1
        status_protocol_error = str(exc)
    try:
        if isinstance(res.get("obj_val", float("inf")), (str, bytes)):
            raise ValueError("obj_val string payload")
        full_obj_raw = float(res.get("obj_val", float("inf")))
    except (TypeError, ValueError, OverflowError):
        full_obj_raw = float("inf")
    timed_out = _diag_bool("timed_out", False)
    interrupted = status == 2

    try:
        if isinstance(res.get("lb", float("-inf")), (str, bytes)):
            raise ValueError("lb string payload")
        lb_raw = float(res.get("lb", float("-inf")))
    except (TypeError, ValueError, OverflowError):
        lb_raw = float("-inf")

    try:
        native_lb_claim = bool(_s3_policy_bit(
            res.get("lb_certified", False),
            label="lb_certified",
            tolerance=0.0,
        ))
    except InvalidS3LagrangianPolicy:
        native_lb_claim = False

    # Decode the two native index spaces strictly, then pass one canonical
    # global policy through the shared exact scorer.  No malformed path entry
    # is skipped and no wrong alpha length is silently replaced with zeros.
    certified_policy = None
    feasible_endpoint_for_lb = None
    incumbent_reason = status_protocol_error
    if incumbent_reason is None and status not in (0, 2):
        incumbent_reason = f"esp_status_{status}"
    if incumbent_reason is None and not np.isfinite(full_obj_raw):
        incumbent_reason = "obj_val_not_finite_binary64"
    if incumbent_reason is None:
        try:
            if "y" not in res or "alpha_inactive" not in res or "path" not in res:
                raise InvalidS3LagrangianPolicy("native_policy_field_missing")
            y_native = _s3_policy_bit(
                res["y"],
                label=f"y[{inps['vehicle']}]",
                tolerance=0.0,
            )
            try:
                alpha_inactive = list(res["alpha_inactive"])
            except TypeError as exc:
                raise InvalidS3LagrangianPolicy(
                    "alpha_inactive_bad_shape"
                ) from exc
            if len(alpha_inactive) != len(inps["vol_inactive"]):
                raise InvalidS3LagrangianPolicy(
                    "alpha_inactive_bad_shape"
                )
            # Inactive alpha is fixed at zero in every reachable Stage-2
            # state; it is deliberately absent from the native free oracle.
            inactive_bits = {
                customer: 0 for customer in inps["inactive_J"]
            }

            try:
                path_local = list(res["path"])
            except TypeError as exc:
                raise InvalidS3LagrangianPolicy("path_bad_shape") from exc
            local2global = list(inps["active_J"]) + [
                probData.numAllnodes - 2,
                probData.numAllnodes - 1,
            ]
            path_global = []
            for position, raw_index in enumerate(path_local):
                local_index = _strict_s3_index(
                    raw_index, label=f"path[{position}]"
                )
                if local_index < 0 or local_index >= len(local2global):
                    raise InvalidS3LagrangianPolicy(
                        f"path[{position}]_out_of_range"
                    )
                path_global.append(local2global[local_index])

            path_active = set(path_global).intersection(inps["active_J"])
            alpha_dict = {
                customer: int(customer in path_active)
                for customer in inps["active_J"]
            }
            alpha_dict.update(inactive_bits)
            candidate = certify_s3_lagrangian_policy(
                probData,
                node,
                pi_value,
                {
                    "y": y_native,
                    "alpha": alpha_dict,
                    "path": path_global,
                },
                binary_tolerance=0.0,
            )
            # Even a feasible decoded point is not accepted as the native
            # incumbent/support unless its independently rebuilt directed-up
            # endpoint is exactly the endpoint carried by this ABI.
            feasible_endpoint_for_lb = candidate["V"]
            if candidate["V"] != full_obj_raw:
                raise InvalidS3LagrangianPolicy(
                    "native_objective_mismatch"
                )
            certified_policy = candidate
        except InvalidS3LagrangianPolicy as exc:
            incumbent_reason = str(exc)

    ub_certified = certified_policy is not None
    # LB channel independent of incumbent payload
    lb_certified, lb_val = _normalize_certified_min_lb(
        lb_raw,
        native_lb_claim,
        incumbent=(
            certified_policy["V"]
            if ub_certified else feasible_endpoint_for_lb
        ),
    )
    exact = status == 0 and ub_certified and lb_certified
    # Status-0 without native LB: incumbent ok, outer needs fallback
    evidence = ("exact" if exact else "certified_interval" if lb_certified and ub_certified
                else "lb_only" if lb_certified else "incumbent_only" if ub_certified
                else "no_usable_evidence")
    record_backend_event("esp", "validated_result", evidence,
                         lb_certified=lb_certified, ub_certified=ub_certified,
                         timed_out=timed_out)

    return {
        "ok": bool(ub_certified or lb_certified),
        "reason": (
            None if (ub_certified or lb_certified)
            else f"esp_no_usable_evidence:{incumbent_reason or 'unknown'}"
        ),
        "V": certified_policy["V"] if ub_certified else None,
        "lb": lb_val,
        "lb_raw": lb_raw,
        "lb_certified": lb_certified,
        "ub_certified": ub_certified,
        "incumbent_policy_certified": ub_certified,
        "incumbent_certification_reason": (
            None if ub_certified else incumbent_reason
        ),
        "obj_incumbent_raw": full_obj_raw,
        "xcp": certified_policy["xcp"] if ub_certified else {},
        "route_cost": certified_policy["route_cost"] if ub_certified else None,
        "y": certified_policy["y"] if ub_certified else None,
        "alpha_dict": (
            certified_policy["alpha_dict"] if ub_certified else {}
        ),
        "path": certified_policy["path"] if ub_certified else [],
        "esp_status": status,
        "status_reason": str(res.get("status_reason", "optimal" if exact else "interrupted")),
        "interrupted": interrupted,
        "label_budget_exhausted": _diag_bool(
            "label_budget_exhausted", False
        ),
        "timed_out": timed_out,
        "exact": exact,
        "n_labels": _diag_int("n_labels"),
        "n_pruned_dom": _diag_int("n_pruned_dom"),
        "n_pruned_lb": _diag_int("n_pruned_lb"),
        "preprocess_time_s": _diag_float("preprocess_time_s"),
        "label_time_s": _diag_float("label_time_s"),
        "cpp_total_time_s": _diag_float("total_time_s", _t_solve),
        "completion_lb_time_s": _diag_float("completion_lb_time_s"),
        "knapsack_setup_time_s": _diag_float("knapsack_setup_time_s"),
        "arc_elimination_time_s": _diag_float("arc_elimination_time_s"),
        "greedy_time_s": _diag_float("greedy_time_s"),
        "frontier_lb_time_s": _diag_float("frontier_lb_time_s"),
        "postprocess_time_s": _diag_float("postprocess_time_s"),
        "n_labels_popped": _diag_int("n_labels_popped"),
        "n_arcs_considered": _diag_int("n_arcs_considered"),
        "n_pruned_arc": _diag_int("n_pruned_arc"),
        "n_pruned_elementary": _diag_int("n_pruned_elementary"),
        "n_pruned_capacity": _diag_int("n_pruned_capacity"),
        "n_dom_checks_forward": _diag_int("n_dom_checks_forward"),
        "n_dom_checks_reverse": _diag_int("n_dom_checks_reverse"),
        "max_bucket_size": _diag_int("max_bucket_size"),
        "n_final_labels": _diag_int("n_final_labels"),
        "n_pending_labels": _diag_int("n_pending_labels"),
        "dssr_iterations": _diag_int("dssr_iterations"),
        "n_customers_dropped": _diag_int("n_customers_dropped"),
        "bound_level": _diag_int("bound_level"),
        "n_escalations": _diag_int("n_escalations"),
        "root_lb": _diag_float("root_lb", float("-inf")),
        "relaxation_lb": _diag_float("relaxation_lb", float("-inf")),
        "t_solve": _t_solve,
    }


# ============================================================
# Stage-2 Branch-and-Price backend (C++)
# ============================================================
def build_s2_bp_static_inputs(probData, node):
    """π-independent inputs (sizes/vehicles/customers); 一次 build 反复用."""
    J = list(probData.J)
    V = list(probData.V)
    n = len(J)
    m = len(V)
    active = np.asarray([int(node.active[j]) for j in J], dtype=np.int32)
    volume = np.asarray([float(node.volume[j]) for j in J], dtype=np.float64)
    cOut = np.asarray([float(node.c_out[j]) for j in J], dtype=np.float64)
    Qv = np.asarray([float(probData.Qv[v]) for v in V], dtype=np.float64)
    succ_list = list(node.successor)
    succ_to_h = {s: h for h, s in enumerate(succ_list)}
    return {
        "n": n, "m": m,
        "numSucc": len(succ_list),
        "J": J, "V": V,
        "successor": succ_list,
        "succ_to_h": succ_to_h,
        "active": active, "volume": volume, "cOut": cOut, "Qv": Qv,
    }


def build_s2_bp_vehicle_types(probData, vehicles=None):
    """Dense canonical type ids in the BPC vehicle order.

    ``V_k`` is also the rank order used by ``purchase_order``.  Reject an
    incomplete or overlapping partition; silently inventing singleton types
    would weaken the requested prefix relaxation while reporting it enabled.
    """
    vehicle_order = list(probData.V if vehicles is None else vehicles)
    positions = {vehicle: index for index, vehicle in enumerate(vehicle_order)}
    if len(positions) != len(vehicle_order):
        raise ValueError("BPC vehicle order contains duplicate IDs")
    type_ids = [-1] * len(vehicle_order)
    for type_id, vehicle_type in enumerate(probData.K):
        canonical_group = list(probData.V_k[vehicle_type])
        represented_group = [
            vehicle for vehicle in canonical_group if vehicle in positions
        ]
        payload_group = [
            vehicle for vehicle in vehicle_order if vehicle in set(canonical_group)
        ]
        if payload_group != represented_group:
            raise ValueError(
                f"BPC vehicle order disagrees with V_k[{vehicle_type!r}]"
            )
        for vehicle in canonical_group:
            position = positions.get(vehicle)
            if position is None:
                continue
            if type_ids[position] != -1:
                raise ValueError(
                    f"vehicle {vehicle!r} occurs in multiple V_k groups"
                )
            type_ids[position] = type_id
    missing = [
        vehicle_order[index]
        for index, type_id in enumerate(type_ids)
        if type_id < 0
    ]
    if missing:
        raise ValueError(f"vehicles missing from V_k: {missing!r}")
    return type_ids


def build_s2_bp_static_route_cuts(probData, node, succ_to_h, vehicles=None):
    """Build static RouteCut payloads for the Stage-2 C++ BPC backend.

    ``vehicles`` may be the whole fleet (backward oracle) or the purchased
    subset (forward solve).  Payload rows use that local vehicle order while
    each row still points at the matching original Stage-3 successor.
    """
    J = list(probData.J)
    V_all = list(probData.V)
    V_used = V_all if vehicles is None else list(vehicles)
    if len(node.successor) != len(V_all):
        raise ValueError(
            "Stage-2 node.successor must contain exactly one Stage-3 node "
            f"per vehicle: got {len(node.successor)} successors for "
            f"{len(V_all)} vehicles"
        )

    original_position = {v: pos for pos, v in enumerate(V_all)}
    active_customers = [j for j in J if node.active[j] == 1]
    depot_start = getattr(probData, "numAllnodes", len(probData.N)) - 2
    capacity_aware = hasattr(node, "volume") and hasattr(probData, "Qv")
    demand = (
        {
            j: Fraction.from_float(float(node.volume[j]))
            for j in active_customers
        }
        if capacity_aware
        else {}
    )
    n = len(J)
    m = len(V_used)
    out = []

    for v_local, v_original in enumerate(V_used):
        if v_original not in original_position:
            raise ValueError(f"unknown vehicle in BPC subset: {v_original!r}")
        succ_ind = node.successor[original_position[v_original]]
        h = succ_to_h.get(succ_ind)
        if h is None:
            raise ValueError(
                f"Stage-3 successor {succ_ind!r} is missing from succ_to_h"
            )

        if capacity_aware:
            capacity = Fraction.from_float(float(probData.Qv[v_original]))
            allowed_predecessors = {
                head: [depot_start]
                + [
                    tail
                    for tail in active_customers
                    if tail != head and demand[tail] + demand[head] <= capacity
                ]
                for head in active_customers
            }
        else:
            all_nodes = list(probData.N)
            allowed_predecessors = {j: all_nodes for j in J}
        incoming = compute_min_incoming_costs(
            customers=J,
            routing_costs=probData.c_routing[v_original],
            allowed_predecessors=allowed_predecessors,
            active=node.active,
        )
        pi_alpha = [[0.0] * n for _ in range(m)]
        for j_local, customer in enumerate(J):
            coefficient = float(incoming.get(customer, 0.0))
            if coefficient > STATIC_CUT_ZERO_TOL:
                pi_alpha[v_local][j_local] = coefficient

        if not any(pi_alpha[v_local]):
            continue
        out.append({
            "succ": int(h),
            "beta": 0.0,
            "piY": [0.0] * m,
            "piAlpha": pi_alpha,
        })
    return out


def build_s2_bp_learned_cuts(
    probData,
    cut_lag,
    succ_to_h,
    *,
    vehicles=None,
):
    """Translate learned S3 cuts to the C++ Stage-2 local index space.

    ``probData.J`` and ``probData.V`` are original model identifiers; neither
    is required to be ``range(n)``.  The C++ payload, however, is dense and
    zero-based.  Canonical variable-name maps make that conversion explicit
    and reject every unknown/malformed nonzero term instead of silently
    dropping it.  For a forward solve, terms belonging to known but unavailable
    vehicles are safely projected out because those variables are fixed at 0.
    """
    J = list(probData.J)
    V_all = list(probData.V)
    V_used = V_all if vehicles is None else list(vehicles)
    n = len(J)
    m = len(V_used)

    if len(set(V_all)) != len(V_all) or len(set(J)) != len(J):
        raise ValueError("BPC translation requires unique customer/vehicle IDs")
    if len(set(V_used)) != len(V_used):
        raise ValueError("BPC vehicle subset contains duplicate IDs")
    unknown_subset = [v for v in V_used if v not in set(V_all)]
    if unknown_subset:
        raise ValueError(f"unknown vehicles in BPC subset: {unknown_subset!r}")

    vehicle_to_local = {v: index for index, v in enumerate(V_used)}
    y_slots = {f"y[{v}]": vehicle_to_local.get(v) for v in V_all}
    alpha_slots = {
        f"alpha[{j},{v}]": (
            None if v not in vehicle_to_local
            else (vehicle_to_local[v], customer_local)
        )
        for v in V_all
        for customer_local, j in enumerate(J)
    }
    if len(y_slots) != len(V_all) or len(alpha_slots) != len(V_all) * len(J):
        raise ValueError(
            "customer/vehicle IDs produce ambiguous learned-cut variable names"
        )

    successor_positions = []
    for successor, raw_position in succ_to_h.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid local BPC successor for {successor!r}: "
                f"{raw_position!r}"
            ) from exc
        if position != raw_position:
            raise ValueError(
                f"invalid local BPC successor for {successor!r}: "
                f"{raw_position!r}"
            )
        successor_positions.append(position)
    if sorted(successor_positions) != list(range(len(successor_positions))):
        raise ValueError(
            "BPC successor mapping must be a zero-based contiguous bijection"
        )

    out = []
    for succ_ind, cuts_for_succ in (cut_lag.get(3) or {}).items():
        h = succ_to_h.get(succ_ind)
        if h is None:
            # Skip cuts for other Stage-2 nodes
            continue
        h_int = int(h)  # the complete mapping was validated above

        for pi_dict, v_val in cuts_for_succ:
            try:
                beta = float(v_val)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"non-numeric BPC cut intercept at successor "
                    f"{succ_ind!r}: {v_val!r}"
                ) from exc
            if not np.isfinite(beta):
                raise ValueError(
                    f"non-finite BPC cut intercept at successor "
                    f"{succ_ind!r}: {v_val!r}"
                )

            piY = [0.0] * m
            piAlpha = [[0.0] * n for _ in range(m)]
            try:
                terms = pi_dict.items()
            except AttributeError as exc:
                raise ValueError(
                    f"BPC cut slope at successor {succ_ind!r} is not a mapping"
                ) from exc
            for variable_name, raw_coefficient in terms:
                try:
                    coefficient = float(raw_coefficient)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"non-numeric BPC cut coefficient at successor "
                        f"{succ_ind!r} for {variable_name!r}: "
                        f"{raw_coefficient!r}"
                    ) from exc
                if not np.isfinite(coefficient):
                    raise ValueError(
                        f"non-finite BPC cut coefficient at successor "
                        f"{succ_ind!r} for {variable_name!r}: "
                        f"{raw_coefficient!r}"
                    )
                if coefficient == 0.0:
                    continue

                if not isinstance(variable_name, str):
                    raise ValueError(
                        f"non-string nonzero BPC cut variable at successor "
                        f"{succ_ind!r}: {variable_name!r}"
                    )
                if variable_name in y_slots:
                    vehicle_local = y_slots[variable_name]
                    if vehicle_local is not None:
                        piY[vehicle_local] = coefficient
                    continue
                if variable_name in alpha_slots:
                    slot = alpha_slots[variable_name]
                    if slot is not None:
                        vehicle_local, customer_local = slot
                        piAlpha[vehicle_local][customer_local] = coefficient
                    continue
                raise ValueError(
                    f"unknown or malformed nonzero BPC cut variable at "
                    f"successor {succ_ind!r}: {variable_name!r}"
                )

            out.append({
                "succ": h_int,
                "beta": beta,
                "piY": piY,
                "piAlpha": piAlpha,
            })
    return out


def build_s2_bp_cuts(probData, node, cut_lag, succ_to_h):
    """Translate learned S3 cuts and append the shared static RouteCuts."""
    out = build_s2_bp_learned_cuts(
        probData,
        cut_lag,
        succ_to_h,
        vehicles=None,
    )
    out.extend(
        build_s2_bp_static_route_cuts(
            probData,
            node,
            succ_to_h,
            vehicles=None,
        )
    )
    return out


def certify_s2_lagrangian_policy(
    probData,
    node,
    cut_lag,
    pi_value,
    raw_policy,
    *,
    binary_tolerance=0.0,
    cuts_payload=None,
    require_assignment_order=False,
    require_purchase_order=False,
):
    """Certify a Stage-2 policy and rebuild its Lagrangian objective upward.

    ``raw_policy`` has dense arrays ``z[m]``, ``y[m]`` and ``alpha[m][n]`` in
    ``probData.V``/``probData.J`` order.  With ``binary_tolerance=0`` (the BPC
    pybind contract), every entry must be exactly 0/1.  A positive tolerance
    may be used for Gurobi MIP noise: the snapped candidate is still accepted
    only after all physical constraints are checked exactly.  The objective
    includes outsourcing, the maximum of every current Stage-3 cut, and
    ``-piZ*z``; exact rationals are converted to one directed-up binary64
    endpoint only at the end.  ``require_assignment_order=True`` and
    ``require_purchase_order=True`` additionally certify the exact canonical
    rows used by the ordered Gurobi Level-Set model; BPC's initial physical
    domain check leaves both false.

    Activation order is not checked here; assignment and purchase order are
    checked exactly when requested.
    """
    customers = list(probData.J)
    vehicles = list(probData.V)
    successors = list(node.successor)
    n = len(customers)
    m = len(vehicles)

    if cuts_payload is None:
        cuts_payload = build_s2_bp_cuts(
            probData,
            node,
            cut_lag,
            {successor: position for position, successor in enumerate(successors)},
        )
    compiled_cuts = compiled_cut_payload(cuts_payload, m, n)

    raw_z = list(raw_policy.get("z", []))
    raw_y = list(raw_policy.get("y", []))
    raw_alpha = list(raw_policy.get("alpha", []))
    if len(raw_z) != m or len(raw_y) != m or len(raw_alpha) != m:
        raise _InvalidS2BpcIncumbent("bad_solution_shape")

    z = [
        _policy_bit(
            raw_z[v_pos],
            label=f"z[{vehicle}]",
            tolerance=binary_tolerance,
        )
        for v_pos, vehicle in enumerate(vehicles)
    ]
    y = [
        _policy_bit(
            raw_y[v_pos],
            label=f"y[{vehicle}]",
            tolerance=binary_tolerance,
        )
        for v_pos, vehicle in enumerate(vehicles)
    ]
    alpha = []
    for v_pos, vehicle in enumerate(vehicles):
        try:
            row = list(raw_alpha[v_pos])
        except TypeError as exc:
            raise _InvalidS2BpcIncumbent(
                f"alpha[{vehicle}]_bad_shape"
            ) from exc
        if len(row) != n:
            raise _InvalidS2BpcIncumbent(f"alpha[{vehicle}]_bad_shape")
        alpha.append([
            _policy_bit(
                row[j_pos],
                label=f"alpha[{customer},{vehicle}]",
                tolerance=binary_tolerance,
            )
            for j_pos, customer in enumerate(customers)
        ])

    outsourced = [0] * n
    for j_pos, customer in enumerate(customers):
        active = _policy_bit(
            node.active[customer],
            label=f"active[{customer}]",
            tolerance=0.0,
        )
        assigned = sum(alpha[v_pos][j_pos] for v_pos in range(m))
        if active:
            if assigned not in (0, 1):
                raise _InvalidS2BpcIncumbent(
                    f"customer_{customer}_assigned_{assigned}_times"
                )
        elif assigned != 0:
            raise _InvalidS2BpcIncumbent(
                f"inactive_customer_{customer}_assigned"
            )
        # Both Stage-2 builders impose sum(alpha)+s=1 for every customer;
        # inactive customers therefore have s=1 (with zero c_out by contract).
        outsourced[j_pos] = 1 - assigned

    for v_pos, vehicle in enumerate(vehicles):
        assigned_positions = [
            j_pos for j_pos in range(n) if alpha[v_pos][j_pos]
        ]
        expected_y = int(bool(assigned_positions))
        if y[v_pos] != expected_y:
            raise _InvalidS2BpcIncumbent(
                f"vehicle_{vehicle}_activation_mismatch"
            )
        if y[v_pos] > z[v_pos]:
            raise _InvalidS2BpcIncumbent(
                f"vehicle_{vehicle}_used_without_z"
            )

        load = sum(
            (
                _finite_binary64_fraction(
                    node.volume[customers[j_pos]],
                    label=f"volume[{customers[j_pos]}]",
                )
                for j_pos in assigned_positions
            ),
            Fraction(0),
        )
        capacity = _finite_binary64_fraction(
            probData.Qv[vehicle], label=f"capacity[{vehicle}]"
        )
        if load > capacity:
            raise _InvalidS2BpcIncumbent(
                f"vehicle_{vehicle}_capacity_violation"
            )

    if (require_assignment_order or require_purchase_order) and m > 1:
        try:
            vehicle_groups = [list(probData.V_k[k]) for k in probData.K]
        except (AttributeError, KeyError, TypeError) as exc:
            raise _InvalidS2BpcIncumbent(
                "assignment_order_metadata_missing"
                if require_assignment_order
                else "purchase_order_metadata_missing"
            ) from exc
        vehicle_position = {vehicle: pos for pos, vehicle in enumerate(vehicles)}
        if len(vehicle_position) != m:
            raise _InvalidS2BpcIncumbent("duplicate_vehicle_identifier")
        for group in vehicle_groups:
            try:
                positions = [vehicle_position[vehicle] for vehicle in group]
            except KeyError as exc:
                raise _InvalidS2BpcIncumbent(
                    "assignment_order_unknown_vehicle"
                    if require_assignment_order
                    else "purchase_order_unknown_vehicle"
                ) from exc
            if require_purchase_order:
                for left, right in zip(positions, positions[1:]):
                    if z[left] < z[right]:
                        raise _InvalidS2BpcIncumbent(
                            "purchase_order_violation_"
                            f"{vehicles[left]}_{vehicles[right]}"
                        )

        if require_assignment_order:
            weights = []
            for customer in customers:
                try:
                    weight = float(np.round(np.log(customer + 2), 4))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise _InvalidS2BpcIncumbent(
                        f"assignment_order_weight_{customer}_invalid"
                    ) from exc
                weights.append(_finite_binary64_fraction(
                    weight, label=f"assignment_order_weight[{customer}]"
                ))
            for group in vehicle_groups:
                positions = [vehicle_position[vehicle] for vehicle in group]
                for left, right in zip(positions, positions[1:]):
                    left_score = sum(
                        (
                            weights[j_pos] * alpha[left][j_pos]
                            for j_pos in range(n)
                        ),
                        Fraction(0),
                    )
                    right_score = sum(
                        (
                            weights[j_pos] * alpha[right][j_pos]
                            for j_pos in range(n)
                        ),
                        Fraction(0),
                    )
                    if left_score < right_score:
                        raise _InvalidS2BpcIncumbent(
                            "assignment_order_violation_"
                            f"{vehicles[left]}_{vehicles[right]}"
                        )

    theta = compiled_cuts.exact_theta(y, alpha, Fraction(0), len(successors))

    stage_cost = sum(
        (
            _finite_binary64_fraction(
                node.c_out[customer], label=f"c_out[{customer}]"
            )
            * outsourced[j_pos]
            for j_pos, customer in enumerate(customers)
        ),
        Fraction(0),
    )
    lagrangian_term = sum(
        (
            _finite_binary64_fraction(
                pi_value.get(f"z[{vehicle},{node.info[1]}]", 0.0),
                label=f"piZ[{vehicle}]",
            )
            * z[v_pos]
            for v_pos, vehicle in enumerate(vehicles)
        ),
        Fraction(0),
    )
    objective = stage_cost + sum(theta, Fraction(0)) - lagrangian_term
    objective_up = _fraction_to_finite_float_up(
        objective, label="stage2_bpc_incumbent_objective"
    )
    xcp = {
        f"z[{vehicle},{node.info[1]}]": float(z[v_pos])
        for v_pos, vehicle in enumerate(vehicles)
    }
    return {
        "V": objective_up,
        # Preserve the exact dyadic value already computed above.  Repeated
        # Level-Set queries for the same certified Stage-2 policy can then
        # change only the affine ``-piZ*z`` term without revalidating the
        # assignment or re-evaluating every Stage-3 cut.  Existing callers
        # continue to use the directed-up binary64 ``V`` endpoint.
        "objective_exact": objective,
        "xcp": xcp,
        "z": z,
        "y": y,
        "alpha": alpha,
        "s": outsourced,
        "theta": [
            _fraction_to_finite_float_up(
                value, label=f"theta[{successors[position]}]"
            )
            for position, value in enumerate(theta)
        ],
    }


def _solve_s2_bpc_core(probData, node, cut_lag, pi_value,
                       static_inps=None, cuts_cache=None,
                       time_limit_s=None, max_nodes=None,
                       max_depth=None, max_colgen_iters=None,
                       root_bound_only=None, max_cutting_rounds=None,
                       use_sr3_cuts=None, use_cover_cuts=None,
                       use_clique_cuts=None, use_purchase_order=False):
    """Stage-2 Lagrangian subproblem via src-ini/customized-subprob/s2backward/bpc.

    ``V`` 是可行 incumbent/UB；``lb`` 仅当 ``lb_certified=True`` 时
    才是 backward 可用的下界。未认证原始值仅放在 ``lb_raw``。
    """
    if stage2_bpc_cpp is None:
        return {"ok": False, "reason": "bpc_module_unavailable"}

    omega, t = node.info
    sin = static_inps if static_inps is not None else build_s2_bp_static_inputs(probData, node)
    n, m = sin["n"], sin["m"]
    V_list = sin["V"]

    piZ = np.asarray(
        [float(pi_value.get(f"z[{v},{t}]", 0.0)) for v in V_list],
        dtype=np.float64,
    )

    if cuts_cache is None:
        cuts_payload = build_s2_bp_cuts(
            probData, node, cut_lag, sin["succ_to_h"]
        )
    else:
        cuts_payload = cuts_cache

    eff_time_limit = float(time_limit_s) if time_limit_s is not None else float(_S2_BP_TIME_LIMIT_S)
    eff_max_nodes = int(max_nodes) if max_nodes is not None else int(_S2_BP_MAX_NODES)
    eff_max_depth = int(max_depth) if max_depth is not None else int(_S2_BP_MAX_DEPTH)
    eff_max_colgen = int(max_colgen_iters) if max_colgen_iters is not None else int(_S2_BP_MAX_CG)
    eff_root_bound_only = (
        bool(root_bound_only)
        if root_bound_only is not None
        else bool(_S2_BPC_ROOT_BOUND_ONLY)
    )
    eff_cut_rounds = (
        int(max_cutting_rounds)
        if max_cutting_rounds is not None
        else int(_S2_BPC_CUT_ROUNDS)
    )
    eff_use_sr3 = (
        bool(use_sr3_cuts)
        if use_sr3_cuts is not None
        else bool(_S2_BPC_USE_SR3_CUTS)
    )
    eff_use_cover = (
        bool(use_cover_cuts)
        if use_cover_cuts is not None
        else bool(_S2_BPC_USE_COVER_CUTS)
    )
    eff_use_clique = (
        bool(use_clique_cuts)
        if use_clique_cuts is not None
        else bool(_S2_BPC_USE_CLIQUE_CUTS)
    )
    eff_use_purchase_order = bool(use_purchase_order)
    if eff_use_purchase_order and not eff_root_bound_only:
        return {
            "ok": False,
            "reason": "bpc_purchase_order_requires_root_bound_only",
        }
    vehicle_types = None
    if eff_use_purchase_order:
        try:
            vehicle_types = build_s2_bp_vehicle_types(probData, V_list)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "reason": "bpc_purchase_order_metadata_invalid",
                "detail": str(exc),
            }

    import time as _time
    _t0 = _time.time()
    abi_supports_root_options = True
    try:
        base_kwargs = dict(
            n=n, m=m, numSucc=sin["numSucc"],
            active=sin["active"].tolist(),
            volume=sin["volume"].tolist(),
            cOut=sin["cOut"].tolist(),
            Qv=sin["Qv"].tolist(),
            piZ=piZ.tolist(),
            cuts=cuts_payload,
            pricing_top_k=int(_S2_BP_TOPK),
            max_nodes=eff_max_nodes,
            max_depth=eff_max_depth,
            max_colgen_iters=eff_max_colgen,
            rc_tol=_S2_BP_RC_TOL,
            int_tol=_S2_BP_INT_TOL,
            time_limit_s=eff_time_limit,
            verbose=False,
        )
        try:
            compatibility_kwargs = dict(
                use_heuristic_pricing=(
                    False
                    if eff_use_purchase_order and eff_root_bound_only
                    else _S2_BPC_USE_HEURISTIC
                ),
                use_vehicle_clustering=_S2_BPC_USE_CLUSTERING,
                use_diving=_S2_BPC_USE_DIVING,
                use_ryan_foster=_S2_BPC_USE_RYAN_FOSTER,
                use_dual_stabilization=_S2_BPC_USE_DUAL_STAB,
                cull_rc_threshold=_S2_BPC_CULL_RC,
                num_threads=_S2_BPC_NUM_THREADS,
                pricing_top_k_root=_S2_BPC_TOPK_ROOT,
                pricing_top_k_shallow=_S2_BPC_TOPK_SHALLOW,
                pricing_top_k_deep=_S2_BPC_TOPK_DEEP,
                use_restricted_mip=_S2_BPC_USE_RESTRICTED_MIP,
                restricted_mip_time_limit=_S2_BPC_RESTRICTED_MIP_TIME_LIMIT,
                use_cut_aging=_S2_BPC_USE_CUT_AGING,
                theta_lower_bound=_S2_BPC_THETA_LOWER_BOUND,
                solve_mode=2,  # LB / backward: tighten the valid root LP bound
            )
            res = _run_native_backend("bpc", stage2_bpc_cpp.solve_stage2_lag,
                **base_kwargs,
                **compatibility_kwargs,
                root_bound_only=eff_root_bound_only,
                max_cutting_rounds=eff_cut_rounds,
                use_sr3_cuts=eff_use_sr3,
                use_cover_cuts=eff_use_cover,
                use_clique_cuts=eff_use_clique,
                vehicle_types=vehicle_types,
                use_purchase_order=eff_use_purchase_order,
            )
        except TypeError as root_options_exc:
            abi_supports_root_options = False
            if eff_use_purchase_order:
                return {
                    "ok": False,
                    "reason": "bpc_extension_missing_purchase_order",
                    "detail": str(root_options_exc),
                    "abi_supports_root_options": False,
                    "abi_supports_purchase_order": False,
                    "root_bound_only": False,
                    "purchase_order_enabled": False,
                }
            requested_new_behavior = (
                eff_root_bound_only
                or eff_cut_rounds != -1
                or not eff_use_sr3
                or not eff_use_cover
                or not eff_use_clique
            )
            if requested_new_behavior:
                return {
                    "ok": False,
                    "reason": "bpc_extension_missing_root_options",
                    "detail": str(root_options_exc),
                    "abi_supports_root_options": False,
                    "root_bound_only": False,
                }
            try:
                # Older ABI: omit root_bound/cut-family kwargs
                res = _run_native_backend("bpc", stage2_bpc_cpp.solve_stage2_lag,
                    **base_kwargs, **compatibility_kwargs
                )
            except TypeError:
                res = _run_native_backend(
                    "bpc", stage2_bpc_cpp.solve_stage2_lag, **base_kwargs,
                )
    except Exception as exc:
        return {"ok": False, "reason": f"bpc_exception:{type(exc).__name__}:{exc}"}
    _t_solve = _time.time() - _t0

    purchase_order_confirmed = bool(res.get("purchase_order_enabled", False))
    if eff_use_purchase_order and not purchase_order_confirmed:
        return {
            "ok": False,
            "reason": "bpc_purchase_order_not_confirmed",
            "abi_supports_root_options": abi_supports_root_options,
            "abi_supports_purchase_order": False,
            "root_bound_only": bool(res.get("root_bound_only", False)),
            "purchase_order_enabled": False,
        }

    backend_metrics = {
        "total_time_s": float(res.get("total_time_s", _t_solve)),
        "rmp_build_time_s": float(res.get("rmp_build_time_s", 0.0)),
        "rmp_solve_time_s": float(res.get("rmp_solve_time_s", 0.0)),
        "pricing_time_s": float(res.get("pricing_time_s", 0.0)),
        "cut_separation_time_s": float(
            res.get("cut_separation_time_s", 0.0)
        ),
        "primal_heuristic_time_s": float(
            res.get("primal_heuristic_time_s", 0.0)
        ),
        "cg_iters_total": int(res.get("cg_iters_total", 0)),
        "lp_solves": int(res.get("lp_solves", 0)),
        "columns_generated": int(res.get("columns_generated", 0)),
        "exact_pricing_calls": int(res.get("exact_pricing_calls", 0)),
        "heuristic_pricing_successes": int(
            res.get("heuristic_pricing_successes", 0)
        ),
        "pricing_certified_nodes": int(
            res.get("pricing_certified_nodes", 0)
        ),
        "pricing_uncertified_nodes": int(
            res.get("pricing_uncertified_nodes", 0)
        ),
        "used_inherited_lb_nodes": int(
            res.get("used_inherited_lb_nodes", 0)
        ),
        "root_lp_before_cuts": float(
            res.get("root_lp_before_cuts", float("inf"))
        ),
        "root_lp_after_cuts": float(
            res.get("root_lp_after_cuts", float("inf"))
        ),
        "root_columns": int(res.get("root_columns", 0)),
        "root_cg_iters": int(res.get("root_cg_iters", 0)),
        "root_pricing_passes_completed": int(
            res.get("root_pricing_passes_completed", 0)
        ),
        "root_cut_rounds_requested": int(
            res.get(
                "root_cut_rounds_requested",
                eff_cut_rounds if abi_supports_root_options else -1,
            )
        ),
        "root_cut_rounds_completed": int(
            res.get("root_cut_rounds_completed", 0)
        ),
        "num_sr3_active": int(res.get("num_sr3_active", 0)),
        "num_cover_active": int(res.get("num_cover_active", 0)),
        "num_clique_active": int(res.get("num_clique_active", 0)),
        "root_bound_only": bool(
            res.get(
                "root_bound_only",
                eff_root_bound_only if abi_supports_root_options else False,
            )
        ),
        "tree_complete": bool(res.get("tree_complete", False)),
        "intentional_root_stop": bool(
            res.get("intentional_root_stop", False)
        ),
        "proof_relaxed": bool(res.get("proof_relaxed", False)),
        "tolerance_bound_prunes": int(
            res.get("tolerance_bound_prunes", 0)
        ),
        "tolerance_integral_closures": int(
            res.get("tolerance_integral_closures", 0)
        ),
        "sr3_cuts_enabled": bool(res.get(
            "sr3_cuts_enabled", eff_use_sr3 if abi_supports_root_options else True,
        )),
        "cover_cuts_enabled": bool(
            res.get(
                "cover_cuts_enabled",
                eff_use_cover if abi_supports_root_options else True,
            )
        ),
        "clique_cuts_enabled": bool(
            res.get(
                "clique_cuts_enabled",
                eff_use_clique if abi_supports_root_options else True,
            )
        ),
        "abi_supports_root_options": abi_supports_root_options,
        "abi_supports_purchase_order": (
            not eff_use_purchase_order or purchase_order_confirmed
        ),
        "purchase_order_enabled": purchase_order_confirmed,
        "termination_reason": str(res.get("termination_reason", "unknown")),
        "abort_reason": str(res.get("abort_reason", "")),
    }

    feasible = bool(res.get("feasible", False))
    timed_out = bool(res.get("timed_out", False))
    lb_raw_obj = res.get("lb", float("-inf"))
    try:
        lb_raw = float(lb_raw_obj)
    except Exception:
        lb_raw = float("-inf")

    # Allow LB-only root certificate
    if not feasible:
        lb_certified, lb_val = _normalize_certified_min_lb(
            lb_raw,
            res.get("lb_certified", False),
            incumbent=None,
        )
        return {
            "ok": lb_certified,
            "reason": None if lb_certified else "bpc_no_incumbent_or_certificate",
            "V": None,
            "lb": lb_val,
            "lb_raw": lb_raw,
            "lb_certified": lb_certified,
            "ub_certified": False,
            "optimality_proven": False,
            "exact": False,
            "incumbent_policy_certified": False,
            "phase2_bundle_compatible": False,
            "incumbent_certification_reason": "no_native_incumbent",
            "xcp": {},
            "y": [],
            "z": [],
            "alpha": [],
            "timed_out": timed_out,
            "nodes_processed": int(res.get("nodes_processed", 0)),
            "t_solve": _t_solve,
            "n": int(n),
            "active_n": int(np.sum(sin["active"])),
            "m": int(m),
            "num_succ": int(sin["numSucc"]),
            "num_cuts": int(len(cuts_payload)),
            **backend_metrics,
        }

    try:
        obj_raw = float(res.get("obj", float("inf")))
    except (TypeError, ValueError, OverflowError):
        obj_raw = float("inf")

    certified_policy = None
    ordered_policy = None
    incumbent_reason = None
    ordered_reason = None
    try:
        # Certify against full current Python cut archive
        certification_cuts = build_s2_bp_cuts(
            probData, node, cut_lag, sin["succ_to_h"]
        )
        certified_policy = certify_s2_lagrangian_policy(
            probData,
            node,
            cut_lag,
            pi_value,
            res,
            binary_tolerance=0.0,
            cuts_payload=certification_cuts,
        )
        try:
            ordered_policy = certify_s2_lagrangian_policy(
                probData,
                node,
                cut_lag,
                pi_value,
                res,
                binary_tolerance=0.0,
                cuts_payload=certification_cuts,
                require_assignment_order=True,
                require_purchase_order=True,
            )
        except _InvalidS2BpcIncumbent as exc:
            ordered_reason = str(exc)
    except _InvalidS2BpcIncumbent as exc:
        incumbent_reason = str(exc)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        # Malformed cut archive: cannot certify support
        incumbent_reason = f"certification_payload:{type(exc).__name__}:{exc}"

    incumbent_value = (
        float(certified_policy["V"])
        if certified_policy is not None
        else None
    )
    ub_certified = certified_policy is not None
    lb_certified, lb_val = _normalize_certified_min_lb(
        lb_raw,
        res.get("lb_certified", False),
        incumbent=incumbent_value,
    )
    proof_relaxed = bool(res.get("proof_relaxed", False))
    # Gap/tolerance closure is never exact, even if LB≈UB numerically
    optimality_proven = bool(
        res.get("optimality_proven", False)
        and not proof_relaxed
        and ordered_policy is not None
        and lb_certified
        and lb_val == float(ordered_policy["V"])
    )
    nodes_proc = int(res.get("nodes_processed", 0))
    if certified_policy is None:
        z_list = []
        y_list = []
        alpha_list = []
        s_list = []
        theta_list = []
        xcp = {}
    else:
        z_list = list(certified_policy["z"])
        y_list = list(certified_policy["y"])
        alpha_list = list(certified_policy["alpha"])
        s_list = list(certified_policy["s"])
        theta_list = list(certified_policy["theta"])
        xcp = dict(certified_policy["xcp"])

    return {
        "ok": bool(ub_certified or lb_certified),
        "reason": (
            None
            if ub_certified
            else (
                f"bpc_invalid_incumbent_lb_only:{incumbent_reason}"
                if lb_certified
                else f"bpc_invalid_incumbent:{incumbent_reason}"
            )
        ),
        "V": incumbent_value,
        "obj_incumbent_raw": obj_raw,
        "lb": lb_val,
        "lb_raw": lb_raw,
        "lb_certified": lb_certified,
        "ub_certified": ub_certified,
        "optimality_proven": optimality_proven,
        "native_optimality_proven": bool(
            res.get("optimality_proven", False)
        ),
        "incumbent_policy_certified": ub_certified,
        "ordered_policy_certified": ordered_policy is not None,
        # The native model is a relaxation. Its incumbent is a valid ordered
        # bundle point only when independently checked against assignment_order.
        "phase2_bundle_compatible": ordered_policy is not None,
        "incumbent_certification_reason": incumbent_reason,
        "ordered_policy_certification_reason": ordered_reason,
        "timed_out": timed_out,
        "nodes_processed": nodes_proc,
        "exact": (
            optimality_proven
        ),
        "n": int(n),
        "active_n": int(np.sum(sin["active"])),
        "m": int(m),
        "num_succ": int(sin["numSucc"]),
        "num_cuts": int(len(cuts_payload)),
        "xcp": xcp,
        "y": y_list,
        "z": z_list,
        "alpha": alpha_list,
        "s": s_list,
        "theta": theta_list,
        "backend_ub_certified": bool(res.get("ub_certified", False)),
        "t_solve": _t_solve,
        **backend_metrics,
    }


def solve_s2_with_bp(probData, node, cut_lag, pi_value,
                     static_inps=None, cuts_cache=None,
                     time_limit_s=None, max_nodes=None,
                     max_depth=None, max_colgen_iters=None, **bpc_options):
    """Stage-2 Lagrangian subproblem (alias)."""
    if not _S2_USE_BP:
        return {"ok": False, "reason": "bp_disabled"}
    if probData is not None and node is not None:
        active_count = sum(
            1 for customer in probData.J
            if int(node.active[customer]) == 1
        )
        if active_count < _PHASE1_S2_BPC_MIN_ACTIVE:
            return {
                "ok": False,
                "reason": "phase1_s2_bpc_below_active_threshold",
                "active_n": active_count,
                "min_active": _PHASE1_S2_BPC_MIN_ACTIVE,
            }
    # Phase 1 uses BPC only as a cheap, exactly-priced root relaxation.
    if "root_bound_only" not in bpc_options:
        bpc_options["root_bound_only"] = _PHASE1_S2_BPC_ROOT_BOUND_ONLY
    if (bpc_options["root_bound_only"]
            and "max_cutting_rounds" not in bpc_options):
        bpc_options["max_cutting_rounds"] = _PHASE1_S2_BPC_ROOT_CUT_ROUNDS
    if "use_purchase_order" not in bpc_options:
        bpc_options["use_purchase_order"] = bool(
            bpc_options["root_bound_only"]
        )
    return _solve_s2_bpc_core(
        probData, node, cut_lag, pi_value,
        static_inps=static_inps, cuts_cache=cuts_cache,
        time_limit_s=time_limit_s, max_nodes=max_nodes,
        max_depth=max_depth, max_colgen_iters=max_colgen_iters,
        **bpc_options,
    )
