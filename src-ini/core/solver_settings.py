"""Shared ordinary solver settings; independent EF options remain explicit."""
from __future__ import annotations
import math
import os


BACKWARD_S3_SELECTOR_CONTRACT = 'lrp_physical_backward_s3_v1'
FORWARD_S2_SELECTOR_CONTRACT = 'lrp_physical_forward_s2_dispatch_v2'


def _backward_setting(suffix, default):
    for prefix in ('LRP_', 'VRP_'):
        raw = os.environ.get(prefix + suffix)
        if raw is not None and str(raw).strip():
            return raw
    return default


def configured_backward_s2_backend(phase):
    """Preserve explicit oracle requests without importing fleet symmetry."""
    if phase not in (1, 2):
        raise ValueError('Backward S2 phase must be 1 or 2')
    value = str(_backward_setting(f'PHASE{phase}_S2_ORACLE',
                                 'auto' if phase == 1 else 'gurobi')).strip().lower()
    if value == 'native':
        value = 'bpc'
    if value == 'fleet_enum':
        raise ValueError('fleet_enum groups interchangeable investment vehicles; LRP facilities '
                         'are distinct physical locations. Use bpc, auto, or gurobi.')
    if value not in ('auto', 'bpc', 'gurobi'):
        raise ValueError(f'PHASE{phase}_S2_ORACLE must be auto, bpc/native, or gurobi')
    return value


def phase1_s2_oracle_policy(prob_data, node):
    """The Investment Phase-1 cheap probe/root policy, on physical LRP nodes.

    DirectDP's fleet-count implementation is deliberately not called: a
    location-specific subset adapter is required before that optional oracle
    can be enabled.  Skipping it retains the certified fixed-RHS LP cut.
    """
    def nonnegative(suffix, default, kind=float):
        value = kind(_backward_setting(suffix, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{suffix} must be finite and nonnegative')
        return value
    def enabled_flag(suffix):
        raw = str(_backward_setting(suffix, '1')).strip().lower()
        if raw not in ('0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'):
            raise ValueError(f'{suffix} must be a boolean (0/1)')
        return raw in ('1', 'true', 'yes', 'on')
    # Both were read by the original Phase-1 wrapper/native call. Neither
    # selector may turn an explicit zero into a full Gurobi strengthening.
    enabled = enabled_flag('PHASE1_S2_USE_BP') and enabled_flag('S2_USE_BP')
    context = getattr(node, 'context', None)
    if context is None:
        active = sum(float(node.active[j]) > .5 for j in prob_data.J)
        period = int(node.info[1])
    else:
        active = sum(float(value) > .5 for value in context.active)
        period = int(context.period)
    minimum = nonnegative('PHASE1_S2_BPC_MIN_ACTIVE', 21, int)
    maximum = nonnegative('PHASE1_S2_AUTO_BPC_MAX_ACTIVE', 49, int)
    reason = ('phase1_s2_bpc_disabled' if not enabled else
              'phase1_s2_bpc_below_active_threshold' if active < minimum else
              'active_customer_size_gate' if maximum == 0 or active > maximum else
              'noninitial_period_gate' if period != 0 else None)
    return dict(backend=configured_backward_s2_backend(1),
                gurobi_probe_seconds=nonnegative('PHASE1_S2_AUTO_GRB_PROBE_S', .1),
                bpc_seconds=nonnegative('PHASE1_S2_AUTO_BPC_TIME_LIMIT_S', 5.),
                bpc_enabled=enabled, bpc_skip_reason=reason, active_customers=active,
                direct_dp_skip_reason='physical_location_subset_adapter_not_available')


def _forward_s2_native_available():
    """The physical BPC equivalent of Final's forward kernel availability gate."""
    for prefix in ('LRP_', 'VRP_'):
        raw = os.environ.get(prefix + 'S2_BP_KERNEL', '').strip().lower()
        if raw:
            if raw in ('0', 'off', 'false', 'no'):
                return False
            break
    # Import lazily: the adapter itself imports this settings module. Availability
    # only loads the native extension; it does not instantiate an optimizer.
    from solvers.forward_stage2_bpc import _load_native, NativeBPCUnavailable
    try:
        _load_native()
    except NativeBPCUnavailable:
        return False
    return True


def forward_s2_requested_mode():
    """Resolve explicit forward intent without probing native availability.

    Phase-specific callers can retain Final's auto-only feasible-probe path;
    general backward/S3 switches do not become forward intent.
    """
    source = 'default'
    for prefix in ('LRP_', 'VRP_'):
        name = prefix + 'S2_FORWARD_SOLVER'
        raw = os.environ.get(name, '').strip().lower()
        if not raw:
            continue
        if raw in ('bpc', 'bp', 'native'):
            return dict(mode='bpc', source=name)
        if raw == 'gurobi':
            return dict(mode=raw, source=name)
        if raw != 'auto':
            raise ValueError(f'{name} must be auto, bpc, or gurobi')
        source = name
        break
    # Dedicated alias already supported by the LRP migration. This is an
    # explicit request, unlike Final's general USE_ESP_BP master switch.
    for prefix in ('LRP_', 'VRP_'):
        name = prefix + 'FORWARD_S2_USE_BP'
        raw = os.environ.get(name, '').strip().lower()
        if not raw:
            continue
        if raw in ('0', 'false', 'no', 'off'):
            return dict(mode='gurobi', source=name)
        if raw in ('1', 'true', 'yes', 'on'):
            return dict(mode='bpc', source=name)
        raise ValueError(f'{name} must be a boolean (0/1)')
    return dict(mode='auto', source=source)


def configured_forward_s2_backend(num_customers=None):
    """Compact auto for physical LRP; Final's forward native path is opt-in."""
    requested = forward_s2_requested_mode()
    if requested['mode'] != 'auto':
        return requested['mode']
    # Match Final's permissive opt-in parsing: only these four true tokens
    # enable the optional attempt; absent/other values keep compact auto.
    for prefix in ('LRP_', 'VRP_'):
        name = prefix + 'S2_BP_AUTO'
        if name not in os.environ:
            continue
        enabled = os.environ[name].strip().lower() in ('1', 'on', 'true', 'yes')
        return 'bpc' if enabled and _forward_s2_native_available() else 'gurobi'
    return 'gurobi'


def forward_s2_bpc_policy(phase):
    """Original BPC stopping controls, distinct from compact Gurobi MIPGap."""
    if phase not in (1, 2):
        raise ValueError('Forward S2 phase must be 1 or 2')
    def setting(suffix, default):
        return os.environ.get('LRP_' + suffix, os.environ.get('VRP_' + suffix, default))
    gap = float(setting(f'FORWARD_S2_BP_GAP_PHASE{phase}', .05 if phase == 1 else .01))
    limit = float(setting('FORWARD_S2_BP_TIME_LIMIT_S', 300.))
    threads = int(setting('S2_BPC_NUM_THREADS', 1))
    if not math.isfinite(gap) or gap < 0:
        raise ValueError('Forward S2 BPC gap must be finite and nonnegative')
    if not math.isfinite(limit) or limit <= 0 or threads < 1:
        raise ValueError('Forward S2 BPC time limit and threads must be positive')
    options = dict(forward_gap=gap, time_limit_s=limit, threads=threads,
        search_contract='lrp_forward_anytime_v1',
        reference_lp=int(setting('FORWARD_S2_BP_REFERENCE_LP', 1)),
        reference_lp_seconds=float(setting('FORWARD_S2_BP_REFERENCE_LP_SECONDS', 2.)),
        accept_timeout=int(setting('FORWARD_S2_BP_ACCEPT_TIMEOUT', 1)),
        pricing_top_k=int(setting('S2_BP_TOPK', 5)),
        max_nodes=int(setting('FORWARD_S2_BP_MAX_NODES', 1000000)),
        max_depth=int(setting('FORWARD_S2_BP_MAX_DEPTH', 100000)),
        max_colgen_iters=int(setting('FORWARD_S2_BP_MAX_CG', 2000000)),
        rc_tol=float(setting('S2_BP_RC_TOL', 1e-7)),
        int_tol=float(setting('S2_BP_INT_TOL', 1e-6)),
        binary_directory=setting('S2_BPC_DIR', None))
    if options['accept_timeout'] not in (0, 1):
        raise ValueError('FORWARD_S2_BP_ACCEPT_TIMEOUT must be 0 or 1')
    if options['reference_lp'] not in (0, 1):
        raise ValueError('FORWARD_S2_BP_REFERENCE_LP must be 0 or 1')
    if not math.isfinite(options['reference_lp_seconds']) or options['reference_lp_seconds'] <= 0:
        raise ValueError('FORWARD_S2_BP_REFERENCE_LP_SECONDS must be finite and positive')
    if any(options[key] < 1 for key in ('pricing_top_k', 'max_nodes', 'max_depth', 'max_colgen_iters')):
        raise ValueError('Forward S2 BPC search limits must be positive')
    if any(not math.isfinite(options[key]) or options[key] <= 0 for key in ('rc_tol', 'int_tol')):
        raise ValueError('Forward S2 BPC tolerances must be finite and positive')
    return options


def configured_backward_s3_backend(phase):
    """Resolve only the physical LRP backward oracle, never a fleet backend.

    An explicit global LRP selector wins, including Compare's all-Gurobi
    request. Otherwise preserve the old phase-specific ESP switches (with
    LRP aliases first). They do not select the forward fixed-route solver.
    """
    if phase not in (1, 2):
        raise ValueError('Backward S3 phase must be 1 or 2')
    if 'LRP_S3_BACKEND' in os.environ:
        backend = os.environ['LRP_S3_BACKEND'].strip().lower()
        if backend not in ('native', 'gurobi'):
            raise ValueError('LRP_S3_BACKEND must be native or gurobi')
        return backend
    for prefix in ('LRP_', 'VRP_'):
        name = f'{prefix}PHASE{phase}_S3_USE_ESP'
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        value = raw.strip().lower()
        if value in ('0', 'false', 'no', 'off'):
            return 'gurobi'
        if value in ('1', 'true', 'yes', 'on'):
            return 'native'
        raise ValueError(f'{name} must be a boolean (0/1)')
    return 'native'


def configured_gurobi_threads():
    """Honor the original VRP name and the explicit LRP alias at every solve."""
    raw = os.environ.get('LRP_GRB_THREADS', os.environ.get('VRP_GRB_THREADS', '1'))
    threads = int(raw)
    if threads < 1:
        raise ValueError('Gurobi thread count must be a positive integer')
    return threads


def phase2_oracle_mip_gap():
    """The original Level Set MIP tolerance, separate from outer phase gap."""
    raw = os.environ.get('LRP_PHASE2_SUB_MIPGAP',
                         os.environ.get('VRP_PHASE2_SUB_MIPGAP', '1e-7'))
    value = float(raw)
    if not math.isfinite(value) or value < 0.:
        raise ValueError('Phase2 oracle MIPGap must be finite and nonnegative')
    return value


def configured_native_probe_seconds():
    """Native allowance within a free-route oracle's shared solve budget."""
    try:
        seconds = float(os.environ.get('LRP_NATIVE_PROBE_SECONDS', '2.0'))
    except (TypeError, ValueError) as exc:
        raise ValueError('LRP_NATIVE_PROBE_SECONDS must be finite and positive') from exc
    if not math.isfinite(seconds) or seconds <= 0.:
        raise ValueError('LRP_NATIVE_PROBE_SECONDS must be finite and positive')
    return seconds


def configured_s3_native_options():
    """Final's free-PCTSP controls, resolved per worker job; explicit APIs stay explicit."""
    result = {}
    for suffix, argument, default, maximum in (
            ('S3_ESP_NG_SIZE', 'ng_size', 8, 256),
            ('S3_ESP_LABEL_BUDGET', 'label_budget', 0, (1 << 64) - 1)):
        name, raw = 'VRP_' + suffix, str(default)
        for prefix in ('LRP_', 'VRP_'):
            if prefix + suffix in os.environ:
                name, raw = prefix + suffix, os.environ[prefix + suffix]
                break
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be an integer in [0, {maximum}]') from exc
        if not 0 <= value <= maximum:
            raise ValueError(f'{name} must be an integer in [0, {maximum}]')
        result[argument] = value
    return result


def configured_s3_native_cap(*, checkpoint=False):
    """Original S3 ESP probe/finalize settings, with LRP aliases preferred.

    Preserve Final's max(0, float(setting)) parsing: zero adds no native cap.
    An explicitly supplied current global override remains the fallback and
    retains its positive-finite contract. Resolve per call for worker jobs.
    """
    suffix = 'FINALIZE_TIME_S' if checkpoint else 'PROBE_TIME_S'
    for prefix in ('LRP_', 'VRP_'):
        key = prefix + 'PHASE2_S3_ESP_' + suffix
        if key in os.environ:
            return max(0., float(os.environ[key])), key
    if 'LRP_NATIVE_PROBE_SECONDS' in os.environ:
        return configured_native_probe_seconds(), 'LRP_NATIVE_PROBE_SECONDS'
    return (10. if checkpoint else 1.), 'Final_default'


def configured_s3_native_time_limit(sub_time_limit, *, checkpoint=False,
                                    remaining_seconds=None):
    """Intersect the S3 native cap, query allowance and shared deadline.

    None is explicitly uncapped; numeric zero is a caller skip sentinel and
    must never reach native's zero=unlimited kernel boundary.
    """
    cap, _ = configured_s3_native_cap(checkpoint=checkpoint)
    full = None if sub_time_limit is None else float(sub_time_limit)
    full = full if full is not None and math.isfinite(full) and full > 0. else None
    limit = full if cap <= 0. else min(full, cap) if full is not None else cap
    if remaining_seconds is not None:
        remaining = float(remaining_seconds)
        if math.isnan(remaining):
            raise ValueError('Shared remaining time cannot be NaN')
        if remaining <= 0.:
            return 0.
        limit = remaining if limit is None else min(limit, remaining)
    return None if limit is None or math.isinf(limit) else limit
