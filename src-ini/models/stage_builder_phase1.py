"""Phase-1 constructor at its original path, using the same LRP physical domain."""
from .stage_builder import (
    MIP_PURPOSE, StageModelBuilder as _StageModelBuilder,
    _add_explicit_s3_to_s2_cuts, _setup_lazy_cuts, _threshold,
    add_explicit_s3_cut, add_s3_to_s2_cut_pools, add_s3_to_s2_cuts,
    apply_stage2_search_params, build_stage1_forward, build_stage2_forward,
    build_stage3_forward, lazy_cut_callback,
)


DEFAULT_LAZY_THRESHOLD = 8


def _resolve_lazy_threshold(override=None):
    if override is not None:
        return _threshold(override)
    import os
    for name in ('LRP_S2_LAZY_THRESHOLD', 'VRP_S2_LAZY_THRESHOLD'):
        raw = os.environ.get(name, '').strip()
        if raw:
            return _threshold(int(raw))
    return DEFAULT_LAZY_THRESHOLD


class StageModelBuilder(_StageModelBuilder):
    """SBC warm-up changes the cut search, never the LRP feasible set.

    dual_lp retains every cut as an ordinary row for Model.relax().
    No vehicle activation/order symmetry is added in Phase 1.
    """

    def __init__(self, prob_data, mip_gap=1e-4, lazy_threshold=None, *, connectivity='mtz', env=None):
        super().__init__(prob_data, mip_gap=mip_gap,
                         lazy_threshold=_resolve_lazy_threshold(lazy_threshold),
                         connectivity=connectivity, env=env)
