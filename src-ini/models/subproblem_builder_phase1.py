"""Phase-1 SBC backward construction, sharing the verified free LRP domains."""
from .subproblem_builder import (
    SubproblemBuilder as _SubproblemBuilder,
    build_stage2_backward, build_stage3_backward,
)


class SubproblemBuilder(_SubproblemBuilder):
    """Keep Phase-1 call sites while rejecting the old restricted y-lift mode."""

    phase = 1

    def build_subproblem(self, stage_no, node, cut_lag, pi_value,
                         force_y_active=False, **kwargs):
        if not isinstance(force_y_active, bool):
            raise TypeError("force_y_active must be boolean")
        if force_y_active:
            raise ValueError(
                "force_y_active restricts the LRP parent domain; no unrestricted "
                "cut may use that bound. A separately certified u-lift interface is required."
            )
        return super().build_subproblem(stage_no, node, cut_lag, pi_value, **kwargs)

    def _build_stage_3_subproblem(self, node, pi_value, force_y_active=False):
        return self.build_subproblem(3, node, {}, pi_value, force_y_active=force_y_active)
