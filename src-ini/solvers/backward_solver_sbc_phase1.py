"""Phase-1 SBC warm-up on the same LRP domains as the production algorithm."""
from models.stage_builder_phase1 import StageModelBuilder
from models.subproblem_builder_phase1 import SubproblemBuilder
from solvers.backward_solver_sbc import BackwardSolverSBC as _BackwardSolverSBC


class BackwardSolverSBC(_BackwardSolverSBC):
    phase = 1

    def __init__(self, prob_data, stage_builder=None, subproblem_builder=None,
                 strengthen_s2=True, strengthen=False, sub_time_limit=60.0,
                 route_lp_separation_time_limit=None):
        stage_builder = stage_builder or StageModelBuilder(prob_data)
        super().__init__(prob_data, stage_builder,
                         subproblem_builder or SubproblemBuilder(
                             prob_data, lazy_threshold=stage_builder.lazy_threshold, env=stage_builder.env),
                         strengthen_s2, strengthen, sub_time_limit,
                         route_lp_separation_time_limit=route_lp_separation_time_limit)
