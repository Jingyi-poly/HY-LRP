"""Phase-1 forward pass on the same LRP physical model as Phase 2."""
from models.stage_builder_phase1 import StageModelBuilder
from solvers.forward_solver import ForwardSolver as _ForwardSolver


class ForwardSolver(_ForwardSolver):
    phase = 1

    def __init__(self, prob_data, stage_builder=None, total_stage=3,
                 sub_time_limit=60.0, period_dedup=True, *, stage2_sub_time_limit=None,
                 stage2_mipgap_schedule=None):
        super().__init__(prob_data, stage_builder or StageModelBuilder(prob_data),
                         total_stage, sub_time_limit, period_dedup,
                         stage2_sub_time_limit=stage2_sub_time_limit)
        self.stage2_mipgap_schedule = stage2_mipgap_schedule
        self.set_iteration(1)

    def set_iteration(self, iteration):
        if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 1:
            raise ValueError('Phase-1 iteration must be a positive integer')
        if self.stage2_mipgap_schedule is None:
            self.stage2_mip_gap = self.stage_builder.mip_gap
        else:
            early, late, switch = self.stage2_mipgap_schedule
            self.stage2_mip_gap = early if iteration < switch else late

    def forward_pass(self, scen_tree, cut_lag, num_processes=1, pool=None,
                     deadline=None, *, iteration=1):
        self.set_iteration(iteration)
        return super().forward_pass(scen_tree, cut_lag, num_processes, pool, deadline)
