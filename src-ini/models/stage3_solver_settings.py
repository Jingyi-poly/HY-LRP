"""Original Stage-3 Gurobi search controls on the heterogeneous LRP domain.

Only solver parameters and branching priorities live here. These settings do
not add fleet symmetry, change physical costs, or weaken the policy audit.
"""
import os


def _setting(suffix, default, kind=int):
    return kind(os.environ.get('LRP_S3_' + suffix,
                               os.environ.get('VRP_S3_' + suffix, default)))


def _apply_stage3_solve_params(model, node_ind=None, phase=None):
    """Keep the original phase split and env names; LRP aliases take priority."""
    if _setting('AGGRESSIVE_CUTS', 1):
        for parameter, suffix, default in (
            ('Cuts', 'CUTS', 2), ('CoverCuts', 'COVER_CUTS', 2),
            ('GomoryPasses', 'GOMORY', 15), ('MIRCuts', 'MIR_CUTS', 2),
            ('FlowCoverCuts', 'FLOW_COVER', 2), ('NumericFocus', 'NUMERIC_FOCUS', 2),
        ):
            model.setParam(parameter, _setting(suffix, default))
        if phase == 2:
            model.setParam('IntFeasTol', _setting('INT_FEAS_TOL', 1e-7, float))
            model.setParam('FeasibilityTol', _setting('FEAS_TOL', 1e-7, float))
            model.setParam('MIPFocus', _setting('MIP_FOCUS', 2))
            model.setParam('Presolve', _setting('PRESOLVE', 2))
            model.setParam('PreCrush', 1)
    # Original priorities were independent of the aggressive-cut switch.
    if _setting('BRANCH_PRIORITY', 1):
        model.getVarByName('u_copy').BranchPriority = _setting('BRANCH_PRIO_Y', 200)
        for variable in model._lrp_variables.get('a_copy', {}).values():
            variable.BranchPriority = _setting('BRANCH_PRIO_ALPHA', 100)
    model.update()
