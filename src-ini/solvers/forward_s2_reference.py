"""A certified continuous S2 bound used only to stop forward primal search."""
from __future__ import annotations

import time
from types import SimpleNamespace

from core.backend_telemetry import backend_call
from core.solve_deadline import SolveDeadlineReached, bounded_solve_time
from core.solver_bounds import extract_verified_fixed_rhs_dual_cut
from core.solver_settings import configured_gurobi_threads
from models.stage_builder import StageModelBuilder


class _CanonicalRow:
    def __init__(self, terms, variables):
        self._terms = tuple(terms.items())
        self._variables = variables

    def size(self):
        return len(self._terms)

    def getVar(self, position):
        return self._variables[self._terms[position][0]]

    def getCoeff(self, position):
        return self._terms[position][1]


class _CanonicalLpDualView:
    """Use solver dual candidates with the complete canonical LP matrix.

    Gurobi can discard extremely small coefficients.  A certificate of its
    stored matrix alone need not bound the original route-cut envelope.  The
    existing exact-residual auditor accepts arbitrary sign-correct candidate
    multipliers, so it can audit the returned Pi against the original rows,
    objective and boxes instead.  No solver coefficient threshold is assumed.

    The canonical route-base substitution already rounds residual slopes
    downward.  Certifying its equality and every cut row therefore also gives
    a valid bound for the complete, unmodified original cut archive.
    """
    def __init__(self, model, specification):
        self._model = model
        specification.validate()
        expanded = specification.expanded_rows()
        native_variables, native_rows = model.getVars(), model.getConstrs()
        if (len(native_variables) != len(specification.names)
                or len(native_rows) != len(expanded)):
            raise ValueError('Reference LP canonical/native dimensions differ')
        if model.getAttr('VarName', native_variables) != specification.names:
            raise ValueError('Reference LP canonical/native variable names differ')
        if any(variable.index != index for index, variable in enumerate(native_variables)):
            raise ValueError('Reference LP native variable indices differ')
        self._variables = [SimpleNamespace(index=index, VarName=name,
            Obj=specification.cost[index], LB=specification.lower[index],
            UB=specification.upper[index]) for index, name in enumerate(specification.names)]
        self._constraints = []
        for native, (index, name, sense, rhs) in zip(native_rows, expanded):
            if native.ConstrName != name or native.Sense != sense:
                raise ValueError('Reference LP canonical/native row mapping differs')
            self._constraints.append(SimpleNamespace(ConstrName=name, Sense=sense,
                RHS=rhs, Pi=float(native.Pi),
                _row=_CanonicalRow(specification.rows[index], self._variables)))
        # LinearMILP has no separate objective constant. Stage costs are
        # represented explicitly in its objective and equality rows.
        self.ObjCon = 0.

    def __getattr__(self, name):
        return getattr(self._model, name)

    def getVars(self):
        return self._variables

    def getConstrs(self):
        return self._constraints

    def getAttr(self, name, objects):
        return [getattr(value, name) for value in objects]

    def getRow(self, constraint):
        return constraint._row


def forward_s2_reference_bound(prob_data, node, cuts, parent, *, phase,
                               time_limit_s, deadline, stage_builder=None):
    """Return a downward-certified LP endpoint, never a partial-RMP value.

    All archived cuts are ordinary rows. The original full assignment model
    is relaxed continuously; its exact-dual audit charges reduced-cost
    residuals to finite boxes, as in the existing SBC LP certification.
    An unavailable certificate only disables this stopping shortcut.
    """
    import gurobipy as gp
    started = time.monotonic()
    stop = min(started + time_limit_s, deadline) if deadline is not None else started + time_limit_s
    fixed = relaxation = None
    diagnostic = dict(executed=False, certified=False, lower_bound=None)
    try:
        bounded_solve_time(time_limit_s, stop)
        builder = stage_builder or StageModelBuilder(prob_data)
        fixed = builder.build_stage_problem(2, node, cuts, parent, learned_cut_purpose='dual_lp')
        bounded_solve_time(time_limit_s, stop)
        relaxation = fixed.relax()
        relaxation.Params.OutputFlag = 0
        relaxation.Params.Threads = configured_gurobi_threads()
        relaxation.Params.FeasibilityTol = 1e-9
        relaxation.Params.OptimalityTol = 1e-9
        relaxation.Params.TimeLimit = bounded_solve_time(time_limit_s, stop)
        with backend_call('gurobi', 'optimize', model=relaxation, phase=phase,
                          path='forward', stage=2, attempt_kind='forward_reference_lp'):
            diagnostic['executed'] = True
            relaxation.optimize()
        diagnostic.update(status=int(relaxation.Status), solver_seconds=float(relaxation.Runtime),
                          variables=relaxation.NumVars, integer_variables=relaxation.NumIntVars)
        if relaxation.Status != gp.GRB.OPTIMAL:
            diagnostic['reason'] = 'reference_lp_not_optimal'
            return None, diagnostic
        # No parent bindings: certify a scalar lower bound for this fixed A.
        # Never substitute ObjVal or a restricted-master objective if this fails.
        try:
            canonical = fixed._lrp_spec.linear
        except AttributeError as exc:
            raise ValueError('Reference LP lacks its original canonical specification') from exc
        audit_view = _CanonicalLpDualView(relaxation, canonical)
        _, lower = extract_verified_fixed_rhs_dual_cut(audit_view, (), label='S2 forward reference LP')
        diagnostic.update(certified=True, lower_bound=lower,
                          objective=float(relaxation.ObjVal),
                          source='full_assignment_LP_exact_dual_audit',
                          matrix_source='canonical_stage_specification')
        return lower, diagnostic
    except SolveDeadlineReached:
        diagnostic['reason'] = 'reference_lp_budget_exhausted'
        return None, diagnostic
    except (gp.GurobiError, ValueError) as exc:
        diagnostic['reason'] = f'reference_lp_certificate_unavailable: {type(exc).__name__}: {exc}'
        return None, diagnostic
    finally:
        diagnostic['wall_seconds'] = time.monotonic() - started
        if relaxation is not None:
            relaxation.dispose()
        if fixed is not None:
            fixed.dispose()
