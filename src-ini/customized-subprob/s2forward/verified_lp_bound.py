"""Fixed-fleet MIP bound stabilization using an exact-verified LP.

The input model must be the synchronized, solved original fixed-fleet model.
Its raw/current and retained prior bounds still use the caller's existing
Gurobi status validation. A larger branch-and-bound value is not certified by
this LP check. No cut, original model, or policy is modified.
"""
from fractions import Fraction as F
import math
import time

from gurobipy import GurobiError
from core.backend_telemetry import backend_call
from core.solver_bounds import extract_verified_fixed_rhs_dual_cut


class VerifiedLPPolicyConflict(RuntimeError):
    pass


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError,TypeError,OverflowError):
        return None
    return number if math.isfinite(number) else None


def tiny_bound_conflict(lower, upper):
    """A <=1e-6 contradiction may trigger re-verification, never acceptance."""
    lower, upper = _finite(lower), _finite(upper)
    return (lower is not None and upper is not None
            and F(0) < F(lower) - F(upper) <= F(1e-6))


def _policy_box(lp, upper):
    """Retain every feasible MIP point costing <= the certified policy UB."""
    if upper is None or upper < 0:
        return 0
    variables = list(lp.getVars())
    constant = _finite(getattr(lp,'ObjCon',0.))
    if (constant is None or constant < 0
            or any(_finite(v.Obj) is None or float(v.Obj) < 0
                   or _finite(v.LB) is None or float(v.LB) < 0 for v in variables)):
        return 0
    tightened = 0
    for variable in variables:
        if float(variable.Obj) <= 0:
            continue
        exact = upper/F(float(variable.Obj))
        try:
            rounded = float(exact)
        except OverflowError:
            continue
        if not math.isfinite(rounded) or rounded >= .5e100:
            continue
        if F(rounded) < exact:
            rounded = math.nextafter(rounded,math.inf)
        if rounded < float(variable.LB):
            raise VerifiedLPPolicyConflict('certified policy upper is below a positive-cost variable floor')
        if rounded < float(variable.UB):
            variable.UB = rounded
            tightened += 1
    return tightened


def _fix_fleet_links(lp):
    """Exact redundant bounds for this fixed LP, never a parametric LP cut."""
    tightened = 0
    for constraint in lp.getConstrs():
        if not str(constraint.ConstrName).startswith('z_prev_eq[') or constraint.Sense != '=':
            continue
        row = lp.getRow(constraint)
        if row.size() != 1 or float(row.getCoeff(0)) != 1.:
            continue
        variable, rhs = row.getVar(0), float(constraint.RHS)
        if not str(variable.VarName).startswith('z[') or rhs not in (0.,1.):
            continue
        if not float(variable.LB) <= rhs <= float(variable.UB):
            raise ValueError('fixed-fleet link contradicts represented variable bounds')
        variable.LB = variable.UB = rhs
        tightened += 1
    return tightened


def stabilize_fixed_fleet_bound(model, raw_bound, *, prior_bound=None,
                                policy_upper_bound=None, time_limit_s=1.,
                                deadline=None, label='Stage2 fixed fleet',
                                recover_missing_bound=False):
    """Return a selected bound and diagnostics from an independent LP check.

    ``policy_upper_bound`` is an independently certified *canonical proxy*
    policy cost for these exact fleet/archive/objective units, not raw ObjVal.
    Prior bounds are allowed only for the same model or a valid strengthening.
    Current/prior candidates are filtered against that policy and combined
    BEFORE stabilization; callers must not max an old prior into the result.
    ``recover_missing_bound`` permits a new verified LP certificate when no
    raw MIP bound survives. It never repairs a raw bound by clipping it to UB.

    Gurobi ``relax`` itself creates a separate model; no second copy is needed.
    Every retained linear row becomes explicit. Omitted callback cuts only
    enlarge this relaxation, so they cannot invalidate its lower bound.
    An optimum-preserving objective box can narrow the relaxation, but keeps
    at least one original integer optimum because that optimum costs <= U.
    Its exact-residual certificate is still an original MIP lower bound.
    """
    started=time.perf_counter()
    upper=None if policy_upper_bound is None else F(policy_upper_bound)
    if upper is not None and not math.isfinite(float(upper)):
        raise ValueError('policy upper must be finite')
    current,prior=_finite(raw_bound),_finite(prior_bound)
    values=[x for x in (current,prior) if x is not None and (upper is None or F(x)<=upper)]
    selected=max(values) if values else None
    report=dict(bound=selected,source='raw_mip' if selected is not None else None,
                raw_bound=current,prior_bound=prior,verified_lp_bound=None,
                status='not_started',lp_status=None,boxed_variables=0,
                fixed_fleet_variables=0,seconds=0.)
    lp=None
    try:
        if selected is None and not recover_missing_bound:
            report['status']='no_accepted_raw_bound'
            return report
        seconds=float(time_limit_s)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('LP allowance must be finite and nonnegative')
        stop=time.monotonic()+min(1.,seconds)
        if deadline is not None:
            if not math.isfinite(float(deadline)):
                raise ValueError('deadline must be finite')
            stop=min(stop,float(deadline))
        if time.monotonic()>=stop:
            report['status']='budget_exhausted'
            return report
        if int(model.ModelSense)!=1 or any(int(getattr(model,name,0)) for name in
                ('NumQNZs','NumQConstrs','NumGenConstrs','NumSOS','NumPWLObjVars','NumScenarios')):
            report['status']='unsupported_model'
            return report
        if int(getattr(model,'NumObj',1))>1:
            report['status']='unsupported_model'
            return report
        lp=model.relax()
        for row in lp.getConstrs():
            row.Lazy=0
        for name,value in (('OutputFlag',0),('Threads',1),('Method',1),
                           ('LazyConstraints',0),('BestBdStop',1e100),
                           ('BestObjStop',-1e100),('Cutoff',1e100),
                           ('IterationLimit',1e100),('WorkLimit',1e100),
                           ('NodeLimit',1e100),('SolutionLimit',2000000000),
                           ('OptimalityTol',1e-9),('FeasibilityTol',1e-9)):
            lp.setParam(name,value)
        report['fixed_fleet_variables']=_fix_fleet_links(lp)
        report['boxed_variables']=_policy_box(lp,upper)
        lp.update()
        remaining=stop-time.monotonic()
        if remaining<=0:
            report['status']='construction_exhausted_budget'
            return report
        lp.setParam('TimeLimit',remaining)
        with backend_call('gurobi', 'optimize', model=lp,
                          attempt_kind='fixed_fleet_lp_verification', stage=2):
            lp.optimize()
        report['lp_status']=int(lp.Status)
        if int(lp.Status)!=2:
            report['status']='lp_not_optimal'
            return report
        slope,verified=extract_verified_fixed_rhs_dual_cut(lp,[],label=label)
        if slope or _finite(verified) is None:
            raise ValueError('fixed-fleet verification did not return a finite constant')
        verified=float(verified)
        report['verified_lp_bound']=verified
        if upper is not None and F(verified)>upper:
            raise VerifiedLPPolicyConflict('verified LP lower exceeds independently certified policy upper')
        if selected is None:
            report.update(bound=verified,source='verified_fixed_lp',status='recovered')
        elif F(selected)<=F(verified)+F(1e-6):
            report.update(bound=verified,source='verified_fixed_lp',status='replaced')
        else:
            report['status']='raw_bound_above_verified_lp'
        return report
    except (GurobiError, AttributeError, TypeError, ValueError, KeyError, OverflowError) as exc:
        report.update(status='lp_verification_unavailable',reason=str(exc))
        return report
    finally:
        if lp is not None:
            lp.dispose()
        report['seconds']=time.perf_counter()-started
