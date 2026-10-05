"""Verify a saved LRP policy against the independent two-stage EF.

The original --state / --replay-forward workflow is retained.  Reconstruct
the saved physical data, audit its actual routes in the EF matrix, then fix
facilities and assignments and let the EF reoptimize routes as before.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from setup.solver_config import bootstrap_solver_environment

bootstrap_solver_environment('bench')


def _replay_forward(inst, config, cut_lag, num_processes, piece_tables=None):
    from algorithms.sddlp import SDDLP
    solver = SDDLP(inst.prob_data, inst.scen_tree, config, cut_lag_init=cut_lag)
    pool = get_context('spawn').Pool(processes=num_processes) if num_processes > 1 else None
    try:
        return solver.forward_solver.forward_pass(
            inst.scen_tree, solver.cut_lag, num_processes, pool=pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _build_run_from_state(state, args):
    from core.instance import LRPInstance
    from core.problem_data import ProblemData
    from core.scenario_tree import build_scenario_tree
    from core.run_snapshot import (algorithm_config_from_snapshot,
        instance_config_from_snapshot, restore_algorithm_environment,
        _validate_lrp_snapshot)

    snapshot = state.get('run_snapshot')
    if not snapshot or snapshot.get('problem_type') != 'stochastic_lrp':
        raise ValueError('A complete LRP state snapshot is required; dimensions alone '
                         'cannot reconstruct physical locations and saved scenarios')
    data = _validate_lrp_snapshot(snapshot)
    if snapshot['instance'].get('config'):
        inst = LRPInstance(instance_config_from_snapshot(snapshot)).build()
    else:
        pd = ProblemData(data)
        inst = SimpleNamespace(prob_data=pd, scen_tree=build_scenario_tree(pd))
    m, n, H, _, S = inst.prob_data.shape
    for flag, actual in (('customers', n), ('facilities', m), ('periods', H), ('scenarios', S)):
        requested = getattr(args, flag, None)
        if requested is not None and requested != actual:
            raise ValueError(f'--{flag}={requested} differs from the saved physical data ({actual})')
    restore_algorithm_environment(snapshot)
    return inst, algorithm_config_from_snapshot(snapshot), snapshot


def _policy_cost_from_x_star(inst, x_star, cost_star):
    from solvers.forward_policy_certification import certify_policy
    return certify_policy(inst.prob_data, inst.scen_tree, x_star)['feasible_upper_bound']


def _ef_policy_vector(builder, x_star):
    """Map actual saved decisions into EF columns; derive only MTZ orders."""
    import numpy as np
    m, n, H, L, S = builder.prob_data.shape
    M = builder.spec
    x = np.zeros(len(M.names))

    def bit(raw):
        raw = float(raw)
        if not math.isfinite(raw) or min(abs(raw), abs(raw - 1)) > 2e-6:
            raise ValueError(f'Non-binary policy decision: {raw}')
        return int(round(raw))

    def put(group, key, raw):
        x[M.groups[group][key]] = bit(raw)

    root = x_star[1][0]
    for i in range(m):
        previous = 0
        for k in range(L):
            current = bit(root[f'A[{i},{k}]'])
            for group, implied in (('A', current), ('o', current * (1-previous)),
                                   ('h', current * previous), ('b', previous * (1-current))):
                put(group, (i,k), root.get(f'{group}[{i},{k}]', implied))
            previous = current
    for s in range(S):
        for t in range(H):
            q = s * H + t
            assignment = x_star[2][q]
            for j in range(n):
                put('e', (j,t,s), assignment[f'e[{j}]'])
            for i in range(m):
                put('u', (i,t,s), assignment[f'u[{i}]'])
                for j in range(n):
                    put('alpha', (i,j,t,s), assignment[f'alpha[{i},{j}]'])
                saved = x_star[3][q*m+i]
                expected_names = {f'r[{i},{v},{w}]' for v in range(n+1)
                                  for w in range(n+1) if v != w}
                if any(str(name).startswith('r[') and name not in expected_names and bit(value)
                       for name, value in saved.items()):
                    raise ValueError('Policy route has an invalid arc or physical facility')
                successor = {}
                for v in range(n+1):
                    for w in range(n+1):
                        if v == w:
                            continue
                        used = bit(saved.get(f'r[{i},{v},{w}]', 0))
                        put('r', (i,v,w,t,s), used)
                        if used:
                            if v in successor:
                                raise ValueError('Policy route has multiple successors')
                            successor[v] = w
                current, seen = successor.get(0, 0), set()
                while current and current not in seen:
                    seen.add(current)
                    x[M.groups['nu'][i,current-1,t,s]] = len(seen)
                    current = successor.get(current, 0)
    return x


def _audit_saved_policy(builder, x_star):
    import numpy as np
    from models.extensive_model_builder import audit_solution
    x = _ef_policy_vector(builder, x_star)
    M, d = builder.spec, builder.prob_data
    audit = audit_solution(d, M, x, float(np.dot(M.cost, x)))
    if not audit['passed']:
        raise ValueError(f'Saved policy fails independent EF audit: {audit["errors"]}')
    # Check fractional capacity data exactly, beyond the matrix audit tolerance.
    for node in audit['nodes']:
        t, s = node['period'], node['scenario']
        for i, assigned in enumerate(node['assigned']):
            load = sum((Fraction.from_float(float(d.arrays['demand'][t,s,j]))
                        for j in assigned), Fraction(0))
            capacity = Fraction.from_float(float(d.arrays['capacity'][i,t])) * node['dispatch'][i]
            if load > capacity:
                raise ValueError(f'Saved policy violates exact facility capacity {(i,t,s)}')
    # Keep the original p*cost arithmetic distinct from stored EF coefficients.
    exact = sum((Fraction.from_float(event['cost']) for event in audit['facility_events']), Fraction(0))
    C = d.route_costs()
    for node in audit['nodes']:
        t, s = node['period'], node['scenario']
        local = sum((Fraction.from_float(float(d.arrays['outsourcing_cost'][t,s,j]))
                     for j in node['outsourced']), Fraction(0))
        for i, route in enumerate(node['routes']):
            local += sum((Fraction.from_float(float(C[t,i,v,w]))
                          for v,w in zip(route,route[1:])), Fraction(0))
        exact += Fraction.from_float(float(d.arrays['scenario_prob'][s])) * local
    matrix = sum((Fraction.from_float(c)*int(x[k])
                  for k,c in enumerate(M.cost) if c), Fraction(0))
    upper = float(exact)
    if Fraction.from_float(upper) < exact:
        upper = math.nextafter(upper, math.inf)
    audit['objective_ub'] = upper
    matrix_upper = float(matrix)
    if Fraction.from_float(matrix_upper) < matrix:
        matrix_upper = math.nextafter(matrix_upper, math.inf)
    audit['matrix_objective_ub'] = matrix_upper
    return x, audit


def evaluate_in_ef(inst, x_star, time_limit, threads):
    from models.extensive_model_builder import ExtensiveModelBuilder
    from core.solver_bounds import certified_gurobi_minimization_lower_bound
    if not math.isfinite(time_limit) or time_limit <= 0 or threads < 1:
        raise ValueError('EF time limit and threads must be positive')
    builder = ExtensiveModelBuilder(inst.prob_data)
    primal, saved_audit = _audit_saved_policy(builder, x_star)
    model = builder.build()
    try:
        model.Params.Threads = threads
        model.Params.MIPGap = 1e-9
        model.Params.FeasibilityTol = 1e-8
        model.Params.IntFeasTol = 1e-8
        model.Params.TimeLimit = time_limit
        variables = model._lrp_variables
        for variable, value in zip(variables, primal):
            variable.Start = value
        n_fixed = 0
        for group in ('A','o','h','b','alpha','e','u'):
            for index in builder.spec.groups[group].values():
                variables[index].LB = variables[index].UB = primal[index]
                n_fixed += 1
        model.optimize()
        cert = builder.certify_rounded_incumbent(model) if model.SolCount else None
        return dict(status=int(model.Status), n_fixed=n_fixed,
            raw_obj=float(model.ObjVal) if model.SolCount else None,
            bound=certified_gurobi_minimization_lower_bound(model),
            certified_obj=cert['objective_ub'] if cert else None,
            capacity_violation=max([0.] + [load-capacity*dispatch
                for node in cert['nodes']
                for load,capacity,dispatch in zip(node['loads'],node['capacities'],node['dispatch'])]) if cert else None,
            saved_policy_ef_audit=saved_audit)
    finally:
        model.dispose()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', required=True)
    ap.add_argument('--customers', type=int, help='optional saved customer-count check')
    ap.add_argument('--facilities', '--vehicles', dest='facilities', type=int,
                    help='optional saved facility-count check (--vehicles is the old spelling)')
    ap.add_argument('--scenarios', type=int, help='optional saved scenario-count check')
    ap.add_argument('--periods', type=int, help='optional saved period-count check')
    ap.add_argument('--processes', type=int, default=6)
    ap.add_argument('--ef-threads', type=int, default=4)
    ap.add_argument('--ef-time-limit', type=float, default=600.)
    ap.add_argument('--replay-forward', action='store_true')
    args = ap.parse_args()
    if args.processes < 1:
        ap.error('--processes must be positive')
    with open(args.state, 'rb') as stream:
        state = pickle.load(stream)
    inst, config, snapshot = _build_run_from_state(state, args)
    print(f"[verify] dump iter={state.get('iteration')} best_lb={state.get('best_lb')} best_ub={state.get('best_ub')}")
    print(f"[verify] reconstructed saved run fingerprint={snapshot['fingerprint']} dimensions={snapshot['dimensions']}")
    policy = state.get('x_best')
    if policy is not None and not args.replay_forward:
        sddp_value = float(state['best_ub'])
        print(f'[verify] policy = dumped x_best (SDDP UB {sddp_value:,.6f})')
    else:
        lb, policy, sddp_value, costs = _replay_forward(inst, config, state['cut_lag'], args.processes)
        own = _policy_cost_from_x_star(inst, policy, costs)
        print(f'[verify] replayed forward on dumped cuts: LB={lb:,.6f} UB={sddp_value:,.6f} (re-accumulated {own:,.6f})')
    if not math.isfinite(sddp_value):
        raise ValueError('Saved/replayed upper bound is nonfinite')
    root = policy[1][0]
    available = {k: [i for i in inst.prob_data.I if root[f'A[{i},{k}]'] > .5]
                 for k in range(inst.prob_data.L)}
    print(f'[verify] Stage-1 facilities by interval: {available}')
    ef = evaluate_in_ef(inst, policy, args.ef_time_limit, args.ef_threads)
    actual = ef['saved_policy_ef_audit']['objective_ub']
    difference = actual - sddp_value
    tolerance = 1e-6 * max(1., abs(sddp_value))
    print(f'[verify] saved routes: independent EF audit PASS, cost={actual:,.6f}, difference={difference:+.6f}')
    print(f"[verify] EF(policy fixed A,alpha,e,u): status={ef['status']} fixed={ef['n_fixed']} raw_obj={ef['raw_obj']} certified_obj={ef['certified_obj']} bound={ef['bound']} cap_violation={ef['capacity_violation']}")
    if sddp_value < actual or abs(difference) > tolerance:
        print('[verify] RESULT: saved SDDP UB differs from its actual policy cost')
        return 1
    saved_lb = state.get('best_lb')
    if saved_lb is not None and float(saved_lb) > actual:
        print('[verify] RESULT: saved LB exceeds the independently audited feasible policy cost')
        return 1
    if ef['certified_obj'] is not None:
        print(f"[verify] route reoptimization difference={ef['certified_obj']-sddp_value:+.6f}; original-route feasibility is certified independently")
    else:
        print('[verify] route reoptimization has no incumbent; saved-route EF feasibility audit remains valid')
    print('[verify] RESULT: POLICY MATCH (feasibility and cost; global LB validity is not checked)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
