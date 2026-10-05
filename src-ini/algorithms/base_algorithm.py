"""Base class shared by SDDP algorithm variants."""

from __future__ import annotations

import math

from core.solution import AlgorithmConfig


class SDDPAlgorithm:
    """Common state for Phase-1 SBC and Phase-2 Lagrangian algorithms."""

    def __init__(self, prob_data, scen_tree, config):
        if isinstance(config, dict):
            config = AlgorithmConfig(**config)
        self.prob_data = prob_data
        self.scen_tree = scen_tree
        self.config = config
        self.lb_history = []
        self.ub_history = []
        self.time_history = []
        self.x_best = None
        self.converged = False

    def solve(self):
        raise NotImplementedError


def cut_archive_fingerprint(cut_lag):
    """Exact fingerprint of the existing tuple-cut archive, without VRP state."""
    return tuple((stage, node, tuple((tuple(sorted(pi.items())), float(intercept))
                 for pi, intercept in cuts))
                 for stage, nodes in sorted(cut_lag.items())
                 for node, cuts in sorted(nodes.items()))


def forward_values_are_finite(values):
    """Reject nonfinite trial states as well as nonfinite outer certificates."""
    if isinstance(values, dict):
        return all(forward_values_are_finite(value) for value in values.values())
    if isinstance(values, (list, tuple)):
        return all(forward_values_are_finite(value) for value in values)
    if isinstance(values, (int, float)):
        return math.isfinite(values)
    return values is None or isinstance(values, (str, bool))


def solver_diagnostics(forward, backward):
    counts = {}
    for solver in (forward, backward):
        for key, value in getattr(solver, 'solve_counts', {}).items():
            counts[key] = counts.get(key, 0) + value
    return {
        'solve_counts': counts,
        'forward_diagnostics': getattr(forward, 'last_forward_diagnostics', {}),
        'backward_diagnostics': getattr(backward, 'last_cut_diagnostics', {}),
        'node_execution': {
            'requested_processes': (getattr(forward, 'last_policy_certificate', None) or {}).get('requested_processes'),
            'forward_worker_pids': list((getattr(forward, 'last_policy_certificate', None) or {}).get('worker_pids', [])),
            'backward_worker_pids': sorted(getattr(backward, 'worker_pids', set())),
        },
        'period_dedup': dict(getattr(forward, 'period_dedup_stats', {})),
    }
