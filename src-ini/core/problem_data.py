"""Validated LRP input shared by the two information-stage EF and SDDP layers."""
from __future__ import annotations

from copy import deepcopy
from math import fsum
from operator import index
from pathlib import Path

import numpy as np

from models.stage_model_core import Instance, load_instance


class ProblemData:
    """LRP arrays with facilities I, customers J and delivery periods T.

    Facility intervals L are distinct from delivery periods T.  There is no
    vehicle-purchase or annual-operation scaling in this problem.
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            source = load_instance(source)
        if isinstance(source, ProblemData):
            source = source.instance
        if not hasattr(source, "arrays") or not hasattr(source, "metadata"):
            raise TypeError("ProblemData requires an LRP instance or instance directory")
        arrays = {}
        for key, value in source.arrays.items():
            value = np.ascontiguousarray(value)
            arrays[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        self.instance = Instance(str(source.name), arrays, deepcopy(source.metadata))
        self.instance.validate()
        self.lrp_instance = self.instance
        self.arrays = self.instance.arrays
        self.metadata = self.instance.metadata
        self.name = self.instance.name
        self.shape = self.instance.shape
        self.m, self.n, self.H, self.L, self.S = self.shape
        self.I = range(self.m)
        self.J = range(self.n)
        self.T = self.H
        self.numCustomers = self.n
        self.numFacilities = self.m
        self.num_scenarios = self.S
        self.information_stages = 2
        self.algorithm_layers = 3
        self.period_to_interval = self.arrays["period_to_interval"]
        self.scenario_prob = self.arrays["scenario_prob"]

    def validate(self):
        self.instance.validate()

    def route_costs(self):
        return self.instance.route_costs()

    def logical_hash(self):
        return self.instance.logical_hash()

    def subset(self, *, num_customers=None, num_facilities=None,
               customer_indices=None, facility_indices=None, T=None, num_scenarios=None):
        """Select existing nodes, periods and full trajectories without resampling.

        Counts retain the first N nodes. Explicit indices are zero-based positions
        in this loaded instance, in the requested order. Capacity, prices and
        scenario trajectories retain their original values. Periods/scenarios
        are prefixes. Retained scenario weights are explicitly conditioned on
        the selected trajectories, never normalized across delivery periods.
        """
        def selection(count, indices, size, label):
            if count is not None and indices is not None:
                raise ValueError(f'Choose {label} count or indices, not both')
            def integer(value):
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f'{label} selection must contain integers, not booleans')
                try:
                    return index(value)
                except TypeError as exc:
                    raise ValueError(f'{label} selection must contain integers') from exc
            if indices is None:
                count = size if count is None else integer(count)
                if not 1 <= count <= size:
                    raise ValueError(f'{label} count must be between 1 and {size}')
                return list(range(count))
            chosen = [integer(value) for value in indices]
            if not chosen or len(set(chosen)) != len(chosen):
                raise ValueError(f'{label} indices must be nonempty and unique')
            if min(chosen) < 0 or max(chosen) >= size:
                raise ValueError(f'{label} index outside [0, {size})')
            return chosen

        fi = selection(num_facilities, facility_indices, self.m, 'facility')
        cj = selection(num_customers, customer_indices, self.n, 'customer')
        periods = selection(T, None, self.H, 'period')
        scenarios = selection(num_scenarios, None, self.S, 'scenario')
        intervals = np.flatnonzero(self.arrays['location_periods'] <= len(periods)).tolist()
        physical_changed = fi != list(self.I) or cj != list(self.J)
        if not physical_changed and len(periods) == self.H and len(scenarios) == self.S:
            return ProblemData(self)
        if np.any(self.arrays['min_open'][intervals] > len(fi)):
            raise ValueError('Selected facilities are fewer than min_open; the strategic requirement is unchanged')
        route_nodes = [0] + [j + 1 for j in cj]
        axes = {
            'active': ((0, periods), (1, scenarios), (2, cj)),
            'demand': ((0, periods), (1, scenarios), (2, cj)),
            'outsourcing_cost': ((0, periods), (1, scenarios), (2, cj)),
            'capacity': ((0, fi), (1, periods)), 'opening_cost': ((0, fi), (1, intervals)),
            'continuation_cost': ((0, fi), (1, intervals)), 'closing_cost': ((0, fi), (1, intervals)),
            'initial_state': ((0, fi),), 'facility_ids': ((0, fi),),
            'facility_xy': ((0, fi),), 'customer_ids': ((0, cj),),
            'customer_xy': ((0, cj),), 'reference_mean_quantity': ((0, cj),),
            'route_cost': ((0, periods), (1, fi), (2, route_nodes), (3, route_nodes)),
            'cost_fc': ((0, periods), (1, fi), (2, cj)),
            'cost_cc': ((0, periods), (1, cj), (2, cj)),
            'location_periods': ((0, intervals),), 'period_to_interval': ((0, periods),),
            'min_open': ((0, intervals),), 'scenario_prob': ((0, scenarios),),
        }
        unknown = set(self.arrays) - set(axes) - {'reference_expected_total'}
        if unknown:
            raise ValueError(f'Unknown array axes for subsetting: {sorted(unknown)}')
        arrays = {}
        for key, axis_selections in axes.items():
            if key not in self.arrays:
                continue
            value = self.arrays[key]
            for axis, chosen in axis_selections:
                value = np.take(value, chosen, axis=axis)
            arrays[key] = value
        provenance = dict(source_name=self.name, source_logical_sha256=self.logical_hash(),
                          facility_indices=fi, customer_indices=cj,
                          period_indices=periods, scenario_indices=scenarios,
                          facility_interval_indices=intervals,
                          capacity_and_cost_policy='retain original selected entries')
        if len(scenarios) != self.S:
            original_weights = [float(value) for value in arrays['scenario_prob']]
            mass = fsum(original_weights)
            arrays['scenario_prob'] = np.array([value / mass for value in original_weights])
            provenance.update(scenario_probability_policy='conditional_on_selected_trajectories',
                              selected_original_probabilities=original_weights,
                              selected_probability_mass=mass)
        if 'reference_expected_total' in self.arrays:
            if physical_changed:
                # This describes the generator's population, not reduced nodes.
                provenance['source_reference_expected_total'] = self.arrays['reference_expected_total'].tolist()
            else:
                arrays['reference_expected_total'] = self.arrays['reference_expected_total'][periods]
        name = f'{self.name}__F{"-".join(map(str, fi))}_C{"-".join(map(str, cj))}'
        if len(periods) != self.H or len(scenarios) != self.S:
            name += f'_T{len(periods)}_S{len(scenarios)}'
        metadata = dict(name=name, kind='subset_of_existing_lrp', subset=provenance,
                        model_flags=deepcopy(self.metadata.get('model_flags', {})),
                        source_metadata=deepcopy(self.metadata))
        result = ProblemData(Instance(name, arrays, metadata))
        result.metadata['dimensions'] = dict(zip(('I', 'J', 'T', 'L', 'S'), result.shape))
        result.metadata['logical_sha256'] = result.logical_hash()
        return result
