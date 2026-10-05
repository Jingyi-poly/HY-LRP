"""Read an existing stochastic LRP instance without altering its data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.problem_data import ProblemData
from core.scenario_tree import build_scenario_tree

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LRP_INSTANCE = REPOSITORY_ROOT / "data/lrp/berlin52_I2_J5_T3_S3_g0_lam2.0"


@dataclass
class LRPConfig:
    instance: str | Path = DEFAULT_LRP_INSTANCE
    out: str | Path | None = None
    data_source: str = "lrp"
    num_customers: int | None = None
    num_facilities: int | None = None
    customer_indices: tuple[int, ...] | None = None
    facility_indices: tuple[int, ...] | None = None
    T: int | None = None
    num_scenarios: int | None = None
    scenario_seed: int | None = None
    location_periods: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.data_source != 'lrp':
            raise ValueError('LRPConfig data_source must be lrp; set instance to an LRP data directory')

    def instance_tag(self):
        tag = Path(self.instance).name
        if self.num_facilities is not None:
            tag += f"_I{self.num_facilities}"
        if self.num_customers is not None:
            tag += f"_J{self.num_customers}"
        if self.facility_indices is not None:
            tag += "_F" + "-".join(map(str, self.facility_indices))
        if self.customer_indices is not None:
            tag += "_C" + "-".join(map(str, self.customer_indices))
        if self.T is not None:
            tag += f'_T{self.T}'
        if self.num_scenarios is not None:
            tag += f'_S{self.num_scenarios}'
        if self.scenario_seed is not None:
            tag += f'_Seed{self.scenario_seed}'
        if self.location_periods is not None:
            tag += '_LP' + '-'.join(map(str, self.location_periods))
        return tag


class LRPInstance:
    def __init__(self, config: LRPConfig | str | Path = DEFAULT_LRP_INSTANCE):
        self.config = config if isinstance(config, LRPConfig) else LRPConfig(config)
        self.cfg = self.config
        self.prob_data = None
        self.scen_tree = None

    def build(self):
        path = Path(self.config.instance).expanduser()
        if not path.exists() and not path.is_absolute():
            path = REPOSITORY_ROOT / path
        self.path = path.resolve()
        self.prob_data = ProblemData(self.path)
        if any(value is not None for value in (self.config.T, self.config.num_scenarios,
                                               self.config.scenario_seed, self.config.location_periods)):
            from core.lrp_generator import configure_horizon
            self.prob_data = configure_horizon(self.prob_data, source_path=self.path,
                T=self.config.T, num_scenarios=self.config.num_scenarios,
                scenario_seed=self.config.scenario_seed, location_periods=self.config.location_periods)
        if any(value is not None for value in (
                self.config.num_customers, self.config.num_facilities,
                self.config.customer_indices, self.config.facility_indices)):
            self.prob_data = self.prob_data.subset(num_customers=self.config.num_customers,
                                                num_facilities=self.config.num_facilities,
                                                customer_indices=self.config.customer_indices,
                                                facility_indices=self.config.facility_indices)
        self.instance = self.prob_data.instance
        self.scen_tree = build_scenario_tree(self.prob_data)
        from core.run_snapshot import attach_instance_snapshot
        attach_instance_snapshot(self.prob_data, self.config, source_path=self.path)
        return self

    def subset(self, **selection):
        """Return a reduced loaded instance and rebuild its matching scenario tree."""
        if self.prob_data is None:
            raise ValueError('Load the instance with build() before selecting a subset')
        from copy import deepcopy
        result = LRPInstance(deepcopy(self.config))
        result.path = self.path
        result.prob_data = self.prob_data.subset(**selection)
        result.instance = result.prob_data.instance
        # Keep the configuration replayable, also after multiple selections.
        same = result.prob_data.logical_hash() == self.prob_data.logical_hash()
        selected = {} if same else result.prob_data.metadata['subset']
        for kind, size in (('facility', self.prob_data.m), ('customer', self.prob_data.n)):
            original = getattr(self.config, f'{kind}_indices')
            if original is None:
                original = tuple(range(size))
            positions = selected.get(f'{kind}_indices', range(size))
            setattr(result.config, f'{kind}_indices', tuple(original[i] for i in positions))
        result.config.num_facilities = result.config.num_customers = None
        if selection.get('T') is not None:
            result.config.T = result.prob_data.H
            if result.config.location_periods is not None:
                result.config.location_periods = tuple(date for date in result.config.location_periods
                                                       if date <= result.prob_data.H)
        if selection.get('num_scenarios') is not None:
            result.config.num_scenarios = result.prob_data.S
        result.config.instance = self.path
        # Replay the composed selection once from the source. Sequential
        # probability conditioning can otherwise introduce roundoff that
        # changes the snapshot hash relative to a fresh build of this config.
        return LRPInstance(result.config).build()
