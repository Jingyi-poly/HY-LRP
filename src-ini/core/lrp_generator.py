"""Reproducible, explicitly NEW LRP instances inspired by Section 4 (2012).

Copied from the supplied generate_lrp.py v0.1.0; its original source SHA256 is
3b239e15b91c9eb55acb5070a4299e9f857dfee1c6685b2b416d8b6b51e44a8e.
Only the optional scenario_seed separates activity/quantity randomness from
the original geometry and cost seed. No author instance is reconstructed exactly.
The repository owns this copy; runtime never imports a parent-directory script.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

VERSION = "0.1.0"
PAPER = {"authors": ["Maria Albareda-Sambola", "Elena Fernández", "Stefan Nickel"],
         "title": "Multiperiod Location-Routing with Decoupled Time Scales",
         "year": 2012, "doi": "10.1016/j.ejor.2011.09.022", "section": "4, p. 253"}


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def seeded(master: int, *tags: Any) -> np.random.Generator:
    """Named streams: adding scenarios does not change costs/capacities or old scenarios."""
    digest = hashlib.sha256(canonical([int(master), *tags]).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))


def read_tsp(path: Path) -> tuple[str, np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8-sig")
    headers: dict[str, str] = {}
    rows = []
    inside = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            inside = True
            continue
        if inside:
            if line == "EOF" or line.endswith("_SECTION"):
                break
            cols = line.split()
            if len(cols) != 3:
                raise ValueError(f"Bad coordinate row: {line!r}")
            rows.append((int(cols[0]), float(cols[1]), float(cols[2])))
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    if headers.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("Only EUC_2D coordinate files are accepted; no silent conversion.")
    if len(rows) != int(headers.get("DIMENSION", -1)) or not rows:
        raise ValueError("DIMENSION does not match coordinate rows.")
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    xy = np.array([r[1:] for r in rows], dtype=float)
    if len(set(ids.tolist())) != len(ids) or not np.isfinite(xy).all():
        raise ValueError("Node IDs must be unique and coordinates finite.")
    return headers.get("NAME", path.stem), ids, xy, hashlib.sha256(raw).hexdigest()


def local_route_matrix(arr: dict[str, np.ndarray], facility: int, period: int) -> np.ndarray:
    """0 is THIS facility; local indices 1..n correspond to customer positions 0..n-1."""
    fc, cc = arr["cost_fc"], arr["cost_cc"]
    n = fc.shape[2]
    out = np.zeros((n + 1, n + 1))
    out[0, 1:] = fc[period, facility]
    out[1:, 0] = fc[period, facility]
    out[1:, 1:] = cc[period]
    return out


def _draw_activity(rng: np.random.Generator, H: int, n: int, q: float, mode: str) -> np.ndarray:
    if mode == "bernoulli":
        return (rng.random((H, n)) < q).astype(np.int8)
    if mode == "fixed_count":
        count = int(math.floor(q * n + 0.5))
        # Draw priorities even when count=0: explicit repeatable per-period subset selection.
        priorities = rng.random((H, n))
        order = np.argsort(priorities, axis=1)
        active = np.zeros((H, n), dtype=np.int8)
        for t in range(H):
            active[t, order[t, :count]] = 1
        return active
    raise ValueError(f"Unknown activity mode {mode!r}")


def generate(cfg: dict[str, Any], base_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    allowed = {"id", "source", "source_kind", "facilities", "customers", "periods", "location_periods",
               "seed", "geometry_rep", "demand_rep", "q", "setup_cv", "kind", "scenarios",
               "activation", "cost_mode", "capacity_ratio", "demand_halfwidth", "outsource_multiplier",
               "min_open", "quantity_mean_bounds", "scenario_seed"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(set(cfg)-allowed)}")
    m, n, H = (int(cfg[k]) for k in ("facilities", "customers", "periods"))
    TL = np.asarray(cfg["location_periods"], dtype=np.int64)
    if min(m, n, H) < 1 or TL.ndim != 1 or not len(TL) or TL[0] != 1:
        raise ValueError("Positive dimensions and nonempty T_L starting at 1 are required.")
    if np.any(np.diff(TL) <= 0) or TL[-1] > H:
        raise ValueError("T_L must be strictly increasing and contained in {1,...,H}.")
    L = len(TL)
    k_of_t = np.searchsorted(TL, np.arange(1, H + 1), side="right") - 1
    q, cv = float(cfg.get("q", .5)), float(cfg.get("setup_cv", .1))
    if not (0 <= q <= 1) or not np.isfinite(cv) or cv < 0:
        raise ValueError("q must be in [0,1] and setup_cv finite/nonnegative.")
    kind = cfg.get("kind", "stochastic")
    if kind not in {"stochastic", "deterministic_regeneration"}:
        raise ValueError("Unknown instance kind.")
    S = int(cfg.get("scenarios", 1))
    if S < 1 or (kind == "deterministic_regeneration" and S != 1):
        raise ValueError("Deterministic regeneration requires exactly one scenario.")
    source = (base_dir / cfg["source"]).resolve()
    name, ids, xy, sha = read_tsp(source)
    source_kind = cfg.get("source_kind", "local_unverified")
    if source_kind == "synthetic_demo" and not name.startswith("SYNTHETIC_DEMO"):
        raise ValueError("Synthetic metadata requires a clearly named synthetic source.")
    if source_kind == "tsplib_local" and name.startswith("SYNTHETIC_DEMO"):
        raise ValueError("Refusing to label a synthetic source as TSPLIB.")
    if m + n > len(ids):
        raise ValueError(f"Need {m+n} distinct nodes but source has {len(ids)}; no replacement/jitter fallback.")
    master = int(cfg.get("seed", 20260920))
    geom_key = [sha, m, n, int(cfg.get("geometry_rep", 0))]
    selected = seeded(master, "geometry", geom_key).permutation(len(ids))[:m+n]
    fi, cj = np.sort(selected[:m]), np.sort(selected[m:])
    node_ix = np.r_[fi, cj]
    coord = xy[node_ix]
    raw_dist = np.sqrt(((coord[:, None, :] - coord[None, :, :]) ** 2).sum(axis=2))
    ei, ej = np.triu_indices(m+n, 1)
    use = ej >= m  # exclude facility--facility pairs
    ei, ej = ei[use], ej[use]
    mode = cfg.get("cost_mode", "paper_affine")
    if mode == "paper_affine":
        edge_dist = np.floor(raw_dist[ei, ej] + .5)  # TSPLIB EUC_2D, not bankers' round
        lo, hi = float(edge_dist.min()), float(edge_dist.max())
        if hi <= lo:
            raise ValueError("Cannot min-max rescale a constant edge-distance set.")
        edges = 10 + 90 * (edge_dist-lo)/(hi-lo)
    elif mode == "metric_control":
        lo, hi = 0., float(raw_dist[ei, ej].max())
        if hi <= 0:
            raise ValueError("All selected coordinates coincide.")
        edges = 100 * raw_dist[ei, ej] / hi
    else:
        raise ValueError("cost_mode must be paper_affine or metric_control.")
    cost_fc = np.zeros((H, m, n)); cost_cc = np.zeros((H, n, n))
    for t in range(H):
        if t:
            noise = seeded(master, "route-increment", geom_key, t).uniform(.99, 1.01, len(edges))
            edges = 1.03 * edges * (noise if mode == "paper_affine" else 1.)
        mat = np.zeros((m+n, m+n)); mat[ei, ej] = edges; mat[ej, ei] = edges
        cost_fc[t] = mat[:m, m:]; cost_cc[t] = mat[m:, m:]
    # Facility costs never use sampled demand and never have a scenario dimension.
    alpha = seeded(master, "facility-alpha", geom_key).uniform(.8, 1.2, m)
    mean_f = 2 * alpha * cost_fc[0].mean() * (1 + q) * n/m
    z = seeded(master, "facility-normal", geom_key).standard_normal(m)
    f0 = mean_f * (1 + cv*z)
    redrawn = 0
    for i in range(m):
        if f0[i] <= 0:
            rng = seeded(master, "negative-setup-redraw", geom_key, i, cv)
            for _ in range(100000):
                f0[i] = float(rng.normal(mean_f[i], cv*mean_f[i])); redrawn += 1
                if f0[i] > 0:
                    break
            else:
                raise RuntimeError("Failed to obtain a positive setup-cost draw.")
    opening = np.zeros((m, L)); opening[:, 0] = f0
    for k in range(1, L):
        noise = seeded(master, "facility-increment", geom_key, int(TL[k-1]), int(TL[k])).uniform(.95, 1.05, m)
        opening[:, k] = opening[:, k-1] * (1.03 ** int(TL[k]-TL[k-1])) * noise
    continuing = np.zeros((m, L)); closing = np.zeros((m, L))
    continuing[:, 1:] = np.ceil(.2 * opening[:, 1:])
    closing[:, 1:] = np.ceil(.5 * opening[:, 1:])
    activation = cfg.get("activation", "fixed_count" if kind == "deterministic_regeneration" else "bernoulli")
    demand_key = [geom_key, int(cfg.get("demand_rep", 0)), "det" if kind == "deterministic_regeneration" else "stoch"]
    scenario_seed = int(cfg.get("scenario_seed", master))
    active = np.zeros((H, S, n), dtype=np.int8)
    for s in range(S):
        active[:, s] = _draw_activity(seeded(scenario_seed, "activity", demand_key, s), H, n, q, activation)
    if kind == "deterministic_regeneration":
        demand = active.astype(float)  # unit placeholders, NEVER a paper capacity parameter
        expected = float((math.floor(q*n+.5)) if activation == "fixed_count" else q*n)
        mean_qty = np.ones(n)
        capacity = np.full((m,H), n, dtype=float)  # redundant, delete constraint in reference model
        outsourcing = np.zeros((H,S,n))  # e must be fixed to 0, NOT left free at zero cost!
        p = np.array([math.ceil(m*int(active[k_of_t==k, 0].sum(axis=1).max())/(n*(1+2*q)))
                      for k in range(L)], dtype=np.int64)
        flags = {"stages": 2, "information": "singleton deterministic demand trajectory",
                 "allow_outsourcing": False, "allow_idle": False,
                 "enforce_facility_capacity": False,
                 "state_semantics": "strict transitions; literal printed F may differ",
                 "singleton_policy": "directed out-and-back; authors' implementation unconfirmed",
                 "quantity_semantics": "unit placeholders ignored by uncapacitated reference model"}
    else:
        bounds = cfg.get("quantity_mean_bounds", [1,10])
        low, high = map(int, bounds)
        if low <= 0 or high < low:
            raise ValueError("quantity_mean_bounds must be positive ordered integers.")
        mean_qty = seeded(master, "mean-quantity", geom_key).integers(low, high+1, size=n).astype(float)
        delta = float(cfg.get("demand_halfwidth", .5))
        rho = float(cfg.get("capacity_ratio", 1.2))
        lam = float(cfg.get("outsource_multiplier", 1.))
        if not 0 <= delta < 1 or not np.isfinite(rho) or rho < 0 or not np.isfinite(lam) or lam < 0:
            raise ValueError("Require 0<=demand_halfwidth<1 and finite nonnegative rho/lambda.")
        demand = np.zeros((H,S,n))
        for s in range(S):
            U = seeded(scenario_seed, "conditional-quantity", demand_key, s).random((H,n))
            demand[:, s] = active[:, s] * mean_qty * (1 + delta*(2*U-1))
        p_act = q if activation == "bernoulli" else math.floor(q*n+.5)/n
        expected = float(p_act * mean_qty.sum())
        w = seeded(master, "capacity-heterogeneity", geom_key).uniform(.8, 1.2, m)
        K = rho * expected * w/w.sum()
        capacity = np.repeat(K[:, None], H, axis=1)
        singleton_reference = 2 * cost_fc.min(axis=1)  # all CANDIDATES, not the selected openings
        outsourcing = lam * singleton_reference[:,None,:] * (.5 + .5*demand/mean_qty) * active
        p_cfg = cfg.get("min_open", 0)
        p = np.full(L,int(p_cfg),dtype=np.int64) if np.isscalar(p_cfg) else np.asarray(p_cfg,dtype=np.int64)
        if p.shape != (L,) or np.any(p<0) or np.any(p>m):
            raise ValueError("min_open must be a feasible scalar or an L-vector.")
        flags = {"stages": 2, "information": "entire demand trajectory revealed after facility-plan commitment",
                 "allow_outsourcing": True, "allow_idle": True, "enforce_facility_capacity": True,
                 "state_semantics": "strict opening/continuation/closure",
                 "singleton_policy": "directed out-and-back", "outsourcing": "whole-order processing+delivery; no own capacity or route",
                 "cross_period_recourse_coupling": False, "cross_scenario_assignment_commitment": False,
                 "vehicles_per_facility_period": 1, "additional_vehicle_capacity": False}
    arrays = {"facility_ids": ids[fi], "customer_ids": ids[cj], "facility_xy": xy[fi], "customer_xy": xy[cj],
              "location_periods": TL, "period_to_interval": k_of_t.astype(np.int64),
              "initial_state": np.zeros(m,dtype=np.int8), "opening_cost": opening,
              "continuation_cost": continuing, "closing_cost": closing, "min_open": p,
              "cost_fc": cost_fc, "cost_cc": cost_cc, "active": active, "demand": demand,
              "capacity": capacity, "outsourcing_cost": outsourcing, "scenario_prob": np.full(S,1/S),
              "reference_mean_quantity": mean_qty, "reference_expected_total": np.full(H,expected)}
    metadata = {"generator_version": VERSION, "numpy_version": np.__version__, "paper": PAPER,
                "original_author_instance": False, "kind": kind,
                "source": {"name":name, "sha256":sha, "kind":source_kind, "path_supplied":cfg["source"]},
                "config": cfg, "dimensions": {"I":m,"J":n,"T":H,"L":L,"S":S},
                "model_flags":flags, "negative_normal_redraws":redrawn,
                "scaling": {"mode":mode,"distance_min":lo,"distance_max":hi},
                "indexing": "arrays are 0-based; location_periods and physical node IDs retain 1-based labels",
                "random_streams": "SHA256-derived PCG64; scenario-indexed activity and quantity streams; no Python hash()",
                "new_conventions": ["see README.md; not an exact regeneration of the original 135 instances"],
                "metric_cut_permission": "not granted by family name; inspect triangle diagnostics on actual matrices"}
    metadata["validation"] = validate(arrays, metadata)
    return arrays, metadata


def validate(a: dict[str,np.ndarray], meta: dict[str,Any]) -> dict[str,Any]:
    m,n,H,L,S=(meta["dimensions"][k] for k in ("I","J","T","L","S"))
    shapes={"active":(H,S,n),"demand":(H,S,n),"outsourcing_cost":(H,S,n),"capacity":(m,H),
            "opening_cost":(m,L),"continuation_cost":(m,L),"closing_cost":(m,L),
            "cost_fc":(H,m,n),"cost_cc":(H,n,n),"min_open":(L,),"scenario_prob":(S,)}
    for key,shape in shapes.items():
        if a[key].shape!=shape or not np.isfinite(a[key]).all() or np.any(a[key]<0):
            raise ValueError(f"Invalid shape or numeric domain: {key}")
    if not np.isin(a["active"],[0,1]).all() or np.any(a["demand"][a["active"]==0]!=0):
        raise ValueError("Activity/demand incompatibility.")
    if not np.isclose(a["scenario_prob"].sum(),1):
        raise ValueError("Scenario probabilities must sum to 1, without normalizing over T.")
    if not np.array_equal(a["cost_cc"],a["cost_cc"].transpose(0,2,1)):
        raise ValueError("Asymmetric customer costs in symmetric generator.")
    if not np.all(np.diagonal(a["cost_cc"],axis1=1,axis2=2)==0):
        raise ValueError("Nonzero customer self arcs.")
    if set(a["facility_ids"].tolist()) & set(a["customer_ids"].tolist()):
        raise ValueError("Facility and customer node sets overlap.")
    worst=0.; bad=0
    # Test exactly the local route graphs, not absent warehouse--warehouse arcs.
    for t in range(H):
        for i in range(m):
            C=local_route_matrix(a,i,t)
            local=0.
            for k in range(n+1):
                local=max(local,float(np.max(C-C[:,k,None]-C[None,k,:])))
            worst=max(worst,local); bad+=int(local>1e-8)
    bounds=None
    if meta["model_flags"]["allow_outsourcing"]:
        bounds=(a["outsourcing_cost"]*a["active"]).sum(axis=2)
    min_active=a["active"].sum(axis=2).min(axis=1)
    return {"status":"passed", "finite_nonnegative":True,"disjoint_physical_nodes":True,
            "full_scenario_weights_sum":float(a["scenario_prob"].sum()),
            "flat_period_scenario_weights_sum":float(H*a["scenario_prob"].sum()),
            "triangle_violating_local_matrices":bad,"max_triangle_violation":worst,
            "empty_period_scenario_nodes":int(np.count_nonzero(a["active"].sum(axis=2)==0)),
            "all_outsourced_recourse_bound":None if bounds is None else bounds.tolist(),
            "sample_demand_exceeds_all_facilities_capacity_nodes":int(np.count_nonzero(a["demand"].sum(axis=2)>a["capacity"].sum(axis=0)[:,None])),
            "mandatory_dispatch_customer_shortage_periods":[int(t+1) for t in range(H)
                    if not meta["model_flags"]["allow_idle"] and a["min_open"][a["period_to_interval"][t]]>min_active[t]]}


def _integer_option(value, label, *, minimum=1):
    from operator import index
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label} must be an integer, not a boolean')
    try:
        answer = index(value)
    except TypeError as exc:
        raise ValueError(f'{label} must be an integer') from exc
    if answer < minimum:
        raise ValueError(f'{label} must be at least {minimum}')
    return answer


def _verified_source_path(metadata, cfg, instance_path):
    """Resolve relocated local geometry only when its original hash matches."""
    source = metadata.get('source', {})
    expected = source.get('sha256')
    if not expected:
        raise ValueError('LRP generation recipe has no source geometry SHA256')
    raw = Path(cfg['source']).expanduser()
    root = Path(__file__).resolve().parents[2]
    candidates = ([raw] if raw.is_absolute() else
        [Path(instance_path) / raw, Path(instance_path).parent / raw, root / raw, root.parent / raw])
    name = str(source.get('name', ''))
    if name and Path(name).name == name:
        candidates += [root / 'data' / (name + '.tsp'), root / 'data' / (name + '.tsp.gz')]
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            _, _, _, digest = read_tsp(path)
        except (OSError, ValueError):
            continue
        if digest == expected:
            return path.resolve()
    raise ValueError('Cannot regenerate LRP data: original source geometry is missing or its SHA256 differs')


def _verify_recipe_arrays(data, regenerated):
    """Allow at most four arithmetic ULPs, never overwrite the stored values.

    Some supplied files were generated with NumPy 2.3.5 and differ by one ULP
    from 2.4.2 in setup costs/capacities. Check any saved original array hashes
    first, so even a one-ULP edit cannot masquerade as version compatibility.
    """
    expected = data.metadata.get('logical_sha256')
    if expected is not None and expected != data.logical_hash():
        raise ValueError('Stored LRP source logical hash differs; refusing to regenerate edited data')
    for key, expected in data.metadata.get('array_sha256', {}).items():
        if key not in data.arrays:
            raise ValueError(f'Stored LRP source array is missing: {key}')
        value = data.arrays[key]
        actual = hashlib.sha256(str(value.dtype).encode() + canonical(list(value.shape)).encode()
                                + value.tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Stored LRP source array hash differs for {key}; refusing to regenerate edited data')
    differences = sorted(set(regenerated) ^ set(data.arrays))
    rounding = {}
    for key in sorted(set(regenerated) & set(data.arrays)):
        new, old = regenerated[key], data.arrays[key]
        if new.dtype != old.dtype or new.shape != old.shape:
            differences.append(key)
            continue
        if np.array_equal(new, old):
            continue
        if old.dtype.kind != 'f':
            differences.append(key)
            continue
        changed = new != old
        delta = np.abs(new[changed] - old[changed])
        ulps = np.maximum(np.abs(np.spacing(new[changed])), np.abs(np.spacing(old[changed])))
        # Numeric zeros are model structure (e.g. forbidden/inactive service),
        # not roundoff candidates. No relative/absolute allclose tolerance.
        if (np.any(new[changed] == 0.) or np.any(old[changed] == 0.)
                or not np.isfinite(delta).all() or np.any(delta > 4. * ulps)):
            differences.append(key)
            continue
        rounding[key] = {'entries': int(np.count_nonzero(changed)),
                         'max_absolute_difference': float(delta.max()),
                         'max_ulps': float((delta / ulps).max())}
    if differences:
        raise ValueError('LRP generation recipe does not reproduce the stored source arrays: '
                         + ', '.join(differences) + '; refusing to replace edited data')
    return rounding


def _preserve_source_values(data, arrays, cfg, *, same_scenario_seed):
    """Anchor generated continuation to actual source values, not rounded copies."""
    old = data.arrays
    m, n, H, L, S = (int(cfg[key]) if key != 'location_periods' else len(cfg[key])
                     for key in ('facilities', 'customers', 'periods', 'location_periods', 'scenarios'))
    shared_h, shared_s = min(H, data.H), min(S, data.S)
    for key in ('facility_ids', 'customer_ids', 'facility_xy', 'customer_xy',
                'reference_mean_quantity', 'initial_state'):
        arrays[key] = np.array(old[key], copy=True)
    arrays['capacity'][:, :shared_h] = old['capacity'][:, :shared_h]
    arrays['reference_expected_total'][:shared_h] = old['reference_expected_total'][:shared_h]
    for t in range(data.H, H):
        arrays['capacity'][:, t] = old['capacity'][:, -1]
        arrays['reference_expected_total'][t] = old['reference_expected_total'][-1]
    for key in ('cost_fc', 'cost_cc'):
        arrays[key][:shared_h] = old[key][:shared_h]
    geom_key = [data.metadata['source']['sha256'], m, n, int(cfg.get('geometry_rep', 0))]
    master = int(cfg.get('seed', 20260920))
    ei, ej = np.triu_indices(m + n, 1)
    keep = ej >= m
    ei, ej = ei[keep], ej[keep]
    for t in range(data.H, H):
        previous = np.zeros((m + n, m + n))
        previous[:m, m:] = arrays['cost_fc'][t - 1]
        previous[m:, m:] = arrays['cost_cc'][t - 1]
        noise = seeded(master, 'route-increment', geom_key, t).uniform(.99, 1.01, len(ei))
        edges = 1.03 * previous[ei, ej] * (noise if cfg.get('cost_mode', 'paper_affine') == 'paper_affine' else 1.)
        current = np.zeros_like(previous)
        current[ei, ej] = edges
        current[ej, ei] = edges
        arrays['cost_fc'][t] = current[:m, m:]
        arrays['cost_cc'][t] = current[m:, m:]
    dates = list(map(int, arrays['location_periods']))
    old_dates = {int(date): k for k, date in enumerate(old['location_periods'])}
    for k, date in enumerate(dates):
        if date in old_dates:
            for key in ('opening_cost', 'continuation_cost', 'closing_cost'):
                arrays[key][:, k] = old[key][:, old_dates[date]]
        else:
            noise = seeded(master, 'facility-increment', geom_key, dates[k - 1], date).uniform(.95, 1.05, m)
            arrays['opening_cost'][:, k] = arrays['opening_cost'][:, k - 1] * (1.03 ** (date - dates[k - 1])) * noise
            arrays['continuation_cost'][:, k] = np.ceil(.2 * arrays['opening_cost'][:, k])
            arrays['closing_cost'][:, k] = np.ceil(.5 * arrays['opening_cost'][:, k])
    if same_scenario_seed:
        for key in ('active', 'demand'):
            arrays[key][:shared_h, :shared_s] = old[key][:shared_h, :shared_s]
    # Recompute quotes using the retained physical costs and quantities. Only
    # an unchanged random stream has an existing quote to preserve exactly.
    reference = 2 * arrays['cost_fc'].min(axis=1)
    arrays['outsourcing_cost'] = (float(cfg.get('outsource_multiplier', 1.))
        * reference[:, None, :] * (.5 + .5 * arrays['demand'] / arrays['reference_mean_quantity'])
        * arrays['active'])
    if same_scenario_seed:
        arrays['outsourcing_cost'][:shared_h, :shared_s] = old['outsourcing_cost'][:shared_h, :shared_s]
    if S == data.S:
        arrays['scenario_prob'] = np.array(old['scenario_prob'], copy=True)


def configure_horizon(data, *, source_path, T=None, num_scenarios=None,
                      scenario_seed=None, location_periods=None):
    """Apply public horizon settings without changing the source dataset.

    Prefix selection conditions the stored probability measure. Extension or
    resampling requires a verified recipe, allowing at most four arithmetic
    ULPs across NumPy versions while retaining the actual stored values.
    Arbitrary fixtures are never extrapolated. Geometry is generated at the
    original I/J sizes, before the caller selects physical node subsets.
    """
    from copy import deepcopy
    from core.problem_data import ProblemData
    from models.stage_model_core import Instance

    H = data.H if T is None else _integer_option(T, 'T')
    S = data.S if num_scenarios is None else _integer_option(num_scenarios, 'num_scenarios')
    seed = None if scenario_seed is None else _integer_option(scenario_seed, 'scenario_seed', minimum=0)
    retained_dates = [int(date) for date in data.arrays['location_periods'] if date <= H]
    dates = (retained_dates if location_periods is None else
             [_integer_option(date, 'location_periods') for date in location_periods])
    if not dates or dates[0] != 1 or any(a >= b for a, b in zip(dates, dates[1:])) or dates[-1] > H:
        raise ValueError('location_periods must start at 1, increase strictly, and lie within 1..T')
    regenerate = H > data.H or S > data.S or seed is not None or dates != retained_dates
    if not regenerate:
        return data.subset(T=H, num_scenarios=S)

    metadata = data.metadata
    original_cfg = metadata.get('config')
    if (metadata.get('generator_version') != VERSION or not isinstance(original_cfg, dict)
            or original_cfg.get('kind', 'stochastic') != 'stochastic'):
        raise ValueError('T/S expansion, scenario_seed and new location_periods require a reproducible LRP generation recipe; '
                         'this fixed instance supports only existing period/scenario prefixes')
    cfg = deepcopy(original_cfg)
    cfg['source'] = str(_verified_source_path(metadata, cfg, source_path))
    original, _ = generate(cfg, Path(source_path))
    rounding = _verify_recipe_arrays(data, original)

    cfg.update(periods=H, scenarios=S, location_periods=dates)
    if seed is not None:
        cfg['scenario_seed'] = seed
    if not np.isscalar(cfg.get('min_open', 0)):
        # A new opening date splits the old interval without changing its
        # minimum availability requirement; the last interval extends in time.
        old_dates = data.arrays['location_periods']
        cfg['min_open'] = [int(data.arrays['min_open'][np.searchsorted(old_dates, date, side='right') - 1])
                           for date in dates]
    name = f'{data.name}__T{H}_S{S}'
    if seed is not None:
        name += f'_Seed{seed}'
    if dates != retained_dates:
        name += '_LP' + '-'.join(map(str, dates))
    cfg['id'] = name
    arrays, generated_metadata = generate(cfg, Path(source_path))
    original_seed = int(original_cfg.get('scenario_seed', original_cfg.get('seed', 20260920)))
    _preserve_source_values(data, arrays, cfg, same_scenario_seed=seed is None or seed == original_seed)
    generated_metadata['validation'] = validate(arrays, generated_metadata)
    generated_metadata['configuration'] = {
        'source_name': data.name, 'source_logical_sha256': data.logical_hash(),
        'source_recipe_reproduced_exactly': not rounding,
        'source_recipe_float_ulp_differences': rounding,
        'source_numpy_version': metadata.get('numpy_version'),
        'source_values_preserved': 'stored fixed values and unchanged trajectory prefixes; only new entries generated',
        'scenario_seed_scope': 'activity and conditional quantity only; geometry/cost/capacity seed unchanged',
        'scenario_probability_policy': 'uniform_empirical_generated_trajectories',
        'location_period_policy': ('source dates; last interval extends' if location_periods is None
                                   else 'explicit dates; retain charges at source dates and generate new charges from preceding date'),
    }
    result = ProblemData(Instance(name, arrays, generated_metadata))
    result.metadata['logical_sha256'] = result.logical_hash()
    return result
