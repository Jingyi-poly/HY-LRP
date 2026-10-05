"""Reproducible instance and algorithm snapshots for solver state dumps."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = 1
_NON_SEMANTIC_ENV = {
    "VRP_EXPERIMENTS_ROOT",
    "VRP_PHASE2_DUMP_STATE_DIR",
    "VRP_COMPARE_LOG_SUFFIX",
    "LRP_EXPERIMENTS_ROOT",
    "LRP_PHASE2_DUMP_STATE_DIR",
    "LRP_COMPARE_LOG_SUFFIX",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, range)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _config_dict(config: Any) -> dict[str, Any]:
    raw = (dict(config) if isinstance(config, Mapping) else
           asdict(config) if is_dataclass(config) else vars(config))
    # JSON fingerprints reject nonfinite numbers. Preserve the original
    # unbounded per-solve option explicitly, without changing finite payloads.
    import math
    result = _jsonable(raw)
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            if value != math.inf:
                raise ValueError(f'Invalid nonfinite configuration value: {key}')
            result[key] = {'__config_float__': '+inf'}
    return result


def _decode_config_dict(raw):
    return {key: (float('inf') if value == {'__config_float__': '+inf'} else value)
            for key, value in raw.items()}


def attach_instance_snapshot(
    prob_data: Any,
    config: Any,
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach the complete build configuration to ``ProblemData``.

    Algorithms receive ``ProblemData`` rather than ``VRPInstance``.  Keeping
    this small, pickle-safe record on it lets every entry point emit the same
    reconstruction metadata without changing solver call signatures.
    """
    resolved = None
    if source_path is not None:
        try:
            resolved = Path(source_path).expanduser().resolve()
        except OSError:
            resolved = Path(source_path).expanduser()
    if hasattr(prob_data, "arrays") and hasattr(prob_data, "logical_hash"):
        source = {"requested": str(getattr(config, "instance", "")),
                  "resolved_path": str(resolved) if resolved is not None else None,
                  "sha256": None}
        if resolved is not None:
            from models.stage_model_core import load_instance
            if not resolved.is_dir():
                raise ValueError(f"LRP source must be an existing instance directory: {resolved}")
            source["files"] = {name: _sha256_file(resolved / name)
                               for name in ("arrays.npz", "metadata.json")}
            source["logical_sha256"] = load_instance(resolved).logical_hash()
            source["sha256"] = source["logical_sha256"]
        snapshot = {"config": _config_dict(config), "source": source,
                    "logical_sha256": prob_data.logical_hash()}
        setattr(prob_data, "_run_instance_snapshot", snapshot)
        return snapshot
    snapshot = {
        "config": _config_dict(config),
        "source": {
            "requested": str(getattr(config, "hfvrp_file", "")),
            "resolved_path": str(resolved) if resolved is not None else None,
            "sha256": _sha256_file(resolved) if resolved is not None else None,
        },
    }
    setattr(prob_data, "_run_instance_snapshot", snapshot)
    return snapshot


def _inferred_instance_snapshot(prob_data: Any, scen_tree: Mapping) -> dict[str, Any]:
    """Best-effort metadata for hand-built test instances."""
    return {
        "config": None,
        "source": {"requested": None, "resolved_path": None, "sha256": None},
        "inferred": {
            "C": int(getattr(prob_data, "numCustomers", len(getattr(prob_data, "J", ())))),
            "V": int(getattr(prob_data, "numVehicles", len(getattr(prob_data, "V", ())))),
            "T": int(getattr(prob_data, "T", 0)),
            "S": len(scen_tree.get(2, ())) // max(1, int(getattr(prob_data, "T", 1))),
        },
    }


@lru_cache(maxsize=4)
def git_state(repo_root: str) -> dict[str, Any]:
    """Return compact revision/dirty metadata without making Git required."""
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": head,
        "dirty": bool(status) if status is not None else None,
        "tracked_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status is not None
            else None
        ),
    }


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("fingerprint", None)
    encoded = json.dumps(
        _jsonable(canonical),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_run_snapshot(prob_data: Any, scen_tree: Mapping, algorithm_config: Any) -> dict[str, Any]:
    """Capture all inputs needed to identify and reconstruct one solver run."""
    if hasattr(prob_data, "arrays") and hasattr(prob_data, "logical_hash"):
        import copy
        import numpy as np
        from core.solver_settings import BACKWARD_S3_SELECTOR_CONTRACT, FORWARD_S2_SELECTOR_CONTRACT
        from models.stage_model_core import Instance
        data = Instance(str(prob_data.name),
                        {key: np.array(value, copy=True) for key, value in prob_data.arrays.items()},
                        copy.deepcopy(prob_data.metadata))
        data.validate()
        attached = copy.deepcopy(getattr(prob_data, "_run_instance_snapshot", {}))
        if attached and attached.get("logical_sha256") != data.logical_hash():
            raise ValueError("Attached LRP source/selection no longer describes the current data")
        instance = dict(attached, name=data.name, arrays=data.arrays, metadata=data.metadata,
                        array_dtypes={key: str(value.dtype) for key, value in data.arrays.items()},
                        logical_sha256=data.logical_hash())
        instance.setdefault("config", None)
        instance.setdefault("source", {"requested": None, "resolved_path": None, "sha256": None})
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION, "problem_type": "stochastic_lrp",
            "complete": True, "instance": instance,
            "dimensions": dict(zip(("I", "J", "T", "L", "S"), data.shape)),
            "semantic_model_fingerprint": data.logical_hash(),
            "algorithm_config": _config_dict(algorithm_config),
            "backward_s3_selector_contract": BACKWARD_S3_SELECTOR_CONTRACT,
            "forward_s2_selector_contract": FORWARD_S2_SELECTOR_CONTRACT,
            "algorithm_environment": {name: value for name, value in sorted(os.environ.items())
                if name.startswith(("LRP_", "VRP_")) and name not in _NON_SEMANTIC_ENV},
            "git": git_state(str(Path(__file__).resolve().parents[2])),
        }
        snapshot["fingerprint"] = canonical_fingerprint(snapshot)
        return snapshot
    from solvers.forward_period_dedup import forward_semantic_fingerprint

    try:
        semantic_fingerprint = forward_semantic_fingerprint(prob_data, scen_tree)
    except (AttributeError, KeyError, TypeError, ValueError):
        semantic_fingerprint = None
    instance = getattr(prob_data, "_run_instance_snapshot", None)
    complete = bool(instance and instance.get("config"))
    if not complete:
        instance = _inferred_instance_snapshot(prob_data, scen_tree)
    dimensions = {
        "C": int(getattr(prob_data, "numCustomers", len(getattr(prob_data, "J", ())))),
        "V": int(getattr(prob_data, "numVehicles", len(getattr(prob_data, "V", ())))),
        "S": len(scen_tree.get(2, ())) // max(1, int(getattr(prob_data, "T", 1))),
        "T": int(getattr(prob_data, "T", 0)),
    }
    environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("VRP_") and name not in _NON_SEMANTIC_ENV
    }
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "complete": complete,
        "instance": instance,
        "dimensions": dimensions,
        "semantic_model_fingerprint": semantic_fingerprint,
        "algorithm_config": _config_dict(algorithm_config),
        "algorithm_environment": environment,
        "git": git_state(str(repo_root)),
    }
    snapshot["fingerprint"] = canonical_fingerprint(snapshot)
    return snapshot


def restore_algorithm_environment(snapshot: Mapping[str, Any]) -> None:
    prefixes = ("LRP_", "VRP_") if snapshot.get("problem_type") == "stochastic_lrp" else ("VRP_",)
    if snapshot.get("problem_type") == "stochastic_lrp":
        _validate_lrp_snapshot(snapshot)
    saved = dict(snapshot.get("algorithm_environment", {}))
    if snapshot.get("problem_type") == "stochastic_lrp":
        from core.solver_settings import BACKWARD_S3_SELECTOR_CONTRACT, FORWARD_S2_SELECTOR_CONTRACT
        forward_contract = snapshot.get('forward_s2_selector_contract')
        if forward_contract is None:
            # Earlier LRP snapshots always executed compact Gurobi S2, even
            # when they recorded unused native selectors. Preserve replay.
            saved['LRP_S2_FORWARD_SOLVER'] = 'gurobi'
        elif forward_contract != FORWARD_S2_SELECTOR_CONTRACT:
            raise ValueError(f'Unknown forward S2 selector contract: {forward_contract}')
        contract = snapshot.get('backward_s3_selector_contract')
        if contract is None:
            # Old LRP bootstrap injected VRP zeros while both physical backward
            # consumers ignored these flags. Their LRP aliases were also unused.
            # Keep the recorded global selector and reproduce that old behavior.
            # Validate the original fingerprint above before filtering a copy.
            for prefix in ('VRP_', 'LRP_'):
                for phase in (1, 2):
                    saved.pop(f'{prefix}PHASE{phase}_S3_USE_ESP', None)
        elif contract != BACKWARD_S3_SELECTOR_CONTRACT:
            raise ValueError(f'Unknown backward S3 selector contract: {contract}')
    for name in tuple(os.environ):
        if (
            name.startswith(prefixes)
            and name not in _NON_SEMANTIC_ENV
            and name not in saved
        ):
            os.environ.pop(name, None)
    for name, value in saved.items():
        if str(name).startswith(prefixes) and name not in _NON_SEMANTIC_ENV:
            os.environ[str(name)] = str(value)


def algorithm_config_from_snapshot(snapshot: Mapping[str, Any]) -> Any:
    from core.solution import AlgorithmConfig

    if snapshot.get('problem_type') == 'stochastic_lrp':
        _validate_lrp_snapshot(snapshot)
        return AlgorithmConfig(**_decode_config_dict(snapshot.get('algorithm_config', {})))
    config = AlgorithmConfig()
    for name, value in _decode_config_dict(snapshot.get("algorithm_config", {})).items():
        setattr(config, name, value)
    return config


def _validate_lrp_snapshot(snapshot: Mapping[str, Any]):
    """Validate a saved LRP payload independently of an on-disk source."""
    import numpy as np
    from models.stage_model_core import Instance
    if snapshot.get("fingerprint") != canonical_fingerprint(snapshot):
        raise ValueError("LRP run snapshot fingerprint mismatch")
    saved = snapshot.get("instance", {})
    try:
        arrays = {key: np.asarray(value, dtype=saved["array_dtypes"][key])
                  for key, value in saved["arrays"].items()}
        data = Instance(saved["name"], arrays, saved["metadata"])
        data.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid physical LRP snapshot arrays") from exc
    logical = data.logical_hash()
    if logical != saved.get("logical_sha256") or logical != snapshot.get("semantic_model_fingerprint"):
        raise ValueError("LRP snapshot arrays differ from the saved logical hash")
    if snapshot.get("dimensions") != dict(zip(("I", "J", "T", "L", "S"), data.shape)):
        raise ValueError("LRP snapshot dimensions differ from its physical arrays")
    return data


def instance_config_from_snapshot(snapshot: Mapping[str, Any]) -> Any:
    """Restore the exact saved source and selection, verifying current data."""
    if snapshot.get("problem_type") == "stochastic_lrp":
        from core.instance import LRPConfig, LRPInstance
        from core.problem_data import ProblemData
        data = _validate_lrp_snapshot(snapshot)
        instance = snapshot["instance"]
        source = instance.get("source", {})
        raw, resolved_raw = instance.get("config"), source.get("resolved_path")
        if not raw or not resolved_raw:
            raise ValueError("In-memory LRP snapshot has no file-based instance configuration; "
                             "rebuild its Instance from the saved arrays and metadata")
        resolved = Path(resolved_raw)
        if not resolved.is_dir():
            raise ValueError(f"Saved LRP source directory is unavailable: {resolved}")
        for name in ("arrays.npz", "metadata.json"):
            expected = source.get("files", {}).get(name)
            if not expected or _sha256_file(resolved / name) != expected:
                raise ValueError(f"Saved LRP source file hash mismatch: {resolved / name}")
        original = ProblemData(resolved)
        if original.logical_hash() != source.get("logical_sha256"):
            raise ValueError("Saved LRP source logical hash mismatch")
        valid = {field.name for field in fields(LRPConfig)}
        kwargs = {name: value for name, value in raw.items() if name in valid}
        kwargs["instance"] = str(resolved)
        for key in ("customer_indices", "facility_indices", "location_periods"):
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        config = LRPConfig(**kwargs)
        selected = LRPInstance(config).build().prob_data
        if selected.logical_hash() != data.logical_hash():
            raise ValueError("Saved LRP source/selection does not reproduce the snapshot data")
        return config
    from core.instance import VRPConfig

    instance = snapshot.get("instance", {})
    raw = instance.get("config")
    if not raw:
        raise ValueError("state dump has no complete instance configuration")
    valid = {field.name for field in fields(VRPConfig)}
    kwargs = {name: value for name, value in raw.items() if name in valid}
    source = instance.get("source", {})
    resolved_raw = source.get("resolved_path")
    if kwargs.get("data_source") == "hfvrp" and resolved_raw:
        resolved = Path(resolved_raw)
        if resolved.exists():
            expected = source.get("sha256")
            actual = _sha256_file(resolved)
            if expected and actual != expected:
                raise ValueError(
                    f"saved instance file hash mismatch: {resolved} "
                    f"(expected {expected}, got {actual})"
                )
            kwargs["hfvrp_file"] = str(resolved)
    return VRPConfig(**kwargs)
