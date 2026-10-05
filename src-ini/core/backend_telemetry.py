"""Observe actual backend calls without changing solver contracts or budgets.

Only ``backend_call`` increments solve counts. Cache hits, dispatch decisions,
fallbacks and skipped work are separate events. Worker tuples keep their shape
and carry a pickle-safe snapshot for the parent to merge exactly once.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import math
import os
import threading
import time
import uuid


_COLLECTOR = ContextVar("backend_telemetry_collector", default=None)
_SCOPE = ContextVar("backend_telemetry_scope", default={})
_DIMENSIONS = ("phase", "path", "stage", "backend", "operation", "kind",
               "outcome", "status", "reason", "attempt_kind")
_MODEL_FLAGS = ("bound_available", "incumbent_available", "optimal_status")


def _safe(value):
    """Restrict diagnostics to portable JSON values (including finite numbers)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    try:
        scalar = value.item()
    except (AttributeError, TypeError, ValueError):
        return str(type(value).__name__)
    return _safe(scalar)


def _finite_attribute(model, name):
    try:
        value = float(getattr(model, name))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _model_outcome(model):
    if model is None:
        return {}
    result = {}
    for source, target in (("Status", "status"), ("SolCount", "solution_count")):
        value = _finite_attribute(model, source)
        if value is not None:
            result[target] = int(value) if value.is_integer() else value
    bound = _finite_attribute(model, "ObjBound")
    # These are observed solver attributes, not Python policy/bound certification.
    result["bound_available"] = bound is not None
    solutions = result.get("solution_count")
    result["incumbent_available"] = None if solutions is None else solutions > 0
    status = result.get("status")
    result["optimal_status"] = None if status is None else status == 2
    if bound is not None:
        result["objective_bound"] = bound
    if result.get("solution_count", 0) > 0:
        value = _finite_attribute(model, "ObjVal")
        if value is not None:
            result["objective_value"] = value
    return result


class BackendTelemetryCollector:
    """Per-process aggregate; public snapshots contain no live solver objects."""

    def __init__(self):
        self.owner_pid = os.getpid()
        self.snapshot_id = uuid.uuid4().hex
        self._rows = {}
        self._merged = set()
        self._lock = threading.RLock()

    def record(self, context, outcome, *, kind, wall_seconds=0.0):
        context, outcome = _safe(context), _safe(outcome)
        dimensions = {key: context.get(key) for key in _DIMENSIONS}
        dimensions.update(kind=kind,
                          outcome=outcome.get("outcome", "completed" if kind == "call" else kind),
                          status=outcome.get("status"),
                          reason=outcome.get("reason", context.get("reason")))
        metadata = {key: value for key, value in context.items() if key not in _DIMENSIONS}
        metrics = {}
        flags = {}
        for key, value in outcome.items():
            if key in ("outcome", "status", "reason"):
                continue
            if isinstance(value, bool) or key in _MODEL_FLAGS:
                flag = "unknown" if value is None else "true" if value else "false"
                flags[key] = {name: int(name == flag) for name in ("true", "false", "unknown")}
            elif value is None:
                continue
            elif isinstance(value, (int, float)):
                metrics[key] = {"min": value, "max": value, "last": value}
            else:
                metadata[key] = value
        row = dict(dimensions, calls=int(kind == "call"), events=int(kind != "call"),
                   wall_seconds=max(0.0, float(wall_seconds)), metadata=metadata,
                   varying_metadata_fields=[], metrics=metrics, flag_counts=flags)
        with self._lock:
            self._merge_row(row)

    def _merge_row(self, row):
        key = json.dumps([row.get(name) for name in _DIMENSIONS], sort_keys=True)
        if key not in self._rows:
            self._rows[key] = _safe(row)
            return
        target = self._rows[key]
        before_count = target["calls"] + target["events"]
        incoming_count = row.get("calls", 0) + row.get("events", 0)
        for name in set(target["flag_counts"]) | set(row.get("flag_counts", {})):
            counts = target["flag_counts"].setdefault(name, {"true": 0, "false": 0, "unknown": before_count})
            incoming = row.get("flag_counts", {}).get(name, {"true": 0, "false": 0, "unknown": incoming_count})
            for flag in ("true", "false", "unknown"):
                counts[flag] += incoming.get(flag, 0)
        for name in ("calls", "events", "wall_seconds"):
            target[name] += row.get(name, 0)
        varying = set(target["varying_metadata_fields"]) | set(row.get("varying_metadata_fields", ()))
        metadata = row.get("metadata", {})
        for name in set(target["metadata"]) | set(metadata):
            if name not in target["metadata"] or name not in metadata or target["metadata"][name] != metadata[name]:
                varying.add(name)
        for name in varying:
            target["metadata"].pop(name, None)
        target["varying_metadata_fields"] = sorted(varying)
        for name, values in row.get("metrics", {}).items():
            if name not in target["metrics"]:
                target["metrics"][name] = dict(values)
            else:
                metric = target["metrics"][name]
                metric.update(min=min(metric["min"], values["min"]),
                              max=max(metric["max"], values["max"]), last=values["last"])

    def merge(self, snapshot):
        if not isinstance(snapshot, dict):
            return False
        token = snapshot.get("snapshot_id")
        if not token:
            return False
        with self._lock:
            if token == self.snapshot_id or token in self._merged:
                return False
            self._merged.add(token)
            for row in snapshot.get("records", ()):
                self._merge_row(row)
        return True

    def snapshot(self):
        with self._lock:
            records = [_safe(self._rows[key]) for key in sorted(self._rows)]
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot_id,
            "calls": sum(row["calls"] for row in records),
            "events": sum(row["events"] for row in records),
            # Sum of backend wall times, not elapsed parallel phase duration.
            "wall_seconds": sum(row["wall_seconds"] for row in records),
            "records": records,
        }


def _current_collector():
    collector = _COLLECTOR.get()
    # A fork inherits ContextVars but must not reuse the parent's aggregates.
    return collector if collector is not None and collector.owner_pid == os.getpid() else None


@contextmanager
def backend_scope(phase=None, path=None, stage=None, operation=None, **metadata):
    values = dict(_SCOPE.get())
    values.update({key: value for key, value in dict(
        phase=phase, path=path, stage=stage, operation=operation, **metadata).items()
                   if value is not None})
    token = _SCOPE.set(values)
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def backend_call(backend, operation=None, *, model=None, **metadata):
    """Place immediately around one actual native/optimize call, not a wrapper."""
    collector = _current_collector()
    outcome = {}
    if collector is None:
        yield outcome
        return
    context = dict(_SCOPE.get(), backend=backend)
    context.update(metadata)
    if operation is not None:
        context["operation"] = operation
    context.setdefault("operation", "solve")
    started = time.perf_counter()
    try:
        yield outcome
    except BaseException as exc:
        outcome["outcome"] = "exception"
        outcome.setdefault("reason", type(exc).__name__)
        raise
    finally:
        seconds = time.perf_counter() - started
        observed = _model_outcome(model)
        observed.update(outcome)
        collector.record(context, observed, kind="call", wall_seconds=seconds)


def record_backend_event(backend, operation, reason, **metadata):
    """A non-solve event; no real backend call is inferred from a model or plan."""
    collector = _current_collector()
    if collector is None:
        return
    context = dict(_SCOPE.get(), backend=backend, operation=operation)
    context.update(metadata)
    kind = metadata.get("event_type", metadata.get("kind", operation))
    if kind == "call":
        kind = "event"  # Even an event named "call" is not a native solve.
    outcome = {"reason": reason}
    for key, value in metadata.items():
        if key in ("status", "outcome") or (
            key not in _DIMENSIONS and isinstance(value, (bool, int, float))
        ):
            outcome[key] = value
            if key not in _DIMENSIONS:
                context.pop(key, None)
    collector.record(context, outcome, kind=str(kind))


class TelemetryTuple(tuple):
    """Tuple contents/length stay unchanged; statistics travel as attributes."""

    def __new__(cls, values, backend_telemetry=None):
        result = super().__new__(cls, values)
        result.backend_telemetry = backend_telemetry
        return result

    def __reduce__(self):
        return type(self), (tuple(self), self.backend_telemetry)


def merge_worker_result(result):
    """Merge one worker snapshot at most once and return its unchanged result."""
    collector = _current_collector()
    if collector is not None:
        collector.merge(getattr(result, "backend_telemetry", None))
    return result


def capture_backend_task(phase, path, stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _current_collector() is not None:
                with backend_scope(phase=phase, path=path, stage=stage):
                    return function(*args, **kwargs)
            collector = BackendTelemetryCollector()
            token = _COLLECTOR.set(collector)
            try:
                with backend_scope(phase=phase, path=path, stage=stage):
                    result = function(*args, **kwargs)
                if isinstance(result, tuple):
                    return TelemetryTuple(result, collector.snapshot())
                return result
            except BaseException as exc:
                # Exception attributes survive normal multiprocessing pickling.
                try:
                    exc.backend_telemetry = collector.snapshot()
                except Exception:
                    pass
                raise
            finally:
                _COLLECTOR.reset(token)
        return wrapped
    return decorate


def print_backend_summary(snapshot, *, phase=None):
    """No-op; telemetry detail stays in the returned snapshot dict only."""
    return


def backend_checkpoint(label):
    """Return the cumulative snapshot without console output.

    Repeated checkpoints never reset the collector or add solver calls.
    """
    collector = _current_collector()
    if collector is None:
        return None
    return collector.snapshot()


def capture_backend_run(phase):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            collector = BackendTelemetryCollector()
            token = _COLLECTOR.set(collector)
            result = None
            try:
                with backend_scope(phase=phase):
                    result = function(*args, **kwargs)
                return result
            except BaseException as exc:
                # Include a failed worker's calls when its exception reaches us.
                merge_worker_result(exc)
                raise
            finally:
                snapshot = collector.snapshot()
                if isinstance(result, dict):
                    result["backend_telemetry"] = snapshot
                _COLLECTOR.reset(token)
        return wrapped
    return decorate


__all__ = ["BackendTelemetryCollector", "TelemetryTuple", "backend_scope",
           "backend_call", "record_backend_event", "capture_backend_run",
           "capture_backend_task", "merge_worker_result", "print_backend_summary",
           "backend_checkpoint"]
