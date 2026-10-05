"""A short physical-policy probe before a full-fleet refinement search."""
from contextlib import contextmanager
import math


def refresh_probe_limit(pd, fleet, *, sub_time_limit, budget, schedule,
                        inner_rounds, has_checkpoint):
    """Return a per-node cap only for a complete, exactly binary final fleet.

    Partial-fleet experiments did not benefit from probing. Neither fleet
    identity nor the original solver's optimality status is approximated here.
    """
    if (not has_checkpoint or inner_rounds <= 0 or budget is None
            or budget.remaining() <= 0 or schedule.rescore_only
            or sub_time_limit is None):
        return None
    limit = float(sub_time_limit)
    if not math.isfinite(limit) or limit <= 10.0:
        return None
    if not pd.V or pd.T <= 0:
        return None
    if not all(fleet.get(f"z[{v},{pd.T - 1}]") == 1.0 for v in pd.V):
        return None
    return 10.0


@contextmanager
def preserve_stall_hints(tables):
    """Undo only probe-created stall hints, never certificates or policies.

    A ten-second probe must not prevent the subsequent original-budget solve.
    Existing stall hints remain in effect, and exact/epsilon-closed table hits
    remain available without another MIP. Source and copied nodes may share a
    state object, so restoration uses copy-on-write for the changed records.
    """
    before = {
        node: {counts: record.get("stalled_at")
               for counts, record in state.get("records", {}).items()}
        for node, state in tables.items() if state is not None
    }
    try:
        yield
    finally:
        for node, state in list(tables.items()):
            if state is None:
                continue
            changed = {}
            for counts, record in state.get("records", {}).items():
                old = before.get(node, {}).get(counts)
                if record.get("stalled_at") != old:
                    changed[counts] = {**record, "stalled_at": old}
            if changed:
                tables[node] = {**state, "records": {**state["records"], **changed}}
