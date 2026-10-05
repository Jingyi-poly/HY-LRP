"""Three computational LRP layers, with only two information stages."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from models.stage_model_core import NodeContext


@dataclass
class NodeType:
    time: int
    index: int
    info: Any
    predecessor: int | None
    successor: list[int] = field(default_factory=list)
    probability: dict[int, float] = field(default_factory=dict)
    multi_coeff: float = 1.0
    context: NodeContext | None = None

    @property
    def active(self):
        return None if self.context is None else self.context.active

    @property
    def volume(self):
        return None if self.context is None else self.context.demand

    @property
    def c_out(self):
        return None if self.context is None else self.context.outsourcing

    @property
    def information_stage(self):
        return 1 if self.time == 1 else 2


def build_scenario_tree(prob_data):
    """S1 facilities -> S2 (scenario, period) -> S3 facility route.

    Root weights are scenario probabilities repeated over delivery periods;
    their sum is T, because period costs are summed. S3 costs add with weight 1.
    """
    root = NodeType(1, 0, None, None)
    tree = {1: [root], 2: [], 3: []}
    for s in range(prob_data.S):
        p = float(prob_data.scenario_prob[s])
        for t in range(prob_data.H):
            q = s * prob_data.H + t
            ctx = NodeContext.from_instance(prob_data.instance, t, s)
            second = NodeType(2, q, (s, t), 0, multi_coeff=p, context=ctx)
            root.successor.append(q)
            root.probability[q] = p
            tree[2].append(second)
            for i in prob_data.I:
                r = q * prob_data.m + i
                tree[3].append(NodeType(3, r, i, q, multi_coeff=p, context=ctx))
                second.successor.append(r)
                second.probability[r] = 1.0
    validate_operating_weights(prob_data, tree)
    return tree


def validate_operating_weights(prob_data, scen_tree):
    """Fail closed when a tree changes the intended objective or node scope."""
    if set(scen_tree) != {1, 2, 3} or len(scen_tree[1]) != 1:
        raise ValueError("LRP needs one facility root and three computational layers")
    root = scen_tree[1][0]
    if root.index != 0 or root.time != 1 or root.predecessor is not None:
        raise ValueError("Invalid LRP root")
    m, n, H, L, S = prob_data.shape
    if len(scen_tree[2]) != H * S or len(scen_tree[3]) != H * S * m:
        raise ValueError("Missing or duplicate LRP recourse nodes")
    if set(root.successor) != set(range(H * S)) or len(root.successor) != H * S:
        raise ValueError("Root successors must cover each scenario-period once")
    if set(root.probability) != set(root.successor):
        raise ValueError("Root probability keys do not match its successors")
    audit = []
    for q, second in enumerate(scen_tree[2]):
        s, t = divmod(q, H)
        expected = float(prob_data.scenario_prob[s])
        context = NodeContext.from_instance(prob_data.instance, t, s)
        if second.index != q or second.info != (s, t) or second.predecessor != 0 or second.time != 2:
            raise ValueError("Invalid scenario-period node ordering")
        if second.context is None or second.context.key != context.key:
            raise ValueError("Scenario-period node has incorrect data context")
        if not math.isfinite(expected) or root.probability[q] != expected or second.multi_coeff != expected:
            raise ValueError("Recourse weight must be the unscaled scenario probability")
        routes = list(range(q * m, (q + 1) * m))
        if second.successor != routes or set(second.probability) != set(routes):
            raise ValueError("Each recourse node must contain one route per facility")
        for i, r in enumerate(routes):
            third = scen_tree[3][r]
            if (third.index, third.time, third.info, third.predecessor) != (r, 3, i, q):
                raise ValueError("Invalid facility-route node ordering")
            if third.context is None or third.context.key != context.key:
                raise ValueError("Route node must share its assignment node context")
            if second.probability[r] != 1.0 or third.multi_coeff != expected:
                raise ValueError("Facility routes add with weight one within each recourse node")
        audit.append({"node": q, "scenario": s, "period": t, "interval": context.interval,
                      "scenario_probability": expected, "multi_coeff": expected})
    return audit
