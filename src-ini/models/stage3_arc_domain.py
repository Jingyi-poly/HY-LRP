"""Shared directed-arc domain for the open Stage-3 vehicle route."""


def stage3_route_arcs(nodes, depot_start, depot_end):
    """Return arcs for a route from ``depot_start`` to ``depot_end``.

    The two depot copies have different roles: no arc may enter the start
    depot or leave the end depot.  The empty start-to-end route is excluded
    because an activated vehicle must serve at least one assigned customer.
    """
    return [
        (i, j)
        for i in nodes
        for j in nodes
        if i != j
        and i != depot_end
        and j != depot_start
        and not (i == depot_start and j == depot_end)
    ]
