import gurobipy as gp
from gurobipy import GRB


def find_subtour(nodes, edges, depot_start, depot_end):
    """Find the shortest disconnected route component in ``O(|N|+|E|)``."""
    if not edges:
        return None

    node_set = set(nodes)
    adjacency = {node: set() for node in node_set}
    for i, j in edges:
        if i in node_set and j in node_set:
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = set()
    shortest_subtour = None
    for start in node_set:
        if start in visited or not adjacency[start]:
            continue
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        if depot_start in component and depot_end in component:
            continue
        if len(component) > 1 and (
            shortest_subtour is None
            or len(component) < len(shortest_subtour)
        ):
            shortest_subtour = list(component)
    return shortest_subtour


def subtour_elimination_callback(model, where):
    """
    Gurobi的lazy constraint回调函数,用于动态添加子回路消除约束

    约束形式:
        对于子回路S,添加约束:sum_{i in S, j in S, i != j} x[i,j] <= |S| - 1
    这个约束确保子回路S内的边数不足以形成闭环。

    模型必备属性:
        model._x: Gurobi的x[i,j]决策变量
        model._N: 所有节点的索引集合（支持非连续索引）
        model._depot_start: 起始仓库的节点索引
        model._depot_end: 终止仓库的节点索引

    Args:
        model (gurobipy.Model): Gurobi模型对象
        where (int): 回调阶段标识,GRB.Callback.MIPSOL表示找到整数解

    """
    if where == GRB.Callback.MIPSOL:
        # 获取当前整数解中x变量的值
        x_vals = model.cbGetSolution(model._x)

        N = model._N

        # 构建选中的边列表（x[i,j] > 0.5 表示边被选中）
        edges = []
        for i in N:
            for j in N:
                if (i, j) in x_vals and x_vals[i, j] > 0.5:
                    edges.append((i, j))

        # 检测是否存在子回路
        subtour = find_subtour(N, edges, model._depot_start, model._depot_end)

        if subtour is not None and len(subtour) < len(N):
            # 添加lazy constraint消除该子回路
            model.cbLazy(
                gp.quicksum(model._x[i, j] for i in subtour for j in subtour
                            if i != j and (i, j) in model._x)
                <= len(subtour) - 1
            )


def optimize_with_callback(model):
    """Solve LRP's explicit connectivity or the existing lazy SEC model."""
    if (getattr(model, '_lrp_all_cuts_explicit', False)
            or getattr(model, '_lrp_learned_cut_dispatch', None) == 'native_attribute'):
        # LRP connectivity and native learned-cut pool need no Python callback.
        # Its own-root r variables do not use the old start/end-depot callback.
        model.optimize()
        return
    model.optimize(subtour_elimination_callback)
