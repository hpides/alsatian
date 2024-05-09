from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge


def dfs_execution_plan(mm_root: MultiModelSnapshotNode) -> [MultiModelSnapshotEdge]:
    # returns an execution order represented as a list of edges which represent the execution of their parent node
    execution_units: [[MultiModelSnapshotEdge]] = []
    current_exec_unit: [MultiModelSnapshotEdge] = []
    stack = list(reversed(mm_root.edges))
    while stack:
        current_edge = stack.pop()

        if not current_edge.parent.snapshot_ids == current_edge.child.snapshot_ids:
            # we start new execution unit once the next added node is not on a linear path
            execution_units.append(current_exec_unit)
            current_exec_unit = []

        current_exec_unit.append(current_edge)

        current_node = current_edge.child
        stack += list(reversed(current_node.edges))

    # add the final exec unit
    execution_units.append(current_exec_unit)
    return execution_units
