from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge

RELEASE = "release"


# def dfs_execution_plan(mm_root: MultiModelSnapshotNode) -> [MultiModelSnapshotEdge]:
#     # returns an execution order represented as a list of edges which represent the execution of their parent node
#     execution_units: [[MultiModelSnapshotEdge]] = []
#     current_exec_unit: [MultiModelSnapshotEdge] = []
#     stack = list(reversed(mm_root.edges))
#     while stack:
#         current_edge = stack.pop()
#
#         if not current_edge.parent.snapshot_ids == current_edge.child.snapshot_ids:
#             # we start new execution unit once the next added node is not on a linear path
#             # which means all nodes on that linear path belong to the same set of snapshots
#             # -> check if snapshot ids are identical, if identical no need to start a new execution unit
#
#             execution_units.append(current_exec_unit)
#             current_exec_unit = []
#
#         current_exec_unit.append(current_edge)
#
#         current_node = current_edge.child
#
#         stack += list(reversed(current_node.edges))
#
#     # add the final exec unit
#     execution_units.append(current_exec_unit)
#     return execution_units

def dfs_execution_plan(mm_root: MultiModelSnapshotNode) -> [MultiModelSnapshotEdge]:
    # returns an execution order represented as a list of edges which represent the execution of their parent node
    execution_units: [[MultiModelSnapshotEdge]] = []
    current_exec_unit: [MultiModelSnapshotEdge] = []

    if len(mm_root.edges) > 1:
        additional_nodes = [RELEASE]
    else:
        additional_nodes = []
    stack = list(reversed(mm_root.edges + additional_nodes))
    while stack:

        if stack[-1] == RELEASE:
            current_exec_unit = _start_new_execution_unit(current_exec_unit, execution_units)
            execution_units.append('RELEASE')
            stack.pop()
        else:
            current_edge = stack.pop()
            if len(current_exec_unit) > 0 and not current_edge.parent.snapshot_ids == current_edge.child.snapshot_ids:
                # we start new execution unit once the next added node is not on a linear path
                # which means all nodes on that linear path belong to the same set of snapshots
                # -> check if snapshot ids are identical, if identical no need to start a new execution unit
                current_exec_unit = _start_new_execution_unit(current_exec_unit, execution_units)

            current_exec_unit.append(current_edge)

            current_node = current_edge.child

            if len(current_node.edges) > 1:
                additional_nodes = [RELEASE]
            else:
                additional_nodes = []

            stack += list(reversed(current_node.edges + additional_nodes))

    # add the final exec unit
    execution_units.append(current_exec_unit)
    return execution_units


def _start_new_execution_unit(exec_unit, execution_units):
    if len(exec_unit) > 0:
        execution_units.append(exec_unit)
        exec_unit = []
    return exec_unit
