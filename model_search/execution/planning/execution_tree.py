from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot, MultiModelSnapshotEdge
from model_search.model_snapshots.rich_snapshot import LayerState, RichModelSnapshot


class Intermediate:
    def __init__(self, _id, size):
        self._id = _id
        self.size = size

    def __str__(self):
        return f"Intermediate: {self._id}"

    def __repr__(self):
        return self.__str__()


class Computation:

    def __init__(self, input, output, execution_unit):
        self.input: Intermediate = input
        self.output: Intermediate = output
        self.execution_unit = execution_unit
        self.duration = None


class Release:
    def __init__(self, intermediate):
        self.intermediate: Intermediate = intermediate

    def __str__(self):
        return f"RELEASE-{self.intermediate}"

    def __repr__(self):
        return self.__str__()


class ExecutionTree:
    def __init__(self, root, edges):
        self.root: Intermediate = root
        self.edges: [Computation] = edges

    def dfs_traversal(self):
        order = []
        stack = [self.root]
        while len(stack) > 0:
            current_node = stack.pop()
            order.append(current_node)
            if not isinstance(current_node, Release):
                # add release operation
                neighbors = [Release(current_node)]
                for edge in self.edges:
                    if edge.input == current_node:
                        neighbors.append(edge.output)
                stack += neighbors

        return order


def execution_tree_from_mm_snapshot(mm_snapshot: MultiModelSnapshot) -> ExecutionTree:
    exec_tree_edges = []
    execution_units: [[MultiModelSnapshotEdge]] = []
    current_exec_unit: [MultiModelSnapshotEdge] = []

    stack = [(Intermediate("input", 42), x) for x in reversed(mm_snapshot.root.edges)]
    exec_tree_root = Intermediate("input", 42)
    current_intermediate = None
    while stack:
        current_intermediate, current_edge = stack.pop()

        current_exec_unit.append(current_edge)

        current_node = current_edge.child

        if len(current_node.edges) > 1 or len(current_node.edges) == 0:
            # if branch -> execution unit ends
            # add a new edge to the execution tree
            computation = Computation(current_intermediate, _get_output_id(current_exec_unit), current_exec_unit)
            exec_tree_edges.append(computation)
            # and start a new execution unit
            current_exec_unit = _start_new_execution_unit(current_exec_unit, execution_units)
            stack += [(current_node.layer_state.id, x) for x in reversed(current_node.edges)]
        else:
            stack += [(current_intermediate, x) for x in reversed(current_node.edges)]

    # add last execution unit
    # computation = Computation(current_intermediate, _get_output_id(current_exec_unit), current_exec_unit)
    # exec_tree_edges.append(computation)



    return ExecutionTree(exec_tree_root, exec_tree_edges)


def _start_new_execution_unit(exec_unit, execution_units):
    execution_units.append(exec_unit)
    exec_unit = []
    return exec_unit


def _get_output_id(exec_unit: [MultiModelSnapshotEdge]):
    return exec_unit[-1].child.layer_state.id


if __name__ == '__main__':
    mm_snapshot = MultiModelSnapshot()
    layer_state_1_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
    layer_state_1_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
    layer_state_1_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
    layer_state_1_4 = LayerState("", "", "sd_hash_1.4", "arch_hash_1")
    layer_state_1_5 = LayerState("", "", "sd_hash_1.5", "arch_hash_1")
    layer_state_1_6 = LayerState("", "", "sd_hash_1.6", "arch_hash_1")
    model_1_layer_states = [layer_state_1_1, layer_state_1_2, layer_state_1_3,
                            layer_state_1_4, layer_state_1_5, layer_state_1_6]

    snapshot1 = RichModelSnapshot("test_arch1", "sd_path1", "sd_hash1", 'test_arch1-sd_path1',
                                  model_1_layer_states)

    layer_state_2_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
    layer_state_2_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
    layer_state_2_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
    layer_state_2_4 = LayerState("", "", "sd_hash_2.4", "arch_hash_1")
    layer_state_2_5 = LayerState("", "", "sd_hash_2.5", "arch_hash_1")
    layer_state_2_6 = LayerState("", "", "sd_hash_2.6", "arch_hash_1")
    model_2_layer_states = [layer_state_2_1, layer_state_2_2, layer_state_2_3,
                            layer_state_2_4, layer_state_2_5, layer_state_2_6]

    snapshot2 = RichModelSnapshot("test_arch2", "sd_path2", "sd_hash2", 'test_arch2-sd_path2',
                                  model_2_layer_states)

    mm_snapshot.add_snapshot(snapshot1)
    mm_snapshot.add_snapshot(snapshot2)

    res = execution_tree_from_mm_snapshot(mm_snapshot)

    print("test")
