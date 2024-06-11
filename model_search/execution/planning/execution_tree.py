from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot, MultiModelSnapshotEdge


class Intermediate:
    def __init__(self, _id, size):
        self._id = _id
        self.size = size

    def __str__(self):
        return f"Intermediate: {self._id}, size: {self.size}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self._id == other._id


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
        node_sequence = []
        edge_sequence = []
        stack = [[None, self.root]]
        while len(stack) > 0:
            current_item = stack.pop()

            if isinstance(current_item, Release):
                edge_sequence.append(current_item)
                node_sequence.append(current_item)
            else:
                current_edge, current_node = current_item
                node_sequence.append(current_node)
                edge_sequence.append(current_edge)
                # Add release operation
                children = []
                for edge in self.edges:
                    if edge.input == current_node:
                        children.append([edge, edge.output])
                children.reverse()
                if len(children) > 0:
                    stack.extend([Release(current_node)] + children)

        return node_sequence, edge_sequence


def execution_tree_from_mm_snapshot(mm_snapshot: MultiModelSnapshot) -> ExecutionTree:
    exec_tree_edges = []
    execution_units: [[MultiModelSnapshotEdge]] = []
    current_exec_unit: [MultiModelSnapshotEdge] = []

    exec_tree_root = Intermediate("input", 42)  # TODO fix the sizes
    stack = [(exec_tree_root, x) for x in reversed(mm_snapshot.root.edges)]

    while stack:
        current_intermediate, current_edge = stack.pop()

        current_exec_unit.append(current_edge)

        current_node = current_edge.child

        if len(current_node.edges) > 1 or len(current_node.edges) == 0:
            # if branch -> execution unit ends
            # add a new edge to the execution tree
            input_intermediate = current_intermediate
            output_intermediate = Intermediate(_get_output_id(current_exec_unit), 42)  # TODO fix size
            computation = Computation(current_intermediate, output_intermediate, current_exec_unit)
            exec_tree_edges.append(computation)
            # and start a new execution unit
            current_exec_unit = _start_new_execution_unit(current_exec_unit, execution_units)
            new_intermediate = Intermediate(current_node.layer_state.id, 42)  # TODO fix the sizes
            stack += [(new_intermediate, x) for x in reversed(current_node.edges)]
        else:
            stack += [(current_intermediate, x) for x in reversed(current_node.edges)]

    return ExecutionTree(exec_tree_root, exec_tree_edges)


def _start_new_execution_unit(exec_unit, execution_units):
    execution_units.append(exec_unit)
    exec_unit = []
    return exec_unit


def _get_output_id(exec_unit: [MultiModelSnapshotEdge]):
    return exec_unit[-1].child.layer_state.id
