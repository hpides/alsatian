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

    def __hash__(self):
        return hash(self._id)


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

    def cost_of_node_sequence(self):
        #TODO
        pass

    def min_intermediate_cost_for_traversal(self):
        possible_traversals = self.generate_all_traversals_nodes()
        traversal_costs = []
        for traversal in possible_traversals:
            max_cost_on_path = max([cost for _,cost in traversal])
            traversal_costs.append(max_cost_on_path)
        return min(traversal_costs)


    def generate_all_traversals_nodes(self):
        result = self._all_traversals({self.root}, set(), set(), [], 0)
        return result

    def _all_traversals(self, choices, saved, released, current_traversal_order, current_cost):
        if len(choices) == 0:
            return [current_traversal_order]
        else:
            # cover all possible choices
            traversal_orders = []
            for choice in choices:
                # create a copy of the elements
                new_choices = choices.copy()
                new_saved = saved.copy()
                new_released = released.copy()

                # making the choice is equivalent to computing this intermediate, once the choice is made

                new_saved.add(choice)
                new_choices.remove(choice)

                # 1) add all children of the choice to the choices
                children = [edge.output for edge in self.edges if edge.input == choice]
                new_choices.update(children)

                # 2) see if making the choice allows us to release some intermediates from saved and move them to released
                # we can release nodes where all their children are in the new saved set
                release = set()
                for node in new_saved:
                    children = set([edge.output for edge in self.edges if edge.input == node])
                    saved_or_released = new_saved.union(new_released)
                    if children.issubset(saved_or_released):
                        release.add(node)
                # release them form saved and add them to released
                new_saved.difference_update(release)
                new_released.update(release)

                # update cost
                # 1) add the cost to save the new intermediate
                new_cost = current_cost + choice.size
                # 2) deduct the cost of the newly released intermediates
                new_cost -= sum([r.size for r in release])
                new_traversal_order = current_traversal_order + [(choice ,new_cost)]

                traversal_orders += self._all_traversals(new_choices, new_saved, new_released, new_traversal_order,
                                                         new_cost)

            return traversal_orders


def flatten_perms(permutations):
    result = []
    for perm in permutations:
        joined_lists = []
        for list_of_tuples in perm:
            joined_lists.append(sum(list_of_tuples, ()))
        result.append(sum(joined_lists, ()))

    return result


def _output_size(exec_unit: [MultiModelSnapshotEdge]):
    last_child = exec_unit[-1].child
    return last_child.layer_state.output_size


def execution_tree_from_mm_snapshot(mm_snapshot: MultiModelSnapshot, input_size) -> ExecutionTree:
    exec_tree_edges = []
    execution_units: [[MultiModelSnapshotEdge]] = []
    current_exec_unit: [MultiModelSnapshotEdge] = []

    exec_tree_root = Intermediate("root-input", input_size)
    stack = [(exec_tree_root, x) for x in reversed(mm_snapshot.root.edges)]

    while stack:
        current_intermediate, current_edge = stack.pop()

        current_exec_unit.append(current_edge)

        current_node = current_edge.child

        if len(current_node.edges) > 1 or len(current_node.edges) == 0:
            # if branch -> execution unit ends
            # add a new edge to the execution tree
            output_intermediate = Intermediate(_get_output_id(current_exec_unit), _output_size(current_exec_unit))
            computation = Computation(current_intermediate, output_intermediate, current_exec_unit)
            exec_tree_edges.append(computation)

            # and start a new execution unit
            current_exec_unit = _start_new_execution_unit(current_exec_unit, execution_units)
            new_intermediate = Intermediate(current_node.layer_state.id, current_node.layer_state.output_size)
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
