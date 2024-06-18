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


def max_cost_of_node_sequence(node_sequence):
    current_cost = 0
    max_cost = 0
    while node_sequence:
        current_node = node_sequence.pop(0)
        if isinstance(current_node, Intermediate):
            current_cost += current_node.size
            # check if we can release sth
            while node_sequence and isinstance(node_sequence[0], Release):
                release = node_sequence.pop(0)
                current_cost -= release.intermediate.size

        if current_cost > max_cost:
            max_cost = current_cost

    return max_cost


class ExecutionTree:
    def __init__(self, root, edges):
        self.root: Intermediate = root
        self.edges: [Computation] = edges

    def dfs_traversal(self, cheapest_path_first=False):
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
                if len(children) > 0:
                    # if cheapest path first is activated -> sort children so that cheapest path is chosen first
                    if cheapest_path_first:
                        children.sort(key=lambda x: x[1].accumulated_path_costs, reverse=True)
                    else:
                        children.reverse()

                    stack.extend([Release(current_node)] + children)

        return node_sequence, edge_sequence

    def cheapest_path_first_traversal(self):
        self._annotate_intermediates_with_accumulated_path_costs()
        return self.dfs_traversal(cheapest_path_first=True)

    def _annotate_intermediates_with_accumulated_path_costs(self, start=None, parent_costs=0):
        if start is None:
            start = self.root

        # traverse down in DFS manner to annotate Intermediates
        children = [edge.output for edge in self.edges if edge.input == start]

        if len(children) == 0:
            start.accumulated_path_costs = parent_costs
            print('test')
        else:
            start.accumulated_path_costs = 0
            for child in children:
                self._annotate_intermediates_with_accumulated_path_costs(child, parent_costs + start.size)
                start.accumulated_path_costs += child.accumulated_path_costs
                print('test')

    def min_intermediate_cost_for_traversal(self):
        possible_traversals = self.generate_all_traversals_nodes()
        traversal_costs = []
        for traversal in possible_traversals:
            max_cost_on_path = max([cost for _, cost in traversal])
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
                new_traversal_order = current_traversal_order + [(choice, new_cost)]

                traversal_orders += self._all_traversals(new_choices, new_saved, new_released, new_traversal_order,
                                                         new_cost)

            return traversal_orders

    def best_traversal(self, available_budget):
        # TODO actually provide a real implementation that
        # returns a map with keys: max items per iteration, and values: the node_sequence and edge_sequences
        # for now just return on partition with "unlimited" budget/number of items

        result = {}
        node_sequence, edge_sequence = self.cheapest_path_first_traversal()
        result[128] = (node_sequence, edge_sequence)
        return result




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
