import unittest

from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot, max_cost_of_node_sequence
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge, \
    MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import LayerState

RELEASE_OUTPUT = 'release'


class TestAllCombinationsExample(unittest.TestCase):

    def setUp(self) -> None:
        self.layer_state0 = LayerState("", "", "Node 0", "Node 0")
        self.layer_state0.output_size = 1
        self.node0 = MultiModelSnapshotNode(self.layer_state0)

        self.layer_state1 = LayerState("", "", "Node 1", "Node 1")
        self.layer_state1.output_size = 1
        self.node1 = MultiModelSnapshotNode(self.layer_state1)

        self.layer_state2 = LayerState("", "", "Node 2", "Node 2")
        self.layer_state2.output_size = 1
        self.node2 = MultiModelSnapshotNode(self.layer_state2)

        self.layer_state3 = LayerState("", "", "Node 3", "Node 3")
        self.layer_state3.output_size = 1
        self.node3 = MultiModelSnapshotNode(self.layer_state3)

        self.layer_state4 = LayerState("", "", "Node 4", "Node 4")
        self.layer_state4.output_size = 1
        self.node4 = MultiModelSnapshotNode(self.layer_state4)

        self.layer_state5 = LayerState("", "", "Node 5", "Node 5")
        self.layer_state5.output_size = 1
        self.node5 = MultiModelSnapshotNode(self.layer_state5)

        self.layer_state6 = LayerState("", "", "Node 6", "Node 6")
        self.layer_state6.output_size = 1
        self.node6 = MultiModelSnapshotNode(self.layer_state6)

        self.layer_state7 = LayerState("", "", "Node 7", "Node 7")
        self.layer_state7.output_size = 0
        self.node7 = MultiModelSnapshotNode(self.layer_state7)

        self.layer_state8 = LayerState("", "", "Node 8", "Node 8")
        self.layer_state8.output_size = 0
        self.node8 = MultiModelSnapshotNode(self.layer_state8)

        self.layer_state9 = LayerState("", "", "Node 9", "Node 9")
        self.layer_state9.output_size = 0
        self.node9 = MultiModelSnapshotNode(self.layer_state9)

        self.layer_state10 = LayerState("", "", "Node 10", "Node 10")
        self.layer_state10.output_size = 0
        self.node10 = MultiModelSnapshotNode(self.layer_state10)

        self.layer_state11 = LayerState("", "", "Node 11", "Node 11")
        self.layer_state11.output_size = 0
        self.node11 = MultiModelSnapshotNode(self.layer_state11)

    def test_two_choice_example_early_release_allowed(self):
        # check test_two_choice_example.jpeg for better understanding
        execution_tree = self._two_choice_execution_tree()

        node_sequences = execution_tree.generate_all_traversals_nodes(early_parent_release=True)

        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            (('root', 0), ('Node 2', 5), ('Node 4', 5), ('Node 6', 0)),
            (('root', 0), ('Node 2', 5), ('Node 6', 5), ('Node 4', 0)),
        }

        self.assertSetEqual(computed_str_set, expected_str_set)
        self.assertEqual(execution_tree.min_intermediate_cost_for_traversal(), 5)

        node_sequence, edge_sequence = execution_tree.dfs_traversal()
        max_cost = max_cost_of_node_sequence(node_sequence)
        self.assertEqual(max_cost, 5)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 5)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 5)

    def test_two_choice_example_max_costs(self):
        # check test_two_choice_example.jpeg for better understanding
        self._two_choice_execution_tree()
        # update costs
        self.layer_state2.output_size = 5
        self.layer_state4.output_size = 2
        self.layer_state6.output_size = 5
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 10)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 12)

    def _two_choice_execution_tree(self):
        # check test_two_choice_example.jpeg for better understanding
        edge0 = MultiModelSnapshotEdge("Edge 0-1", self.node0, self.node1)
        edge1 = MultiModelSnapshotEdge("Edge 1-2", self.node1, self.node2)
        edge2 = MultiModelSnapshotEdge("Edge 2-3", self.node2, self.node3)
        edge3 = MultiModelSnapshotEdge("Edge 3-4", self.node3, self.node4)
        edge4 = MultiModelSnapshotEdge("Edge 2-5", self.node2, self.node5)
        edge5 = MultiModelSnapshotEdge("Edge 5-6", self.node5, self.node6)
        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1])
        self.node2.edges.extend([edge2, edge4])
        self.node3.edges.extend([edge3])
        self.node5.edges.extend([edge5])

        # update costs
        self.layer_state2.output_size = 5
        self.layer_state4.output_size = 0
        self.layer_state6.output_size = 0
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

        return execution_tree

    def test_three_branch_example_early_parent_release_allowed(self):
        # check test_three_branch_example.jpeg for better understanding
        execution_tree = self._three_branch_execution_tree()

        node_sequences = execution_tree.generate_all_traversals_nodes(early_parent_release=True)

        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            (('root', 0), ('Node 2', 5), ('Node 4', 5), ('Node 6', 5), ('Node 8', 0)),
            (('root', 0), ('Node 2', 5), ('Node 4', 5), ('Node 8', 5), ('Node 6', 0)),
            (('root', 0), ('Node 2', 5), ('Node 6', 5), ('Node 4', 5), ('Node 8', 0)),
            (('root', 0), ('Node 2', 5), ('Node 6', 5), ('Node 8', 5), ('Node 4', 0)),
            (('root', 0), ('Node 2', 5), ('Node 8', 5), ('Node 4', 5), ('Node 6', 0)),
            (('root', 0), ('Node 2', 5), ('Node 8', 5), ('Node 6', 5), ('Node 4', 0)),

        }

        self.assertSetEqual(computed_str_set, expected_str_set)
        self.assertEqual(execution_tree.min_intermediate_cost_for_traversal(), 5)

        node_sequence, edge_sequence = execution_tree.dfs_traversal()
        max_cost = max_cost_of_node_sequence(node_sequence)
        self.assertEqual(max_cost, 5)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 5)

    def test_three_branch_example_max_costs(self):
        # check test_three_branch_example.jpeg for better understanding
        execution_tree = self._three_branch_execution_tree()
        # update costs
        self.layer_state2.output_size = 5
        self.layer_state4.output_size = 2
        self.layer_state6.output_size = 4
        self.layer_state8.output_size = 3
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 9)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 14)

    def _three_branch_execution_tree(self):
        # check test_three_branch_example.jpeg for better understanding
        edge0 = MultiModelSnapshotEdge("Edge 0-1", self.node0, self.node1)
        edge1 = MultiModelSnapshotEdge("Edge 1-2", self.node1, self.node2)
        edge2 = MultiModelSnapshotEdge("Edge 2-3", self.node2, self.node3)
        edge3 = MultiModelSnapshotEdge("Edge 3-4", self.node3, self.node4)
        edge4 = MultiModelSnapshotEdge("Edge 2-5", self.node2, self.node5)
        edge5 = MultiModelSnapshotEdge("Edge 5-6", self.node5, self.node6)
        edge6 = MultiModelSnapshotEdge("Edge 2-7", self.node2, self.node7)
        edge7 = MultiModelSnapshotEdge("Edge 7-8", self.node7, self.node8)
        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1])
        self.node2.edges.extend([edge2, edge4, edge6])
        self.node3.edges.extend([edge3])
        self.node5.edges.extend([edge5])
        self.node7.edges.extend([edge7])
        # update costs
        self.layer_state2.output_size = 5
        self.layer_state4.output_size = 0
        self.layer_state6.output_size = 0
        self.layer_state8.output_size = 0
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
        return execution_tree

    def test_two_then_two_branch_example_early_parent_release_allowed(self):
        # check test_two_then_three_branch_example.jpeg for better understanding
        execution_tree = self._two_then_two_branch_execution_tree()
        node_sequences = execution_tree.generate_all_traversals_nodes(early_parent_release=True)
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            # first go to node 2
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 4', 8), ('Node 6', 5), ('Node 10', 0)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 6', 8), ('Node 4', 5), ('Node 10', 0)),

            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 10', 3), ('Node 6', 3), ('Node 4', 0)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 10', 3), ('Node 4', 3), ('Node 6', 0)),

            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 4', 8), ('Node 10', 3), ('Node 6', 0)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 6', 8), ('Node 10', 3), ('Node 4', 0)),

            (('root', 0), ('Node 1', 5), ('Node 10', 5), ('Node 2', 3), ('Node 4', 3), ('Node 6', 0)),
            (('root', 0), ('Node 1', 5), ('Node 10', 5), ('Node 2', 3), ('Node 6', 3), ('Node 4', 0)),

        }

        self.assertSetEqual(computed_str_set, expected_str_set)
        self.assertEqual(execution_tree.min_intermediate_cost_for_traversal(early_parent_release=True), 5)

        node_sequence, edge_sequence = execution_tree.dfs_traversal()
        max_cost = max_cost_of_node_sequence(node_sequence)
        self.assertEqual(max_cost, 8)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 8)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 8)

    def test_two_then_two_branch_example_early_parent_release_not_allowed(self):
        # check test_two_then_three_branch_example.jpeg for better understanding
        execution_tree = self._two_then_two_branch_execution_tree()
        node_sequences = execution_tree.generate_all_traversals_nodes(early_parent_release=False)
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            # first go to node 2
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 4', 8), ('Node 6', 8), ('Node 10', 5)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 6', 8), ('Node 4', 8), ('Node 10', 5)),

            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 10', 8), ('Node 6', 3), ('Node 4', 3)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 10', 8), ('Node 4', 3), ('Node 6', 3)),

            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 4', 8), ('Node 10', 8), ('Node 6', 3)),
            (('root', 0), ('Node 1', 5), ('Node 2', 8), ('Node 6', 8), ('Node 10', 8), ('Node 4', 3)),

            (('root', 0), ('Node 1', 5), ('Node 10', 5), ('Node 2', 8), ('Node 4', 3), ('Node 6', 3)),
            (('root', 0), ('Node 1', 5), ('Node 10', 5), ('Node 2', 8), ('Node 6', 3), ('Node 4', 3)),

        }

        self.assertSetEqual(computed_str_set, expected_str_set)
        self.assertEqual(execution_tree.min_intermediate_cost_for_traversal(early_parent_release=False), 8)

        node_sequence, edge_sequence = execution_tree.dfs_traversal()
        max_cost = max_cost_of_node_sequence(node_sequence)
        self.assertEqual(max_cost, 8)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 8)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 8)

    def test_two_then_two_branch_example_max_costs(self):
        # check test_two_then_three_branch_example.jpeg for better understanding
        self._two_then_two_branch_execution_tree()
        # update costs
        self.layer_state1.output_size = 5
        self.layer_state2.output_size = 3
        self.layer_state4.output_size = 2
        self.layer_state6.output_size = 1
        self.layer_state10.output_size = 4
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

        execution_tree.annotate_intermediates_with_max_path_costs()
        acc_max_costs = execution_tree.root.accumulated_path_costs_without_persisting_leafs
        self.assertEqual(acc_max_costs, 10)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 12)

    def _two_then_two_branch_execution_tree(self):
        edge0 = MultiModelSnapshotEdge("Edge 0-1", self.node0, self.node1)
        edge1 = MultiModelSnapshotEdge("Edge 1-2", self.node1, self.node2)
        edge2 = MultiModelSnapshotEdge("Edge 2-3", self.node2, self.node3)
        edge3 = MultiModelSnapshotEdge("Edge 3-4", self.node3, self.node4)
        edge4 = MultiModelSnapshotEdge("Edge 2-5", self.node2, self.node5)
        edge5 = MultiModelSnapshotEdge("Edge 5-6", self.node5, self.node6)
        edge8 = MultiModelSnapshotEdge("Edge 1-9", self.node1, self.node9)
        edge9 = MultiModelSnapshotEdge("Edge 9-10", self.node9, self.node10)
        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1, edge8])
        self.node2.edges.extend([edge2, edge4])
        self.node3.edges.extend([edge3])
        self.node5.edges.extend([edge5])
        self.node9.edges.extend([edge9])
        # update costs
        self.layer_state1.output_size = 5
        self.layer_state2.output_size = 3
        self.layer_state4.output_size = 0
        self.layer_state6.output_size = 0
        self.layer_state10.output_size = 0
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
        return execution_tree

    def test_two_then_three_branch_example_early_parent_release_allowed(self):
        # check test_two_then_three_branch_example.jpeg for better understanding

        edge0 = MultiModelSnapshotEdge("Edge 0-1", self.node0, self.node1)
        edge1 = MultiModelSnapshotEdge("Edge 1-2", self.node1, self.node2)

        edge2 = MultiModelSnapshotEdge("Edge 2-3", self.node2, self.node3)
        edge3 = MultiModelSnapshotEdge("Edge 3-4", self.node3, self.node4)

        edge4 = MultiModelSnapshotEdge("Edge 2-5", self.node2, self.node5)
        edge5 = MultiModelSnapshotEdge("Edge 5-6", self.node5, self.node6)

        edge6 = MultiModelSnapshotEdge("Edge 2-7", self.node2, self.node7)
        edge7 = MultiModelSnapshotEdge("Edge 7-8", self.node7, self.node8)

        edge8 = MultiModelSnapshotEdge("Edge 1-9", self.node1, self.node9)
        edge9 = MultiModelSnapshotEdge("Edge 9-10", self.node9, self.node10)

        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1, edge8])
        self.node2.edges.extend([edge2, edge4, edge6])
        self.node3.edges.extend([edge3])
        self.node5.edges.extend([edge5])
        self.node7.edges.extend([edge7])
        self.node9.edges.extend([edge9])

        # update costs
        self.layer_state0.output_size = 0
        self.layer_state1.output_size = 0
        self.layer_state2.output_size = 0
        self.layer_state3.output_size = 0
        self.layer_state4.output_size = 0
        self.layer_state5.output_size = 0
        self.layer_state6.output_size = 0
        self.layer_state7.output_size = 0
        self.layer_state8.output_size = 0
        self.layer_state9.output_size = 0
        self.layer_state10.output_size = 0

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
        node_sequences = execution_tree.generate_all_traversals_nodes(early_parent_release=True)
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 6', 0), ('Node 8', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 8', 0), ('Node 6', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 4', 0), ('Node 8', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 8', 0), ('Node 4', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 4', 0), ('Node 6', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 6', 0), ('Node 4', 0), ('Node 10', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 10', 0), ('Node 4', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 8', 0), ('Node 10', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 10', 0), ('Node 6', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 8', 0), ('Node 4', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 10', 0), ('Node 8', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 10', 0), ('Node 8', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 10', 0), ('Node 4', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 6', 0), ('Node 10', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 4', 0), ('Node 10', 0), ('Node 6', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 4', 0), ('Node 6', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 6', 0), ('Node 4', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 4', 0), ('Node 10', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 4', 0), ('Node 8', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 8', 0), ('Node 6', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 10', 0), ('Node 6', 0), ('Node 8', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 6', 0), ('Node 8', 0), ('Node 10', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 4', 0), ('Node 10', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 2', 0), ('Node 8', 0), ('Node 6', 0), ('Node 10', 0), ('Node 4', 0)),

            # first go to node 10
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 4', 0), ('Node 6', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 4', 0), ('Node 8', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 6', 0), ('Node 4', 0), ('Node 8', 0)),
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 6', 0), ('Node 8', 0), ('Node 4', 0)),
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 8', 0), ('Node 4', 0), ('Node 6', 0)),
            (('root', 0), ('Node 1', 0), ('Node 10', 0), ('Node 2', 0), ('Node 8', 0), ('Node 6', 0), ('Node 4', 0)),

        }

        self.assertSetEqual(computed_str_set, expected_str_set)
        self.assertEqual(execution_tree.min_intermediate_cost_for_traversal(), 0)

    def test_two_then_two_branch_example_cheapest_path_first(self):
        # check test_two_then_two_branch_example.jpeg for better understanding
        # check test_two_then_three_branch_example.jpeg for better understanding
        self._two_then_two_branch_execution_tree()

        # update costs
        self.layer_state1.output_size = 5
        self.layer_state2.output_size = 3
        self.layer_state4.output_size = 2
        self.layer_state6.output_size = 1
        self.layer_state10.output_size = 4

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0

        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
        node_seq, edge_seq = execution_tree.cheapest_path_first_traversal()

        computed_seq = self._simplify_node_seq(node_seq)
        expected_seq = ['root', 'Node 1', 'RELEASE-root', 'Node 10', 'Node 2', 'RELEASE-Node 1', 'Node 6', 'Node 4',
                        'RELEASE-Node 2']
        self.assertListEqual(computed_seq, expected_seq)

        cost_seq, node_sequence, edge_sequence = \
            execution_tree.dfs_traversal(cheapest_path_first=True, return_costs=True)
        max_cost = max(cost_seq)
        self.assertEqual(max_cost, 12)

    def _simplify_node_seq(self, node_sequence):
        result = []
        for step in node_sequence:
            s = str(step).replace("Intermediate: ", "")
            split = s.split("-")
            if split[0] == 'RELEASE':
                result.append(f'{split[0]}-{split[1]}')
            else:
                result.append(split[0])

        return result

    def _simplify_node_cost_seq(self, node_sequence):
        result = []
        for step, cost in node_sequence:
            s = str(step).replace("Intermediate: ", "")
            split = s.split("-")
            result.append((split[0], cost))

        return result

    def _to_string_set(self, node_sequences):
        result = set()
        for seq in node_sequences:
            simplified = self._simplify_node_cost_seq(seq)
            result.add(tuple(simplified))

        return result
