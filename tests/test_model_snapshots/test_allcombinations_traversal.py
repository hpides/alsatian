import unittest

from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge, \
    MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import LayerState

RELEASE_OUTPUT = 'release'


class TestAllCombinationsExample(unittest.TestCase):

    def setUp(self) -> None:
        self.node0 = MultiModelSnapshotNode(LayerState("", "", "Node 0", "Node 0"))
        self.node1 = MultiModelSnapshotNode(LayerState("", "", "Node 1", "Node 1"))
        self.node2 = MultiModelSnapshotNode(LayerState("", "", "Node 2", "Node 2"))
        self.node3 = MultiModelSnapshotNode(LayerState("", "", "Node 3", "Node 3"))
        self.node4 = MultiModelSnapshotNode(LayerState("", "", "Node 4", "Node 4"))
        self.node5 = MultiModelSnapshotNode(LayerState("", "", "Node 5", "Node 5"))
        self.node6 = MultiModelSnapshotNode(LayerState("", "", "Node 6", "Node 6"))
        self.node7 = MultiModelSnapshotNode(LayerState("", "", "Node 7", "Node 7"))
        self.node8 = MultiModelSnapshotNode(LayerState("", "", "Node 8", "Node 8"))
        self.node9 = MultiModelSnapshotNode(LayerState("", "", "Node 9", "Node 9"))
        self.node10 = MultiModelSnapshotNode(LayerState("", "", "Node 10", "Node 10"))

    def test_two_choice_example(self):
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

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 3 * 224 * 224)
        node_sequences = execution_tree.generate_all_traversals_nodes()
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            ('root', 'Node 2', 'Node 4', 'Node 6'),
            ('root', 'Node 2', 'Node 6', 'Node 4'),
        }

        self.assertSetEqual(computed_str_set, expected_str_set)

    def test_three_branch_example(self):
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

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 3 * 224 * 224)
        node_sequences = execution_tree.generate_all_traversals_nodes()
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            ('root', 'Node 2', 'Node 4', 'Node 6', 'Node 8'),
            ('root', 'Node 2', 'Node 4', 'Node 8', 'Node 6'),
            ('root', 'Node 2', 'Node 6', 'Node 4', 'Node 8'),
            ('root', 'Node 2', 'Node 6', 'Node 8', 'Node 4'),
            ('root', 'Node 2', 'Node 8', 'Node 4', 'Node 6'),
            ('root', 'Node 2', 'Node 8', 'Node 6', 'Node 4'),

        }

        self.assertSetEqual(computed_str_set, expected_str_set)

    def test_two_then_two_branch_example(self):
        # check test_two_then_two_branch_example.jpeg for better understanding

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

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 3 * 224 * 224)
        node_sequences = execution_tree.generate_all_traversals_nodes()
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            # first go to node 2
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 6', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 4', 'Node 10'),

            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 4', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 6', 'Node 4'),

            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 10', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 10', 'Node 4'),

            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 4', 'Node 6'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 6', 'Node 4'),


        }

        self.assertSetEqual(computed_str_set, expected_str_set)

    def test_two_then_three_branch_example(self):
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

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 3 * 224 * 224)
        node_sequences = execution_tree.generate_all_traversals_nodes()
        print('test')
        computed_str_set = self._to_string_set(node_sequences)
        expected_str_set = {
            # first go to node 2
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 6', 'Node 8', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 8', 'Node 6', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 4', 'Node 8', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 8', 'Node 4', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 4', 'Node 6', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 6', 'Node 4', 'Node 10'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 10', 'Node 4', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 8', 'Node 10', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 10', 'Node 6', 'Node 4'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 8', 'Node 4', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 10', 'Node 8', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 10', 'Node 8', 'Node 4'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 10', 'Node 4', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 6', 'Node 10', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 4', 'Node 10', 'Node 6', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 4', 'Node 6', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 6', 'Node 4', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 4', 'Node 10', 'Node 8'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 4', 'Node 8', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 8', 'Node 6', 'Node 4'),
            ('root', 'Node 1', 'Node 2', 'Node 10', 'Node 6', 'Node 8', 'Node 4'),
            ('root', 'Node 1', 'Node 2', 'Node 6', 'Node 8', 'Node 10', 'Node 4'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 4', 'Node 10', 'Node 6'),
            ('root', 'Node 1', 'Node 2', 'Node 8', 'Node 6', 'Node 10', 'Node 4'),
            # first go to node 9
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 4', 'Node 6', 'Node 8'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 4', 'Node 8', 'Node 6'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 6', 'Node 4', 'Node 8'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 6', 'Node 8', 'Node 4'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 8', 'Node 4', 'Node 6'),
            ('root', 'Node 1', 'Node 10', 'Node 2', 'Node 8', 'Node 6', 'Node 4'),
        }

        self.assertSetEqual(computed_str_set, expected_str_set)

    def _simplify(self, node_sequence):
        result = []
        for step in node_sequence:
            s = str(step).replace("Intermediate: ", "")
            split = s.split("-")
            result.append(split[0])

        return result

    def _to_string_set(self, node_sequences):
        result = set()
        for seq in node_sequences:
            simplified = self._simplify(seq)
            result.add(tuple(simplified))

        return result
