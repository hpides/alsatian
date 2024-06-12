import unittest

from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot, Release, \
    max_cost_of_node_sequence
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge, \
    MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import LayerState

RELEASE_OUTPUT = 'release'


class TestDFSTraversal(unittest.TestCase):

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

    def test_simple_example(self):
        # check dfs_traversal_example.jpeg for better understanding

        # Create edges
        # model 1
        edge0 = MultiModelSnapshotEdge("Edge 0-1", self.node0, self.node1)
        edge1 = MultiModelSnapshotEdge("Edge 1-2", self.node1, self.node2)
        edge4 = MultiModelSnapshotEdge("Edge 2-5", self.node2, self.node5)
        edge8 = MultiModelSnapshotEdge("Edge 5-9", self.node5, self.node9)

        # model 2
        edge3 = MultiModelSnapshotEdge("Edge 2-4", self.node2, self.node4)
        edge7 = MultiModelSnapshotEdge("Edge 4-8", self.node4, self.node8)

        # model 3
        edge2 = MultiModelSnapshotEdge("Edge 1-3", self.node1, self.node3)
        edge5 = MultiModelSnapshotEdge("Edge 3-6", self.node3, self.node6)
        edge6 = MultiModelSnapshotEdge("Edge 6-7", self.node3, self.node7)

        # model 4
        edge9 = MultiModelSnapshotEdge("Edge 1-10", self.node1, self.node10)

        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1, edge2])
        self.node2.edges.extend([edge4, edge3])
        self.node3.edges.extend([edge5])
        self.node4.edges.append(edge7)
        self.node5.edges.append(edge8)
        self.node6.edges.append(edge6)
        self.node1.edges.append(edge9)

        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.root = self.node0
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
        node_sequence, edge_sequence = execution_tree.dfs_traversal()

        ex_units = []
        for e in edge_sequence:
            if e is not None:
                if isinstance(e, Release):
                    ex_units.append((RELEASE_OUTPUT, e.intermediate._id))
                else:
                    ex_units.append(e.execution_unit)

        expected_ex_units = [
            [edge0],
            [edge1],
            [edge4, edge8],
            [edge3, edge7],
            (RELEASE_OUTPUT, self.node2.layer_state.id),
            [edge2, edge5, edge6],
            [edge9],
            (RELEASE_OUTPUT, self.node1.layer_state.id),
            (RELEASE_OUTPUT, "root-input")
        ]

        self.assertEqual(expected_ex_units, ex_units)

        max_cost = max_cost_of_node_sequence(node_sequence)
        self.assertEqual(max_cost, 2)
