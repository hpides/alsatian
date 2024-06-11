import unittest

from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot, Release
from model_search.model_snapshots.dfs_traversal import RELEASE_OUTPUT
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge, \
    MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import LayerState


class TestDFSTraversal(unittest.TestCase):

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
        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot)
        node_sequence, edge_sequence = execution_tree.dfs_traversal()

        # ex_units = dfs_execution_plan(self.node0)
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
