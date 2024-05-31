import unittest

from model_search.model_snapshots.dfs_traversal import dfs_execution_plan, RELEASE_OUTPUT
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge


class TestDFSTraversal(unittest.TestCase):

    def setUp(self) -> None:
        self.node0 = MultiModelSnapshotNode(layer_state="Node 0")
        self.node1 = MultiModelSnapshotNode(layer_state="Node 1")
        self.node2 = MultiModelSnapshotNode(layer_state="Node 2")
        self.node3 = MultiModelSnapshotNode(layer_state="Node 3")
        self.node4 = MultiModelSnapshotNode(layer_state="Node 4")
        self.node5 = MultiModelSnapshotNode(layer_state="Node 5")
        self.node6 = MultiModelSnapshotNode(layer_state="Node 6")
        self.node7 = MultiModelSnapshotNode(layer_state="Node 7")
        self.node8 = MultiModelSnapshotNode(layer_state="Node 8")
        self.node9 = MultiModelSnapshotNode(layer_state="Node 9")

    def test_simple_example(self):
        # check dfs_traversal_example.jpeg for better understanding

        self.node0.snapshot_ids = [1, 2, 3]
        self.node1.snapshot_ids = [1, 2, 3]
        self.node2.snapshot_ids = [1, 2]
        self.node3.snapshot_ids = [3]
        self.node4.snapshot_ids = [2]
        self.node5.snapshot_ids = [1]
        self.node6.snapshot_ids = [3]
        self.node7.snapshot_ids = [3]
        self.node8.snapshot_ids = [2]
        self.node9.snapshot_ids = [1]

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

        # Connect edges to nodes
        self.node0.edges.extend([edge0])
        self.node1.edges.extend([edge1, edge2])
        self.node2.edges.extend([edge4, edge3])
        self.node3.edges.extend([edge5, edge6])
        self.node4.edges.append(edge7)
        self.node5.edges.append(edge8)

        ex_units = dfs_execution_plan(self.node0)
        expected_ex_units = [
            [edge0],
            [edge1],
            [edge4, edge8],
            [edge3, edge7],
            (RELEASE_OUTPUT, self.node2),
            [edge2, edge5, edge6],
            (RELEASE_OUTPUT, self.node3),
            (RELEASE_OUTPUT, self.node1)
        ]

        self.assertEqual(expected_ex_units, ex_units)
