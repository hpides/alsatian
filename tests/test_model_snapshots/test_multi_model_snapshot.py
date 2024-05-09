import unittest

from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState

ID_1 = 'id1'


class TestTensorCachingService(unittest.TestCase):
    def setUp(self):
        self.layer_state_1_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_1_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_1_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_1_4 = LayerState("", "", "sd_hash_1.4", "arch_hash_1")
        self.layer_state_1_5 = LayerState("", "", "sd_hash_1.5", "arch_hash_1")
        self.layer_state_1_6 = LayerState("", "", "sd_hash_1.6", "arch_hash_1")
        self.model_1_layer_states = [self.layer_state_1_1, self.layer_state_1_2, self.layer_state_1_3,
                                     self.layer_state_1_4, self.layer_state_1_5, self.layer_state_1_6]

        self.snapshot1 = RichModelSnapshot("test_arch1", "sd_path1", "sd_hash1", 'test_arch1-sd_path1', self.model_1_layer_states)

        self.layer_state_2_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_2_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_2_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_2_4 = LayerState("", "", "sd_hash_2.4", "arch_hash_1")
        self.layer_state_2_5 = LayerState("", "", "sd_hash_2.5", "arch_hash_1")
        self.layer_state_2_6 = LayerState("", "", "sd_hash_2.6", "arch_hash_1")
        self.model_2_layer_states = [self.layer_state_2_1, self.layer_state_2_2, self.layer_state_2_3,
                                     self.layer_state_2_4, self.layer_state_2_5, self.layer_state_2_6]
        self.snapshot2 = RichModelSnapshot("test_arch2", "sd_path2", "sd_hash2", 'test_arch2-sd_path2', self.model_2_layer_states)

        self.layer_state_3_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_3_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_3_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_3_4 = LayerState("", "", "sd_hash_2.4", "arch_hash_1")
        self.layer_state_3_5 = LayerState("", "", "sd_hash_3.5", "arch_hash_1")
        self.layer_state_3_6 = LayerState("", "", "sd_hash_3.6", "arch_hash_1")
        self.model_3_layer_states = [self.layer_state_3_1, self.layer_state_3_2, self.layer_state_3_3,
                                     self.layer_state_3_4, self.layer_state_3_5, self.layer_state_3_6]
        self.snapshot3 = RichModelSnapshot("test_arch3", "sd_path3", "sd_hash3", 'test_arch3-sd_path3', self.model_3_layer_states)

        self.layer_state_4_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_4_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_4_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_4_4 = LayerState("", "", "sd_hash_4.4", "arch_hash_1")
        self.layer_state_4_5 = LayerState("", "", "sd_hash_4.5", "arch_hash_1")
        self.layer_state_4_6 = LayerState("", "", "sd_hash_4.6", "arch_hash_1")
        self.model_4_layer_states = [self.layer_state_4_1, self.layer_state_4_2, self.layer_state_4_3,
                                     self.layer_state_4_4, self.layer_state_4_5, self.layer_state_4_6]
        self.snapshot4 = RichModelSnapshot("test_arch4", "sd_path4", "sd_hash4", 'test_arch4-sd_path4', self.model_4_layer_states)

    def tearDown(self):
        pass

    def test_add_snapshot_to_empty_mm(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)

        current_node = mm_snapshot.root
        for i in range(6):
            self.assertEqual(mm_snapshot.root.snapshot_ids, ["test_arch1-sd_path1"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[i])
            current_node = current_node.edges[0].child

        # last node has no child
        self.assertEqual(len(current_node.edges), 0)

    def test_add_snapshot_to_one_model_mm(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        mm_snapshot.add_snapshot(self.snapshot2)

        self._check_for_snapshot_1_and_2(mm_snapshot)

    def _check_for_snapshot_1_and_2(self, mm_snapshot):
        current_node = mm_snapshot.root
        # for the first three layers no children because merged, afterward split
        layer_count = 0
        for i in range(layer_count, layer_count + 3):
            self.assertEqual(mm_snapshot.root.snapshot_ids, ["test_arch1-sd_path1", "test_arch2-sd_path2"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[i])
            current_node = current_node.edges[0].child
            layer_count += 1
        # now current node should be split point
        self.assertEqual(len(current_node.edges), 2)
        self.assertEqual(current_node.snapshot_ids, ["test_arch1-sd_path1", "test_arch2-sd_path2"])
        self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[layer_count])
        child_nodes = [edge.child for edge in current_node.edges]
        layer_count += 1
        current_node = child_nodes[0]
        snapshot_ids = ["test_arch1-sd_path1"]
        layer_states = self.model_1_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)
        current_node = child_nodes[1]
        snapshot_ids = ["test_arch2-sd_path2"]
        layer_states = self.model_2_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)

    def _check_tail(self, current_node, layer_start, num_layers, layer_states, snapshot_ids):
        for i in range(layer_start, layer_start + num_layers - 1):
            self.assertEqual(current_node.snapshot_ids, snapshot_ids)
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, layer_states[i])
            current_node = current_node.edges[0].child
        # last node has no child
        self.assertEqual(len(current_node.edges), 0)

    def test_add_same_snapshot_twice(self):
        # add the first model
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        # add the second model
        snapshot = RichModelSnapshot("test_arch2", "sd_path2", "sd_hash2", "test_arch2-sd_path2", self.model_1_layer_states)
        mm_snapshot.add_snapshot(snapshot)

        current_node = mm_snapshot.root
        snapshot_ids = ["test_arch1-sd_path1", "test_arch2-sd_path2"]
        layer_states = self.model_1_layer_states
        self._check_tail(current_node, 0, 7, layer_states, snapshot_ids)

    def test_add_snapshot_to_two_model_mm_different_split_point(self):
        # add the first model
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)

        # add the second model
        mm_snapshot.add_snapshot(self.snapshot2)

        # add the third model

        mm_snapshot.add_snapshot(self.snapshot3)

        self._check_for_snapshots_1_2_3(mm_snapshot)

    def _check_for_snapshots_1_2_3(self, mm_snapshot):
        current_node = mm_snapshot.root
        # for the first three layers no children because merged, afterward split
        layer_count = 0
        for i in range(layer_count, layer_count + 3):
            self.assertEqual(mm_snapshot.root.snapshot_ids,
                             ["test_arch1-sd_path1", "test_arch2-sd_path2", "test_arch3-sd_path3"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[i])
            current_node = current_node.edges[0].child
            layer_count += 1
        # now current node should be split point
        self.assertEqual(len(current_node.edges), 2)
        self.assertEqual(current_node.snapshot_ids,
                         ["test_arch1-sd_path1", "test_arch2-sd_path2", "test_arch3-sd_path3"])
        self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[layer_count])
        self.assertEqual(current_node.edges[1].child.layer_state, self.model_2_layer_states[layer_count])
        child_nodes = [edge.child for edge in current_node.edges]
        layer_count += 1
        current_node = child_nodes[0]
        snapshot_ids = ["test_arch1-sd_path1"]
        layer_states = self.model_1_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)
        current_node = child_nodes[1]
        self.assertEqual(len(current_node.edges), 2)
        self.assertEqual(current_node.snapshot_ids, ["test_arch2-sd_path2", "test_arch3-sd_path3"])
        self.assertEqual(current_node.edges[0].child.layer_state, self.model_2_layer_states[layer_count])
        self.assertEqual(current_node.edges[1].child.layer_state, self.model_3_layer_states[layer_count])
        child_nodes = [edge.child for edge in current_node.edges]
        layer_count += 1
        current_node = child_nodes[0]
        snapshot_ids = ["test_arch2-sd_path2"]
        layer_states = self.model_2_layer_states
        self._check_tail(current_node, layer_count, 2, layer_states, snapshot_ids)
        current_node = child_nodes[1]
        snapshot_ids = ["test_arch3-sd_path3"]
        layer_states = self.model_3_layer_states
        self._check_tail(current_node, layer_count, 2, layer_states, snapshot_ids)

    def test_add_snapshot_to_two_model_mm_same_split_point(self):
        # add the first model
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)

        # add the second model
        mm_snapshot.add_snapshot(self.snapshot2)

        # add the third model

        mm_snapshot.add_snapshot(self.snapshot4)

        current_node = mm_snapshot.root
        # for the first three layers no children because merged, afterward split
        layer_count = 0
        for i in range(layer_count, layer_count + 3):
            self.assertEqual(mm_snapshot.root.snapshot_ids,
                             ["test_arch1-sd_path1", "test_arch2-sd_path2", "test_arch4-sd_path4"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[i])
            current_node = current_node.edges[0].child
            layer_count += 1

        # now current node should be split point
        self.assertEqual(len(current_node.edges), 3)
        self.assertEqual(current_node.snapshot_ids,
                         ["test_arch1-sd_path1", "test_arch2-sd_path2", "test_arch4-sd_path4"])
        self.assertEqual(current_node.edges[0].child.layer_state, self.model_1_layer_states[layer_count])
        child_nodes = [edge.child for edge in current_node.edges]
        layer_count += 1

        current_node = child_nodes[0]
        snapshot_ids = ["test_arch1-sd_path1"]
        layer_states = self.model_1_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)

        current_node = child_nodes[1]
        snapshot_ids = ["test_arch2-sd_path2"]
        layer_states = self.model_2_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)

        current_node = child_nodes[2]
        snapshot_ids = ["test_arch4-sd_path4"]
        layer_states = self.model_4_layer_states
        self._check_tail(current_node, layer_count, 3, layer_states, snapshot_ids)

    def test_add_1_2_3_prune_3(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        mm_snapshot.add_snapshot(self.snapshot2)
        self._check_for_snapshot_1_and_2(mm_snapshot)

        mm_snapshot.add_snapshot(self.snapshot3)
        mm_snapshot.prune_snapshot(self.snapshot3.id)
        self._check_for_snapshot_1_and_2(mm_snapshot)

    def test_add_1_2_4_prune_4(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        mm_snapshot.add_snapshot(self.snapshot2)
        self._check_for_snapshot_1_and_2(mm_snapshot)

        mm_snapshot.add_snapshot(self.snapshot3)
        mm_snapshot.prune_snapshot(self.snapshot3.id)
        self._check_for_snapshot_1_and_2(mm_snapshot)

        mm_snapshot.add_snapshot(self.snapshot4)
        mm_snapshot.prune_snapshot(self.snapshot4.id)
        self._check_for_snapshot_1_and_2(mm_snapshot)

    def test_add_1_2_3_4_prune_4(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        mm_snapshot.add_snapshot(self.snapshot2)
        mm_snapshot.add_snapshot(self.snapshot3)
        self._check_for_snapshots_1_2_3(mm_snapshot)

        mm_snapshot.add_snapshot(self.snapshot4)
        mm_snapshot.prune_snapshot(self.snapshot4.id)
        self._check_for_snapshots_1_2_3(mm_snapshot)
