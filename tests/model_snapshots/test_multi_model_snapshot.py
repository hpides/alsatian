import unittest

from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState

ID_1 = 'id1'


class TestTensorCachingService(unittest.TestCase):
    def setUp(self):
        self.layer_state_1_1 = LayerState("", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_1_2 = LayerState("", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_1_3 = LayerState("", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_1_4 = LayerState("", "sd_hash_1.4", "arch_hash_1")
        self.layer_state_1_5 = LayerState("", "sd_hash_1.5", "arch_hash_1")
        self.layer_state_1_6 = LayerState("", "sd_hash_1.6", "arch_hash_1")

        self.layer_state_2_1 = LayerState("", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_2_2 = LayerState("", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_2_3 = LayerState("", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_2_4 = LayerState("", "sd_hash_2.4", "arch_hash_1")
        self.layer_state_2_5 = LayerState("", "sd_hash_2.5", "arch_hash_1")
        self.layer_state_2_6 = LayerState("", "sd_hash_2.6", "arch_hash_1")

    def tearDown(self):
        pass

    def test_add_snapshot_to_empty_mm(self):
        mm_snapshot = MultiModelSnapshot()
        layer_states = [self.layer_state_1_1, self.layer_state_1_2, self.layer_state_1_3, self.layer_state_1_4,
                        self.layer_state_1_5, self.layer_state_1_6]
        snapshot = RichModelSnapshot("test_arch", "sd_path", "sd_hash", layer_states)

        mm_snapshot.add_snapshot(snapshot)

        current_node = mm_snapshot.root
        for i in range(6):
            self.assertEqual(mm_snapshot.root.model_ids, ["test_arch-sd_path"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, layer_states[i])
            current_node = current_node.edges[0].child

        # last node has no child
        self.assertEqual(len(current_node.edges), 0)

    def test_add_snapshot_to_one_model_mm(self):
        # add the first model
        mm_snapshot = MultiModelSnapshot()
        model_1_layer_states = [self.layer_state_1_1, self.layer_state_1_2, self.layer_state_1_3, self.layer_state_1_4,
                                self.layer_state_1_5, self.layer_state_1_6]
        snapshot = RichModelSnapshot("test_arch1", "sd_path1", "sd_hash1", model_1_layer_states)
        mm_snapshot.add_snapshot(snapshot)

        # add the second model
        model_2_layer_states = [self.layer_state_2_1, self.layer_state_2_2, self.layer_state_2_3, self.layer_state_2_4,
                                self.layer_state_2_5, self.layer_state_2_6]
        snapshot = RichModelSnapshot("test_arch2", "sd_path2", "sd_hash2", model_2_layer_states)
        mm_snapshot.add_snapshot(snapshot)

        current_node = mm_snapshot.root
        # for the first three layers no children because merged, afterward split
        layer_count = 0
        for i in range(layer_count, layer_count + 3):
            self.assertEqual(mm_snapshot.root.model_ids, ["test_arch1-sd_path1", "test_arch2-sd_path2"])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, model_1_layer_states[i])
            current_node = current_node.edges[0].child
            layer_count += 1

        # now current node should be split point
        self.assertEqual(len(current_node.edges), 2)
        self.assertEqual(current_node.model_ids, ["test_arch1-sd_path1", "test_arch2-sd_path2"])
        self.assertEqual(current_node.edges[0].child.layer_state, model_1_layer_states[layer_count])
        child_nodes = [edge.child for edge in current_node.edges]
        layer_count += 1

        current_node = child_nodes[0]
        model_id = "test_arch1-sd_path1"
        layer_states = model_1_layer_states
        self._check_tail(current_node, layer_count, layer_states, model_id)

        current_node = child_nodes[1]
        model_id = "test_arch2-sd_path2"
        layer_states = model_2_layer_states
        self._check_tail(current_node, layer_count, layer_states, model_id)

    def _check_tail(self, current_node, layer_count, layer_states, model_id):
        for i in range(layer_count, layer_count + 2):
            self.assertEqual(current_node.model_ids, [model_id])
            # test only one child
            self.assertEqual(len(current_node.edges), 1)
            # test if referenced node has correct hashes
            self.assertEqual(current_node.edges[0].child.layer_state, layer_states[i])
            current_node = current_node.edges[0].child
        # last node has no child
        self.assertEqual(len(current_node.edges), 0)
