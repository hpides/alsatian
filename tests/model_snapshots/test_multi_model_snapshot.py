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
