import unittest

from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot
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

        self.snapshot1 = RichModelSnapshot("test_arch1", "sd_path1", "sd_hash1", 'test_arch1-sd_path1',
                                           self.model_1_layer_states)

        self.layer_state_2_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_2_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_2_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_2_4 = LayerState("", "", "sd_hash_2.4", "arch_hash_1")
        self.layer_state_2_5 = LayerState("", "", "sd_hash_2.5", "arch_hash_1")
        self.layer_state_2_6 = LayerState("", "", "sd_hash_2.6", "arch_hash_1")
        self.model_2_layer_states = [self.layer_state_2_1, self.layer_state_2_2, self.layer_state_2_3,
                                     self.layer_state_2_4, self.layer_state_2_5, self.layer_state_2_6]
        self.snapshot2 = RichModelSnapshot("test_arch2", "sd_path2", "sd_hash2", 'test_arch2-sd_path2',
                                           self.model_2_layer_states)

        self.layer_state_3_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_3_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_3_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_3_4 = LayerState("", "", "sd_hash_2.4", "arch_hash_1")
        self.layer_state_3_5 = LayerState("", "", "sd_hash_3.5", "arch_hash_1")
        self.layer_state_3_6 = LayerState("", "", "sd_hash_3.6", "arch_hash_1")
        self.model_3_layer_states = [self.layer_state_3_1, self.layer_state_3_2, self.layer_state_3_3,
                                     self.layer_state_3_4, self.layer_state_3_5, self.layer_state_3_6]
        self.snapshot3 = RichModelSnapshot("test_arch3", "sd_path3", "sd_hash3", 'test_arch3-sd_path3',
                                           self.model_3_layer_states)

        self.layer_state_4_1 = LayerState("", "", "sd_hash_1.1", "arch_hash_1")
        self.layer_state_4_2 = LayerState("", "", "sd_hash_1.2", "arch_hash_1")
        self.layer_state_4_3 = LayerState("", "", "sd_hash_1.3", "arch_hash_1")
        self.layer_state_4_4 = LayerState("", "", "sd_hash_1.4", "arch_hash_1")
        self.layer_state_4_5 = LayerState("", "", "sd_hash_4.5", "arch_hash_1")
        self.layer_state_4_6 = LayerState("", "", "sd_hash_4.6", "arch_hash_1")
        self.model_4_layer_states = [self.layer_state_4_1, self.layer_state_4_2, self.layer_state_4_3,
                                     self.layer_state_4_4, self.layer_state_4_5, self.layer_state_4_6]
        self.snapshot4 = RichModelSnapshot("test_arch4", "sd_path4", "sd_hash4", 'test_arch4-sd_path4',
                                           self.model_4_layer_states)

    def tearDown(self):
        pass

    def test_add_snapshot_to_empty_mm(self):
        mm_snapshot = MultiModelSnapshot()
        mm_snapshot.add_snapshot(self.snapshot1)
        mm_snapshot.add_snapshot(self.snapshot2)
        mm_snapshot.add_snapshot(self.snapshot3)
        mm_snapshot.add_snapshot(self.snapshot4)

        exec_tree = execution_tree_from_mm_snapshot(mm_snapshot)

        # expected execution tree in picture: expected_execution_tree.jpeg
        self.assertEqual(len(exec_tree.edges), 7)
        all_outputs_expected = {"arch_hash_1-sd_hash_1.3", "arch_hash_1-sd_hash_1.4", "arch_hash_1-sd_hash_2.4",
                                "arch_hash_1-sd_hash_1.6", "arch_hash_1-sd_hash_2.6", "arch_hash_1-sd_hash_3.6",
                                "arch_hash_1-sd_hash_4.6"}
        all_outputs_actual = set([])
        for edge in exec_tree.edges:
            splits = str(edge.output).split("-")
            simplified_out = splits[0] + '-' + splits[1]
            all_outputs_actual.add(simplified_out.replace("Intermediate: ", ""))

        self.assertSetEqual(all_outputs_expected, all_outputs_actual)
