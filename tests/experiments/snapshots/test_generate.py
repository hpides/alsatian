import unittest

from custom.models.init_models import initialize_model
from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.model_names import RESNET_18
from global_utils.model_operations import state_dict_equal, split_model_in_two

ID_1 = 'id1'


class TestGenerateSnapshots(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_with_hardcoded_layer_indices(self):
        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [5, 7, 9]
        split_idxs = [len(pre_trained_model) - i for i in retrain_idxs]
        snaps = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, retrain_idxs, use_same_base=True)

        self.assertTrue(state_dict_equal(pre_trained_model.state_dict(), snaps[0].state_dict()))

        for i in range(1, 4):
            self.assertFalse(state_dict_equal(pre_trained_model.state_dict(), snaps[i].state_dict()))
            self._compare_base_and_new(pre_trained_model, split_idxs, snaps, i)

    def _compare_base_and_new(self, pre_trained_model, split_idxs, snaps, idx):
        base_first, base_second = split_model_in_two(pre_trained_model, split_idxs[idx - 1])
        new_first, new_second = split_model_in_two(snaps[idx], split_idxs[idx - 1])
        self.assertTrue(state_dict_equal(base_first.state_dict(), new_first.state_dict()))
        self.assertFalse(state_dict_equal(base_second.state_dict(), new_second.state_dict()))
