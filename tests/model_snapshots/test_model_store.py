import random
import unittest

import numpy as np
import torch

from custom.models.init_models import initialize_model
from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.model_names import RESNET_18
from global_utils.model_operations import state_dict_equal
from model_search.model_management.model_store import ModelStore


class TestModelStore(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Define test save path
        self.save_path = '/Users/nils/uni/programming/model-search-paper/tmp_dir'

        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [5, 7, 9]
        split_idxs = [len(pre_trained_model) - i for i in retrain_idxs]
        save_path = '/Users/nils/uni/programming/model-search-paper/tmp_dir'
        self.snapshots = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, save_path=save_path,
                                            retrain_idxs=retrain_idxs, use_same_base=True)

        self.model_store = ModelStore(self.save_path)

    def test_get_model(self):
        self.model_store.add_snapshot(self.snapshots[0])
        retrieved_snapshot = self.model_store.get_snapshot(self.snapshots[0].id)
        self.assertEqual(self.snapshots[0], retrieved_snapshot)

    def test_get_composed_model(self):
        # # Test whether get_composed_model method returns a sequential model
        # layer_state_ids = [list(self.model_store.layers.keys())[0]]
        # composed_model = self.model_store.get_composed_model(layer_state_ids)
        # self.assertIsInstance(composed_model, torch.nn.Sequential)

    def test_add_snapshot(self):
        # # Test whether add_snapshot method correctly adds a snapshot to the model store
        # self.assertEqual(len(self.model_store.models), 1)
        # self.assertEqual(len(self.model_store.layers), 1)


if __name__ == '__main__':
    unittest.main()
