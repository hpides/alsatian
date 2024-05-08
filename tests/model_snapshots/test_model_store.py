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
        self.save_path = '/mount-fs/tmp-dir'

        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [5, 7, 9]
        self.snapshots = generate_snapshots(RESNET_18, 1, RetrainDistribution.HARD_CODED, save_path=self.save_path,
                                            retrain_idxs=retrain_idxs, use_same_base=True)

        self.model_store = ModelStore(self.save_path)

    def test_get_snapshot(self):
        self.model_store.add_snapshot(self.snapshots[0])
        retrieved_snapshot = self.model_store.get_snapshot(self.snapshots[0].id)
        self.assertEqual(self.snapshots[0], retrieved_snapshot)

    def test_get_model(self):
        self.model_store.add_snapshot(self.snapshots[0])
        retrieved_snapshot = self.model_store.get_snapshot(self.snapshots[0].id)
        sd1 = self.snapshots[0].init_model_from_snapshot().state_dict()
        sd2 = retrieved_snapshot.init_model_from_snapshot().state_dict()
        self.assertTrue(state_dict_equal(sd1, sd2))

    def test_get_composed_model(self):
        self.model_store.add_snapshot(self.snapshots[0])
        retrieved_snapshot = self.model_store.get_snapshot(self.snapshots[0].id)
        sd1 = self.snapshots[0].init_model_from_snapshot().state_dict()
        retrieved_model = self.model_store.get_composed_model([ls.id for ls in retrieved_snapshot.layer_states])
        self.assertTrue(state_dict_equal(sd1, retrieved_model.state_dict()))


if __name__ == '__main__':
    unittest.main()
