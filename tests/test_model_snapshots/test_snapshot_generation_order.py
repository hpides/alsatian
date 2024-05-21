import os
import unittest

from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import read_json_to_dict, write_json_to_file
from global_utils.model_names import RESNET_18
from model_search.model_management.model_store import model_store_from_dict, ModelStore


class TestSnapshotGenerationOrder(unittest.TestCase):

    def _generate_models(self, save_path, json_file_path):
        model_snapshots = generate_snapshots(RESNET_18, 5, RetrainDistribution.TOP_LAYERS,
                                             save_path=save_path, use_same_base=True)

        # add the snapshots to a model store
        model_store = ModelStore(save_path)
        for snapshot in model_snapshots:
            model_store.add_snapshot(snapshot)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, json_file_path)

        return model_snapshots, model_store

    def _read_snapshots(self, json_file_path):
        model_store_dict = read_json_to_dict(json_file_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())

        return model_snapshots, model_store

    def setUp(self) -> None:
        self.tmp_dir = './tmp_dir'
        self.json_file_path = os.path.join(self.tmp_dir, 'model_store.json')

    def test_simple_example(self):
        gen_model_snapshots, gen_model_store = self._generate_models(self.tmp_dir, self.json_file_path)
        read_model_snapshots, read_model_store = self._read_snapshots(self.json_file_path)

        self.assertEqual(gen_model_snapshots, read_model_snapshots)



