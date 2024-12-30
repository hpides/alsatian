import os

from experiments.dummy_experiments.dummy_models import TwoBlockModel, get_sequential_two_block_model
from experiments.main_experiments.snapshots.hugging_face.generate_hf_snapshots import get_existing_model_store
from experiments.main_experiments.snapshots.synthetic.generate import generate_snapshot
from global_utils.json_operations import write_json_to_file
from model_search.model_management.model_store import ModelStore

model_store_save_path = "/mount-fs/dummy_model_store"

def get_three_random_two_bock_models():
    return get_existing_model_store(model_store_save_path)

if __name__ == '__main__':
    snapshots = []
    snapshot_save_path = "/mount-fs/dummy_snapshots"

    architecture_name = "dummy_two_block_model"
    for i in range(3):
        input_channels = 3
        conv_output_channels = 16  # Number of output channels for the convolutional layer
        output_size = 2  # Number of classes for output
        model = get_sequential_two_block_model()
        snapshot = generate_snapshot(architecture_name, model, snapshot_save_path)
        snapshots.append(snapshot)

    model_store = ModelStore(model_store_save_path)
    for snapshot in snapshots:
        model_store.add_snapshot(snapshot)

    model_store_json_path = os.path.join(model_store_save_path, 'model_store.json')
    model_store_dict = model_store.to_dict()
    write_json_to_file(model_store_dict, model_store_json_path)