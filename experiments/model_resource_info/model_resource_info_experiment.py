import configparser
import os

from custom.models.init_models import initialize_model
from custom.models.split_indices import SPLIT_INDEXES
from experiments.opt_parameters.data_loading.experiment_args import ExpArgs
from global_utils.model_names import VISION_MODEL_CHOICES
from global_utils.model_resource_info import model_resource_info
from global_utils.write_results import write_measurements_and_args_to_json_file

if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), './config.ini')
    config_section = 'model-reuse-info-exp-params-des-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    for model_name in VISION_MODEL_CHOICES:
        model = initialize_model(model_name, pretrained=True, sequential_model=True)
        info = model_resource_info(model, SPLIT_INDEXES[model_name], [3, 224, 224], inference_time=True)

        exp_id = f'{config_section}-batch_size-{exp_args.batch_size}'
        write_measurements_and_args_to_json_file(
            measurements=info,
            args=exp_args.get_dict(),
            dir_path=exp_args.result_dir,
            file_id=exp_id
        )
