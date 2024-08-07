import torch

from custom.models.init_models import initialize_model
from global_utils.json_operations import write_json_to_file
from global_utils.model_names import VISION_MODEL_CHOICES, BERT
from global_utils.model_resource_info import layer_output_sizes, bert_layer_output_sizes

if __name__ == '__main__':
    result = {}
    for model_name in VISION_MODEL_CHOICES:
        model = initialize_model(model_name, pretrained=True, sequential_model=True, features_only=True)
        result[model_name] = layer_output_sizes(model, [3, 224, 224])

    for model_name in [BERT]:
        model = initialize_model(model_name, pretrained=True, sequential_model=True, features_only=True)
        input_data = (
            torch.randint(size=[1] + [256], dtype=torch.int64, low=0, high=100),
            torch.ones(size=[1] + [256])
        )
        result[model_name] = bert_layer_output_sizes(model, input_data=input_data)
    write_json_to_file(result, 'outputs/layer_output_infos.json')

