from custom.models.init_models import initialize_model
from global_utils.json_operations import write_json_to_file
from global_utils.model_names import VISION_MODEL_CHOICES
from global_utils.model_resource_info import layer_output_sizes

if __name__ == '__main__':
    result = {}
    for model_name in VISION_MODEL_CHOICES:
        model = initialize_model(model_name, pretrained=True, sequential_model=True, features_only=True)
        result[model_name] = layer_output_sizes(model, [3, 224, 224])
        write_json_to_file(result, 'outputs/layer_output_infos.json')
