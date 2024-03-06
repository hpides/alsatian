from custom.models.init_models import initialize_model
from global_utils.model_names import VISION_MODEL_CHOICES

if __name__ == '__main__':
    for model_name in VISION_MODEL_CHOICES:
        model = initialize_model(model_name, sequential_model=True)
        with open(f'./model_descriptions/{model_name}-layers.txt', "w") as file:
            file.write(str(model))
