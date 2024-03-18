from custom.models.init_models import initialize_model
from global_utils.model_names import VISION_MODEL_CHOICES

if __name__ == '__main__':
    for m_name in VISION_MODEL_CHOICES:
        model = initialize_model(m_name, pretrained=True)
        model = initialize_model(m_name, pretrained=True, sequential_model=True)
