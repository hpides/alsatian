from transformers import AutoModelForObjectDetection, AutoModelForImageClassification
from transformers.models.resnet.modeling_resnet import ResNetEncoder, ResNetModel

from custom.models.custom_resnet import resnet50, resnet101, resnet18
from custom.models.sequential_hf_dinov2 import get_sequential_dinov2_model
from custom.models.sequential_hf_vit import get_sequential_vit_model
from custom.models.split_indices import SPLIT_INDEXES
from global_utils.model_names import RESNET_50, RESNET_101, RESNET_18
from global_utils.model_operations import transform_to_sequential, split_model_in_two

SENSE_TIME_DEFORMABLE_DETR = "SenseTime/deformable-detr"

GOOGLE_VIT_BASE_PATCH16_224_IN21K = "google/vit-base-patch16-224-in21k"

FACEBOOK_DINOV2_BASE = "facebook/dinov2-base"

FACEBOOK_DINOV2_LARGE = "facebook/dinov2-large"

FACEBOOK_DETR_RESNET_101 = "facebook/detr-resnet-101"

MICROSOFT_TABLE_STRUCTURE_RECOGNITION = "microsoft/table-transformer-structure-recognition"

MICROSOFT_TABLE_TRANSFORMER_DETECTION = "microsoft/table-transformer-detection"

MICROSOFT_RESNET_152 = "microsoft/resnet-152"

MICROSOFT_RESNET_18 = "microsoft/resnet-18"

FACEBOOK_DETR_RESNET_50_DC5 = "facebook/detr-resnet-50-dc5"

CONDITIONAL_DETR_RESNET_50 = "microsoft/conditional-detr-resnet-50"

FACEBOOK_DETR_RESNET_50 = "facebook/detr-resnet-50"

ALL_HF_MODELS = [MICROSOFT_RESNET_18, MICROSOFT_RESNET_152, MICROSOFT_TABLE_STRUCTURE_RECOGNITION,
                 MICROSOFT_TABLE_TRANSFORMER_DETECTION, FACEBOOK_DETR_RESNET_101, FACEBOOK_DETR_RESNET_50_DC5,
                 CONDITIONAL_DETR_RESNET_50, FACEBOOK_DINOV2_BASE, FACEBOOK_DINOV2_LARGE,
                 SENSE_TIME_DEFORMABLE_DETR, GOOGLE_VIT_BASE_PATCH16_224_IN21K, FACEBOOK_DETR_RESNET_50]

RESNET_50_MODELS = [FACEBOOK_DETR_RESNET_50, CONDITIONAL_DETR_RESNET_50, FACEBOOK_DETR_RESNET_50_DC5,
                    SENSE_TIME_DEFORMABLE_DETR]
MICROSOFT_RESNETS = [MICROSOFT_RESNET_152, MICROSOFT_RESNET_18]
MICROSOFT_TABLE_TRANSFORMERS = [MICROSOFT_TABLE_TRANSFORMER_DETECTION,
                                MICROSOFT_TABLE_STRUCTURE_RECOGNITION]
PYTORCH_RESNETS = RESNET_50_MODELS + MICROSOFT_TABLE_TRANSFORMERS + [FACEBOOK_DETR_RESNET_101]

DINO_V2_MODELS = [FACEBOOK_DINOV2_BASE, FACEBOOK_DINOV2_LARGE]


def initialize_hf_model(hf_base_model_id, hf_model_id, hf_cache_dir):
    """
    Method that returns a sequential feature extractor/backbone of a hugging face model
    :param hf_base_model_id: the hugging face model id of the base model (the model the hf model was fine-tuned from)
    :param hf_model_id: the hf model id to load the weights from
    :param hf_cache_dir: the cache dir for the hugging face snapshot
    :return: a sequential feature extractor/backbone of a hugging face model
    """
    if (hf_base_model_id in PYTORCH_RESNETS):
        if hf_base_model_id == FACEBOOK_DETR_RESNET_101:
            model = resnet101()
            model_name = RESNET_101
        elif hf_base_model_id in MICROSOFT_TABLE_TRANSFORMERS:
            model = resnet18()
            model_name = RESNET_18
        else:
            model = resnet50()
            model_name = RESNET_50

        print(f"HF CACHING DIR: {hf_cache_dir}")
        print(f"HF MODEL ID: {hf_model_id}")
        hf_model = AutoModelForObjectDetection.from_pretrained(hf_model_id, cache_dir=hf_cache_dir)
        backbone_model = hf_model.model.backbone.conv_encoder.model
        hf_model_sd = backbone_model.state_dict()
        missing_keys, unexpected_keys = model.load_state_dict(hf_model_sd, strict=False)
        # HF model is a backbone and naturally misses these layers
        if not missing_keys == ['fc.weight', 'fc.bias'] and not len(unexpected_keys) == 0:
            print(f'hf_identifier: {hf_model_id}')
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys: {unexpected_keys}')
            print(f"HF MODEL ID: {hf_model_id}")
            assert False

        model = transform_to_sequential(model)
        split_index = SPLIT_INDEXES[model_name][0]
        first, _ = split_model_in_two(model, split_index)
        model = first
    elif hf_base_model_id in MICROSOFT_RESNETS:
        model = get_sequential_microsoft_resnet(hf_model_id, hf_cache_dir)
        model_name = hf_base_model_id
    elif hf_base_model_id == GOOGLE_VIT_BASE_PATCH16_224_IN21K:
        model = get_sequential_vit_model(model_id=hf_model_id, hf_caching_dir=hf_cache_dir)
        model_name = hf_base_model_id
    elif hf_base_model_id in DINO_V2_MODELS:
        model = get_sequential_dinov2_model(model_id=hf_model_id, hf_caching_dir=hf_cache_dir)
        model_name = hf_base_model_id
    else:
        raise NotImplementedError

    return model_name, model


def get_sequential_microsoft_resnet(hf_model_id, hf_cache_dir=None):
    hf_model = AutoModelForImageClassification.from_pretrained(hf_model_id, cache_dir=hf_cache_dir)
    seq_model = transform_to_sequential(hf_model, split_classes=[ResNetModel, ResNetEncoder])
    first, second = split_model_in_two(seq_model, 7)
    model = first
    return model
