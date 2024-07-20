import torch
from torch import nn

from custom.models.custom_mobilenet import mobilenet_v2, MobileNet_V2_Weights
from custom.models.custom_resnet import *
from custom.models.efficientnet import efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_L_Weights, \
    EfficientNet_V2_S_Weights
from custom.models.sequential_bert import get_sequential_bert_model
from custom.models.split_indices import SPLIT_INDEXES
from custom.models.vision_transformer import Encoder, vit_b_16, vit_b_32, vit_l_16, vit_l_32, ViT_L_32_Weights, \
    ViT_L_16_Weights, ViT_B_32_Weights, ViT_B_16_Weights
from global_utils.model_names import *
from global_utils.model_operations import transform_to_sequential, split_model_in_two


def initialize_model(model_name, pretrained=False, new_num_classes=None, features_only=False, sequential_model=False,
                     freeze_feature_extractor=False):
    # init base model
    if model_name == RESNET_18:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if pretrained else resnet18()
    elif model_name == RESNET_34:
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) if pretrained else resnet34()
    elif model_name == RESNET_50:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) if pretrained else resnet50()
    elif model_name == RESNET_101:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1) if pretrained else resnet101()
    elif model_name == RESNET_152:
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1) if pretrained else resnet152()
    elif model_name == MOBILE_V2:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1) if pretrained else mobilenet_v2()
    elif model_name == VIT_B_16:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1) if pretrained else vit_b_16()
    elif model_name == VIT_B_32:
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1) if pretrained else vit_b_32()
    elif model_name == VIT_L_16:
        model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1) if pretrained else vit_l_16()
    elif model_name == VIT_L_32:
        model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1) if pretrained else vit_l_32()
    elif model_name == EFF_NET_V2_S:
        model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) if pretrained else efficientnet_v2_s()
    elif model_name == EFF_NET_V2_L:
        model = efficientnet_v2_l(
            weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) if pretrained else efficientnet_v2_l()
    elif model_name == BERT and features_only:
        model = get_sequential_bert_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # replace classification layer
    if new_num_classes:
        if model_name in RESNETS:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, new_num_classes)
        elif model_name == MOBILE_V2:
            model.classifier = nn.Sequential(
                nn.Dropout(p=model.dropout),
                nn.Linear(model.last_channel, new_num_classes),
            )
        elif model_name in TRANSFORMER_MODELS:
            model.heads = model._heads(model.hidden_dim, new_num_classes, model.representation_size)
        elif model_name in EFF_NETS:
            model.classifier = nn.Sequential(
                nn.Dropout(p=model.dropout, inplace=True),
                nn.Linear(model.lastconv_output_channels, new_num_classes),
            )
        elif model_name == BERT:
            model: torch.nn.Sequential = model
            model.append(
                torch.nn.Sequential(
                    nn.Dropout(p=0.1, inplace=False),
                    nn.Linear(in_features=768, out_features=new_num_classes, bias=True)
                )
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    if freeze_feature_extractor:
        # freeze all parameters later unfreeze specific ones
        for param in model.parameters():
            param.requires_grad = False

        if model_name in RESNETS:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == MOBILE_V2:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif model_name in TRANSFORMER_MODELS:
            for param in model.heads.parameters():
                param.requires_grad = True
        elif model_name in EFF_NETS:
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown model: {model_name}")

    if sequential_model or features_only:
        split_cls = None
        if model_name in TRANSFORMER_MODELS:
            split_cls = [Encoder, torch.nn.Sequential]

        model = transform_to_sequential(model, split_classes=split_cls)

    if features_only:
        split_index = SPLIT_INDEXES[model_name][0]
        first, _ = split_model_in_two(model, split_index)
        model = first

    return model

if __name__ == '__main__':
    bert_model = initialize_model(BERT, features_only=True, sequential_model=True)
    print("test")
