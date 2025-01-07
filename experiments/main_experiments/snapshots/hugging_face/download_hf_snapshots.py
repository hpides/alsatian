import os

import torch

from experiments.main_experiments.snapshots.hugging_face.init_hf_models import *
from global_utils.model_names import RESNET_50

HF_MODEL_CHOICES = [RESNET_50]


def download_snapshot(base_model_id, fine_tuned_model_ids, hf_cache_dir: str,
                      num_models: int = -1) -> [torch.nn.Module]:
    hf_model_ids = [base_model_id] + fine_tuned_model_ids
    if num_models > 0:
        hf_model_ids = hf_model_ids[num_models]

    i =0
    for hf_model_id in hf_model_ids:
        print(i)
        initialize_hf_model(base_model_id, hf_model_id, hf_cache_dir)
        i += 1


def get_snapshot_ids(snapshot_id_file_path):
    with open(snapshot_id_file_path, "r") as file:
        fine_tuned_model_ids = [line.strip() for line in file]
        return fine_tuned_model_ids


if __name__ == '__main__':

    hf_caching_dir = "/mount-fs/hf-caching-dir"
    # model_names = [MICROSOFT_RESNET_18, MICROSOFT_RESNET_152, MICROSOFT_TABLE_STRUCTURE_RECOGNITION,
    #                MICROSOFT_TABLE_TRANSFORMER_DETECTION, FACEBOOK_DETR_RESNET_101, FACEBOOK_DETR_RESNET_50_DC5,
    #                CONDITIONAL_DETR_RESNET_50, FACEBOOK_DINOV2_BASE, FACEBOOK_DINOV2_LARGE,
    #                SENSE_TIME_DEFORMABLE_DETR, GOOGLE_VIT_BASE_PATCH16_224_IN21K, FACEBOOK_DETR_RESNET_50]
    model_names = [FACEBOOK_DETR_RESNET_50]

    for model_name in model_names:
        fine_tuned_model_ids_file = os.path.join("./hf-model-ids", f"{model_name.replace('/', '-')}.txt")

        fine_tuned_model_ids = get_snapshot_ids(fine_tuned_model_ids_file)

        model_ids = fine_tuned_model_ids

        if model_name == FACEBOOK_DETR_RESNET_50:
            snapshots = download_snapshot(model_name, model_ids, hf_caching_dir, 250)
        elif model_name == GOOGLE_VIT_BASE_PATCH16_224_IN21K:
            snapshots = download_snapshot(model_name, model_ids, hf_caching_dir, 100)
        else:
            snapshots = download_snapshot(model_name, model_ids, hf_caching_dir)
