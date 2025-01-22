import os

from experiments.main_experiments.snapshots.hugging_face.generate_hf_snapshots import generate_hf_snapshots, \
    build_model_store, get_existing_model_store
from global_utils.json_operations import write_json_to_file

ORDERED = "ordered"


def _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path, snapshot_specs=None):
    snapshot_save_path = os.path.join(snapshot_save_base_path, model_ids_file.replace(".txt", ""))
    model_store_json_path = os.path.join(snapshot_save_path, 'model_store.json')
    if os.path.exists(model_store_json_path):
        return get_existing_model_store(snapshot_save_path)
    else:
        if snapshot_specs is None:
            snapshot_specs = [[model_ids_file, -1, ORDERED]]
        return generate_model_store(snapshot_specs, snapshot_save_base_path, hf_caching_path, model_store_json_path)


def get_100_ordered_google_vit_base_patch16_224_in21k(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "google-vit-base-patch16-224-in21k.txt"
    snapshot_specs = [[model_ids_file, 100, ORDERED]]
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path, snapshot_specs=snapshot_specs)


def get_232_facebook_detr_resnet_50_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "facebook-detr-resnet-50.txt"
    snapshot_specs = [[model_ids_file, 232, ORDERED]]
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path, snapshot_specs=snapshot_specs)


def get_microsoft_conditional_detr_resnet_50_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "microsoft-conditional-detr-resnet-50.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_facebook_detr_resnet_50_dc5_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "facebook-detr-resnet-50-dc5.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_SenseTime_deformable_detr_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "SenseTime-deformable-detr.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_facebook_detr_resnet_101_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "facebook-detr-resnet-101.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_microsoft_resnet_18_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "microsoft-resnet-18.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_microsoft_resnet_152_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "microsoft-resnet-152.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_microsoft_table_transformer_detection_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "microsoft-table-transformer-detection.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_microsoft_table_transformer_structure_recognition_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "microsoft-table-transformer-structure-recognition.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_facebook_dinov2_base_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "facebook-dinov2-base.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


def get_facebook_dinov2_large_snapshot_set(snapshot_save_base_path, hf_caching_path):
    model_ids_file = "facebook-dinov2-large.txt"
    return _get_models(hf_caching_path, model_ids_file, snapshot_save_base_path)


# def get_dummy_combined_snapshot_set(snapshot_save_base_path, hf_caching_path, model_store_save_path):
#     model_store_id = "dummy-combined-model-store"
#     model_store_json_path = os.path.join(model_store_save_path, f"{model_store_id}.json")
#
#     if os.path.exists(model_store_json_path):
#         # execute just to see if we get any errors
#         return get_existing_model_store(model_store_save_path, json_file_name=f"{model_store_id}.json")
#     else:
#         snapshot_specs = [
#             ["microsoft-resnet-18.txt", 2, ORDERED],
#             ["microsoft-resnet-101.txt", 2, ORDERED],
#         ]
#         return generate_model_store(snapshot_specs, snapshot_save_base_path, hf_caching_path, model_store_json_path)


def generate_model_store(snapshot_specs, snapshot_save_base_path, hf_caching_path, model_store_json_path):
    for snapshot_spec in snapshot_specs:
        model_file, num_models, sampling = snapshot_spec

        snapshot_id_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-model-ids", model_file)
        with open(snapshot_id_file_path, "r") as file:
            fine_tuned_model_ids = [line.strip() for line in file]

        if num_models > -1:
            if sampling == ORDERED:
                fine_tuned_model_ids = fine_tuned_model_ids[:num_models]
            else:
                raise NotImplementedError

        base_model_id = model_file.replace(".txt", "").replace("-", "/", 1)
        number_models = len(fine_tuned_model_ids) + 1  # +1 to also include the base model

        snapshot_save_path = os.path.join(snapshot_save_base_path, model_file.replace(".txt", ""))
        os.makedirs(snapshot_save_path, exist_ok=True)

        snapshots = generate_hf_snapshots(
            base_model_id, fine_tuned_model_ids, snapshot_save_path, hf_caching_path, number_models)

        model_store = build_model_store(snapshot_save_path, snapshots)
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, model_store_json_path)

        model_snapshots = list(model_store.models.values())

    return model_snapshots, model_store


if __name__ == '__main__':
    snapshot_save_base_path = "/mount-fs/hf-snapshots/"
    hf_caching_path = "/mount-fs/hf-caching-dir"
    num_model_snapshots = 0


    def count_and_print_snapshots(name, snapshots):
        global num_model_snapshots
        snapshot_count = len(snapshots)  # Assuming `snapshots` is a list or similar collection
        num_model_snapshots += snapshot_count
        print(f"{name}: {snapshot_count} snapshots")


    # Extract and count snapshots
    model_snapshots, model_store = get_100_ordered_google_vit_base_patch16_224_in21k(snapshot_save_base_path,
                                                                                     hf_caching_path)
    count_and_print_snapshots("Google ViT Base Patch16 224 IN21K", model_snapshots)

    model_snapshots, model_store = get_232_facebook_detr_resnet_50_snapshot_set(snapshot_save_base_path,
                                                                                hf_caching_path)
    count_and_print_snapshots("Facebook DETR ResNet-50", model_snapshots)

    model_snapshots, model_store = get_microsoft_conditional_detr_resnet_50_snapshot_set(snapshot_save_base_path,
                                                                                         hf_caching_path)
    count_and_print_snapshots("Microsoft Conditional DETR ResNet-50", model_snapshots)

    model_snapshots, model_store = get_facebook_detr_resnet_50_dc5_snapshot_set(snapshot_save_base_path,
                                                                                hf_caching_path)
    count_and_print_snapshots("Facebook DETR ResNet-50 DC5", model_snapshots)

    model_snapshots, model_store = get_SenseTime_deformable_detr_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("SenseTime Deformable DETR", model_snapshots)

    model_snapshots, model_store = get_facebook_detr_resnet_101_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Facebook DETR ResNet-101", model_snapshots)

    model_snapshots, model_store = get_microsoft_resnet_18_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Microsoft ResNet-18", model_snapshots)

    model_snapshots, model_store = get_microsoft_resnet_152_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Microsoft ResNet-152", model_snapshots)

    model_snapshots, model_store = get_microsoft_table_transformer_detection_snapshot_set(snapshot_save_base_path,
                                                                                          hf_caching_path)
    count_and_print_snapshots("Microsoft Table Transformer Detection", model_snapshots)

    model_snapshots, model_store = get_microsoft_table_transformer_structure_recognition_snapshot_set(
        snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Microsoft Table Transformer Structure Recognition", model_snapshots)

    model_snapshots, model_store = get_facebook_dinov2_base_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Facebook DINOv2 Base", model_snapshots)

    model_snapshots, model_store = get_facebook_dinov2_large_snapshot_set(snapshot_save_base_path, hf_caching_path)
    count_and_print_snapshots("Facebook DINOv2 Large", model_snapshots)

    # Print the total number of snapshots
    print(f"Total number of snapshots: {num_model_snapshots}")
