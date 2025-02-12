import json
import os
from collections import Counter
from itertools import groupby

SIMILARITY_IDENTICAL = 'LayerSimilarity.IDENTICAL'

IDENTICAL_LAYER_COUNT = 'first_identical_layer_count'

SIMILARITY_GROUPS = 'similarity_groups'

SIMILARITY_HISTOGRAM = 'similarity_histogram'

SIMILARITY_LIST = 'similarity_list'

FILE_PATH = "file_path"


def list_files_containing_string(directory, x):
    """
    List all absolute file paths in the given directory where the file name contains the string x.

    :param directory: The directory to search in.
    :param x: The string to search for in file names.
    :return: A list of absolute file paths.
    """
    matching_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if x in file:
                file_id = file.split(";")[3].replace(".txt", "")
                file_path = os.path.abspath(os.path.join(root, file))
                matching_files[file_id] = {FILE_PATH: file_path}

    return matching_files


def get_model_similarity_list(file_path):
    """
    Parses the given text file, skips the first line, and extracts the enum value from the second line.

    :param file_path: Path to the text file.
    :return: A tuple containing the enum value and its corresponding string representation.
    """
    similarity_list = []
    with open(file_path, 'r') as file:
        print(file_path)
        lines = file.readlines()

        # Skip the first line
        if len(lines) < 2:
            raise ValueError("File does not have enough lines for parsing.")

        for line in lines[1:]:
            json_line = json.loads(line)
            similarity = json_line["comp-level"]
            similarity_list.append(similarity)

        return similarity_list


def generate_model_similarity_report(base_model, model_analysis_files, output_path):
    files = list_files_containing_string(model_analysis_files, base_model)

    for file in list(files.keys()):
        similarity_list = get_model_similarity_list(files[file][FILE_PATH])
        files[file][SIMILARITY_LIST] = similarity_list
        files[file][SIMILARITY_GROUPS] = [(key, len(list(group))) for key, group in groupby(similarity_list)]
        histogram = dict(Counter(similarity_list))
        files[file][SIMILARITY_HISTOGRAM] = histogram
        files[file][IDENTICAL_LAYER_COUNT] = 0 if SIMILARITY_IDENTICAL not in histogram else histogram[
            SIMILARITY_IDENTICAL]

    sorted_files = dict(sorted(files.items(), key=lambda x: x[1][IDENTICAL_LAYER_COUNT], reverse=True))

    file_path = os.path.join(output_path, f"compare-{base_model}.json")
    with open(file_path, 'w') as json_file:
        json.dump(sorted_files, json_file, indent=4)

    short_report = ""
    for file, v in sorted_files.items():
        histogram = v[SIMILARITY_HISTOGRAM]
        short_report += f"{file}: {files[file][IDENTICAL_LAYER_COUNT]} / {sum(histogram.values())} \n"

    file_path = os.path.join(output_path, f"SHORT-compare-{base_model}.json")
    with open(file_path, 'w') as text_file:
        text_file.write(short_report)


if __name__ == '__main__':

    MODEL_ANALYSIS_FILES = "/Users/nils/Downloads/out" # path to comparison files
    OUTPUT_PATH = "./similarity_reports" # path where similarity reports are written to

    BASE_MODEL_NAMES = [
        "facebook-detr-resnet-50;",
        "nvidia-segformer-b5-finetuned-cityscapes-1024-1024;",
        "nvidia-segformer-b1-finetuned-cityscapes-1024-1024;",
        "nvidia-segformer-b2-finetuned-ade-512-512;",
        "nvidia-segformer-b1-finetuned-ade-512-512;",
        "nvidia-segformer-b0-finetuned-ade-512-512;",
        "microsoft-table-transformer-detection;",
        "hustvl-yolos-tiny;",
        "hustvl-yolos-small;",
        "microsoft-table-transformer-structure-recognition;",
        "microsoft-conditional-detr-resnet-50;",
        "google-vit-large-patch16-224-in21k;",
        "facebook-dinov2-large;",
        "facebook-dinov2-base;",
        "google-vit-huge-patch14-224-in21k;",
        "google-mobilenet_v2_1.0_224;",
        "facebook-detr-resnet-101;",
        "google-vit-base-patch16-224-in21k;",
        "SenseTime-deformable-detr;",
        "facebook-detr-resnet-50-dc5;",
        "google-efficientnet-b0;",
        "microsoft-resnet-50;",
        "microsoft-resnet-101;",
        "microsoft-resnet-18;",
        "microsoft-resnet-152;",
        "microsoft-trocr-large-printed;",
        "Salesforce-blip-image-captioning-base;",
        "naver-clova-ix-donut-base-finetuned-cord-v2;",
        "microsoft-trocr-base-printed;",
        "nlpconnect-vit-gpt2-image-captioning;",
        "microsoft-git-base;",
        "microsoft-trocr-base-stage1;",
        "naver-clova-ix-donut-base;",
    ]

    for base_model in BASE_MODEL_NAMES:
        generate_model_similarity_report(base_model, MODEL_ANALYSIS_FILES, OUTPUT_PATH)
