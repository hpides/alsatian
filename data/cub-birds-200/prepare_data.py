import os.path
import shutil

def count_jpg_files_recursively(directory):
    jpg_file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                jpg_file_count += 1
    return jpg_file_count

def create_map_from_file(file_path):
    result_map = {}

    with open(file_path, 'r') as file:
        for line in file:
            value, key = line.strip().split(' ', 1)
            key = key.split("/")[1]
            result_map[key] = int(value)

    return result_map


def read_file_into_lists(file_path):
    test_ids = []
    train_ids = []

    with open(file_path, 'r') as file:
        for line in file:
            value, key = map(int, line.split())
            if key == 0:
                train_ids.append(value)
            elif key == 1:
                test_ids.append(value)

    return test_ids, train_ids


def copy_and_rename_directory(src, dst, new_name):
    # Create the destination path with the new name
    dst_path = os.path.join(dst, new_name)

    # Copy the source directory to the destination path
    shutil.copytree(src, dst_path)


def delete_non_valid_files(root_dir, valid_ids, id_map):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if ".jpg" in filename and id_map[filename] not in valid_ids:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


if __name__ == '__main__':
    root_dir = "/Users/nils/uni/programming/model-search-paper/data/cub-birds-200/CUB_200_2011/CUB_200_2011"
    images_dir = os.path.join(root_dir, "images")
    train_test_split_file = os.path.join(root_dir, "train_test_split.txt")
    id_mapping_file = os.path.join(root_dir, "images.txt")

    target_dir = "/Users/nils/uni/programming/model-search-paper/data/cub-birds-200/prepared_data"

    train_ids, test_ids = read_file_into_lists(train_test_split_file)
    id_map = create_map_from_file(id_mapping_file)

    # # copy directory twice
    copy_and_rename_directory(images_dir, target_dir, "train")
    copy_and_rename_directory(images_dir, target_dir, "test")

    train_data = os.path.join(target_dir, "train")
    delete_non_valid_files(train_data, train_ids, id_map)
    test_data = os.path.join(target_dir, "test")
    delete_non_valid_files(test_data, test_ids, id_map)

    assert count_jpg_files_recursively(train_data) == 5994
    assert count_jpg_files_recursively(test_data) == 5794
