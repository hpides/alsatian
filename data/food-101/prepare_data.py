import os
import shutil

from data.stanford_dogs.prepare_data import count_jpg_files_recursively


def extract_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by '/' and take the second part
            id = int(line.strip().split('/')[1])
            ids.append(id)
    return ids


def copy_and_rename_directory(src, dst, new_name):
    # Create the destination path with the new name
    dst_path = os.path.join(dst, new_name)

    # Copy the source directory to the destination path
    shutil.copytree(src, dst_path)


def delete_non_valid_files(root_dir, valid_ids):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if ".jpg" in filename and int(filename.replace(".jpg", "")) not in valid_ids:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


if __name__ == '__main__':
    root_dir = "/Users/nils/uni/programming/model-search-paper/data/food-101/food-101"
    train_ids = extract_ids(os.path.join(root_dir, "meta", "train.txt"))
    test_ids = extract_ids(os.path.join(root_dir, "meta", "test.txt"))

    images_dir = os.path.join(root_dir, "images")

    target_dir = "/Users/nils/uni/programming/model-search-paper/data/food-101/prepared_data"

    # # copy directory twice
    copy_and_rename_directory(images_dir, target_dir, "train")
    copy_and_rename_directory(images_dir, target_dir, "test")

    train_data = os.path.join(target_dir, "train")
    delete_non_valid_files(train_data, train_ids)
    test_data = os.path.join(target_dir, "test")
    delete_non_valid_files(test_data, test_ids)

    assert count_jpg_files_recursively(train_data) > 10000
    # assert count_jpg_files_recursively(test_data) > 10000
