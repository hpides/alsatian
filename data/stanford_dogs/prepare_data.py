import os
import shutil

import scipy.io


def get_file_names(mat_file):
    result = []
    mat = scipy.io.loadmat(mat_file)
    for item in mat['file_list']:
        result.append(item[0][0].split("/")[1])
    return result

def copy_and_rename_directory(src, dst, new_name):
    # Create the destination path with the new name
    dst_path = os.path.join(dst, new_name)

    # Copy the source directory to the destination path
    shutil.copytree(src, dst_path)

def delete_non_valid_files(root_dir, valid_file_names):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if ".jpg" in filename and filename not in valid_file_names:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

def count_jpg_files_recursively(directory):
    jpg_file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                jpg_file_count += 1
    return jpg_file_count

if __name__ == '__main__':
    train_file_names = get_file_names('/Users/nils/uni/programming/model-search-paper/data/stanford_dogs/lists/train_list.mat')
    test_file_names = get_file_names('/Users/nils/uni/programming/model-search-paper/data/stanford_dogs/lists/test_list.mat')

    print(len(train_file_names))
    print(len(test_file_names))

    images_root_dir = "/Users/nils/uni/programming/model-search-paper/data/stanford_dogs/Images"
    target_dir = "/Users/nils/uni/programming/model-search-paper/data/stanford_dogs/prepared_data"

    copy_and_rename_directory(images_root_dir, target_dir, "train")
    copy_and_rename_directory(images_root_dir, target_dir, "test")

    train_data = os.path.join(target_dir, "train")
    delete_non_valid_files(train_data, train_file_names)
    test_data = os.path.join(target_dir, "test")
    delete_non_valid_files(test_data, test_file_names)

    assert count_jpg_files_recursively(train_data) == 12000
    assert count_jpg_files_recursively(test_data) == 8580
