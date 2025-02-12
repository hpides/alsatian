import os
import re


def clean_file_name(file_name):
    # Define a regular expression to match characters not allowed in file names
    invalid_chars_regex = r'[<>:"/\\|?*\x00-\x1F\x7F\s(),]'

    # Remove invalid characters from the file name
    cleaned_file_name = re.sub(invalid_chars_regex, '', file_name)

    return cleaned_file_name


def parsable_as_list(base_save_path):
    return "," in base_save_path


def to_path_list(paths_string):
    path_strings = paths_string.split(",")
    return [ os.path.abspath(path) for path in path_strings]

def clear_directory(directory_path, delete_subdirectories=False):
    """
    Deletes all files in the specified directory.

    Parameters:
    - directory_path (str): Path to the directory to clear.
    - delete_subdirectories (bool): If True, also delete subdirectories. Defaults to False.

    Returns:
    - None
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Delete files or symlinks
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path) and delete_subdirectories:
            shutil.rmtree(item_path)  # Delete subdirectory
            print(f"Deleted directory: {item_path}")

    print(f"Directory '{directory_path}' has been cleared.")
