import json
import os

def get_raw_data(root_dir, search_strings):
    matches = extract_files_by_name(root_dir, search_strings)
    assert len(matches) == 1
    return parse_json_file(matches[0])

def all_strings_in_file_name(search_strings, file):
    for search_string in search_strings:
        if search_string not in file:
            return False
    return True


def extract_files_by_name(directory_path, search_strings):
    matching_files = []

    # Check if the given directory path exists
    if not os.path.exists(directory_path):
        print("Error: Directory does not exist.")
        return matching_files

    # Iterate through all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if all_strings_in_file_name(search_strings, file):
                matching_files.append(os.path.join(root, file))

    return matching_files


def parse_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
            return data_dict
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in '{file_path}': {e}")
        return None
