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
