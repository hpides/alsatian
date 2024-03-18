import os


def average_file_size(directory):
    total_size = 0
    num_files = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                num_files += 1

    if num_files > 0:
        return total_size / num_files
    else:
        return 0  # Avoid division by zero if there are no files


if __name__ == '__main__':
    # Example usage:
    directory = "/Users/nils/uni/programming/model-search-paper/data/imagenette2/train"  # Replace this with the path to your directory
    avg_size = average_file_size(directory)
    print(f"Average file size in directory '{directory}': {avg_size} bytes")
