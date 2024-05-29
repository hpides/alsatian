import os
from datetime import datetime


def file_up_to_date(base_path, accepted_duration):
    files = [f for f in os.listdir(base_path) if f.startswith('active-')]
    if len(files) == 0:
        return False

    file_name = files[0].replace("active-", "")
    file_time = datetime.strptime(file_name, '%Y-%m-%d_%H-%M-%S')

    # Get the current time
    current_time = datetime.now()

    # Calculate the difference in seconds
    time_difference = (current_time - file_time).total_seconds()

    return time_difference <= accepted_duration


if __name__ == '__main__':
    file_up_to_date('/Users/nils/uni/programming/model-search-paper/experiments/prevent_caching', 5)
