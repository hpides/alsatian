import os
import re
import subprocess
import time
from datetime import datetime

from global_utils.deterministic import TRUE

FLAG_FLUSH_CACHES = "flag-flush-caches"

LIMIT_IO = 'limit_io'


def clear_caches_and_check_io_limit():
    if LIMIT_IO in os.environ and os.environ[LIMIT_IO] == TRUE:
        # check if script for clearing the caches is active
        assert active_file_up_to_date('/mount-fs', 10), \
            "script to clear caches does not seem to be active (look into prevent_caching directory for info)"
        # clear caches by writing file
        write_empty_file(os.path.join('/mount-fs', FLAG_FLUSH_CACHES))
        # in the worst case it takes 1 sec for the watch script on the host machine to realize
        time.sleep(10)
        print("write:", FLAG_FLUSH_CACHES)
        # also check if the I/O speed is limited
        assert check_read_speed_below_threshold('/mount-fs', mb_s_threshold=200), \
            "I/O limit for docker container does not seem to be active (look into prevent_caching directory for info)"


def check_read_speed_below_threshold(base_path, mb_s_threshold=200):
    speed_value, speed_unit = get_read_speed(base_path)

    # Convert speed to MB/s for comparison
    if speed_unit == "KB":
        speed_in_mb = speed_value / 1024
    elif speed_unit == "MB":
        speed_in_mb = speed_value
    elif speed_unit == "GB":
        speed_in_mb = speed_value * 1024
    else:
        raise ValueError("Unexpected unit in read speed.")

    is_below_threshold = speed_in_mb < mb_s_threshold
    return is_below_threshold


def get_read_speed(base_path):
    tmp_file_name = '1GB_random_file'
    file_path = f'{base_path}/{tmp_file_name}'
    # check if file exists otherwise write it
    if not os.path.exists(file_path):
        # create file
        cmd = ["dd", "if=/dev/zero", f"of={file_path}", "bs=1M", "count=1024"]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        time.sleep(10)

    # read file
    assert os.path.exists(f'{file_path}')
    cmd = ['dd', f'if={file_path}', 'of=/dev/null']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    # Output from dd is usually sent to stderr
    output = result.stderr

    # Use regex to find the read speed in the output
    match = re.search(r'(\d+\.?\d*\s+[KMG]?B/s)', output)

    if match:
        match_str = match.group(1)
        print('measured:', match_str)
        split = match_str.split(" ")
        read_speed = float(split[0])
        unit = split[1].replace("/s", "")
        return read_speed, unit
    else:
        raise ValueError("Read speed not found in the command output.")


def write_empty_file(file_path):
    with open(file_path, 'w') as f:
        pass


def active_file_up_to_date(base_path, accepted_duration):
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
    # active_file_up_to_date('/Users/nils/uni/programming/model-search-paper/experiments/prevent_caching', 5)

    base_path = '/mount-fs/io-test'
    # speed = get_read_speed(base_path)
    # print(f"Read Speed: {speed}")
    print("below:", check_read_speed_below_threshold(base_path, 200))
