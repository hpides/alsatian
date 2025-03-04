import os
import subprocess
import time
from datetime import datetime

FLAG_FLUSH_CACHES = "flag-flush-caches"
BASE_PATH = "/fs/nils-strassenburg/docker-mounted/mount-fs"


def execute_command(command):
    subprocess.run(command, shell=True)


def renew_active_file(base_path):
    # delete olf file if existing
    files = [f for f in os.listdir(base_path) if f.startswith('active-')]
    for old_file in files:
        os.remove(os.path.join(base_path, old_file))

    # Get the current timestamp
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(base_path, f"active-{current_timestamp}")

    # Create a new file with the current timestamp
    with open(file_name, 'w') as f:
        pass

    print(f"Created file: {file_name}")


def empty_caches_if_triggered(flash_caches_flag_filename):
    command = "echo 3 | sudo tee /proc/sys/vm/drop_caches"
    print("looking for:", flash_caches_flag_filename)
    if os.path.exists(flash_caches_flag_filename):
        print(f"File '{flash_caches_flag_filename}' found. Executing command...")
        execute_command(command)
        os.remove(flash_caches_flag_filename)
        print("delete caches cmd executed")


def main():
    base_path = BASE_PATH
    flash_caches_flag_filename = os.path.join(base_path, FLAG_FLUSH_CACHES)

    renew_active_file(base_path)
    renew_active_file_counter = 0

    while True:
        # check if caches should be emptied every 1 second
        empty_caches_if_triggered(flash_caches_flag_filename)
        time.sleep(1)

        # every 5 seconds also give some indication that the script is still running and has not crashed
        renew_active_file_counter += 1
        if renew_active_file_counter == 5:
            renew_active_file(base_path)
            renew_active_file_counter = 0
        print(renew_active_file_counter)


if __name__ == "__main__":
    main()
