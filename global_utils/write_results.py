import json
import os
import subprocess
from datetime import datetime

from global_utils.dependencies import get_pip_dependency_dict
from global_utils.env_info import get_system_info


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        return ""


def write_measurements_and_args_to_json_file(measurements, args, dir_path, file_id=""):
    # Open file in write mode
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S#")
    file_path = os.path.join(dir_path, f'{timestamp}{file_id}.json')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    commit_hash = get_git_commit_hash()

    results = {'git-hash': commit_hash,
               'pip-dependencies': get_pip_dependency_dict(),
               'system_info': get_system_info(),
               'measurements': measurements
               }
    if isinstance(args, dict):
        results['args'] = str(args)
    else:
        results['args'] = vars(args)

    with open(file_path, 'w') as json_file:
        # Write dictionary as JSON to the file
        json.dump(results, json_file)
