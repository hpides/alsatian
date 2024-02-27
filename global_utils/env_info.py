import os
import platform
import subprocess
import psutil

def get_cpuset_cpus():
    try:
        cpuset_cpus = subprocess.check_output(['cat', '/sys/fs/cgroup/cpuset.cpus']).decode().strip()
        return cpuset_cpus
    except subprocess.CalledProcessError:
        return "Unknown"

def get_system_info():
    system_info = {}

    # Operating System
    system_info['OS'] = platform.system()

    # CPU information
    cpu_info = {}
    cpu_info['type'] = platform.processor()
    cpu_info['cores'] = psutil.cpu_count(logical=False)
    cpu_info['cores (logical)'] = psutil.cpu_count(logical=True)
    cpu_info['cpus avail to docker'] = get_cpuset_cpus()
    system_info['CPU'] = cpu_info

    memory_info = {
        'total_memory': psutil.virtual_memory().total,
        'available_memory': psutil.virtual_memory().available
    }
    system_info['memory'] = memory_info


    # GPU information (for NVIDIA GPUs)
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).decode().strip().split(',')
        system_info['GPU'] = {
            'type': gpu_info[0].strip(),
            'driver_version': gpu_info[1].strip()
        }
    except subprocess.CalledProcessError:
        system_info['GPU'] = "No NVIDIA GPU detected"


    return system_info

if __name__ == '__main__':
    # Example usage:
    system_info = get_system_info()
    print(system_info)
