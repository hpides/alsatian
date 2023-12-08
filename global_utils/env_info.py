import importlib


def get_system_info():
    # Check if psutil and GPUtil are available
    try:
        importlib.import_module('psutil')
        importlib.import_module('GPUtil')
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required libraries using 'pip install psutil gputil'")
        return None

    import psutil
    import GPUtil

    # Get CPU information
    cpu_info = {
        'cpu_cores': psutil.cpu_count(logical=False),
        'total_cpu_threads': psutil.cpu_count(logical=True)
    }

    # Get memory information
    memory_info = {
        'total_memory': psutil.virtual_memory().total,
        'available_memory': psutil.virtual_memory().available
    }

    # Get GPU information
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [
            {
                'gpu_id': gpu.id,
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_free': gpu.memoryFree
            }
            for gpu in gpus
        ]
    except Exception as e:
        gpu_info = f"Error retrieving GPU information: {e}"

    system_info = {
        'cpu': cpu_info,
        'memory': memory_info,
        'gpu': gpu_info
    }

    return system_info


if __name__ == "__main__":
    system_info = get_system_info()
    print(system_info)
