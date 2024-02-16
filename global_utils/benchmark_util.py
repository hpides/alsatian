import time
from typing import Dict

import torch

CPU = 'cpu'
CUDA = 'cuda'


class Task:

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_taken = None

    def set_time_taken(self):
        self.time_taken = self.start.elapsed_time(self.end)

    def record_start(self):
        self.start.record()

    def record_end(self):
        self.end.record()


class Benchmarker:
    def __init__(self, device: torch.device):
        assert isinstance(device, torch.device), "device needs to be a valid device "
        self.device = device
        self.tasks: Dict[str, Task] = {}

    def register_cuda_start(self, task_id):
        task = Task()
        task.record_start()
        self.tasks[task_id] = task

    def register_cuda_end(self, task_id):
        self.tasks[task_id].record_end()

    def sync_and_summarize_tasks(self):
        torch.cuda.synchronize()
        for task in self.tasks.values():
            if task.time_taken is None:
                task.set_time_taken()

    def benchmark_end_to_end(self, method, *args, **kwargs):
        start_time = time.perf_counter()

        # actually executing sth
        result = method(*args, **kwargs)

        # make sure that all the GPU tasks are finished as well
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return elapsed, result

    def micro_benchmark_gpu(self, method, *args, **kwargs):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # actually executing sth
        result = method(*args, **kwargs)

        ender.record()
        torch.cuda.synchronize()  # WAIT FOR GPU SYNC
        elapsed = starter.elapsed_time(ender)
        elapsed = elapsed * 10 ** 6  # times are in ms, convert to ns to be consistent
        return elapsed, result

    def micro_benchmark_cpu(self, method, *args, **kwargs):
        start_time = time.perf_counter()

        # actually executing sth
        result = method(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return elapsed, result

    def micro_benchmark(self, method, *args, **kwargs):
        """
        This method takes another method, the arguments for that method and a device. Depending on the device it benchmarks
        the method given its parameters. The measurements are done in nano seconds (ns)
        """
        if self.device.type == CUDA:
            elapsed, result = self.micro_benchmark_gpu(method, *args, **kwargs)
        else:
            elapsed, result = self.micro_benchmark_cpu(method, *args, **kwargs)

        return elapsed, result

    def warm_up_gpu(self):
        if self.device.type == CPU:
            return

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        # Create an instance of the model
        model = SimpleModel().to(self.device)

        # Define input data
        batch_size = 32
        input_data = torch.randn(batch_size, 10).to(self.device)

        # Warm-up the GPU by running forward pass
        with torch.no_grad():
            for _ in range(100):  # Run forward pass multiple times for warming up
                output = model(input_data)
