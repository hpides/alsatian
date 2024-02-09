import time

import torch

CPU = 'cpu'
CUDA = 'cuda'


class Benchmarker:
    def __init__(self, device: torch.device):
        assert isinstance(device, torch.device), "device needs to be a valid device "
        self.device = device

    def benchmark_gpu(self, method, *args, **kwargs):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # actually executing sth
        result = method(*args, **kwargs)

        ender.record()
        torch.cuda.synchronize()  # WAIT FOR GPU SYNC
        elapsed = starter.elapsed_time(ender)
        elapsed = elapsed * 10 ** 6  # times are in ms, convert to ns to be consistent
        return elapsed, result

    def benchmark_cpu(self, method, *args, **kwargs):
        start_time = time.time_ns()

        # actually executing sth
        result = method(*args, **kwargs)

        end_time = time.time_ns()
        elapsed = end_time - start_time
        return elapsed, result

    def benchmark(self, method, *args, **kwargs):
        """
        This method takes another method, the arguments for that method and a device. Depending on the device it benchmarks
        the method given its parameters. The measurements are done in nano seconds (ns)
        """
        if self.device.type == CUDA:
            elapsed, result = self.benchmark_gpu(method, *args, **kwargs)
        else:
            elapsed, result = self.benchmark_cpu(method, *args, **kwargs)

        return elapsed, result
