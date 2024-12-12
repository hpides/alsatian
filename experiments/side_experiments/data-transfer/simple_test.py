import torch
import time
import statistics

# Calculate the number of elements for a 5MB tensor
num_elements = (20 * 1024 * 1024) // 4

# Create a random tensor in CPU memory
tensor_cpu = torch.randn(num_elements, dtype=torch.float32)

# Ensure a CUDA-capable GPU is available
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
else:
    transfer_times = []  # List to store transfer times

    for _ in range(20):  # Repeat the experiment 20 times
        # Start timing
        start_time = time.time()
        tensor_gpu = tensor_cpu.to('cuda')  # Transfer to GPU
        end_time = time.time()

        # Calculate transfer time and store it
        transfer_time = (end_time - start_time) * 1000  # Convert to milliseconds
        transfer_times.append(transfer_time)

    # Calculate statistics
    avg_time = sum(transfer_times) / len(transfer_times)
    std_dev = (sum((x - avg_time) ** 2 for x in transfer_times) / len(transfer_times)) ** 0.5
    median_time = statistics.median(transfer_times)

    # Print results
    print(f"Tensor size: {tensor_cpu.element_size() * tensor_cpu.nelement() / (1024 * 1024):.2f} MB")
    print(f"Average time to transfer to GPU: {avg_time:.3f} ms")
    print(f"Median time to transfer to GPU: {median_time:.3f} ms")
    print(f"Standard deviation: {std_dev:.3f} ms")
