# Optimal Parameters experiments

- in this set of experiments we investigate what are good parameters to use for a given set of models and dataset
- for now we focus on:
    - **models**: vision models
    - **data**: imagenet like data, read from disk vs already decoded in memory
    - **parameters**: number of workers, batch size

## Hardware Setup

- **Hardware**
    - DES GPU server
    - 2 GPUs (NVIDIA RTX A5000, 24GB)
    - AMD Ryzen Threadripper PRO 3995WX 64-Cores, 128 logical cores
- **Our Setup**
    - docker container
    - access to one GPU, and **32 logical cores (default) OR 64 logical cores**
    - GPU Driver Version: 535.129.03 CUDA Version: 12.2
    - Python 3.8.10
    - PyTorch 12.1.1, torchvision 0.16.1

## Max Batch size

- before we optimize batch size and number of workers, we want to know what the max batch size is
- for this we use the script [find_max_batch_size.py](opt_parameters%2Ffind_max_batch_size.py)
- on the DES GPU server with the hardware described above we get the following results
- {'resnet152': 1024, 'resnet101': 1024, 'resnet50': 1024, 'resnet34': 1024, 'resnet18': 2048, 'mobilenet_v2': 1024, '
  eff_net_v2_s': 1024, 'eff_net_v2_l': 2048, 'vit_b_16': 2048, 'vit_l_16': 2048, 'vit_b_32': 2048, 'vit_l_32': 4096}
- after this short analysis we decide to (for now) use a max batch size 1024 for all models

## Impact of number of workers and batch size

- we use the script [worker_batch_size_impact.py](worker_batch_size_impact.py) for our experiment
- range of workers: 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48
- range of batch sizes: 32, 128, 256, 512, 1024
    - NVIDIA advises to use a multiple of 256:
    - https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html