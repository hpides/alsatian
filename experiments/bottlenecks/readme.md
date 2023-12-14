# Bottleneck experiments

- in these experiments we investigate where the bottlenecks are when executing current state-of-the-art model strategies
  on different amounts of data

## Hardware

- for the experiments we had the following setup
- **Hardware**
    - DES GPU server
    - 2 GPUs (NVIDIA RTX A5000, 24GB)
    - AMD Ryzen Threadripper PRO 3995WX 64-Cores, 128 logical cores
- **Our Setup**
    - docker container
    - access to one GPU, and 32 logical cores
    - GPU Driver Version: 535.129.03 CUDA Version: 12.2
    - Python 3.8.10
    - PyTorch 12.1.1, torchvision 0.16.1

## Analysis

- we execute ...
- 