# Fine Tune VS Search Time Experiment

- in this experiment we want to get insights in how long a typical fine-tuning process takes and compare it to the time
  it takes to search through a set of models
- we perform the following experiments
  - transfer learning (feature-extraction) ([details](./fine_tuning))
  - transfer learning (full fine-tuning) ([details](./fine_tuning))
  - extracting features (no training at all) ([details](./calc_features))

## Hardware
- for the experiments we had the following setup
- **Hardware**
  - DES GPU server
  - 2 GPUs (NVIDIA RTX A5000, 24GB) 
  - AMD Ryzen Threadripper PRO 3995WX 64-Cores, 128 logical cores
- **Our Setup**
  - docker container
  - access to one GPU, and 32 logical cores 
  - GPU Driver Version: 535.129.03   CUDA Version: 12.2
  - Python 3.8.10
  - PyTorch 12.1.1, torchvision 0.16.1

