# Bottleneck Analysis

- related work shows that in the search process feature extraction is the bottleneck
    - e.g. Huang et al., Li et al.
- Li et al. also show that
    - training a linear classifier is approx and order mag faster than the feature extraction
    - NLEEP is multiple orders of mag faster than feature extraction and also training a linear classifier
    - NLEEP approx 10x faster than LEEP
- Huang et al.
    - LEEP among the fastest
    - other scores are up to approx 10x slower
    - (combining this with findings form above) all the "optimized scores" should still be much more efficient than
      training a linear classifier

- thus for our experiments we take the training of a linear layer as the upper bound

## Times of interest

- we want to analyze where in the process of model search the bottlenecks are
- therefore we execute some characteristic workloads and track the following times
    - time for training the final linear classifier
    - time of the accumulated data loading
    - time for the pure inference (without data loading)
    - time to load the model form disk
    - time to initialize the model
- we want to see how the times behave for different numbers of
    - samples used for inference
    - across different models
- **assumptions**
    - lets assume the models are saved on a HDD with a read speed of 200MB/s
        - we can easily adjust these numbers at a later point in time
        - we will just use the size of the snapshot and will divide it by 200MB/s to get the estimated time

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

