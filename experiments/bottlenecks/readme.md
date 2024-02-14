# Model Search Bottlenecks

## Experiment Overview

- **goal**
    - When searching through different DL models the standard procedure (as proposed in many papers asuch as SHiFT)consist of two steps:
        - (1) extracting features and
        - (2) ranking/scoring the model on the extracted features
    - In this experiment we want to analyze where the bottlenecks are for this procedure. We will first look at entire models and after that at reduced and composed models
- **workload**
    - we will use a simple feature extraction and model ranking pipeline
    - we will execute for a total of 50 batches (first few batches to warm up the GPU, then get some measurements)
    - we will vary the workload by
        - domain / model architecture
        - full/partially/joined models
        - (batch size): could use the max batch size working for all/most models on our GPU
        - hardware: GPU setup and CPU setup
- **overview of folder structure**
    - TODO

## Hardware setup

### Reduced DES GPU server setup

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

## Conducted Experiments

### Models (Feature Extraction)
- model architectures
    - efficientNet, mobilenet, resnet18, resnet 50, resnet152, ViT
    - BERT, RoBERTa
- model configurations
    - full, last 50%, 25%, 3#, 1# layers
    - joint model with multiple heads: last 50%, last 3#

### Models (Model Scoring)
- one fully connected layer with
    - 1000, 100, 2 output classes

### Datasets
- imagemodels: imagenet
- text: TBD


## Analysis

- notes of analysing the experiments -> used as foundation to write the text in the paper
- also include plots in markdown files
    - ![Alt text](path_to_image_file.png){:width="300px"}



