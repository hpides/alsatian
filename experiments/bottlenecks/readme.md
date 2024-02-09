# EXPERIMENT NAME

## Experiment Overview

- **goal**
    - In this set of experiments we want to find out where the bottlenecks are for the model search workload as
      described in SHiFT
    - questions to be answered are
        - Q1: what is the bottleneck: feature extraction or the final ranking of the model based on extracted features (
          e.g. by training a FC layer)
        - Q2: where are the bottlenecks for the sub-steps
            - Q2.1: where are the bottlenecks for feature extraction
            - Q2.2: where are the bottlenecks for ranking
- **workload**
    - the high level workload should be a search query over a selected model as, for example, described in SHiFT
    - to have more reliable results we will vary the workload by 
      - domain / model architecture
      - dataset size
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

## Models
- we use the following models ...
- we use the following data

- overview of what experiments are conducted

## Analysis

- notes of analysing the experiments -> used as foundation to write the text in the paper
- also include plots in markdown files
    - ![Alt text](path_to_image_file.png){:width="300px"}



