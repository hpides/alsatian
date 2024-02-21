# Model Search Bottlenecks

## Experiment Overview

- **goal**
    - When searching through different DL models the standard procedure (as proposed in many papers asuch as SHiFT)
      consist of two steps:
        - (1) extracting features and
        - (2) ranking/scoring the model on the extracted features
    - In this experiment we want to analyze where the bottlenecks are for this procedure. We will first look at entire
      models and after that at reduced and composed models
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

## Max batch size

- as a first experiment we wanted to find out the maximum batch size each model can handle on the given hardware
- for this we use the script [find_max_batch_size.py](opt_parameters%2Ffind_max_batch_size.py)
- on the DES GPU server with the hardware described above we get the following results
- {'resnet152': 1024, 'resnet101': 1024, 'resnet50': 1024, 'resnet34': 1024, 'resnet18': 2048, 'mobilenet_v2': 1024, '
  eff_net_v2_s': 1024, 'eff_net_v2_l': 2048, 'vit_b_16': 2048, 'vit_l_16': 2048, 'vit_b_32': 2048, 'vit_l_32': 4096}
- so far we only focused on the vision models, and the hardware listed above, other settings are future work/WiP

## Measure batch size impact

- while one could assume that the largest batch size is always the best since we can max out the GPU, we show that for
  our setting this is actually not true
- we use the code in [batch_size_impact.py](opt_parameters%2Fbatch_size_impact.py) to test the impact on different batch
  sizes on the end to end time and different sub steps such as: inference, load_data, and data to gpu
- the results for the DES GPU server (and reduced resources in docker container) can be found here
    - [2024-02-20-13:30:29#batch_size_impact.json](opt_parameters%2Fresults%2Fbatch_size_impact%2F2024-02-20-13%3A30%3A29%23batch_size_impact.json)
- so far we only focused on the vision models, and the hardware listed above, other settings are future work/WiP
- looking at the plot below, we see that batch size 128 seems to be the best batch size, at least for the vision models
  and our setup, so we will form now one us this batch size for our experiments
![des_gpu_vision_measurements.png](opt_parameters%2Feval%2Fplots%2Fdes_gpu_vision_measurements.png)

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

- imagemodels: ImageNette (subclass of Imagenet dataset)
    - https://github.com/fastai/imagenette (we use the full sized version)
- text: TBD

## Analysis

- notes of analysing the experiments -> used as foundation to write the text in the paper
- also include plots in markdown files
    - ![Alt text](path_to_image_file.png){:width="300px"}



