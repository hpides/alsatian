# Optimal Parameters experiments

- in this set of experiments we investigate what are good parameters to use for a given set of models and dataset
- for now, we focus on:
    - **models**: vision models
    - **data**: imagenet like data, read from disk
    - **parameters we want to find out**:
        - number of workers used for data loading
        - best batch size

## Hardware Setup

- we use our groups GPU server with the setup descried in [readme.md](..%2F..%2Fexp_environment%2Freadme.md)
- depending on the experiment, we use 32 or 64 (virtual) cores

## Max Batch size

- before we optimize batch size and number of workers, we want to know what the max batch size is
- for this we use the script [find_max_batch_size.py](find_max_batch_size.py)
- on the DES GPU server with the hardware described above we get the following results
- {'resnet152': 1024, 'resnet101': 1024, 'resnet50': 1024, 'resnet34': 1024, 'resnet18': 2048, 'mobilenet_v2': 1024, '
  eff_net_v2_s': 1024, 'eff_net_v2_l': 2048, 'vit_b_16': 2048, 'vit_l_16': 2048, 'vit_b_32': 2048, 'vit_l_32': 4096}
- after this short analysis we decide to (for now) use a max batch size 1024 for all models

## Data loading

- for a detailed analysis of how batch size, number of workers, the inference time and the type of dataset affect the
  delay induced by loading data see here [data_loading](data_loading)

# OLD

## Impact of number of workers and batch size

- to investigate the influence of batch sizes and number ofr workers we use the
  script [worker_batch_size_impact.py](worker_batch_size_impact.py)
- it can be started on the server using and adjusting the script [run_exp_on_server.sh](run_exp_on_server.sh)
- we use the configuration `opt-params-des-gpu` as specified in [config.ini](config.ini)
- the search space is
    - range of workers: 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48
    - range of batch sizes: 32, 128, 256, 512, 1024
        - NVIDIA advises to use a multiple of 256:
        - https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
- we execute the script for 10*1024 items in the dataset
    - this means all runs process the same amount of data, but the number of batches processed is different

### Results

- we executed experiments in two setups one with 32 available CPUs, one with 64 available CPUs available to docker
    - see [readme.md](..%2F..%2Fexp_environment%2Freadme.md) for details
- the results can be found here (for now only internal access)
    - 32 CPUs: https://nextcloud.hpi.de/f/11738389
    - 64 CPUs: https://nextcloud.hpi.de/f/11749084

### Analysis

- we use the code in [plotting.py](eval%2Fplotting.py) to generate the plots

### End-to-end times

- **impact number of workers**
- general trend
    - the batch size influences the end-to-end time, but the analysis below is valid independednt of the used batch size
    - when increasing the number of workers the end-to-end time goes down until it reaches minimum
    - after reaching the minimum the end-to-end time increases marginally again -> so using more workers does not do
      any harm
    - increasing the number of CPUs does not have a significant impact on the end-to-end times or the general trend
        - 32 (virtual) CPUs seem to be enough to not be CPU-compute bottlenecked?
- model dependent optimal num workers
    - for our largest model VIT_L_16, we see that the number of workers has almost no impact
    - for larger models like the VIT_L_32 RENSET_152 the end to end time is shortest for 2 or 4 workers respectively
    - for smaller models like the EFFICIENTNET_2_S, MOBILENET_V2, or RESNET_18 the shortest end to end time is reached
      using 12 workers, while using more workers increases the end to end times marginally

- **impact batch size**
- general trend
    - increasing the number of CPUs does not have a significant impact on the end-to-end times or the general trend
        - 32 (virtual) CPUs seem to be enough to not be CPU-compute bottlenecked
- EFF_L: 128 and 256 almost identical lowest
- EFF_S: 128 lowest consistently
- MOB_V2: almost always 32 best, 128 close
- RES_18: 32 best, 128 close
- RES_34: 128 best
- RES_50: 128 best
- RES_101: 128 best, 256 close
- RES_152: 256 best, 128 close
- VIT_b_16: 128 best
- VIT_b_32: 128 best
- VIT_l_16: 256 best
- VIT_l_32: 256 best

## Detailed Analysis: batch sizes -> inference times

- with inference time in this case we mean processing the forward pass on the GPU, this explicitly excludes the time
  to load data or the model to the GPU
- expected behaviour: inference time/item: decreases with increased batch size
- investigation
    - we plot for every model and batch size the average time to process one item (e.g. take inference time for batch
      size 32 and divide by 32)
    - we see that for all models and increased batch size decreases the inference time
    - **TODO** conform this in new run where

## Detailed Analysis - Data to device + Batch Sizes

- time depends on: memory allocation, latency, throughput
- expected: assuming constant troughput + latency, with larger batch size allocation and latency should be amortized
  more -> larger batch size shorter time per item (unless allocating more GPU memory is very expensive)
- TODO

## Detailed Analysis - Load data + Batch Sizes + Num workers

- (1) with fixed batch size, up until a certain number of workers the load data time per item should improve
    - TODO
- (2) for fixed number of workers, the load data time per item should ?? increase slighlty because higher canche of
  straggeler (not all images have same size)

- experiment we should to: load data form memory vs load data from SSD AND DATA preprocessed and not preprocessed
