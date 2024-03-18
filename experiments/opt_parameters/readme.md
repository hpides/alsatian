# Optimal Parameters experiments

- in this set of experiments we analyse the effect of the parameters batch size and number of workers on data loading
  and inference time
- next to getting insight that will help throughout the paper, we also want to select reasonable parameters for our
  bottleneck analysis which marks the motivation and entry point of our paper

## Hardware Setup

- we use our groups GPU server with the setup descried in [readme.md](..%2F..%2Fexp_environment%2Freadme.md) with 64
  core

## Results

- the results are structured in three directories, every directory comes with its own readme, experiments, plots, and
  short analysis
- [inf_times](inf_times): analyses the effect of the batch size on the inference time (dataloading excluded)
- [data_loading](data_loading): analyses the effect of batch size and number of workers on data loading
- [num_workers_and_batch_size](num_workers_and_batch_size): analyses the combined effect of number of workers and batch
  size on the end to end time execution time of an inference workload