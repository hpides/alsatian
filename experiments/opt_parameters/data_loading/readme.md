# Detailed Analysis: batch sizes & number of workers -> load data times

#### Pytorch data loader

- parallel data loading: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    - interesting parameters to look at: pin_memory & pin_memory_device (seems like we can load the batches into a
      pinned memory region on the GPU)
- when setting the `num_workers` parameter of the dataloader to a positive value *n*, *n* many worker processes will
  be started that load batches in parallel
- **every worker is responsible for loading one batch** but there are discussions of how to adjust that if loading
  the first batch is the bottleneck in the training process
    - https://discuss.pytorch.org/t/the-samples-in-one-batch-are-loaded-in-parallel-while-enumerating-dataloader/159163
    - https://discuss.pytorch.org/t/multiple-workers-for-single-batch/126473

#### Effect of number of workers and inference time

- for now let's assume we hava a fixed batch size and 2 worker processes
- the situation is visualized below
- ![data_loader-Page-1.png](data_loader-Page-1.png)
- at fist the dataloader is initialized and two additional worker processes are spawned
- each worker process loads a batch independently
- this means, the GPU waits longest to get the first batch (b1), but batch (b2) is ready almost immediately after so the
  delay for loading b2 is very small
- while waiting for the first batch the GPU idles, weather or not the GPU idles also while waiting for the second batch
  depends on the duration of the inference:
    - if the model is large and the inference takes long, a new batch will always be ready
    - if the model is small and the inference is faster than loading the data, the GPU might not have to wait for b2,
      but it will most likely wait for b3 because one worker process needs longer to load one batch than performing
      inference on two tuples
- to verify this behaviour, we perform the following experiment:

#### Experiment

- we run a micro benchmark using the following script [data_loading_experiment.py](data_loading_experiment.py)
- it can be executed on the server by adapting the
  script [run_data_loading_exp_on_server.sh](run_data_loading_exp_on_server.sh)
- the script runs over the follwoing configurations
    - *batch size*: 32, 128, 256, 512, 1024
    - *number of workers*: 1, 2, 4, 8, 32, 48, 64
    - *sleep duration*: to simulate the time it would take to perform inference
        - None -> no sleep -> simulate small model/no inference cost
        - 2s -> simulates a larger model/longer inference time
    - dataset_type
        - preprocessed_ssd: saves data in the shape (3,224,224) on SSD, meaning there is no data loading require
            - here a single item is 3*224*224*4 Byte ~= 0.6MB large, meaning batch has the following sizes:
                - 32: 19.2 MB
                - 128: 76.8 MB
                - 256: 153 MB
                - 512: 306 MB
                - 1024: 612 MB
        - imagenette: uses the imagenette dataset with the standard inference pre-processing pipeline for imagenet
          models
            - not every image has the same size ~=117 KB -> 0.12 MB
            - this means the data that has to be loaded form disk is approx 5X smaller

#### Results

- the experiment was executed on the hardware as described in [readme.md](..%2F..%2F..%2Fexp_environment%2Freadme.md)
  and is also documented in every single result file
- the single result files can be found here: https://nextcloud.hpi.de/f/11781487

##### Effect of batch size

- when not overlapping data loading, the time it takes to load roughly grows linear with the batch size
- example here for no sleep and loading imagenet data (left) and loading preprocessed data from SSD (right)

<p float="left">
  <img src="./plots/batch_size_impact/workers-1-sleep-None-data-imagenette.png" width="400" />
  <img src="./plots/batch_size_impact/workers-1-sleep-None-data-preprocessed_ssd.png" width="400" /> 
</p>

##### Effect of number of workers

- for **1 worker** we see an almost constant load time per batch over time (see above)
    - when loading actual images we see more fluctuation because not all images have same size
    - when loading preprocessed data from SSD more or less constant because all items have exact same size
- for **2 or more workers** we see the behaviour that we predicted above
    - assume the inference is faster than loading a batch (sleep None), then we see peaks every n batches because we
      have to little
      overlap (left image)
    - when the inference takes longer (sleep 2) than loading a batch, loading first batch has high time, then almost
      instantaneous (right image)
- **large batch size can lead to data loading bottleneck**
    - what we also see is that the larger the batch size the higher the peaks become and the higher the time to load the
      very first batch, if overall data loading is the bottleneck, it can be the case that (even though inference with a
      larger batch size might be faster) a smaller batch size is the better choice for end to end time because then the
      peaks, and especially the time to load the first batch are shorter

<p float="left">
  <img src="./plots/batch_size_impact/workers-2-sleep-None-data-imagenette.png" width="400" />
  <img src="./plots/batch_size_impact/workers-2-sleep-2-data-imagenette.png" width="400" /> 
</p>

- varying the number of workers more we can observe
    - the more workers, the longer it takes to load the initial batch because of initializing processes, compete for
      resources
    - long run the more workers, the lower the number of spikes because of limited parallel preprocessing
        - 1 worker almost constant time
        - 2 workers, peaks every 2 batches
        - 4 workers peaks every 4 batches, ...
- <p float="left">
  <img src="./plots/workers_impact/batch_size-256-sleep-None-data-preprocessed_ssd.png" width="600" />

</p>

##### Effect of sleep time

- see above
- when no sleep -> data loading is bottleneck, we see delays
- when sleep long enough -> inference bottleneck, data loading almost instantaneous

##### Effect of dataset type

- loading actual images, slower and more fluctuations
- loading preprocessed data faster and almost no fluctuations
- **interesting**: a preprocessed item is 5X as large as the average image but the data loading of the preprocessed data
  is still significantly shorter -> when loading actual images we are compute bottlenecked
- for images see section effect on batch szie above
