# Choosing parameters for batch size and number of workers

#### Experiment

- when performing inference two parameters that can significantly influence the runtime are:
    - number of workers: indicating the number of processes that are used in parallel to load and transform the input
      data
    - batch size: the number of data items that are processed at once on the GPU
- we already investigated the effects of ... in separate experiments
    - number of workers on data loading time ([data_loading](..%2Fdata_loading))
    - batch size on pure inference time ([inf_times](..%2Finf_times))
- in this experiment we want to investigate the joint effect and chose a good tuple of batch size and number of workers
  for our bottleneck experiments. We explicitly *do not* want to find the absolut best configuration for every single
  model, but rather a configuration that gives us reasonably good results for all models
- we distinguish between two settings
    - (1) the data (in our case image data) is stored as images and needs to be decoded and normalized before feeding it
      to the model which is part of the data loading process
    - (2) the data is already preprocessed and can be accessed in item granularity as binary data, this means the
      process is mainly I/O bound because there is no heavy data transformation to be done
- for every setting run an experiment measuring the end to end time and sub-times and vary the following parameters
    - data (two options see above)
    - models architecture
    - batch size
    - number of workers

#### Results

- the experiment was executed on the hardware as described in [readme.md](..%2F..%2F..%2Fexp_environment%2Freadme.md)
  and is also documented in every single result file
- the single result files can be found here: https://nextcloud.hpi.de/f/11798644
- plotting code can be found in [eval](eval), the actual plots in [plots](plots)
- we have three types of plots
    - (1) fixed batch size varying the number of workers
    - (2) fixed number of workers varying the batch size
    - (3) fixed model vary number of workers and batch size

#### Analysis

- general comment:
    - when optimizing for end to end time and choosing batch size and number of workers we have to consider
      that we should set the number of workers large enough to not be bottlenecked by data loading or at least limit
      this
      bottleneck
    - so naively one could say lets just set the number of workers to the highest possible number to be safe, but we see
      in the detailed analysis that the more workers we use the higher the overhead is to load the first batch which
      might be a significant delay when considering runs that only have a low number of batches to process in total
    - this time/delay to load the first batch is heavily influenced by the first batch size because every worker loads
      one batch independently, so doubling the batch size might double the size to load the first batch

- looking at the plots we can see trends, but no single clear winner in terms of batch_size workers combination
- this is why we take the top 3 configurations for every model, count them and sort the output
    - code in [top_n_configs.py](eval%2Ftop_n_configs.py)
- imagenette data (non preprocessed)
    - we see batch size **128** is best, and number of workers: 8, **12**, 16
    - we decided for the ones marked bold (12 also reasonably covers 16 but is not too much higher than 8)

```
{'batch_size': 128, 'num_workers': 8}: 6
{'batch_size': 128, 'num_workers': 12}: 5
{'batch_size': 128, 'num_workers': 16}: 5
{'batch_size': 256, 'num_workers': 4}: 4
{'batch_size': 128, 'num_workers': 4}: 4
{'batch_size': 256, 'num_workers': 2}: 3
{'batch_size': 256, 'num_workers': 8}: 2
{'batch_size': 32, 'num_workers': 16}: 2
{'batch_size': 32, 'num_workers': 12}: 2
{'batch_size': 128, 'num_workers': 2}: 2
{'batch_size': 256, 'num_workers': 1}: 1
```

- preprocessed data
    - we see batch size **256** is best, and number of workers: **2**, 1, 4
    - we decided for the ones marked bold which is the top ranked configuration

```
{'batch_size': 256, 'num_workers': 2}: 8
{'batch_size': 256, 'num_workers': 1}: 7
{'batch_size': 256, 'num_workers': 4}: 6
{'batch_size': 128, 'num_workers': 2}: 5
{'batch_size': 512, 'num_workers': 1}: 3
{'batch_size': 1024, 'num_workers': 1}: 2
{'batch_size': 128, 'num_workers': 1}: 2
{'batch_size': 128, 'num_workers': 4}: 1
{'batch_size': 512, 'num_workers': 2}: 1
{'batch_size': 1024, 'num_workers': 2}: 1
```




