## Experiment

- We just want to get basic insights into intermediate and parameter sizes of models we will use for our evaluation
- We might extend this experiment in the future to analyze models not only on a block but on a layer granularity

#### Results
Find plotted results under [plots](eval%2Fplots). We analyze 
- the **inference time** per block in plots with the prefix: `gpu_inf_times`
- the number of **parameters per block** in plots with the prefix: `num_params` and `num_params_mb`
- the **block output size** (assuming the model gets (224,224,3)-shaped inputs) in plots with the prefix: `output_size_mb`

- the experiment environment is documented in every single result file
- the single result files can be found here: TODO insert path once we can publish it