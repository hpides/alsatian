# Experiments

This readme gives an overview of the experiments we ran for our paper. Not all of them made it in the actual paper.
- [Bottlenecks](main_experiments/bottlenecks): analyzes current bottlenecks when searching through models
- [Model search](main_experiments/model_search): contains all the material for the experiments that search through a set
  of models
- [Hyperparameters](side_experiments%2Fopt_parameters%2Freadme.md): analysis of the effect of the parameters _batch
  size_ and _number of
  workers_ on data loading and inference time.
- [Model resource info](side_experiments%2Fmodel_resource_info%2Freadme.md): basic analysis of inference times,
  parameters sizes and intermediate outputs on a block level granularity.
- [Merged model vs sequential execution](micro_benchmarks%2Fseq_vs_merged_model%2Freadme.md): microbenchmark analyzing the inference
  time of one merge pytorch model or the sequential execution of individual modules


