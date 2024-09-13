# Model Search Bottlenecks

## Experiment Overview

- **goal**
    - When searching through different DL models the standard procedure (as proposed in many papers such as SHiFT)
      consist of two steps:
        - (1) extracting features and
        - (2) ranking/scoring the model on the extracted features
    - In this experiment we want to analyze where the bottlenecks are for this procedure. We will first look at entire
      models and after that at reduced and composed models
- **workload**
    - we will use a simple feature extraction and model ranking pipeline
    - we will vary the workload by
        - model architecture
        - number of items
        - if the dataset is preprocessed or not
        - what portion of model we execute inference over
- we run the experiment by running the
  script [prepare_exp_variants.py](model_rank%2Fon_server_exec%2Fprepare_exp_variants.py) to generate the files
    - [run-all-exp.sh](model_rank%2Fon_server_exec%2Frun-all-exp.sh)
    - [tmp-config.ini](model_rank%2Fon_server_exec%2Ftmp-config.ini)
- we need to adjust the script [run_exp_template.sh](model_rank%2Fon_server_exec%2Frun_exp_template.sh) for the target
  setup, copy over the files generated in the previous step and then execute the script run-all-exp.sh
- ImageNette (subclass of Imagenet dataset)
    - https://github.com/fastai/imagenette (we use the full sized version)

#### Results

- the experiment was executed on the hardware as described in [setup](..%2F..%2F..%2Fsetup)[readme.md](..%2F..%2Fexp_environment%2Freadme.md)
  and is also documented in every single result file
- the results of the experiment should be placed in [results](model_rank%2Fresults) to be available for plotting
- the results that we show in the paper can be downloaded from our nextcloud



