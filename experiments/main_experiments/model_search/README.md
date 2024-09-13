# Model Search Experiments

We run several types of experiments in the following we give a short description of what they measure and how to run
them

## General

Every experiment (group) consists of:

- a python script that runs the set of experiments e.g. different model architectures etc.
- a section in the [config.ini](config.ini) to define the default parameters
- a start script (saved in [start-scripts](start-scripts)) that is used to run the experiment in a freshly cloned repo

Results of experiments in the paper:

- we are happy to share the result files used for the plots in the paper, please reach out to us so that we can share
  a link to ur next cloud. (Internal link: https://nextcloud.hpi.de/index.php/f/11738380)

### Running an existing experiment:

- we run our experiments in a docker container
    - a prebuild image (`slin96/model-search-experiments:latest`) can be
      found [here](https://hub.docker.com/repository/docker/slin96/model-search-experiments/general)
    - instructions on how to start the container or build a new one can be
      found [here](..%2F..%2F..%2Fsetup%2Fdocker_setup)
    - to have a realistic setup and prevent caching of result from previous runs, we
        - limit the I/O of the docker container for specific directories
        - and run a script on the host machines that empties caches when triggered by the experiment script
        - see [readme.md](..%2Fprevent_caching%2Freadme.md) for details

- to run an existing experiments we need to ssh into the target machine, make sure the model snapshots and datasets are
  available and then run the start script of our choice
    - we give detailed instructions on how to run individual experiments below

### Run a new experiment

- to run a new experiment we:
- chose a python run script,
  e.g. [run_experiment_set_synthetics_snapshots.py](run_experiment_set_synthetics_snapshots.py) or create a new one that
  reflects the experiment we want to execute
- generate a new config by generating an entirely new config file or add a new section to an existing config file
- generate a new run script by copying the [start-exp-template.sh](start-scripts%2Fstart-exp-template.sh)
  and set the following parameters (see example below):
    - the GitHub access token can be generated on the GitHub web page, but might not be needed anymore once the repo is
      public
      ``` 
      repository_url=github.com/slin96/model-search-paper.git
      script_to_start=experiments/model_search/run_experiment_set_bert.py
      github_access_token_file=./access_token
      python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
      branch=synsthetic-snapshot-median-run
      repo_name=model-search-paper
      config_file=experiments/model_search/config.ini
      config_section=des-gpu-bert-synthetic
      ```

### Model Snapshots

- in our experiments we use synthetic model snapshots as well as trained model snapshots
- synthetic snapshots
    - when using synthetic model snapshots, the snapshots will be automatically generated when running the experiment
      for the first time (can take up to hours) and will be saved/cached in the directory specified in the config file
      so that they can be reused for following runs
    - to generate snapshots manually without executing a model search experiment, we can adjust the main function
      in [generate_set.py](..%2Fsnapshots%2Fsynthetic%2Fgenerate_set.py)
- trained snapshots
    - short description here [trained](..%2Fsnapshots%2Ftrained)
- our scripts are designed such that once snapshots are generated (which can take a significant amount of time) they are
  reused. This implies (a) you have to delete existing snapshots to make sure you generate new ones (b) you can get our
  generated snapshots upon request if you want to reproduce our experiments

### Datasets

- for our experiments in the paper we use subsets of the _Imagenette_ and the _Imagewoof_ datasets
    - details on our used dataset can be found [here](..%2F..%2F..%2Fdata)
    - to specify how many items of the full dataset we use for search is specified using the config
      parameters `num_train_items` and `num_test_items`

