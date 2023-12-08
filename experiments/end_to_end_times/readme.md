# Fine Tune VS Search Time Experiment

- in this experiment we want to get insights in how long a typical fine-tuning process takes and compare it to the time
  it takes to search through a set of models
- we perform the following experiments

## Hardware

- setup 1: Groups GPU server
  - we run on our groups GPU server
  - inside a docker container that has access to 1 GPU, and 32 of the CPUs
- setup 2: slurm 
  - TBD

## Run inside docker container
- sync the data using the deployment option of PyCharm (after setting up ssh interpreter just go into settings and deployment, find out whats the path, e.g.: `/tmp/pycharm_project_635`)
- log into the docker container and start a screen session where you run the experiment script
  - `screen -S nils`
  - `export PYTHONPATH=${PYTHONPATH}:/tmp/pycharm_project_635/` (insert here the path of the deployment)
  - `/home/nils/.virtualenvs/model-search-paper/bin/python /tmp/pycharm_project_635/experiments/fine_tune_vs_search_time/fine_tuning_exp.py --result_dir /tmp/pycharm_project_635/experiments/fine_tune_vs_search_time/results --data_dir /tmp/pycharm_project_635/data/imagenette2 --device cuda --new_num_classes 10 --use_defined_parameter_sets --fine_tuning_variant SPECIFY` 

- afterward the results are written into the result_dir, use scp or the deployment function of PyCharm do download the results

## Results and plotting
