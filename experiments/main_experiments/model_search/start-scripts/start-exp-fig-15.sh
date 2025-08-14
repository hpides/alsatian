#!/bin/sh

# Assigning command line arguments to variables
repository_url=https://github.com/hpides/alsatian.git
script_to_start=experiments/main_experiments/model_search/run_experiment_fig_15.py
python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
branch=reproducibility
repo_name=alsatian
config_file=experiments/main_experiments/model_search/config.ini
config_section=des-gpu-imagenette-huggingface-search-fig15


git clone $repository_url

# Change directory to the cloned repository
repo_name=$(basename "$repository_url" .git)
cd $repo_name

git pull
git checkout $branch
git pull 

# Check if the specified script exists
if [ ! -f "$script_to_start" ]; then
    echo "Script '$script_to_start' not found in the repository."
    exit 1
fi

# Set the PYTHONPATH to include the current directory
export PYTHONPATH=$PWD

# Execute the specified Python script
$python_dir $script_to_start --config_file $config_file --base_config_section $config_section
