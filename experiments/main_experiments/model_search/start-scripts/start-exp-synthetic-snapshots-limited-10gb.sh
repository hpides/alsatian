#!/bin/sh

# Assigning command line arguments to variables
repository_url=https://github.com/hpides/alsatian.git
script_to_start=experiments/main_experiments/model_search/run_experiment_set_limited_mem-10gb.py
branch=reproducibility
python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
repo_name=alsatian
config_file=experiments/main_experiments/model_search/config.ini
config_section=debug-des-gpu-out-of-memory-10gb

# Clone the GitHub repository
if [ -z "$github_access_token" ]; then
    git clone $repository_url
else
    git clone https://$github_access_token@$repository_url
fi

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
