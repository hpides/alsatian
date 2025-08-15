#!/bin/sh

# Assigning command line arguments to variables
repository_url=https://github.com/hpides/alsatian.git
branch=reproducibility
python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
repo_name=alsatian
script_to_start=experiments/side_experiments/model_resource_info/model_resource_info_experiment.py

git clone $repository_url

# Change directory to the cloned repository
repo_name=$(basename "$repository_url" .git)
cd $repo_name

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
$python_dir $script_to_start