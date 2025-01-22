#!/bin/sh

# Assigning command line arguments to variables
repository_url=github.com/slin96/alsatian.git
script_to_start=experiments/main_experiments/snapshots/hugging_face/generate_hf_snapshots.py
github_access_token_file=./access_token
python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
branch=hf-models
repo_name=model-search-paper
config_file=experiments/main_experiments/snapshots/hugging_face/config.ini
config_section=generate-resnets-50

# Function to read the access token from file
read_access_token() {
    local token_file="$1"
    if [ -f "$token_file" ]; then
        echo "$(cat $token_file)"
    else
        echo "Error: Access token file not found: $token_file"
        exit 1
    fi
}

github_access_token=$(read_access_token "$github_access_token_file")

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
$python_dir $script_to_start --config_file $config_file --config_section $config_section
