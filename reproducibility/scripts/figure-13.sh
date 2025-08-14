#!/bin/bash
set -e  # Exit immediately on error
set -u  # Treat unset variables as errors

# Clone the repository and checkout the desired branch
cd /mount-ssd/
if [ ! -d "alsatian" ]; then
    git clone https://github.com/hpides/alsatian.git
fi
cd alsatian
git fetch --all
git checkout reproducibility
git pull

# Copy the script execution directory
mkdir -p /mount-ssd/script-execution/fig13
cp -r /mount-ssd/alsatian/experiments/main_experiments/model_search/start-scripts \
      /mount-ssd/script-execution/fig13

# Download the model snapshots
mkdir -p /mount-fs/snapshot-sets/
cd /mount-fs/snapshot-sets/

#for model in resnet18 resnet152 eff_net_v2_l vit_l_32; do
#  if [ ! -d "$model" ]; then
#    if [ ! -f "$model.tar" ]; then
#      wget "https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/snapshot-sets/${model}.tar"
#    else
#      echo "$model.tar already exists, skipping download."
#    fi
#    tar -xf "$model.tar"
#  else
#    echo "$model directory already exists, skipping extraction."
#  fi
#done


# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data

if [ ! -d "image-woof" ]; then
    if [ ! -f "image-woof.tar" ]; then
        wget https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/datasets/image-woof.tar
    fi
    tar -xf image-woof.tar
fi
echo "✅ Download dataset completed successfully."

# Create results directory and caching directory
mkdir -p /mount-fs/results/fig13/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig13/start-scripts
sh start-exp-trained-models.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig13/"

mkdir -p /mount-fs/plots/fig13/
cd /mount-fs/plots/fig13/

# Assigning command line arguments to variables
repository_url=https://github.com/hpides/alsatian.git
script_to_start=experiments/main_experiments/model_search/eval/bert/plot_trained_snapshots.py
branch=reproducibility
python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
repo_name=alsatian

rm -rf alsatian
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
$python_dir $script_to_start

echo "✅ Plots done"
echo "plots can be found under /mount-fs/plots/fig13"
