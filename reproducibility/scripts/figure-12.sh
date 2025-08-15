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
mkdir -p /mount-ssd/script-execution/fig12
cp /mount-ssd/alsatian/experiments/side_experiments/model_resource_info/model_resource_info_exp_on_server.sh \
      /mount-ssd/script-execution/fig12



# Create results directory and caching directory
mkdir -p /mount-fs/results/fig12/

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig12
sh /mount-ssd/script-execution/fig12/model_resource_info_exp_on_server.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig13/"

#mkdir -p /mount-fs/plots/fig12/
#cd /mount-fs/plots/fig12/
#
## Assigning command line arguments to variables
#repository_url=https://github.com/hpides/alsatian.git
#script_to_start=experiments/main_experiments/model_search/eval/trained_snapshots/plot_trained_snapshots.py
#branch=reproducibility
#python_dir=/home/nils/.virtualenvs/model-search-paper/bin/python
#repo_name=alsatian
#
#rm -rf alsatian
#git clone $repository_url
#
## Change directory to the cloned repository
#repo_name=$(basename "$repository_url" .git)
#cd $repo_name
#
#git pull
#git checkout $branch
#git pull
#
## Check if the specified script exists
#if [ ! -f "$script_to_start" ]; then
#    echo "Script '$script_to_start' not found in the repository."
#    exit 1
#fi
#
## Set the PYTHONPATH to include the current directory
#export PYTHONPATH=$PWD
#
## Execute the specified Python script
#$python_dir $script_to_start
#
#echo "✅ Plots done"
#echo "plots can be found under /mount-fs/plots/fig13"
