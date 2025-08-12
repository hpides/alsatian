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
mkdir -p /mount-ssd/script-execution/fig10
cp -r /mount-ssd/alsatian/experiments/main_experiments/model_search/start-scripts \
      /mount-ssd/script-execution/fig10

# Download the model snapshots
mkdir -p /mount-fs/snapshot-sets/
cd /mount-fs/snapshot-sets/

if [ ! -d resnet18 ]; then
  if [ ! -f resnet18.tar ]; then
      wget https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/snapshot-sets/resnet18.tar
  else
      echo "resnet18.tar already exists, skipping download."
  fi
  tar -xf resnet18.tar
else
    echo "resnet18 directory already exists, skipping extraction."
fi

# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data
if [ ! -f "imagenette2.tgz" ]; then
    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
fi
tar -xzf imagenette2.tgz

# Create results directory and caching directory
mkdir -p /mount-fs/results/fig10/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig10/start-scripts
sh start-exp-fig-10.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig10/"
#
#mkdir -p /mount-fs/plots/fig10
#cd /mount-ssd/script-execution/fig10/on_server_exec
#sh plot.sh
#
#echo "✅ Plots done"
#echo "plots can be found under /mount-fs/plots/fig10"


