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
mkdir -p /mount-ssd/script-execution/fig17
cp -r /mount-ssd/alsatian/experiments/main_experiments/model_search/start-scripts \
      /mount-ssd/script-execution/fig17


# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data

if [ ! -d "aclImdb" ]; then
    if [ ! -f "aclIMDB.tar" ]; then
        wget https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/datasets/aclIMDB.tar
    fi
    tar -xf aclIMDB.tar
fi
echo "✅ Download dataset completed successfully."

# Download the model snapshots
mkdir -p /mount-fs/snapshot-sets/
cd /mount-fs/snapshot-sets/

for model in bert; do
  if [ ! -d "$model" ]; then
    if [ ! -f "$model.tar" ]; then
      wget "https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/snapshot-sets/${model}.tar"
    else
      echo "$model.tar already exists, skipping download."
    fi
    tar -xf "$model.tar"
  else
    echo "$model directory already exists, skipping extraction."
  fi
done
echo "✅ Download models completed successfully."



# Create results directory and caching directory
mkdir -p /mount-fs/results/fig17/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig17/start-scripts
sh start-exp-synthetic-bert.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig17/"
#
#mkdir -p /mount-fs/plots/fig10/2000
#mkdir -p /mount-fs/plots/fig10/8000
#cd /mount-ssd/script-execution/fig10/start-scripts
#sh plot-fig-10.sh
#
#
#echo "✅ Plots done"
#echo "plots can be found under /mount-fs/plots/fig10"
#
#
