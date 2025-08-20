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
mkdir -p /mount-ssd/script-execution/fig16
cp -r /mount-ssd/alsatian/experiments/main_experiments/model_search/start-scripts \
      /mount-ssd/script-execution/fig16

# Download the model snapshots
mkdir -p /mount-fs/snapshot-sets/
cd /mount-fs/snapshot-sets/

for model in vit_l_32; do
  mkdir -p "/mount-fs/snapshot-sets/$model"
  cd "/mount-fs/snapshot-sets/$model" || exit
  for chunk in FIFTY_PERCENT; do
    tar_file="${model}_${chunk}.tar"
    unpack_dir="${chunk}"

    if [ ! -f "$tar_file" ]; then
      wget "https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/snapshot-sets/${tar_file}"
    else
      echo "$tar_file already exists, skipping download."
    fi

    if [ ! -d "$unpack_dir" ]; then
      tar -xf "$tar_file"
    else
      echo "$unpack_dir already exists, skipping extraction."
    fi
  done
done


# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data

if [ ! -d "imagenette2" ]; then
    if [ ! -f "imagenette2.tgz" ]; then
        wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    fi
    tar -xzf imagenette2.tgz
fi

# Create results directory and caching directory
mkdir -p /mount-fs/results/fig16/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig16/start-scripts
sh start-exp-synthetic-snapshots-limited-5gb.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig16/"
echo "TO CONTINUE WITH THE 10GB EXPERIMENT SWITCH DOCKER CONTAINERS"

