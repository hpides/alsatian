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

for model in resnet18 resnet152; do
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

for model in eff_net_v2_l vit_l_32; do
  mkdir -p "/mount-fs/snapshot-sets/$model"
  cd "/mount-fs/snapshot-sets/$model" || exit
  for chunk in TOP_LAYER FIFTY_PERCENT TWENTY_FIVE_PERCENT; do
    tar_file="${model}_${chunk}.tar"
    unpack_dir="${model}_${chunk}"

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
mkdir -p /mount-fs/results/fig10/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig10/start-scripts
sh start-exp-fig-10.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig10/"

mkdir -p /mount-fs/plots/fig10/2000
mkdir -p /mount-fs/plots/fig10/8000
cd /mount-ssd/script-execution/fig10/start-scripts
sh plot-fig-10.sh
echo "✅ Figure 10 plots done"
echo "plots can be found under /mount-fs/plots/fig10"

mkdir -p /mount-fs/plots/fig11
cd /mount-ssd/script-execution/fig10/start-scripts
sh plot-fig-11.sh
echo "✅ Figure 11 plots done"
echo "plots can be found under /mount-fs/plots/fig11"
