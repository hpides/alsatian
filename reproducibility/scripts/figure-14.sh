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
mkdir -p /mount-ssd/script-execution/fig14
cp -r /mount-ssd/alsatian/experiments/main_experiments/model_search/start-scripts \
      /mount-ssd/script-execution/fig14

# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data

if [ ! -d "imagenette2" ]; then
    if [ ! -f "imagenette2.tgz" ]; then
        wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    fi
    tar -xzf imagenette2.tgz
fi

# Download the model snapshots
cd /mount-fs

if [ ! -d "hf-caching-dir" ]; then
    if [ ! -f "hf-caching-dir.tar" ]; then
         wget "https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/hf-snapshots/hf-caching-dir.tar"
    fi
    tar -xf  hf-caching-dir.tar
fi

mkdir -p /mount-fs/hf-snapshots
cd /mount-fs/hf-snapshots

for model in \
  SenseTime-deformable-detr \
  facebook-detr-resnet-101 \
  facebook-detr-resnet-50-dc5 \
  facebook-detr-resnet-50 \
  facebook-dinov2-base \
  facebook-dinov2-large \
  google-vit-base-patch16-224-in21k \
  hf-microsoft-resnet-152 \
  hf-microsoft-resnet-18 \
  microsoft-conditional-detr-resnet-50 \
  microsoft-resnet-152 \
  microsoft-resnet-18 \
  microsoft-table-transformer-detection \
  microsoft-table-transformer-structure-recognition \
  resnet-50-test \
  resnet-50
do
  if [ ! -d "$model" ]; then
    if [ ! -f "$model.tar" ]; then
      wget "https://data-engineering-systems.s3.openhpicloud.de/nils-strassenburg/alsatian/hf-snapshots/${model}.tar"
    else
      echo "$model.tar already exists, skipping download."
    fi
    tar -xf "$model.tar"
  else
    echo "$model directory already exists, skipping extraction."
  fi
done


# Create results directory and caching directory
mkdir -p /mount-fs/results/fig14/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig14/start-scripts
sh start-exp-fig-14.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig14/"

# CODE FOR PLOTTING
# CODE FOR PLOTTING
# CODE FOR PLOTTING
# CODE FOR PLOTTING

echo "✅ Plots done"
echo "plots can be found under /mount-fs/plots/fig14"


