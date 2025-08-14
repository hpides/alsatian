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

# Download the model snapshots
mkdir -p /mount-fs/trained-snapshots/
cd /mount-fs/trained-snapshots/

# CODE FOR DOWNLOAD HF CACHING DIR HERE

# CODE FOR DOWNLOAD MODEL STORES HERE


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
mkdir -p /mount-fs/results/fig15/
mkdir -p /mount-ssd/cache-dir

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig14/start-scripts
sh start-exp-fig-15.sh

echo "✅ Experiments done"
echo "results can be found under /mount-fs/results/fig15/"

# CODE FOR PLOTTING
# CODE FOR PLOTTING
# CODE FOR PLOTTING
# CODE FOR PLOTTING

echo "✅ Plots done"
echo "plots can be found under /mount-fs/plots/fig15"


