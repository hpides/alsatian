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

# Copy the script execution directory
mkdir -p /mount-ssd/script-execution/fig5
cp -r /mount-ssd/alsatian/experiments/main_experiments/bottlenecks/model_rank/on_server_exec \
      /mount-ssd/script-execution/fig5

# Navigate to the execution directory
cd /mount-ssd/script-execution/fig5/on_server_exec

# Download and extract Imagenette2 dataset
mkdir -p /mount-ssd/data
cd /mount-ssd/data
if [ ! -f "imagenette2.tgz" ]; then
    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
fi
tar -xzf imagenette2.tgz

# Create results directory
mkdir -p /mount-fs/results/bottleneck-analysis/

echo "✅ Setup completed successfully."

cd /mount-ssd/script-execution/fig5/on_server_exec
sh run-all-exp.sh bottleneck_analysis-model-resnet18-items-96-split-None-dataset_type-imagenette

echo "✅ Experiments done"

mkdir -p /mount-fs/plots/bottleneck-analysis


