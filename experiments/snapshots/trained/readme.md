# Trained model snapshots

- in this part of the repo we describe how we generate our trained snapshots
- for every architecture, we generate multiple as follows
  - we select a fixed base snapshots (model pretrained on ImageNet as provided by PyTorch)
  - we train all models with the same hyper-parameters: LR, optimizer, etc
  - then for every model we want to generate we select
    - a dataset out of the set of dataset (described below)
    - and draw a random retrain index from the given distribution to determine how many layers we want to freeze
  
## Model architectures
- for this experiment, we limit ourselves to the following to following architectures as provided in our repo
  - ResNet18, ResNet152, EfficientNet_V2_Large, ViT_l_32

## Datasets
- the paper "Guided Recommendation for Model Fine-Tuning" by Li et al. has a nice overview of available datasets in their appendix
  - couldn't find the version with appendix online anymore, so uploaded together with a list of links for the datasets here (internal access only)
    - https://github.com/slin96/fine-tune-models?tab=readme-ov-file#tested-datasets
