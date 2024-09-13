# Trained model snapshots

- in this part of the repo we describe how we generate our trained snapshots
- for every architecture, we generate multiple snapshots as follows
    - we select a fixed base snapshots (model pretrained on ImageNet as provided by PyTorch)
    - then for every model we want to generate we select
        - a dataset out of the set of dataset (described below)
        - and draw a random retrain index from the given distribution to determine how many layers we want to freeze
            - the distribution is the same as for the synthetic snapshots
        - we freeze all layers that should be frozen and fine-tune the model (we alywas use the same hyper-parameters:
          LR, optimizer, etc.)
- we use the [generate_trained_snapshots.py](generate_trained_snapshots.py) script to generate the trained models
    - see exact models we generated are saved in (internal use for now: /mnt/external-ssd/trained-snapshots/ or
      /mnt/external-wd-hdd/trained-snapshots/)
- to build our model store that we use for our experiments from the fine-tuned snapshots, we
  use [build_trained_model_store.py](build_trained_model_store.py)
    - when running the training, we generated more snapshots for the efficient net than actually needed, so we delete
      some in the building process
    - to run the script you might have to adjust the `snapshot_base_path` variable

## Model architectures

- for this experiment, we limit ourselves to the following architectures as provided in our repo
    - ResNet18, ResNet152, EfficientNet_V2_Large, ViT_l_32

## Datasets

- the paper "Guided Recommendation for Model Fine-Tuning" by Li et al. has a nice overview of available datasets in
  their appendix
    - couldn't find the version with appendix online anymore, so uploaded together with a list of links for the datasets
      here (internal access only)
        - https://github.com/slin96/fine-tune-models?tab=readme-ov-file#tested-datasets
- out of the list we chose the following datasets and if necessary provide a script to transform the data into a
  representation compatible with the PyTorch Imagefolder dataset
    - [cub-birds-200](..%2F..%2F..%2F..%2Fdata%2Fcub-birds-200)
    - [food-101](..%2F..%2F..%2F..%2Fdata%2Ffood-101)
    - [image-woof](..%2F..%2F..%2F..%2Fdata%2Fimage-woof)
    - [stanford-cars](..%2F..%2F..%2F..%2Fdata%2Fstanford-cars)
    - [stanford_dogs](..%2F..%2F..%2F..%2Fdata%2Fstanford_dogs)
