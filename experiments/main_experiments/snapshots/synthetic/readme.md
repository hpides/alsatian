# Synthetic Snapshots

- we use synthetic snapshots for many experiments by this we mean the snapshots have real architectures e.g. ResNet50
  but their parameters/weights are randomly initialized
- for our experiments this makes sense because
    - the computational complexity of performing inference over a given snapshot does not depend on the exact values of
      the parameters
    - generating synthetic snapshots is cheap because it does not require training, this allows use to generate a large
      number of them

## Generating Synthetic snapshots

- for all of our experiments the snapshots will be automatically generated if not provided in a path already
- to manually generate a synthetic set of snapshots, you can adjust and run [generate_set.py](generate_set.py)
- the generation works as follows:
    - the first model in the snapshot set is always a model pretraiend on imagenet
    - the next models are generated by drawing from a specified truncated normal distribution how many of the new model
      should be adjusted
    - then a model is randomly selected form the set of already generated models, all weights are copied over to a new
      model, and the last n layers (according to the distribution form above) are randomly initialized