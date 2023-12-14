# Fine Tune VS Search Time Experiment

- in this experiment we want to get insights in how long a typical fine-tuning process takes and compare it to the time
  it takes to search through a set of models
- we perform the following experiments
    - transfer learning (feature-extraction) ([details](./fine_tuning))
    - transfer learning (full fine-tuning) ([details](./fine_tuning))
    - extracting features (no training at all) ([details](./calc_features))

## Hardware

- for the experiments we had the following setup
- **Hardware**
    - DES GPU server
    - 2 GPUs (NVIDIA RTX A5000, 24GB)
    - AMD Ryzen Threadripper PRO 3995WX 64-Cores, 128 logical cores
- **Our Setup**
    - docker container
    - access to one GPU, and 32 logical cores
    - GPU Driver Version: 535.129.03 CUDA Version: 12.2
    - Python 3.8.10
    - PyTorch 12.1.1, torchvision 0.16.1

## Brief Analysis

### Improvements of approximate model search

- lets assume 1000 samples and a split of 800/200
- lets take resent 50 as an example (afterwards check if same trends apply for other models)

- resnet50 (1000 samples)
    - fine tuning (10 epochs) -> 26.4s
    - just feature extraction -> 1.3s
    - lets assume a naive training of 10 epochs as the naive model search
    - -> we are approx 20X faster, (SHiFT reports numbers of 20-45X, but they also train 20 epochs)

- resnet50 (4000 samples)
    - fine tuning (10 epochs) -> 91s
    - just feature extraction -> 3.9s
    - lets assume a naive training of 10 epochs as the naive model search
    - 23X faster


- vit_l_16 (1000 samples)
    - fine tuning (10 epochs) -> 282.4s
    - just feature extraction -> 11.5s
    - 25X faster


- vit_l_16 (4000 samples)
    - fine tuning (10 epochs) -> 1108s
    - just feature extraction -> 45.5s
    - 24.3X faster

- conclusion
    - **we can confirm trends form the shift paper: with approximate search we are significantly faster**

### End to end time distribution

- we want to show that with optimized search, the bottleneck of the end to end process is currently the search part

- estimated training time
    - lets assume we have 4000 data points
    - for example for ResNet50, we measure for 10 epochs: 91s
    - lest assume normally we would train for 20-30 epochs and try 5 different hyper parameter sets (with pruning)
    - 2.5 full runs, each taking 200s -> 500s

- estimated search
    - searching one model costs 4s
    - so fine-tuning takes as long as searching through 125 models
    - if we make assumptions of just one training run a 200s -> its only 50 models
    - but current model stores have 50K + models
    - **this shows when doing a naive approximate search --> search is the bottleneck (because time scales linearly)**

### Discussion of absolute numbers

- **our numbers are only for 4000 samples**
- assuming 500s per model -> 100 models 50000s -> 833 -> 13h
- 13h scale up to 10-15 X samples -> 130 - 195
- SHiFt reports 250 h, but also older, less powerful GPU
    - NVIDIA GeForce RTX 2080 Ti (GeekBench 15% slower than our GPU)
    - https://technical.city/en/video/GeForce-RTX-2080-Ti-vs-RTX-A5000#benchmarks (don't know how reliable the numbers
      are but just to get a rough estimate)
    - 195 * 1.15 -> 224h (pretty close to SHiFT numbers)
    - **problem**: SHiFT assumes only 20 epochs so we would assume 200s, not 500s
      - with this our results are approx 2.5x faster than what SHiFT presents, but:
        - they use different framework, different setup, maybe slow data load


































