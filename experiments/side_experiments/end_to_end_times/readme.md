# Fine Tune VS Search Time Experiment

- in this experiment we want to get insights in how long a typical fine-tuning process takes and compare it to the time
  it takes to search through a set of models
- we perform the following experiments
    - transfer learning (feature-extraction) ([details](fine_tuning))
    - transfer learning (full fine-tuning) ([details](fine_tuning))
    - extracting features (no training at all) ([details](calc_features))

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

- let's assume 1000 samples and a split of 800/200
- let's take resent 50 as an example (afterwards check if same trends apply for other models)

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

- **conclusions**
    - **approximate search is significantly faster than the baseline of full fine-tuning**
    - **we can confirm trends form the SHiFT paper**

### End to end time distribution

- we want to show that with optimized search, the bottleneck of the end-to-end process is currently the search part

- estimated training time
    - lets assume we have 4000 data points
    - for example for ResNet50, we measure for 10 epochs: 91s
    - lets assume different scenarios for the final training steps
        - S1: we have a model hat converges relatively fast
            - we perform a training of 10 epochs and no hyper paramater tuning
            - **91s**
        - S2: we train for 20-30 epochs and try 5 different hyperparameter sets (with pruning of non-promising runs)
            - 2.5 full runs, each taking 200s
            - approx **500s**

- estimated search
    - searching one model costs 4s, so fine-tuning takes as long as searching through
        - S1: 91/4 -> if we have more than **22 models** to search through search is the bottleneck
        - S2: 500/4 -> if we have more than **125 models** to search through search is the bottleneck
        - *comment*: most often the model development process is iterative and thus the initial full fine-tuning is
          often not that extensive. In addition, we might not train all layer of the model or have input data already
          cached which speeds up the training. Thus, **we assume to be closer to S1**
    - but current model stores have 50K + models
    - **this shows when doing a naive approximate search --> search is the bottleneck (because time scales linearly)**

### Discussion of absolute numbers
- our numbers: 91s per 4000 samples, 10 epochs (keep in mind this is for Renet50)
- shift assumes dataset sizes of 10-15X larger, 20 epochs, and 100 models
- so: approx: 200s per model (2* 91s + fact they use a less powerfull GPU, https://technical.city/en/video/GeForce-RTX-2080-Ti-vs-RTX-A5000#benchmarks)
- -> scale to 10-15X larger data -> 2000-3000s per model
- consider 100 models: 200 000s - 300 000s for search -> **approx 56 - 83h**

- shift numbers
  - **they report 250h - 300h**
  - comparison to our numbers:
    - average inference time across vision models
    - (9+8+8+9+8+9+11+11+10+11+9+9+11+9+8+9+11+11+10+9+9+19+16+22+13+16+22+13+12+15+17+21+32+44+79+151+13+25+21+12+12+12+12+11+12+17+10+10+10+10+11+11+12+12+11+14+14+17+85+55+23+11+11+12+14+11+12+13+14+73+33+74+63+33+21+26+13+11+14+18+6+6+6+6+6+8+6+6+7+6+6+6+6+7+6+7+8+8+9+8+9)/100
    - 17.63
    - resnet50 has inference time of 12 in the table
    - so scale up numbers from above by 1.5
    - 65 - 83h -> scaled (1.5) -> 97.5 - 124h
    - **our numbers are 2-3x lower (we are faster)**

- **possible reasons for diff**
  - diff in GPU/setup is more significant than we assumed
  - PyTorch might be faster than TensorFlow
  - they say they have for every model a **dockerized setup** --> probably large overhead starting a model
  - also it is beneficial for them to report high numbers for the baseline -> maybe process is not fully optimized


































