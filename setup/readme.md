# Setup

We developed MOSIX/Alsatian using the following hardware and software setup.
To speed up development on a new machine, we provide the following [docker image](https://hub.docker.com/repository/docker/slin96/model-search-experiments/general)
    - for details on the docker setup look into [docker_setup](docker_setup)
    - the container has the requirements installed that are listed in our [requirements.txt](requirements.txt)

# Groups GPU Server setup
- unless otherwise noted the information can be extracted by executing the
  script [env_info.py](..%2Fglobal_utils%2Fenv_info.py)

## Hardware Setup

- GPU
  - 2 * NVIDIA RTX A5000
  - driver_version: 535.161.07
- CPU
  - AMD Ryzen Threadripper PRO 3995WX 64-Cores
  - 2.4761 GHz
- SSD
  - 2* SAMSUNG MZVL21T0HCLR-00BL7
  - one used for home directory
  - commands used
    - `lsblk`
    - `lsblk -o KNAME,TYPE,SIZE,MODE`
- HDD
  - RAID 5 setup -> 1.8 TB
  - 3 * WDC WD10EZEX-08W
  - commands used
    - `lsblk`
    - `lsblk -o KNAME,TYPE,SIZE,MODE`

## Software setup
- docker version on host machine: 24.0.2, build cb74dfc
- we develop in a docker container running on the server (infos see [docker_setup](docker_setup))
  - base image: nvidia/cuda:11.3.0-devel-ubuntu20.04
  - use only half of the CPUs (--cpuset-cpus="64-127")
    - e.g.: docker update --cpuset-cpus="64-127" <container name>
  - use only one GPU (--gpus device=1)
- most relevant version numbers:
  - python 3.8
  - pytorch 2.2.0
  - torchvision 0.17.0

