# Execution Environment

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
- python 3.8
- pytorch 2.2.0
- torchvision 0.17.0
- docker Docker version 24.0.2, build cb74dfc

## Hardware access restrictions
- run in docker image
  - base image: nvidia/cuda:11.3.0-devel-ubuntu20.04
- use only half of the CPUs (--cpuset-cpus="64-127")
  - e.g.: docker update --cpuset-cpus="64-127" <container name>
- use only one GPU (--gpus device=0)



