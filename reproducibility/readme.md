# Reproducibility Instructions

These are the reproducibility instructions for our SIGMOD 2025 paper.
You can find the PDF [here](../alsatian-single-column.pdf).
We structure the reproducibility by figure numbers in the paper.

## Hardware

We execute all experiments on a server with the following specs, but only use a subset of its resources (see software
setup).

- CPU
    - AMD Ryzen Threadripper PRO 3995WX 64-Cores
    - 2.4761 GHz
- GPU
    - 2 * NVIDIA RTX A5000
    - driver_version: 535.161.07
- SSD
    - 2* SAMSUNG MZVL21T0HCLR-00BL7
    - one used for home directory
- HDD
    - RAID 5 setup -> 1.8 TB
    - 3 * WDC WD10EZEX-08W

## Software setup

We use the docker container: `slin96/model-search-experiments` for our experiments. Depending on the plot we have to
start it in different ways. For example to prevent caching or limit its memory resources.

You can find the dockerfile that we used to build the container [here](../setup/docker_setup/Dockerfile)

For all experiments, we mount two directories from the host machine in the docker.

- one that is mounted on the SSD (`/mount-ssd` inside the container)
- one that is mounted on the HDD (`/mount-fs` inside the container)

#### Figure 5 (Bottlenecks)
- start plain docker container (see below under Reoccurring Steps - Start plain docker container)
- open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
- copy the [figure-5.sh](scripts/figure-5.sh) script on the container and run it

#### Figure 10 (End-to-end times synthetic models)
- if not still running, start plain docker container (see below under Reoccurring Steps - Start plain docker container)
- open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
- TODODOD start clear caches script
- copy the [figure-10.sh](scripts/figure-10.sh) script on the container and run it



## Reoccurring Steps

#### Start plain docker container
- to start the container by using: [start-plain-container.sh](scripts/start-plain-container.sh)
    - **important**: adjust the following fields according to your setup
        - ```
      --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
      --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
      --gpus device=1 \
      --cpuset-cpus="64-127" \
    ```

#### Start clear caches script
- TODO