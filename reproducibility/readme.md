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

- start plain docker container
- to start the container run: [start-plain-container.sh](scripts/start-plain-container.sh)
    - **important**: adjust the following fields according to your setup
        - ```
          --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
          --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
          --gpus device=1 \
          --cpuset-cpus="64-127" \
          ```

- open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
- copy the [figure-5.sh](scripts/figure-5.sh) script on the container and run it

#### Figure 10 & 11(End-to-end times synthetic models)

- start a docker container with limited IO
    - to start the container run: [start-container-limited-io.sh](scripts/start-container-limited-io.sh)
        - **important**: adjust the following fields according to your setup
          ```
          --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
          --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
          --gpus device=1 \
          --cpuset-cpus="64-127" \
          --device-read-bps=/dev/md127:200mb \
          ```
        - the first four fields can be copied form the setup used for figure 5
            - for the last line `/dev/md127` must be replaced with
            - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
            - the output on our machine is:
            ```
              Filesystem
              /dev/md127
            ```
            - for details see this readme: [readme.md](../experiments/main_experiments/prevent_caching/readme.md)

- **on the host machine** start the clear caches script (see below under Reoccurring Steps - Start clear caches script)

- open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
- copy the [figure-10.sh](scripts/figure-10-11.sh) script on the container and run it

- after or while the download section of the script is running you can remove the `.tar` files that were already unpacked
- without the tar files the directory structure of `/mount-fs/snapshot-sets` for this experiment should look like this
```
.
├── eff_net_v2_l
│   ├── FIFTY_PERCENT
│   ├── TOP_LAYERS
│   └── TWENTY_FIVE_PERCENT
├── resnet152
│   ├── FIFTY_PERCENT
│   ├── TOP_LAYERS
│   └── TWENTY_FIVE_PERCENT
├── resnet18
│   ├── FIFTY_PERCENT
│   ├── LAST_ONE_LAYER
│   ├── TOP_LAYERS
│   └── TWENTY_FIVE_PERCENT
└── vit_l_32
    ├── FIFTY_PERCENT
    ├── TOP_LAYERS
    └── TWENTY_FIVE_PERCENT
```

##  Figure 13
- after download the folder structure of `/mount-fs/trained-snapshots` looks like this:
```
.
├── cub-birds-200
├── food-101
├── image-woof
├── modelstore_savepath
│   ├── eff_net_v2_l-model-store
│   ├── resnet152-model-store
│   ├── resnet18-model-store
│   └── vit_l_32-model-store
├── stanford-cars
└── stanford-dogs
```

## Reoccurring Steps

#### Start clear caches script

- for our experiments, we assume that when models are loaded for the first time they are loaded from cold/cloud storage
- thus we limit the I/O when reading from the HDD mounted directory to 200MB/s
- and we need to prevent caching of files between runs (otherwise a second run has an unfair advantage because the files
  are already in memory)
- next to using a docker container with limited I/O, we start the following script on the host machine to clean the
  caches when triggered by a script inside the container, and also monitor the I/O speed before every
- run the following script and make sure it runs all the time during experiment execution
    - [host_watch_script.py](scripts/host_watch_script.py)
    - **important** adjust the following line before running the script:
      `BASE_PATH = <PATH ON HOST MACHINE THAT IS MOUNTED TO /mount-fs INSIDE CONTAINER>`
    - `sudo python3 host_watch_script.py`
- for details see this readme: [readme.md](../experiments/main_experiments/prevent_caching/readme.md)









