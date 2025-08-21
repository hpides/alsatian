# Reproducibility Instructions

- These are the reproducibility instructions for our SIGMOD 2025 paper.
- You can find the PDF [here](../alsatian-single-column.pdf).
- We first provide general info on hardware and software. Afterward, we have one section per experiment in the paper.

## General Info

### Approach Naming

- the working title of our approach was `mosix`, in the final paper we use the name `Alsatian`
- the naming of plots and result files still uses the name `mosix`

### Number of Experiments

- For our paper we executed every experiment 3-5 times to investigate the variance in execution time and report median
  values
- On our setup we find the variance to be low. Thus, we only do one run per experiment for this reproducibility
  submission. This is mainly to limit the execution time for the reviewers (which will be long even with one run per
  experiment).

### Hardware

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

## Reproducing Experiments

- **We recommend to reproduce the experiments in the order provided in this readme**.
- Reason 1: As part of the experiment execution we download large sets of model snapshots. In our case these were so
  large that we were not able to fit all models for all experiments on our server at the same time. We thus ordered
  the experiments in a way that consecutive experiments can avoid re-downloading datasets and that once one experiment
  group is finished the downloads of previous experiments can be deleted.
- Reason 2: We ordered the experiments in increasing duration (so first experiments take a short amount of time, later
  ones take longer)

- All figures mentioned under one bullet point share a large part of data (if there is only one figure per bullet point
  there might be sharing of datasets but no sharing of model snapshots)
    - Figure 5
    - Figure 12
    - Figure 17
    - Figure 13
    - Figure 10, Figure 11, Figure 16
    - Figure 14, Figure 15

- **General Setup**
    - as a starting point for all experiments, we assume you have a cloned version of our repo
    - **and you are in the reproducibility branch**
    - we further assume you have two empty directories created that can be mounted into the docker container
        - one on a hard drive (will later be mounted to `/mount-fs` in the docker container)
        - one on an SSD (will later be mounted to `/mount-ssd` in the docker container)

#### Figure 5 (Bottlenecks)

- **start plain docker container**
    - adjust the following fields in [start-plain-container.sh](scripts/start-plain-container.sh)
    ```
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_SSD_ON_HOST_MACHINE>,target=/mount-ssd \
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_HDD_ON_HOST_MACHINE>,target=/mount-fs \
    --gpus device=<THE_ID_OF_THE_GPU_YOU_WANT_TO_USE> \
    --cpuset-cpus="<THE_ID_RANGE_OF_CPUS_YOU_WANT_TO_USE>" \
    ```
    - start the container by executing [start-plain-container.sh](scripts/start-plain-container.sh)

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-5.sh](scripts/figure-5.sh) to the container (e.g. by placing it in one of the
      mounted directories)
    - execute `figure-5.sh`
    - the plots are saved under: `/mount-fs/plots/bottleneck-analysis` **in the docker container** (and thus should be
      in your mounted directory on the host machine)

#### Figure 12 (Model Micro Benchmark)

- **start plain docker container (same as in Fig 5)**
    - adjust the following fields in [start-plain-container.sh](scripts/start-plain-container.sh)
    ```
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_SSD_ON_HOST_MACHINE>,target=/mount-ssd \
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_HDD_ON_HOST_MACHINE>,target=/mount-fs \
    --gpus device=<THE_ID_OF_THE_GPU_YOU_WANT_TO_USE> \
    --cpuset-cpus="<THE_ID_RANGE_OF_CPUS_YOU_WANT_TO_USE>" \
    ```
    - start the container by executing [start-plain-container.sh](scripts/start-plain-container.sh)

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-12.sh](scripts/figure-12.sh) to the container (e.g. by placing it in one of the
      mounted directories)
    - execute `figure-12.sh`
    - the plots are saved under: `/mount-fs/plots/fig12` **in the docker container** (and thus should be
      in your mounted directory on the host machine)

- **comparing reproduced results with results in paper**
    - there is usually some variation in the numbers depending on how and where you split the models
    - the important messages to see in the plots
        - for Vit-L-32 both the number of parameters and the inference time is more or less stable
        - for ResNet-152
            - the number of parameters increases in the stages and there is a high peak at the end
            - the inf time is higher in the beginning and then flattens out

#### Figure 17

- **start a docker container with LIMITED I/O (different to previous experiments)**
    - take the script [start-container-limited-io.sh](scripts/start-container-limited-io.sh) as a starting point and
      adjust the following fields
    ```
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_SSD_ON_HOST_MACHINE>,target=/mount-ssd \
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_HDD_ON_HOST_MACHINE>,target=/mount-fs \
    --gpus device=<THE_ID_OF_THE_GPU_YOU_WANT_TO_USE> \
    --cpuset-cpus="<THE_ID_RANGE_OF_CPUS_YOU_WANT_TO_USE>" \
    ```
    - the first four fields can be copied form the setup used for figure 5
        - the last line `/dev/md127`, must be adjusted to your setup
        - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
        - the output on our machine is:
          ```
            Filesystem
            /dev/md127
          ```
        - for further details see [readme.md](../experiments/main_experiments/prevent_caching/readme.md)

- **on the host machine** start the clear caches script
    - (see below under Reoccurring Steps - Start clear caches script)

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-17.sh](scripts/figure-17.sh) to the container (e.g. by placing it in one of the
      mounted directories)
    - execute `figure-17.sh`
    - the plots are saved under: `/mount-fs/plots/fig17` **in the docker container** (and thus should be
      in your mounted directory on the host machine)

- **troubleshooting**
    - after all downloads are complete, the folder structure of `/mount-fs/snapshot-sets` looks like this:
      ```
        .
        ├── bert
        │   ├── FIFTY_PERCENT
        │   ├── TOP_LAYERS
        │   └── TWENTY_FIVE_PERCENT
      ```
    - this experiment is the only one that uses the `bert` snapshots, meaning if you run out of space you can
      delete the directory after the plots were generated

#### Figure 13

- **start a docker container with LIMITED I/O (same as figure17)**
    - take the script [start-container-limited-io.sh](scripts/start-container-limited-io.sh) as a starting point and
      adjust the following fields
    ```
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_SSD_ON_HOST_MACHINE>,target=/mount-ssd \
    --mount type=bind,source=<ABS_PATH_TO_DIR_ON_HDD_ON_HOST_MACHINE>,target=/mount-fs \
    --gpus device=<THE_ID_OF_THE_GPU_YOU_WANT_TO_USE> \
    --cpuset-cpus="<THE_ID_RANGE_OF_CPUS_YOU_WANT_TO_USE>" \
    ```
    - the first four fields can be copied form the setup used for figure 5
        - the last line `/dev/md127`, must be adjusted to your setup
        - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
        - the output on our machine is:
          ```
            Filesystem
            /dev/md127
          ```
        - for further details see [readme.md](../experiments/main_experiments/prevent_caching/readme.md)

- **on the host machine** start the clear caches script
    - (see below under Reoccurring Steps - Start clear caches script)

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-13.sh](scripts/figure-13.sh) to the container (e.g. by placing it in one of the
      mounted directories)
    - execute `figure-13.sh`
    - the plots are saved under: `/mount-fs/plots/fig13` **in the docker container** (and thus should be
      in your mounted directory on the host machine)

- **troubleshooting**
    - for this experiment we download large amounts of data, if you run out of space you can delete downloaded `.tar`
      file after they were unpacked
    - after all downloads are complete, the folder structure of `/mount-fs/trained-snapshots` looks like this:
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
    - this experiment is the only one that uses the `trained-snapshots`, meaning if you run out of space you can
      delete the directory after the plots were generated

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
            - the last line `/dev/md127`, must be adjusted to your setup
            - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
            - the output on our machine is:
            ```
              Filesystem
              /dev/md127
            ```
            - for details see this readme: [readme.md](../experiments/main_experiments/prevent_caching/readme.md)


- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-10-11.sh](scripts/figure-10-11.sh) to the container (e.g. by placing it in one
      of the
      mounted directories)
    - execute `figure-10-11.sh`
    - the plots are saved under: `/mount-fs/plots/fig10` and `/mount-fs/plots/fig11` **in the docker container** (and
      thus should be
      in your mounted directory on the host machine)

- **troubleshooting**
    - without the tar files the directory structure of `/mount-fs/snapshot-sets` for this experiment should look like
      this

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
        - this experiment is the only one that uses the `/mount-fs/snapshot-sets/eff_net_v2_l`,
          `/mount-fs/snapshot-sets/resnet18`, `/mount-fs/snapshot-sets/resnet152`,
          `/mount-fs/snapshot-sets/vit_l_32/TOP_LAYERS`,
          and `/mount-fs/snapshot-sets/vit_l_32/TWENTY_FIVE_PERCENT` meaning if you run out of space you can
          delete the directory after the plots were generated

#### Figure 16

- **general info**
    - in this experiment we investigate the effect of different caching budgets by limiting the amount of available DRAM
    - we consider three scenarios: 64GB, 10GB, and 5GB of available main memory
    - we model each scenario with a separate docker container and start one experiment in each of them (sequentially,
      not at the same time)

##### 64 GB experiment

- to start the container
  run: [start-container-limited-io-and-memory-64.sh](scripts/start-container-limited-io-and-memory-64.sh)
    - **important**: adjust the following fields according to your setup
      ```
      --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
      --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
      --gpus device=1 \
      --cpuset-cpus="64-127" \
      --device-read-bps=/dev/md127:200mb \
      ```
    - the first four fields can be copied form the setup used for figure 5
        - the last line `/dev/md127`, must be adjusted to your setup
        - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
        - the output on our machine is:
        ```
          Filesystem
          /dev/md127
        ```
        - for details see this readme: [readme.md](../experiments/main_experiments/prevent_caching/readme.md)

- **on the host machine** start the clear caches script
    - (see below under Reoccurring Steps - Start clear caches script)

- **run the experiment for 64 GB**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID-WITH-64GB-MEMORY> bash`
    - copy the [figure-16-64gb.sh](scripts/figure-16-64gb.sh) to the container (e.g. by placing it in
      one of the
      mounted directories)
    - execute `figure-16-64gb.sh`

##### 10GB (and 5GB) experiment

- **to start the corresponding container (one of the following)**:
    - **important** adjust the fields `mount type` and so on as described above before running the scripts
    - [start-container-limited-io-and-memory-10.sh](scripts/start-container-limited-io-and-memory-10.sh)
    - [start-container-limited-io-and-memory-5.sh](scripts/start-container-limited-io-and-memory-5.sh)
- **on the host machine** start the clear caches script
    - (see below under Reoccurring Steps - Start clear caches script)
- **run the experiment for 10GB (and then do the same for 5GB)**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID-WITH-10GB-MEMORY> bash`
    - copy the [figure-16-10gb.sh](scripts/figure-16-10gb.sh) to the container (e.g. by placing it in
      one of the
      mounted directories)
    - execute `figure-16-10gb.sh`
    - then repeat the steps using the 5gb scripts 

##### Plotting

- use any of the docker containers started above
- copy the script [figure-16-plot.sh](scripts/figure-16-plot.sh) to the container and run it

##### troubleshooting

- after all downloads are complete, the folder structure of `/mount-fs/snapshot-sets` looks like this:
  ```
      .
      └── vit_l_32
          ├── FIFTY_PERCENT
  ```
- this experiment and Figure 10 & 11 are the only experiments that use `/mount-fs/snapshot-sets`, meaning if you run out
  of space you can delete the directory after the plots were generated

#### Figure 14 & Figure 15

- **general comment regarding reproducibility**
    - Between the initial paper submission and the reproducibility submission a small number of models we used for our
      experiments have been deleted from HuggingFace. While Alsatian is able to recover these snapshots form our backup
      directory SHiFT and Baseline fail to do so and will just skip these snapshots. When we reproduced the results as
      part of preparing the reproducibility submission this had the effect that the performance improvements of Alsatian
      over the Baseline and SHiFT slightly degrade (because Alsatian is searching through a couple more models). Still
      the trends should be very similar.

- **start a docker container with limited IO (and UNLIMITED memory, unlike for Figure 16)**
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
            - the last line `/dev/md127`, must be adjusted to your setup
            - we used the following cmd: `df --output=source /fs/nils-strassenburg/docker-mounted/mount-fs/`
            - the output on our machine is:
            ```
              Filesystem
              /dev/md127
            ```
            - for details see this readme: [readme.md](../experiments/main_experiments/prevent_caching/readme.md)

- **on the host machine** start the clear caches script
    - (see below under Reoccurring Steps - Start clear caches script)

##### Figure 14

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-14.sh](scripts/figure-14.sh) to the container (e.g. by placing it in one
      of the mounted directories)
    - execute `scripts/figure-14.sh`
    - the plots are saved under: `/mount-fs/plots/fig14` **in the docker container** (and
      thus should be in your mounted directory on the host machine)

##### Figure 15

- **run the experiment & generate plots**
    - open the containers bash: `sudo docker exec -it <CONTAINER-ID> bash`
    - copy the [figure-15.sh](scripts/figure-15.sh) to the container (e.g. by placing it in one
      of the mounted directories)
    - execute `scripts/figure-15.sh`
    - the plots are saved under: `/mount-fs/plots/fig15` **in the docker container** (and
      thus should be in your mounted directory on the host machine)

##### troubleshooting

- during the execution of the baseline and SHiFT there should be no downloads of models from hugging face since we
  provide the hf-caching dir 
- after all downloads are complete, the folder structure of `/mount-fs/hf-snapshots` looks like this:
  ```
    .
    ├── facebook-detr-resnet-101
    ├── facebook-detr-resnet-50
    ├── facebook-detr-resnet-50-dc5
    ├── facebook-dinov2-base
    ├── facebook-dinov2-large
    ├── google-vit-base-patch16-224-in21k
    ├── hf-microsoft-resnet-152
    ├── hf-microsoft-resnet-18
    ├── microsoft-conditional-detr-resnet-50
    ├── microsoft-resnet-152
    ├── microsoft-resnet-18
    ├── microsoft-table-transformer-detection
    ├── microsoft-table-transformer-structure-recognition
    ├── resnet-50
    └── SenseTime-deformable-detr
  ```
- this experiment and Figure 14 & 15 are the only experiments that use `/mount-fs/hf-snapshots`, meaning if you run out
  of space you can delete the directory after the plots were generated

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









