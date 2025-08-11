#!/bin/sh

cd "$(dirname "$0")"

DOCKER="nils-pytorch"

docker run -d -p 3318:22 \
           --ipc=host \
           -h $DOCKER \
           --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/repro-mount-ssd,target=/mount-ssd \
           --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/repro-mount-fs,target=/mount-fs \
           --gpus device=1 \
           --cpuset-cpus="64-127" \
           --restart always \
           --name $DOCKER \
           slin96/model-search-experiments:latest