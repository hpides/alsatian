#!/bin/sh

cd "$(dirname "$0")"

DOCKER="nils-pytorch"

# When setting --memory swap to the same valu as memory, the container will not be allowed to use any swap
docker run -d -p 3340:22 \
           --ipc=host \
           -h $DOCKER-5GB \
           --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/repro-mount-ssd,target=/mount-ssd \
           --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/repro-mount-fs,target=/mount-fs \
           --gpus device=1 \
           --cpuset-cpus="64-127" \
           --restart always \
           --name $DOCKER-5GB \
           --memory=5g \
           --memory-swap=0g \
           --device-read-bps=/dev/md127:200mb \
           slin96/model-search-experiments:latest