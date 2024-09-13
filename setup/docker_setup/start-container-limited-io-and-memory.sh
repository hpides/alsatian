#!/bin/sh

cd "$(dirname "$0")"

DOCKER="nils-pytorch-installed-env"

# When setting --memory swap to the same valu as memory, the container will not be allowed to use any swap
docker run -d -p 3343:22 \
           --ipc=host \
           -h $DOCKER-2GB \
           --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
           --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
           --gpus device=1 \
           --cpuset-cpus="64-127" \
           --restart always \
           --name $DOCKER-2GB \
           --memory=2g \
           --memory-swap=2g \
           --device-read-bps=/dev/md127:200mb \
           $DOCKER