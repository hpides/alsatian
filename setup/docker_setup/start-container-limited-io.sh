#!/bin/sh

cd "$(dirname "$0")"

DOCKER="nils-pytorch-installed-env"

docker run -d -p 3341:22 \
           --ipc=host \
           -h $DOCKER \
           --mount type=bind,source=/home/fgrabl/nils-strassenburg/docker-mounted/mount-ssd,target=/mount-ssd \
           --mount type=bind,source=/fs/nils-strassenburg/docker-mounted/mount-fs,target=/mount-fs \
           --gpus device=1 \
           --cpuset-cpus="64-127" \
           --restart always \
           --name $DOCKER \
           --device-read-bps=/dev/md127:200mb \
           $DOCKER