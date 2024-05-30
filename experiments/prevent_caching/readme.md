
# Example usage
- we briefly explain how we can limit the I/O speed of a specific storage device inside a docker container

### Indentify device

- as a first step we have to find out what storage device we want to use
- to get an overview run: `lsblk` in terminal, example output:

```
...
sda                         8:0    0 931.5G  0 disk
└─md127                     9:127  0   1.8T  0 raid5 /fs
sdb                         8:16   0 931.5G  0 disk
└─md127                     9:127  0   1.8T  0 raid5 /fs
sdc                         8:32   0 931.5G  0 disk
└─md127                     9:127  0   1.8T  0 raid5 /fs
sr0                        11:0    1  1024M  0 rom
nvme1n1                   259:0    0 953.9G  0 disk
└─nvme1n1p1               259:1    0 953.9G  0 part  /home
nvme0n1                   259:2    0 953.9G  0 disk
├─nvme0n1p1               259:3    0     1G  0 part  /boot/efi
├─nvme0n1p2               259:4    0   1.5G  0 part  /boot
└─nvme0n1p3               259:5    0 951.3G  0 part
  ├─ubuntu--vg-ubuntu--lv 253:0    0   900G  0 lvm   /
  └─ubuntu--vg-lv--swap   253:1    0     8G  0 lvm   [SWAP]
```

- our case we want to limit the read speed of the `/fs` path -> our devcie is `md127`

### Docker container with limited I/O

- we mount the directory we want to limit the access to into the docker conatiner
    - create directory on host: `mkdir /fs/nils-strassenburg/io-tests/`
- we use the flag `--device-read-bps` to limit I/O speed
- cmd to run a container with limit of 200MB/s
    - `docker run -it --rm --device-read-bps=/dev/md127:200mb -v /fs/nils-strassenburg/io-tests:/io-tests ubuntu /bin/bash`
- to test this we can run the following commands
    - write dummy file: `dd if=/dev/zero of=/io-tests/tmp-file bs=1M count=1024 conv=fdatasync`
    - test read spead: `dd if=/io-tests/tmp-file of=/dev/null`

### Flushing Caches

- the goal is to limit the read speed of the HDD to x MB/s (e.g. 200MB/s)
- in theory applying the steps above (limit the read speed of the container we are running in) should be enough
- **BUT**: on our server we have RAM as well as a fast SSD (with free space) that the OS automatically uses to cache
  files which in turn increases the read speed again, so we have to flush the caches every time
    - you can observe this by the following cmd in the docker container from above:
        - write dummy file: `dd if=/dev/zero of=/io-tests/tmp-file bs=1M count=1024 conv=fdatasync`
        - test read spead: `dd if=/io-tests/tmp-file of=/dev/null`
        - test read spead (again this time should be a lot faster because
          cached): `dd if=/io-tests/tmp-file of=/dev/null`
        - flush caches: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
        - test read spead (should be slow again): `dd if=/io-tests/tmp-file of=/dev/null`
- the problem with flushing the caches though is that we can not do this from within the docker container

- so we run the script [host_watch_script.py](host_watch_script.py) on the host machine and the start the container
- the script checks every second if the docker container has written a specific file (flag) into a mounted directory
- if this is the case the script triggers emptying the caches on the host
- in addition, the script also writes an active file into the shared directory so that the container has a chance to
  figure out if the script to empty the caches on command is activated (the logic for the check is implemented
  in [watch_utils.py](watch_utils.py))





















