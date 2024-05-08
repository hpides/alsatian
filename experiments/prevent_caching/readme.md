- on our GPU server we have the problem, that limiting the read speed is difficult because we have a lot of RAM as well
  as a fast SSD that is used to cache files
- to prevent this we can run the following command before starting to read to flush the caches
    - `echo 3 | sudo tee /proc/sys/vm/drop_caches`
- the problem is that inside a docker container, we do not have sudo rights to flush the caches of the host machine
- so we do the following:
    - we run the following python script on the host machine:
      ```python
      import os
      import subprocess
      import time
      
      def execute_command(command):
          subprocess.run(command, shell=True)
      
      def main():
          filename = "/fs/nils-strassenburg/io-tests/flush-caches-flag"
          command = "echo 3 | sudo tee /proc/sys/vm/drop_caches"
      
          while True:
              if os.path.exists(filename):
                  print(f"File '{filename}' found. Executing command...")
                  execute_command(command)
                  os.remove(filename)
                  print("Command executed and file deleted.")
              time.sleep(1)
      
      if __name__ == "__main__":
          main()
      ```
        - and mount the directory `/fs/nils-strassenburg/io-tests/flush-caches-flag` (can also be renamed) into the
          docker container
        - **to trigger a flush of the caches**: we write a file using `touch flush-caches-flag` in the mounted directory
          which will then trigger the execution of the command in the python script 
        - **to additionally limit the read speed of the docker container, we can use the following parameter**
          - `--device-read-bps=/dev/sdX:/dev/sdX:200mb`
        - **verify setup**
          - use an ubuntu container
          - `docker run -it --rm --device-read-bps=/dev/md127:200mb -v /fs/nils-strassenburg/io-tests:/io-tests ubuntu /bin/bash`
          - write dummy file: `dd if=/dev/zero of=/io-tests/tmp-file bs=1M count=1024 conv=fdatasync`
          - test read spead: `dd if=/io-tests/tmp-file of=/dev/null`