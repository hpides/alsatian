FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# SSH setup
RUN apt update && apt install -y \
    software-properties-common \
    libssl-dev openssh-server rsyslog

# General packages
RUN apt update && apt install -y \
    less nano vim emacs htop tree git rsync

# Tools for development
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update && apt install -y gcc-11 g++-11 gdb cmake

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11

RUN apt install -y \
    sudo \
    libcudnn8=8.2.1.32-1+cuda11.3 \
    screen \
    tmux \
    python3 \
    python3-pip \
    python3-venv

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the password.txt file into the container
COPY password.txt /app/password.txt
RUN PASSWORD=$(cat /app/password.txt) && \
    echo "Password read from file: $PASSWORD" && \
    useradd -ms /bin/bash nils && \
    echo "nils:${PASSWORD}" | chpasswd

EXPOSE 22
ENTRYPOINT service ssh restart && service rsyslog start && tail -f /dev/null