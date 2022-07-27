FROM arrival-sim:0.7.1.188691

ENV DEBIAN_FRONTEND=noninteractive

# Update keyring
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

# General updates & install python3.9
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev \
        # Pillow dependencies
        libjpeg-dev \
        zlib1g-dev \
    && wget https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz \
    && tar -xf Python-3.9.13.tgz \
    && cd Python-3.9.13 \
    && ./configure --enable-optimizations \
    && make -j 12 \
    && make altinstall

# Set working directory
WORKDIR /home/

# Install requirements
RUN python3.9 -m pip install l2r

# Add local files
COPY main.py main.py
