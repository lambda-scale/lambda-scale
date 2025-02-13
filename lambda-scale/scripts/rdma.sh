#!/bin/bash
set -x

apt-get install -y \
    vim \
    zsh \
    git \
    wget \
    curl \
    cmake \
    libssl-dev \
    net-tools \
    tk \
    iproute2 \
    iputils-ping
    
apt-get install -y \
    autotools-dev \
    libnl-route-3-dev \
    autoconf \
    flex \
    libltdl-dev \
    graphviz \
    libgfortran5 \
    chrpath \
    gfortran

apt-get install -y  \    
    libnl-3-dev \
    kmod \
    libfuse2 \
    libusb-1.0-0 \
    pkg-config \
    udev \
    lsof

apt-get install -y \    
    automake \
    ethtool \
    bison \
    m4 \
    swig \
    libpci3 \
    libelf1 \
    pciutils \
    libnuma1 \
    libmnl0

pip install protobuf==3.20.3

cd /jiachaobo && \
    wget https://content.mellanox.com/ofed/MLNX_OFED-23.10-2.1.3.1/MLNX_OFED_LINUX-23.10-2.1.3.1-ubuntu22.04-x86_64.tgz
    
cd /jiachaobo && \
    git clone https://github.com/pciutils/pciutils.git && \
    cd pciutils && \
    git checkout v3.13.0

cd /jiachaobo && \
    git clone https://github.com/linux-rdma/perftest.git && \
    cd perftest && \
    git checkout 24.04.0-0.41

apt-get install -y openssh-server

cd /jiachaobo && git clone https://github.com/mpmilano/mutils.git

cd /jiachaobo && git clone https://github.com/nlohmann/json.git

cd /jiachaobo && git clone https://github.com/ofiwg/libfabric.git

cd /jiachaobo && git clone https://github.com/mpmilano/mutils-containers.git



echo " "
echo "-----------------------------------"
echo "Installing RDMC-GDR dependencies..."
echo "-----------------------------------"
cd ~
bash /jiachaobo/RDMC-GDR/scripts/deploy/install-json.sh
bash /jiachaobo/RDMC-GDR/scripts/deploy/install-mutils.sh
bash /jiachaobo/RDMC-GDR/scripts/deploy/install-mutils-containers.sh
bash /jiachaobo/RDMC-GDR/scripts/deploy/install-libfabric.sh

echo " "
echo "---------------------"
echo "Build RDMC-GDR lib..."
echo "---------------------"
cd /jiachaobo/RDMC-GDR/
./build.sh Debug