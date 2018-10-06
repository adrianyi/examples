#!/bin/bash
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev
rm -rf /var/lib/apt/lists/*
curl -s -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p /opt/conda
rm ~/miniconda.sh
export PATH=/opt/conda/bin:$PATH
export CMAKE_PREFIX_PATH=$(dirname $(which conda))/../

git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch
git clone https://github.com/pytorch/vision.git /opt/vision

conda create --name clusterone -y python=3.6
source activate clusterone
conda install -y -c conda-forge openmpi
conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -y -c mingfeima mkldnn
conda install -y -c pytorch magma-cuda92
conda clean -ya

cd /opt/pytorch
python setup.py install
cd /opt/vision
python setup.py install
cd /code/
