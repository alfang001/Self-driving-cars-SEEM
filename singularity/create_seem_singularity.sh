#!/bin/bash

# Script to install Singularity 4.3.1

set -e

# Function to print messages
print_message() {
    echo ">>> $1"
}

# Check if script is run as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

print_message "Installing Singularity 4.3.1"

# Install dependencies
print_message "Installing dependencies..."
# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
   autoconf \
   automake \
   cryptsetup \
   fuse \
   fuse2fs \
   git \
   libfuse-dev \
   libglib2.0-dev \
   libseccomp-dev \
   libtool \
   pkg-config \
   runc \
   squashfs-tools \
   squashfs-tools-ng \
   uidmap \
   wget \
   zlib1g-dev

# Install Go
print_message "Installing Go..."
export VERSION=1.21.0 OS=linux ARCH=amd64
wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz
rm go$VERSION.$OS-$ARCH.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> /etc/profile.d/go.sh
source /etc/profile.d/go.sh

# Verify Go installation
if ! command -v go &> /dev/null; then
    echo "Go installation failed"
    exit 1
fi

# Download and install Singularity
print_message "Downloading and installing Singularity 4.3.1..."
wget https://github.com/sylabs/singularity/releases/download/v4.1.3/singularity-ce-4.1.3.tar.gz
tar -xzf singularity-ce-4.1.3.tar.gz
cd singularity-ce-4.1.3

# Configure and compile Singularity
./mconfig
make -C builddir
make -C builddir install

# Verify Singularity installation
if ! command -v singularity &> /dev/null; then
    echo "Singularity installation failed"
    exit 1
fi

# Clean up
cd ..
rm -rf singularity-ce-4.1.3 singularity-ce-4.1.3.tar.gz

print_message "Singularity 4.1.3 has been successfully installed!"
singularity --version

# Now creating the image itself
# sudo singularity build pytorch_container.sif docker://pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
# sudo singularity build vlfm_container.sif vlfm.def
sudo singularity build seem_container.sif seem.def