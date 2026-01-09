# Base image with Fortran, C compiler and scientific computing tools
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    gcc \
    g++ \
    cmake \
    make \
    git \
    libx11-dev \
    libopenmpi-dev \
    openmpi-bin \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge only, no TOS issues)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy environment configuration
COPY environment.yml /workspace/

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate conda environment by default
ENV CONDA_DEFAULT_ENV=airfoil-optim-env
ENV PATH="/opt/conda/envs/airfoil-optim-env/bin:${PATH}"

# Copy xfoil source
COPY xfoil /workspace/xfoil

# Build xfoil
RUN cd /workspace/xfoil && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Set environment variables for xfoil
ENV PATH="/usr/local/bin:${PATH}"

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
