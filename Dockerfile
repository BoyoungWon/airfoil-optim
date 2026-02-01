# =============================================================================
# Airfoil Optimization Environment
# =============================================================================
# Contains: XFoil, SU2, NeuralFoil (Python environment)
#
# Build:
#   docker-compose build
#   or: docker build -t airfoil-optim .
#
# Run:
#   docker-compose up -d
#   docker exec -it airfoil-optim bash
#
# Note: First build takes ~30-60 minutes due to SU2 compilation
# =============================================================================

FROM ubuntu:22.04

LABEL maintainer="airfoil-optim"
LABEL description="Airfoil analysis and optimization environment with XFoil, SU2, NeuralFoil"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gfortran \
    gcc \
    g++ \
    cmake \
    make \
    git \
    pkg-config \
    ccache \
    # XFoil dependencies
    libx11-dev \
    # MPI (for SU2)
    libopenmpi-dev \
    openmpi-bin \
    # Scientific libraries
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    # SU2 build dependencies
    swig \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-mpi4py \
    # HDF5 for CGNS
    libhdf5-dev \
    libhdf5-openmpi-dev \
    # Eigen3 (for SU2)
    libeigen3-dev \
    # Utilities
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install meson and ninja for SU2 build
RUN pip3 install --no-cache-dir meson ninja

WORKDIR /workspace

# =============================================================================
# Stage 1: Build XFoil
# =============================================================================
COPY xfoil /workspace/xfoil

RUN cd /workspace/xfoil && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /workspace/xfoil/build

# =============================================================================
# Stage 2: Build SU2
# =============================================================================
COPY SU2 /opt/su2

# Setup Eigen (avoid GitLab download issues)
RUN rm -rf /opt/su2/externals/eigen && \
    mkdir -p /opt/su2/externals/eigen && \
    cp -r /usr/include/eigen3/* /opt/su2/externals/eigen/ && \
    touch /opt/su2/externals/eigen/d71c30c47858effcbd39967097a2d99ee48db464

RUN cd /opt/su2 && \
    python3 meson.py setup build \
        --prefix=/usr/local \
        -Denable-tecio=true \
        -Denable-cgns=true \
        -Denable-autodiff=false \
        -Denable-directdiff=false \
        -Denable-pywrapper=false \
        -Dwith-mpi=enabled \
        -Dwith-omp=true \
        --buildtype=release && \
    cd build && ninja -j$(nproc) && ninja install && \
    rm -rf /opt/su2/build

ENV SU2_RUN=/usr/local/bin
ENV SU2_HOME=/opt/su2

# =============================================================================
# Stage 3: Python Environment (Miniforge/Conda)
# =============================================================================
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment
COPY environment.yml /workspace/
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate environment
ENV CONDA_DEFAULT_ENV=airfoil-optim
ENV PATH="/opt/conda/envs/airfoil-optim/bin:${PATH}"

# =============================================================================
# Stage 4: Install NeuralFoil
# =============================================================================
COPY neuralfoil /workspace/neuralfoil
RUN cd /workspace/neuralfoil && pip install --no-cache-dir .

# =============================================================================
# Environment Setup
# =============================================================================
ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/scripts:/workspace/solvers:/workspace/neuralfoil"
ENV XFOIL_PATH="/usr/local/bin/xfoil"

# MPI settings (allow root in container)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Default shell
SHELL ["/bin/bash", "-c"]

# Copy project files
COPY scripts /workspace/scripts
COPY solvers /workspace/solvers
COPY scenarios /workspace/scenarios
COPY input /workspace/input
COPY requirements.txt /workspace/

# Verify installation
RUN echo "=== Verifying Installation ===" && \
    which xfoil && xfoil -h 2>&1 | head -5 || true && \
    which SU2_CFD && SU2_CFD -h 2>&1 | head -5 || true && \
    python -c "from neuralfoil import get_aero_from_kulfan_parameters; print('NeuralFoil OK')" && \
    python -c "import numpy, scipy, pandas; print('Core packages OK')" && \
    echo "=== Installation Complete ==="

WORKDIR /workspace
CMD ["/bin/bash"]
