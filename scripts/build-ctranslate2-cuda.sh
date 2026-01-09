#!/bin/bash
# CTranslate2 CUDA Build Script for ARM64
# This script builds CTranslate2 with CUDA support from source

set -e

echo "=========================================="
echo "CTranslate2 CUDA Build for ARM64"
echo "=========================================="

# Step 1: Install cuDNN
echo ""
echo "[1/5] Installing cuDNN..."
sudo apt-get update
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Verify cuDNN installation
echo "Verifying cuDNN installation..."
ldconfig -p | grep cudnn || echo "Warning: cuDNN not in ldconfig"

# Step 2: Clone CTranslate2
echo ""
echo "[2/5] Cloning CTranslate2 repository..."
BUILD_DIR="/tmp/ctranslate2-build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

# Step 3: Build with CMake
echo ""
echo "[3/5] Building with CMake (CUDA enabled)..."
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DOPENMP_RUNTIME=NONE \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Build with parallel jobs
make -j$(nproc)

# Step 4: Install C++ library
echo ""
echo "[4/5] Installing C++ library..."
sudo make install
sudo ldconfig

# Step 5: Build and install Python package
echo ""
echo "[5/5] Building Python wheel..."
cd "$BUILD_DIR/CTranslate2/python"

# Install build requirements
/home/innojini/dev/voice.man/.venv-py312/bin/python -m pip install --upgrade pip wheel setuptools 2>/dev/null || \
    uv pip install wheel setuptools -p /home/innojini/dev/voice.man/.venv-py312

# Set environment for finding the built library
export CTRANSLATE2_ROOT=/usr/local

# Build wheel
/home/innojini/dev/voice.man/.venv-py312/bin/python setup.py bdist_wheel

# Install the wheel
WHEEL_FILE=$(ls dist/*.whl | head -1)
echo "Installing wheel: $WHEEL_FILE"
uv pip install "$WHEEL_FILE" -p /home/innojini/dev/voice.man/.venv-py312 --force-reinstall

# Step 6: Verify installation
echo ""
echo "=========================================="
echo "Verifying CUDA support..."
/home/innojini/dev/voice.man/.venv-py312/bin/python -c "
import ctranslate2
cuda_count = ctranslate2.get_cuda_device_count()
print(f'CUDA devices: {cuda_count}')
if cuda_count > 0:
    print('SUCCESS: CTranslate2 CUDA support enabled!')
else:
    print('WARNING: CUDA support may not be working')
"

echo ""
echo "Build complete!"
echo "=========================================="
