#!/bin/bash
# ARM64 PyTorch/WhisperX CUDA Dependencies Installation Script
# Platform: ARM64 (aarch64) - NVIDIA GB10
# CUDA Version: 13.0 (Driver 580.95.05)
# Python Version: 3.12 or 3.13
#
# This script installs PyTorch with CUDA support for ARM64 architecture
# and WhisperX with all required dependencies for speaker diarization.
#
# Usage:
#   ./scripts/install-cuda-deps.sh
#
# Prerequisites:
#   - NVIDIA driver installed (nvidia-smi working)
#   - Python 3.12 or 3.13 virtual environment activated
#   - FFmpeg installed (sudo apt install ffmpeg)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ARM64 PyTorch/WhisperX CUDA Installation ===${NC}"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}Error: This script is for ARM64 (aarch64) only. Detected: $ARCH${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

if [[ "$PYTHON_VERSION" != "3.12" && "$PYTHON_VERSION" != "3.13" ]]; then
    echo -e "${RED}Error: Python 3.12 or 3.13 required. Detected: $PYTHON_VERSION${NC}"
    exit 1
fi

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Please install NVIDIA driver.${NC}"
    exit 1
fi

echo -e "${GREEN}NVIDIA GPU detected:${NC}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

# Detect CUDA version from driver
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
echo -e "${YELLOW}CUDA Driver Version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')${NC}"

# Determine the appropriate CUDA wheel index
# For CUDA 13.0, use cu130 wheels (available for torch 2.9.x)
# For CUDA 12.x, use cu129 wheels (available for torch 2.8.x+)
CUDA_WHEEL_VERSION="cu130"
TORCH_VERSION="2.9.1"

echo -e "${GREEN}Using PyTorch CUDA wheel: ${CUDA_WHEEL_VERSION}${NC}"
echo -e "${GREEN}Target PyTorch version: ${TORCH_VERSION}${NC}"

# Step 1: Uninstall existing torch packages to avoid conflicts
echo -e "${YELLOW}Step 1: Removing existing torch packages...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Step 2: Install PyTorch with CUDA support from official wheel index
echo -e "${YELLOW}Step 2: Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_WHEEL_VERSION}...${NC}"
pip install \
    torch==${TORCH_VERSION}+${CUDA_WHEEL_VERSION} \
    torchaudio==${TORCH_VERSION}+${CUDA_WHEEL_VERSION} \
    torchvision==0.24.1+${CUDA_WHEEL_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}

# Step 3: Verify CUDA is available
echo -e "${YELLOW}Step 3: Verifying CUDA availability...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA is not available!')
"

# Step 4: Install WhisperX dependencies
echo -e "${YELLOW}Step 4: Installing WhisperX dependencies...${NC}"

# Install faster-whisper and ctranslate2 first (CTranslate2 needs specific version for ARM64)
pip install ctranslate2>=4.5.0 faster-whisper>=1.1.1

# Install pyannote-audio (must be <4.0.0 for WhisperX compatibility)
pip install "pyannote-audio>=3.3.2,<4.0.0"

# Install WhisperX
pip install whisperx>=3.1.6

# Step 5: Install remaining dependencies from pyproject.toml
echo -e "${YELLOW}Step 5: Installing project dependencies...${NC}"
pip install -e ".[dev]"

# Step 6: Final verification
echo -e "${YELLOW}Step 6: Final verification...${NC}"

python -c "
import sys
print('=== Verification Results ===')

# Check torch
try:
    import torch
    cuda_status = 'CUDA available' if torch.cuda.is_available() else 'CPU only'
    print(f'torch {torch.__version__} ({cuda_status})')
except ImportError as e:
    print(f'torch: FAILED - {e}')

# Check torchaudio
try:
    import torchaudio
    print(f'torchaudio {torchaudio.__version__}')
except ImportError as e:
    print(f'torchaudio: FAILED - {e}')

# Check torchvision
try:
    import torchvision
    print(f'torchvision {torchvision.__version__}')
except ImportError as e:
    print(f'torchvision: FAILED - {e}')

# Check pyannote
try:
    from pyannote.audio import Pipeline
    import pyannote.audio
    print(f'pyannote-audio {pyannote.audio.__version__}')
except ImportError as e:
    print(f'pyannote-audio: FAILED - {e}')

# Check whisperx
try:
    import whisperx
    print(f'whisperx installed')
except ImportError as e:
    print(f'whisperx: FAILED - {e}')

# Check faster_whisper
try:
    import faster_whisper
    print(f'faster-whisper installed')
except ImportError as e:
    print(f'faster-whisper: FAILED - {e}')

print('=== Verification Complete ===')
"

echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Get Hugging Face token: https://huggingface.co/settings/tokens"
echo "2. Accept pyannote model agreements:"
echo "   - https://huggingface.co/pyannote/segmentation-3.0"
echo "   - https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "3. Set environment variable: export HF_TOKEN=your_token"
echo ""
echo "Test with:"
echo "  python -c \"from pyannote.audio import Pipeline; print('OK')\""
