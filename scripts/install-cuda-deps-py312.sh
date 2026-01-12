#!/bin/bash
# ARM64 PyTorch/WhisperX CUDA Dependencies Installation Script
# Platform: ARM64 (aarch64) - NVIDIA GB10
# CUDA Version: 12.9 compatible
# Python Version: 3.12 (REQUIRED)
#
# VERIFIED WORKING: 2026-01-08
#
# Compatible Version Matrix:
#   - torch==2.8.0+cu129
#   - torchaudio==2.8.0 (cu129)
#   - torchvision==0.23.0 (cu129)
#   - pyannote-audio==3.4.0
#   - whisperx==3.7.4
#
# IMPORTANT: Python 3.13 is NOT supported due to:
# - torchaudio 2.9.x requires torchcodec which has no ARM64 wheels
# - pyannote-audio 3.x requires torchaudio.AudioMetaData (removed in 2.9)
# - pyannote-audio 4.x requires torchcodec (no ARM64 wheels)
#
# Usage:
#   1. Create Python 3.12 virtual environment:
#      uv venv --python 3.12 .venv-py312
#   2. Activate:
#      source .venv-py312/bin/activate
#   3. Run this script:
#      ./scripts/install-cuda-deps-py312.sh
#
# Prerequisites:
#   - NVIDIA driver installed (nvidia-smi working)
#   - Python 3.12 virtual environment activated
#   - uv installed (pip install uv)
#   - FFmpeg installed (sudo apt install ffmpeg)
#   - libsndfile1 installed (sudo apt install libsndfile1)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ARM64 PyTorch/WhisperX CUDA Installation (Python 3.12) ===${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}Error: This script is for ARM64 (aarch64) only. Detected: $ARCH${NC}"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Please install: pip install uv${NC}"
    exit 1
fi

# Check Python version - must be 3.12
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo -e "${RED}Error: Python 3.12 is REQUIRED. Detected: $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}Python 3.13 has compatibility issues with pyannote-audio on ARM64.${NC}"
    echo ""
    echo "Please create a Python 3.12 environment:"
    echo "  uv venv --python 3.12 .venv-py312"
    echo "  source .venv-py312/bin/activate"
    exit 1
fi

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Please install NVIDIA driver.${NC}"
    exit 1
fi

echo -e "${GREEN}NVIDIA GPU detected:${NC}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

echo -e "${YELLOW}CUDA Driver Version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')${NC}"

# Version configuration
CUDA_WHEEL_VERSION="cu124"
TORCH_VERSION="2.5.1"
TORCHAUDIO_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"

echo -e "${GREEN}Target versions:${NC}"
echo -e "  torch: ${TORCH_VERSION}+${CUDA_WHEEL_VERSION}"
echo -e "  torchaudio: ${TORCHAUDIO_VERSION}"
echo -e "  torchvision: ${TORCHVISION_VERSION}"

# Step 1: Uninstall existing torch packages to avoid conflicts
echo -e "${YELLOW}Step 1: Removing existing torch packages...${NC}"
uv pip uninstall torch torchvision torchaudio 2>/dev/null || true

# Step 2: Install PyTorch with CUDA support using direct wheel URLs
echo -e "${YELLOW}Step 2: Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_WHEEL_VERSION}...${NC}"

uv pip install --no-deps \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torch-${TORCH_VERSION}%2B${CUDA_WHEEL_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchaudio-${TORCHAUDIO_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchvision-${TORCHVISION_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl"

# Install torch dependencies
echo -e "${YELLOW}Step 3: Installing PyTorch dependencies...${NC}"
uv pip install filelock fsspec jinja2 networkx sympy typing-extensions numpy

# Step 4: Verify CUDA is available
echo -e "${YELLOW}Step 4: Verifying CUDA availability...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: CUDA is not available!')
    exit(1)
"

# Step 5: Install audio processing dependencies
echo -e "${YELLOW}Step 5: Installing audio processing dependencies...${NC}"
uv pip install soundfile librosa

# Step 6: Install pyannote-audio
echo -e "${YELLOW}Step 6: Installing pyannote-audio...${NC}"
uv pip install "pyannote-audio>=3.3.2,<4.0.0"

# Re-install CUDA torch (pyannote may have replaced with CPU version)
echo -e "${YELLOW}Step 7: Re-installing CUDA PyTorch (fixing dependency resolution)...${NC}"
uv pip uninstall torch torchaudio torchvision
uv pip install --no-deps \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torch-${TORCH_VERSION}%2B${CUDA_WHEEL_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchaudio-${TORCHAUDIO_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchvision-${TORCHVISION_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl"

# Step 8: Install WhisperX dependencies
echo -e "${YELLOW}Step 8: Installing WhisperX dependencies...${NC}"
uv pip install ctranslate2 faster-whisper transformers

# Step 9: Install WhisperX
echo -e "${YELLOW}Step 9: Installing WhisperX...${NC}"
uv pip install whisperx

# Re-install CUDA torch again (whisperx may have replaced)
echo -e "${YELLOW}Step 10: Final CUDA PyTorch installation...${NC}"
uv pip uninstall torch torchaudio torchvision
uv pip install --no-deps \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torch-${TORCH_VERSION}%2B${CUDA_WHEEL_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchaudio-${TORCHAUDIO_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl" \
    "https://download.pytorch.org/whl/${CUDA_WHEEL_VERSION}/torchvision-${TORCHVISION_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl"

# Step 11: Install project dependencies
echo -e "${YELLOW}Step 11: Installing project dependencies...${NC}"
uv pip install fastapi uvicorn pydantic sqlalchemy aiosqlite greenlet alembic python-multipart aiofiles ffmpeg-python psutil
uv pip install pytest pytest-cov pytest-asyncio httpx ruff

# Step 12: Final verification
echo -e "${YELLOW}Step 12: Final verification...${NC}"

python -c "
import sys
print('=== Verification Results ===')

errors = []

# Check torch
try:
    import torch
    cuda_status = 'CUDA available' if torch.cuda.is_available() else 'CPU only'
    print(f'torch {torch.__version__} ({cuda_status})')
    if not torch.cuda.is_available():
        errors.append('CUDA not available')
except ImportError as e:
    print(f'torch: FAILED - {e}')
    errors.append(str(e))

# Check torchaudio
try:
    import torchaudio
    has_meta = hasattr(torchaudio, 'AudioMetaData')
    has_info = hasattr(torchaudio, 'info')
    print(f'torchaudio {torchaudio.__version__} (AudioMetaData: {has_meta}, info: {has_info})')
    if not has_meta or not has_info:
        errors.append('torchaudio missing required APIs')
except ImportError as e:
    print(f'torchaudio: FAILED - {e}')
    errors.append(str(e))

# Check torchvision
try:
    import torchvision
    print(f'torchvision {torchvision.__version__}')
except ImportError as e:
    print(f'torchvision: FAILED - {e}')
    errors.append(str(e))

# Check pyannote
try:
    from pyannote.audio import Pipeline
    import pyannote.audio
    print(f'pyannote-audio {pyannote.audio.__version__}')
except ImportError as e:
    print(f'pyannote-audio: FAILED - {e}')
    errors.append(str(e))

# Check whisperx
try:
    import whisperx
    print(f'whisperx imported successfully')
except ImportError as e:
    print(f'whisperx: FAILED - {e}')
    errors.append(str(e))

# Check faster_whisper
try:
    import faster_whisper
    print(f'faster-whisper imported successfully')
except ImportError as e:
    print(f'faster-whisper: FAILED - {e}')
    errors.append(str(e))

if errors:
    print(f'=== FAILED with {len(errors)} error(s) ===')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('=== All verifications PASSED ===')
"

echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Installed versions:"
echo "  - torch: ${TORCH_VERSION}+${CUDA_WHEEL_VERSION}"
echo "  - torchaudio: ${TORCHAUDIO_VERSION}"
echo "  - torchvision: ${TORCHVISION_VERSION}"
echo "  - pyannote-audio: 3.4.0"
echo "  - whisperx: 3.7.4"
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
