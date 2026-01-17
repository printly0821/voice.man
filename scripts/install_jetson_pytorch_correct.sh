#!/bin/bash
# Jetson Orin용 올바른 PyTorch CUDA 설치 스크립트
# Reference: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

set -e

echo "=== Jetson Orin용 PyTorch CUDA 설치 (NVIDIA 공식 빌드) ==="

# JetPack 버전 확인
echo "Step 1: JetPack 버전 확인..."
JETPACK_VERSION=$(cat /etc/nv_tegra_release 2>/dev/null | grep "R" | head -1 | awk '{print $2}')
if [ -z "$JETPACK_VERSION" ]; then
    echo "  JetPack 버전을 확인할 수 없습니다. 기본값 사용."
    JETPACK_VERSION="6.0"
fi
echo "  JetPack 버전: $JETPACK_VERSION"

# Python 버전 확인
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python 버전: $PYTHON_VERSION"

# 기존 PyTorch 제거
echo ""
echo "Step 2: 기존 PyTorch 제거..."
uv pip uninstall torch torchvision torchaudio -y || true

# NVIDIA PyTorch wheel 다운로드 (Jetson 전용)
echo ""
echo "Step 3: NVIDIA PyTorch wheel 다운로드..."

# JetPack 6.0 / Python 3.12용 PyTorch 버전
PYTORCH_VERSION="2.5.0"
TORCHAUDIO_VERSION="2.5.0"
TORCHVISION_VERSION="0.20.0"

WHEEL_DIR="/tmp/jetson_wheels"
mkdir -p "$WHEEL_DIR"
cd "$WHEEL_DIR"

# PyTorch wheel 다운로드 (NVIDIA 저장소)
echo "  다운로드 중: torch-${PYTORCH_VERSION}+nv24.09..."
wget -q --show-progress --timeout=30 --tries=3 \
    https://nvidia.box.com/shared/static/bs84fn6t24j33s8tjd1p35n42s50s4f2.whl \
    -O torch-${PYTORCH_VERSION}.whl || {
    echo "  오류: NVIDIA wheel 다운로드 실패"
    echo "  대안: NVIDIA 포럼에서 직접 다운로드 필요"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    exit 1
}

# torchaudio wheel 다운로드
echo "  다운로드 중: torchaudio-${TORCHAUDIO_VERSION}+nv24.09..."
wget -q --show-progress --timeout=30 --tries=3 \
    https://nvidia.box.com/shared/static/gbqst2s0t9g0t30e4sgtg8tp30z3z2p8.whl \
    -O torchaudio-${TORCHAUDIO_VERSION}.whl || {
    echo "  경고: torchaudio 다운로드 실패 (선택 사항)"
}

# torchvision wheel 다운로드
echo "  다운로드 중: torchvision-${TORCHVISION_VERSION}+nv24.09..."
wget -q --show-progress --timeout=30 --tries=3 \
    https://nvidia.box.com/shared/static/gp4w25tcw7wqzh13t7lvgt4fc2w2j12m.whl \
    -O torchvision-${TORCHVISION_VERSION}.whl || {
    echo "  경고: torchvision 다운로드 실패 (선택 사항)"
}

# PyTorch 설치
echo ""
echo "Step 4: PyTorch 설치..."
uv pip install "$WHEEL_DIR/torch-${PYTORCH_VERSION}.whl"

# torchaudio 설치 (파일이 존재하면)
if [ -f "$WHEEL_DIR/torchaudio-${TORCHAUDIO_VERSION}.whl" ]; then
    uv pip install "$WHEEL_DIR/torchaudio-${TORCHAUDIO_VERSION}.whl"
fi

# torchvision 설치 (파일이 존재하면)
if [ -f "$WHEEL_DIR/torchvision-${TORCHVISION_VERSION}.whl" ]; then
    uv pip install "$WHEEL_DIR/torchvision-${TORCHVISION_VERSION}.whl"
fi

# CUDA 확인
echo ""
echo "Step 5: CUDA 확인..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('ERROR: CUDA를 사용할 수 없습니다!')
    exit(1)
" || {
    echo ""
    echo "=== 설치 실패 ==="
    echo "CUDA가 활성화되지 않았습니다."
    echo ""
    echo "수동 설치 방법:"
    echo "1. NVIDIA 포럼 방문: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "2. JetPack 버전에 맞는 wheel 파일 다운로드"
    echo "3. uv pip install <wheel-file>.whl"
    exit 1
}

echo ""
echo "=== 설치 완료 ==="
echo "CUDA PyTorch가 성공적으로 설치되었습니다!"
