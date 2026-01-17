# Jetson Orin용 PyTorch CUDA 설치 가이드

## 문제 분석

### 현재 문제
```
No SGEMM backend on CPU
```
- PyPI의 PyTorch는 CPU-only 버전입니다
- Whisper large-v3 모델은 CUDA가 필수입니다

### 원인
Jetson ARM64 아키텍처에서는 NVIDIA에서 별도로 빌드한 PyTorch wheel을 사용해야 합니다.
PyPI의 표준 PyTorch는 CUDA 기능이 포함되어 있지 않습니다.

## 해결 방법

### 방법 1: NVIDIA Developer Forum에서 직접 다운로드 (권장)

1. **NVIDIA 포럼 방문**
   - URL: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
   - JetPack 버전에 맞는 wheel 파일 링크 확인

2. **JetPack 버전 확인**
   ```bash
   cat /etc/nv_tegra_release
   ```

3. **Wheel 파일 다운로드 및 설치**

   JetPack 6.0 / Python 3.12 기준:
   ```bash
   # 다운로드 디렉토리 생성
   mkdir -p ~/tmp/pytorch_jetsom
   cd ~/tmp/pytorch_jetsom

   # PyTorch wheel 다운로드 (NVIDIA Box 또는 포럼 링크 사용)
   wget -O torch.whl <NVIDIA_FORUM_PYTORCH_URL>

   # 설치
   uv pip install torch.whl

   # 확인
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### 방법 2: 자동 설치 스크립트 사용

```bash
# 스크립트 실행
bash /home/innojini/dev/voice.man/scripts/install_jetson_pytorch_correct.sh
```

### 방법 3: JetPack SDK 내장 PyTorch 사용

Jetson에 JetPack SDK가 설치되어 있다면 이미 PyTorch가 포함되어 있을 수 있습니다.

```bash
# 시스템 전체 PyTorch 확인
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 이미 설치되어 있다면 가상환경에 symlink 생성
uv pip install --no-index torch torchvision torchaudio
```

## 설치 확인

```bash
python3 << 'EOF'
import torch

print("=== PyTorch CUDA 확인 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    # 테스트 연산
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"\nGPU 연산 테스트: 성공")
else:
    print("\n오류: CUDA를 사용할 수 없습니다!")
    exit(1)
EOF
```

## GPU 벤치마크 실행

설치가 완료되면:

```bash
# GPU 최적화 벤치마크 실행
cd /home/innojini/dev/voice.man
python scripts/benchmark_forensic_gpu_optimization.py
```

## 참고 자료

- [NVIDIA PyTorch for Jetson Forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Downloads](https://developer.nvidia.com/embedded/downloads)
- [WhisperX GPU Requirements](https://github.com/m-bain/whisperX)

## 문제 해결

### "No SGEMM backend on CPU" 오류
- 원인: CPU-only PyTorch 설치됨
- 해결: NVIDIA CUDA PyTorch wheel로 재설치

### "HTTP 403 Forbidden" 오류
- 원인: PyTorch wheel repository가 액세스를 거부
- 해결: NVIDIA 포럼에서 직접 다운로드

### "weights_only" 오류 (PyTorch 2.6+)
- 원인: PyTorch 2.6부터 보안 강화
- 해결: PyTorch 2.5.x 사용 또는 weights_only=False 설정
