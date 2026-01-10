# GPU 기반 F0 추출 사용자 가이드

**프로젝트**: voice.man 음성포렌식 분석
**모듈**: GPU-Accelerated F0 Extraction (SPEC-GPUAUDIO-001)
**최종 업데이트**: 2026-01-10
**안정성**: ✅ 프로덕션 레벨

---

## 목차

1. [개요](#개요)
2. [설치 및 환경 설정](#설치-및-환경-설정)
3. [빠른 시작](#빠른-시작)
4. [기본 사용법](#기본-사용법)
5. [고급 사용법](#고급-사용법)
6. [성능 최적화](#성능-최적화)
7. [트러블슈팅](#트러블슈팅)
8. [모범 사례](#모범-사례)

---

## 개요

GPU 기반 F0 추출은 음성 신호의 기본 주파수(F0, Fundamental Frequency)를 GPU를 이용해 고속으로 추출하는 기술입니다.

### 주요 특징

| 특징 | 설명 |
|-----|------|
| **성능** | CPU 대비 **114배 빠름** (1.76ms vs 200ms/window) |
| **정확도** | **99.0%** 유효 F0 값 추출 |
| **확장성** | 180K+ 윈도우 안정적 처리 |
| **신뢰도** | 평균 신뢰도 **0.82** (0-1 범위) |
| **메모리** | 메모리 누수 **0건** (확인됨) |
| **안정성** | Graceful error handling (극한 상황에서도 crash 없음) |

### 언제 사용할까?

✅ **GPU 권장** (대규모 처리):
- 180개 이상 오디오 파일
- 실시간 또는 준실시간 처리 필요
- 배치 처리 가능한 시스템
- 처리 속도가 중요한 경우

❌ **CPU 권장** (소규모 처리):
- 100개 미만 오디오 파일
- GPU 메모리 제약
- 낮은 지연 시간 필요
- 비용 최소화 필요

---

## 설치 및 환경 설정

### 필수 요구사항

```bash
# Python 3.8+
python --version

# GPU 환경 (NVIDIA CUDA 12.1+)
nvidia-smi

# 필수 패키지
pip install torch torchcrepe librosa numpy
```

### 기본 설치

```bash
# voice.man 프로젝트 디렉토리에서
cd voice.man

# 개발 환경 설치
pip install -e .

# GPU 지원 확인
python -c "import torch; print(f'GPU 사용 가능: {torch.cuda.is_available()}')"
```

### GPU 설정 확인

```python
import torch

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"PyTorch 버전: {torch.__version__}")
```

### CPU Fallback 설정

환경에 GPU가 없는 경우 자동으로 CPU fallback이 작동합니다:

```python
from src.voice_man.services.forensic.audio_feature_service import AudioFeatureService

# GPU 자동 감지 및 fallback
service = AudioFeatureService(use_gpu=True)
# GPU 없으면 자동으로 CPU 사용 (에러 없음)
```

---

## 빠른 시작

### 1. 단일 오디오 파일 처리

```python
import librosa
from src.voice_man.services.forensic.audio_feature_service import AudioFeatureService

# 서비스 초기화 (GPU 자동 사용)
service = AudioFeatureService(use_gpu=True)

# 오디오 로드
audio_file = "/path/to/audio.wav"
audio, sr = librosa.load(audio_file, sr=16000)

# F0 추출
f0, confidence = service.extract_f0(audio, sr)

print(f"추출된 F0 값: {len(f0)} frames")
print(f"유효 F0: {len(f0[~np.isnan(f0)])}")
print(f"평균 주파수: {np.mean(f0[~np.isnan(f0)]):.2f} Hz")
```

### 2. 배치 처리 (여러 파일)

```python
import numpy as np
from pathlib import Path
import librosa

# 오디오 파일 목록
audio_dir = Path("/path/to/audio_files")
audio_files = sorted(list(audio_dir.glob("*.wav")))

# 모든 윈도우 추출
all_windows = []
sr = None

for audio_file in audio_files:
    audio, sr = librosa.load(str(audio_file), sr=16000)

    # 1초 윈도우로 분할
    window_size = sr
    num_windows = len(audio) // window_size

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio[start:end]
        all_windows.append(window)

# 배치 F0 추출
audio_windows = np.array(all_windows)
f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(
    audio_windows, sr, fmin=50.0, fmax=550.0
)

print(f"처리된 윈도우: {len(all_windows)}")
print(f"유효 F0 비율: {100 * len(f0_batch[~np.isnan(f0_batch)]) / len(f0_batch):.1f}%")
```

### 3. 결과 저장

```python
import json

# F0 및 신뢰도 저장
results = {
    "f0": f0.tolist(),
    "confidence": confidence.tolist(),
    "num_frames": len(f0),
    "sample_rate": sr,
    "valid_ratio": float(len(f0[~np.isnan(f0)]) / len(f0))
}

with open("f0_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("F0 결과 저장 완료")
```

---

## 기본 사용법

### AudioFeatureService 클래스

메인 인터페이스로, F0 추출과 신뢰도 계산을 담당합니다.

#### 초기화

```python
from src.voice_man.services.forensic.audio_feature_service import AudioFeatureService

# GPU 사용 (자동 fallback)
service = AudioFeatureService(use_gpu=True)

# CPU만 사용
service_cpu = AudioFeatureService(use_gpu=False)
```

#### 메서드: `extract_f0()`

**단일 오디오 F0 추출**

```python
f0, confidence = service.extract_f0(
    audio,           # numpy array, shape (n_samples,)
    sr,              # sampling rate (Hz)
    fmin=50.0,       # minimum frequency (Hz)
    fmax=550.0       # maximum frequency (Hz)
)

# 반환값
# f0: numpy array, shape (n_frames,), unit: Hz (NaN for unvoiced)
# confidence: numpy array, shape (n_frames,), range: [0, 1]
```

**매개변수**:
- `audio` (numpy.ndarray): 오디오 신호
- `sr` (int): 샘플링 레이트 (권장: 16000 Hz)
- `fmin` (float): 최소 주파수 (기본값: 50 Hz)
- `fmax` (float): 최대 주파수 (기본값: 550 Hz)

**반환값**:
- `f0` (numpy.ndarray): F0 값 (주파수, Hz)
- `confidence` (numpy.ndarray): 신뢰도 (0-1)

**예제**:

```python
import numpy as np

# 간단한 테스트
audio = np.random.randn(16000) * 0.1  # 1초 랜덤 노이즈
f0, conf = service.extract_f0(audio, sr=16000)

print(f"F0 통계:")
print(f"  - 유효 값: {len(f0[~np.isnan(f0)])} frames")
print(f"  - 평균: {np.nanmean(f0):.2f} Hz")
print(f"  - 범위: {np.nanmin(f0):.2f} - {np.nanmax(f0):.2f} Hz")
print(f"신뢰도 통계:")
print(f"  - 평균: {np.mean(conf[~np.isnan(f0)]):.4f}")
print(f"  - 높음(>0.8): {100*len(conf[conf>0.8])/len(conf):.1f}%")
```

### GPUAudioBackend 클래스

배치 처리 최적화를 위한 GPU 백엔드입니다.

#### 초기화

```python
from src.voice_man.services.forensic.gpu.backend import GPUAudioBackend

backend = GPUAudioBackend(use_gpu=True)
```

#### 메서드: `extract_f0_batch()`

**배치 오디오 F0 추출 (최적화됨)**

```python
f0_batch, conf_batch = backend.extract_f0_batch(
    audio_windows,   # numpy array, shape (batch_size, n_samples)
    sr,              # sampling rate (Hz)
    fmin=50.0,       # minimum frequency (Hz)
    fmax=550.0       # maximum frequency (Hz)
)

# 반환값
# f0_batch: numpy array, shape (batch_size,)
# conf_batch: numpy array, shape (batch_size,)
```

**매개변수**:
- `audio_windows` (numpy.ndarray): 배치 오디오, shape (batch_size, n_samples)
- `sr` (int): 샘플링 레이트
- `fmin`, `fmax` (float): 주파수 범위

**반환값**:
- `f0_batch` (numpy.ndarray): F0 값 배치
- `conf_batch` (numpy.ndarray): 신뢰도 배치

**성능 정보**:

```
배치 크기별 처리 시간 (183개 파일, 74,446개 윈도우):
- 처리 시간: 131초 (2분 11초)
- 윈도우당 시간: 1.76ms
- 처리 속도: 568 윈도우/초
```

---

## 고급 사용법

### 오디오 윈도우 처리 패턴

#### 패턴 1: 고정 길이 윈도우

```python
def extract_windows_fixed(audio_file, window_duration=1.0):
    """고정 길이 윈도우 추출 (1초)"""
    audio, sr = librosa.load(audio_file, sr=16000)

    window_size = int(window_duration * sr)
    windows = []

    for i in range(0, len(audio) - window_size, window_size):
        window = audio[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)

    return np.array(windows), sr

# 사용 예제
windows, sr = extract_windows_fixed("audio.wav", window_duration=1.0)
f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(windows, sr)
```

#### 패턴 2: 슬라이딩 윈도우

```python
def extract_windows_sliding(audio_file, window_duration=0.5, hop_duration=0.25):
    """슬라이딩 윈도우 추출 (오버랩 가능)"""
    audio, sr = librosa.load(audio_file, sr=16000)

    window_size = int(window_duration * sr)
    hop_size = int(hop_duration * sr)
    windows = []

    for start in range(0, len(audio) - window_size, hop_size):
        window = audio[start:start + window_size]
        windows.append(window)

    return np.array(windows), sr, hop_size

# 사용 예제
windows, sr, hop_size = extract_windows_sliding(
    "audio.wav",
    window_duration=0.5,  # 500ms
    hop_duration=0.25     # 250ms (50% overlap)
)
f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(windows, sr)
```

#### 패턴 3: Concatenate-Extract-Split (최적화)

GPU 메모리를 최대한 활용하는 전략입니다:

```python
def extract_f0_optimized(audio_files, sr=16000):
    """모든 오디오를 연결하여 한 번에 처리 (GPU 최적화)"""

    # 1단계: 모든 윈도우 수집
    all_windows = []
    for audio_file in audio_files:
        audio, _ = librosa.load(audio_file, sr=sr)
        window_size = sr
        for i in range(0, len(audio) - window_size, window_size):
            all_windows.append(audio[i:i + window_size])

    # 2단계: 배치 처리 (한 번의 GPU 호출)
    windows_array = np.array(all_windows)
    f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(windows_array, sr)

    # 3단계: 결과 저장
    return f0_batch, conf_batch

# 사용 예제
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
f0_all, conf_all = extract_f0_optimized(audio_files)

print(f"처리된 총 윈도우: {len(f0_all)}")
print(f"유효 F0: {len(f0_all[~np.isnan(f0_all)])}")
```

### 오디오 전처리

#### 정규화

```python
def normalize_audio(audio, target_db=-20.0):
    """오디오 정규화"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        # RMS 정규화
        audio = audio * (10 ** (target_db / 20)) / rms
    return audio

# 사용
audio = librosa.load("audio.wav", sr=16000)[0]
audio_normalized = normalize_audio(audio)
f0, conf = service.extract_f0(audio_normalized, sr=16000)
```

#### 고주파 노이즈 제거

```python
import scipy.signal as signal

def remove_high_freq_noise(audio, sr, cutoff_freq=4000):
    """고주파 노이즈 제거 (저역 통과 필터)"""
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    b, a = signal.butter(4, normalized_cutoff, btype='low')
    audio_filtered = signal.filtfilt(b, a, audio)

    return audio_filtered

# 사용
audio = librosa.load("audio.wav", sr=16000)[0]
audio_filtered = remove_high_freq_noise(audio, sr=16000)
f0, conf = service.extract_f0(audio_filtered, sr=16000)
```

### 결과 후처리

#### 중값 필터링 (스무딩)

```python
from scipy.ndimage import median_filter

def smooth_f0(f0, median_filter_size=5):
    """F0 값 스무딩 (중값 필터)"""
    # NaN을 임시값으로 치환
    f0_filled = np.where(np.isnan(f0), np.nanmedian(f0), f0)
    f0_smoothed = median_filter(f0_filled, size=median_filter_size)

    # 원래 NaN 위치 복원
    f0_smoothed[np.isnan(f0)] = np.nan

    return f0_smoothed

# 사용
f0, conf = service.extract_f0(audio, sr=16000)
f0_smoothed = smooth_f0(f0, median_filter_size=5)
```

#### 신뢰도 기반 필터링

```python
def filter_by_confidence(f0, confidence, min_confidence=0.8):
    """신뢰도가 낮은 F0 값 제거"""
    f0_filtered = f0.copy()
    f0_filtered[confidence < min_confidence] = np.nan
    return f0_filtered

# 사용
f0, conf = service.extract_f0(audio, sr=16000)
f0_high_conf = filter_by_confidence(f0, conf, min_confidence=0.8)

print(f"높은 신뢰도 F0: {len(f0_high_conf[~np.isnan(f0_high_conf)])}")
```

---

## 성능 최적화

### 메모리 효율적인 배치 처리

```python
def batch_process_large_dataset(audio_files, batch_size=100, sr=16000):
    """대규모 데이터셋 배치 처리 (메모리 효율)"""

    all_f0_results = []
    all_conf_results = []

    # 배치 단위로 처리
    for batch_idx in range(0, len(audio_files), batch_size):
        batch_files = audio_files[batch_idx:batch_idx + batch_size]

        # 이번 배치의 윈도우 수집
        batch_windows = []
        for audio_file in batch_files:
            audio, _ = librosa.load(audio_file, sr=sr)
            window_size = sr
            for i in range(0, len(audio) - window_size, window_size):
                batch_windows.append(audio[i:i + window_size])

        # 배치 처리
        windows_array = np.array(batch_windows)
        f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(windows_array, sr)

        all_f0_results.extend(f0_batch)
        all_conf_results.extend(conf_batch)

        print(f"처리됨: {batch_idx + len(batch_files)}/{len(audio_files)}")

    return np.array(all_f0_results), np.array(all_conf_results)

# 사용
audio_files = [...]  # 1000개 이상 파일
f0_all, conf_all = batch_process_large_dataset(audio_files, batch_size=100)
```

### 성능 모니터링

```python
import time

def benchmark_f0_extraction(audio, sr):
    """F0 추출 성능 측정"""

    # GPU 사용
    service_gpu = AudioFeatureService(use_gpu=True)
    start = time.time()
    f0_gpu, _ = service_gpu.extract_f0(audio, sr)
    gpu_time = time.time() - start

    # CPU 사용
    service_cpu = AudioFeatureService(use_gpu=False)
    start = time.time()
    f0_cpu, _ = service_cpu.extract_f0(audio, sr)
    cpu_time = time.time() - start

    print(f"GPU 시간: {gpu_time:.4f}초")
    print(f"CPU 시간: {cpu_time:.4f}초")
    print(f"성능 향상: {cpu_time/gpu_time:.1f}배")

    return gpu_time, cpu_time

# 사용
audio, sr = librosa.load("audio.wav", sr=16000)
gpu_time, cpu_time = benchmark_f0_extraction(audio, sr)
```

---

## 트러블슈팅

### 문제 1: CUDA/GPU 감지 안 됨

```python
# 확인
import torch
print(torch.cuda.is_available())  # False인 경우

# 해결 방법
# 1. NVIDIA GPU 드라이버 확인
# nvidia-smi 명령어 실행

# 2. PyTorch 재설치 (CUDA 지원 버전)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 환경 변수 확인
import os
print(os.environ.get('CUDA_VISIBLE_DEVICES'))  # None일 수 있음

# 해결: GPU 명시 지정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0 사용
```

### 문제 2: 메모리 부족 (CUDA OOM)

```python
# 증상: RuntimeError: CUDA out of memory

# 해결 방법 1: 배치 크기 감소
windows = np.array(all_windows[:1000])  # 1000개씩만 처리
f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(windows, sr)

# 해결 방법 2: GPU 메모리 정리
import torch
torch.cuda.empty_cache()

# 해결 방법 3: CPU fallback 사용
service = AudioFeatureService(use_gpu=False)  # CPU 강제 사용
```

### 문제 3: NaN 값이 많은 경우

```python
# 원인 분석
f0, conf = service.extract_f0(audio, sr)
nan_ratio = len(f0[np.isnan(f0)]) / len(f0)

print(f"NaN 비율: {100 * nan_ratio:.1f}%")

if nan_ratio > 0.5:
    # 오디오 검사
    print(f"오디오 RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"오디오 범위: {np.min(audio):.4f} - {np.max(audio):.4f}")

    # 해결 방법
    # 1. 정규화
    audio = audio / np.max(np.abs(audio))

    # 2. 노이즈 제거
    audio = remove_high_freq_noise(audio, sr)

    # 3. 증폭
    audio = audio * 2.0
```

### 문제 4: 느린 처리 속도

```python
# 원인 확인
# 1. GPU 사용 여부 확인
service = AudioFeatureService(use_gpu=True)
print(service._use_gpu)  # False면 GPU 작동 안 함

# 2. 배치 크기 확인
windows = np.array(all_windows)
print(f"배치 크기: {len(windows)}")  # 작으면 GPU 효율 낮음

# 해결 방법
# - 더 큰 배치 크기 사용 (메모리 허용 범위 내)
# - GPU 메모리 정리
torch.cuda.empty_cache()

# - 배치 처리 확인
f0_batch, _ = service.gpu_backend.extract_f0_batch(windows, sr)
# 단일 처리보다 훨씬 빠름
```

---

## 모범 사례

### ✅ 권장 패턴

```python
# 1. 서비스 초기화 (한 번)
service = AudioFeatureService(use_gpu=True)

# 2. 오디오 파일 수집
audio_files = list(Path("./audio").glob("*.wav"))

# 3. 배치 처리 (메모리 효율)
batch_size = 100
all_results = []

for batch_idx in range(0, len(audio_files), batch_size):
    batch_files = audio_files[batch_idx:batch_idx + batch_size]

    # 윈도우 추출
    batch_windows = []
    for audio_file in batch_files:
        audio, sr = librosa.load(audio_file, sr=16000)
        # 윈도우 추출...

    # 배치 F0 추출
    f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(
        np.array(batch_windows), sr
    )

    # 결과 저장
    for f0, conf in zip(f0_batch, conf_batch):
        all_results.append({"f0": f0, "confidence": conf})

# 4. 결과 분석
print(f"총 처리된 윈도우: {len(all_results)}")
```

### ❌ 피해야 할 패턴

```python
# 1. 매번 서비스 재초기화 (비효율)
for audio_file in audio_files:
    service = AudioFeatureService()  # ❌ 반복 초기화
    f0, conf = service.extract_f0(audio, sr)

# 2. 모든 파일을 메모리에 로드 (메모리 오버플로우)
all_audio = [librosa.load(f)[0] for f in audio_files]  # ❌ 위험
windows = np.concatenate([...])  # ❌ 메모리 부족

# 3. 오류 처리 없음 (프로덕션 부적합)
f0, conf = service.extract_f0(audio, sr)  # ❌ 예외 처리 없음
```

### 프로덕션 코드 예제

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def process_audio_dataset(audio_dir, output_dir, batch_size=100):
    """프로덕션 레벨 배치 처리"""

    # 초기화
    service = AudioFeatureService(use_gpu=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 파일 수집
    audio_files = sorted(list(Path(audio_dir).glob("*.wav")))
    logger.info(f"처리할 파일: {len(audio_files)}개")

    # 배치 처리
    processed = 0
    for batch_idx in range(0, len(audio_files), batch_size):
        try:
            batch_files = audio_files[batch_idx:batch_idx + batch_size]
            batch_windows = []
            window_metadata = []

            # 윈도우 추출
            for file_idx, audio_file in enumerate(batch_files):
                try:
                    audio, sr = librosa.load(str(audio_file), sr=16000)
                    window_size = sr

                    for win_idx in range(0, len(audio) - window_size, window_size):
                        window = audio[win_idx:win_idx + window_size]
                        batch_windows.append(window)
                        window_metadata.append({
                            "file": audio_file.name,
                            "file_idx": file_idx,
                            "window_idx": win_idx // window_size
                        })
                except Exception as e:
                    logger.warning(f"파일 처리 실패: {audio_file.name} - {e}")
                    continue

            if not batch_windows:
                logger.warning(f"배치 {batch_idx//batch_size + 1}: 윈도우 없음")
                continue

            # 배치 F0 추출
            f0_batch, conf_batch = service.gpu_backend.extract_f0_batch(
                np.array(batch_windows), sr
            )

            # 결과 저장
            for f0, conf, meta in zip(f0_batch, conf_batch, window_metadata):
                result_file = output_dir / f"{meta['file']}_w{meta['window_idx']:04d}.json"

                result = {
                    "file": meta['file'],
                    "window_index": meta['window_idx'],
                    "f0": float(f0) if not np.isnan(f0) else None,
                    "confidence": float(conf),
                    "valid": not np.isnan(f0)
                }

                with open(result_file, "w") as f:
                    json.dump(result, f)

            processed += len(batch_files)
            logger.info(f"처리됨: {processed}/{len(audio_files)}")

        except Exception as e:
            logger.error(f"배치 처리 실패: {e}", exc_info=True)
            continue

    logger.info("처리 완료")

# 실행
process_audio_dataset("./audio", "./results")
```

---

## 요약

### 주요 학습 포인트

| 항목 | 권장사항 |
|-----|--------|
| **GPU 사용** | use_gpu=True (자동 fallback) |
| **배치 크기** | 100-1000개 윈도우 |
| **오디오 길이** | 1초 이상 (권장: 2초+) |
| **음역대** | 50-550Hz (최적: 100-500Hz) |
| **신뢰도 필터** | 0.8 이상 권장 |
| **메모리** | 배치 처리 시 선형 증가 |

### 지원 연락처

문제 발생 시:
1. VALIDATION_PHASE_*.md 문서 참조
2. API_REFERENCE.md의 트러블슈팅 섹션 확인
3. GitHub Issues에 보고

---

**최종 업데이트**: 2026-01-10
**유지보수**: voice.man 팀
**라이선스**: MIT

