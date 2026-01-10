# GPU F0 추출 API 레퍼런스

**프로젝트**: voice.man 음성포렌식 분석
**모듈**: GPU-Accelerated F0 Extraction
**버전**: 1.0.0
**API 안정성**: ✅ Stable

---

## 목차

1. [개요](#개요)
2. [핵심 클래스](#핵심-클래스)
3. [AudioFeatureService](#audiofeatureservice)
4. [GPUAudioBackend](#gpuaudiobackend)
5. [TorchCrepeExtractor](#torchcrepeextractor)
6. [예외 처리](#예외-처리)
7. [성능 메트릭](#성능-메트릭)

---

## 개요

GPU F0 추출 API는 3계층 아키텍처로 구성됩니다:

```
┌─────────────────────────────┐
│   AudioFeatureService       │  (사용자 인터페이스)
│  - extract_f0()            │
│  - gpu_backend 접근         │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│   GPUAudioBackend           │  (GPU 최적화 레이어)
│  - extract_f0_batch()      │
│  - fallback 관리            │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│   TorchCrepeExtractor       │  (GPU 드라이버)
│  - PyTorch/Crepe 사용       │
│  - 배치 처리 지원            │
└─────────────────────────────┘
```

---

## 핵심 클래스

### 클래스 계층도

```
ForensicService (베이스)
  └── AudioFeatureService
        ├── gpu_backend: GPUAudioBackend
        └── _use_gpu: bool

GPUAudioBackend
  ├── crepe_extractor: TorchCrepeExtractor
  ├── is_gpu_available: bool
  └── device: str

TorchCrepeExtractor
  ├── model: str ("full" | "tiny")
  ├── device: str ("cuda" | "cpu")
  └── model: nn.Module (torchcrepe)
```

---

## AudioFeatureService

음성 기반 법의학 분석을 위한 메인 서비스 클래스입니다.

### 위치

```python
from src.voice_man.services.forensic.audio_feature_service import AudioFeatureService
```

### 클래스 정의

```python
class AudioFeatureService(ForensicService):
    """음성 특성 추출 서비스 (GPU 지원)"""

    def __init__(self, use_gpu: bool = True):
        """
        AudioFeatureService 초기화

        파라미터:
            use_gpu (bool): GPU 사용 여부
                - True: GPU 시도, 실패 시 CPU fallback
                - False: CPU만 사용

        속성:
            _use_gpu (bool): GPU 사용 여부
            gpu_backend (GPUAudioBackend): GPU 백엔드 인스턴스
        """
```

### 메서드

#### 1. `extract_f0()`

**단일 오디오의 F0(기본 주파수) 추출**

```python
def extract_f0(
    self,
    audio: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 550.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    오디오 신호에서 F0(기본 주파수)를 추출합니다.

    파라미터:
        audio (np.ndarray):
            - 형태: (n_samples,)
            - 타입: float32 또는 float64
            - 값 범위: 일반적으로 [-1.0, 1.0], 자동 정규화
            설명: 단일 채널 오디오 신호

        sr (int):
            - 단위: Hz
            - 권장값: 16000
            - 지원값: 8000 이상
            설명: 오디오 샘플링 레이트

        fmin (float):
            - 기본값: 50.0
            - 단위: Hz
            - 범위: 20 ~ 400
            설명: F0 최소값 (하한)

        fmax (float):
            - 기본값: 550.0
            - 단위: Hz
            - 범위: 100 ~ 2000
            설명: F0 최대값 (상한)

    반환값:
        Tuple[np.ndarray, np.ndarray]:
            (f0, confidence)

            f0 (np.ndarray):
                - 형태: (n_frames,)
                - 타입: float32
                - 값: Hz (주파수)
                - NaN: 음성이 아닌 부분 (unvoiced)
                설명: 각 프레임의 기본 주파수

            confidence (np.ndarray):
                - 형태: (n_frames,)
                - 타입: float32
                - 값 범위: [0.0, 1.0]
                - 1.0: 매우 높은 신뢰도 (유성음)
                - 0.0: 신뢰도 없음 (무성음)
                설명: F0 추출의 신뢰도 점수

    예외:
        ValueError: audio 형태 또는 sr이 유효하지 않은 경우
        RuntimeError: GPU/CPU 처리 중 오류 발생

    예제:
        >>> import librosa
        >>> from src.voice_man.services.forensic.audio_feature_service import AudioFeatureService
        >>>
        >>> service = AudioFeatureService(use_gpu=True)
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> f0, confidence = service.extract_f0(audio, sr)
        >>>
        >>> print(f"F0 프레임: {len(f0)}")
        >>> print(f"유효 F0: {len(f0[~np.isnan(f0)])}")
        >>> print(f"평균 주파수: {np.nanmean(f0):.2f} Hz")

    참고:
        - 자동 fallback: GPU 오류 시 CPU 사용
        - 프레임 길이: hop_length = sr // 100 (10ms)
        - 처리 시간:
          * GPU: ~1.76ms (1초 오디오)
          * CPU: ~200ms (1초 오디오)
    """
```

#### 2. 속성

```python
@property
def gpu_backend(self) -> GPUAudioBackend:
    """GPU 백엔드 인스턴스 접근"""

@property
def _use_gpu(self) -> bool:
    """GPU 사용 여부"""
```

---

## GPUAudioBackend

GPU 최적화 배치 처리를 위한 백엔드입니다.

### 위치

```python
from src.voice_man.services.forensic.gpu.backend import GPUAudioBackend
```

### 클래스 정의

```python
class GPUAudioBackend:
    """GPU 기반 오디오 처리 백엔드"""

    def __init__(self, use_gpu: bool = True):
        """
        GPUAudioBackend 초기화

        파라미터:
            use_gpu (bool): GPU 사용 여부

        속성:
            crepe_extractor (TorchCrepeExtractor): CREPE 모델
            is_gpu_available (bool): GPU 사용 가능 여부
            device (str): 사용 장치 ("cuda" 또는 "cpu")
        """
```

### 메서드

#### 1. `extract_f0_batch()`

**배치 오디오의 F0 추출 (최적화)**

```python
def extract_f0_batch(
    self,
    audio_windows: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 550.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    배치 오디오 윈도우에서 F0를 추출합니다 (GPU 최적화).

    이 메서드는 Concatenate-Extract-Split 전략을 사용하여
    모든 윈도우를 연결하고 한 번에 처리하여 GPU 효율을 최대화합니다.

    파라미터:
        audio_windows (np.ndarray):
            - 형태: (batch_size, n_samples)
            - 타입: float32 또는 float64
            설명: 배치 오디오 윈도우 배열

        sr (int):
            - 단위: Hz
            - 권장값: 16000
            설명: 샘플링 레이트 (모든 윈도우 동일)

        fmin (float):
            - 기본값: 50.0
            - 단위: Hz
            설명: F0 최소값

        fmax (float):
            - 기본값: 550.0
            - 단위: Hz
            설명: F0 최대값

    반환값:
        Tuple[np.ndarray, np.ndarray]:
            (f0_batch, confidence_batch)

            f0_batch (np.ndarray):
                - 형태: (batch_size,)
                - 값: 각 윈도우의 F0 (Hz)
                - NaN: 음성 부재

            confidence_batch (np.ndarray):
                - 형태: (batch_size,)
                - 값: 각 윈도우의 신뢰도 [0, 1]

    성능:
        - 처리 속도: ~1.76ms/윈도우 (GPU)
        - 메모리 사용: ~2GB (74,446개 윈도우)
        - 배치 크기별:
          * 100 윈도우: 176ms
          * 1,000 윈도우: 1.76초
          * 10,000 윈도우: 17.6초

    예제:
        >>> import numpy as np
        >>> import librosa
        >>>
        >>> # 오디오 로드 및 윈도우 생성
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> window_size = sr  # 1초 윈도우
        >>> windows = []
        >>> for i in range(0, len(audio) - window_size, window_size):
        ...     windows.append(audio[i:i+window_size])
        >>> audio_windows = np.array(windows)
        >>>
        >>> # 배치 F0 추출
        >>> backend = GPUAudioBackend(use_gpu=True)
        >>> f0_batch, conf_batch = backend.extract_f0_batch(audio_windows, sr)
        >>>
        >>> print(f"처리된 윈도우: {len(f0_batch)}")
        >>> print(f"유효 F0: {len(f0_batch[~np.isnan(f0_batch)])}")

    최적화 팁:
        1. 배치 크기 증가:
           - 작은 배치 (< 100): GPU 효율 낮음
           - 중간 배치 (100-1000): 권장
           - 큰 배치 (> 1000): 메모리 주의

        2. 메모리 관리:
           - GPU OOM 시 배치 크기 감소
           - torch.cuda.empty_cache() 사용
           - CPU fallback 고려

        3. 정확도:
           - fmin=50, fmax=550 권장
           - 범위 외 음역대는 NaN 반환

    참고:
        - 자동 fallback: GPU 오류 시 CPU 사용
        - Concatenate-Extract-Split 전략:
          * 모든 윈도우 연결 (메모리 효율)
          * 한 번의 torchcrepe 호출 (속도)
          * 윈도우별 F0 분할 (정확성)
    """
```

#### 2. 속성

```python
@property
def is_gpu_available(self) -> bool:
    """GPU 사용 가능 여부"""

@property
def device(self) -> str:
    """사용 중인 장치 ('cuda' 또는 'cpu')"""

@property
def crepe_extractor(self) -> TorchCrepeExtractor:
    """CREPE 추출기 인스턴스"""
```

---

## TorchCrepeExtractor

PyTorch 기반 CREPE 모델 래퍼입니다.

### 위치

```python
from src.voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor
```

### 클래스 정의

```python
class TorchCrepeExtractor:
    """PyTorch CREPE 기반 F0 추출기"""

    def __init__(
        self,
        model: str = "full",
        device: str = "cuda"
    ):
        """
        TorchCrepeExtractor 초기화

        파라미터:
            model (str):
                - "full": 정확도 우선 (기본값)
                - "tiny": 속도 우선 (경량)
                설명: 모델 크기

            device (str):
                - "cuda": GPU 사용
                - "cpu": CPU 사용
                설명: 실행 장치

        속성:
            model (nn.Module): Crepe 신경망 모델
            device (str): 현재 장치
            is_gpu_available (bool): GPU 사용 가능 여부
        """
```

### 메서드

#### 1. `extract_f0()`

```python
def extract_f0(
    self,
    audio: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 550.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 오디오의 F0 추출

    파라미터:
        audio (np.ndarray): 오디오 신호 (n_samples,)
        sr (int): 샘플링 레이트
        fmin (float): F0 최소값
        fmax (float): F0 최대값

    반환값:
        (f0, confidence): F0 값과 신뢰도
    """
```

#### 2. `extract_f0_batch()`

```python
def extract_f0_batch(
    self,
    audio: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 550.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    배치 오디오의 F0 추출 (최적화)

    파라미터:
        audio (np.ndarray): 배치 오디오 (batch_size, n_samples)
        sr (int): 샘플링 레이트
        fmin (float): F0 최소값
        fmax (float): F0 최대값

    반환값:
        (f0_batch, confidence_batch): 배치 F0 및 신뢰도

    구현:
        1. 모든 윈도우 연결 (메모리 절약)
        2. 한 번의 torchcrepe 호출 (속도)
        3. 윈도우별 결과 분할 (정확성)
    """
```

#### 3. 속성

```python
@property
def is_gpu_available(self) -> bool:
    """GPU 사용 가능 여부"""
```

---

## 예외 처리

### 표준 예외

#### ValueError

```python
# 원인: 잘못된 오디오 형태
try:
    f0, conf = service.extract_f0(
        audio.reshape(-1, 1),  # 2D 배열
        sr=16000
    )
except ValueError as e:
    print(f"오디오 형태 오류: {e}")
    # 해결: 1D 배열 사용
    f0, conf = service.extract_f0(audio, sr=16000)
```

#### RuntimeError

```python
# 원인: GPU 메모리 부족
try:
    f0_batch, _ = backend.extract_f0_batch(large_batch, sr)
except RuntimeError as e:
    print(f"GPU 오류: {e}")
    # 해결 방법 1: 배치 크기 감소
    f0_batch, _ = backend.extract_f0_batch(large_batch[:100], sr)
    # 해결 방법 2: CPU fallback
    service_cpu = AudioFeatureService(use_gpu=False)
    f0, _ = service_cpu.extract_f0(audio, sr)
```

### 예외 처리 패턴

```python
import logging
import numpy as np

logger = logging.getLogger(__name__)

def safe_extract_f0(audio, sr, max_retries=2):
    """안전한 F0 추출 (재시도 로직)"""

    service = AudioFeatureService(use_gpu=True)

    for attempt in range(max_retries):
        try:
            f0, conf = service.extract_f0(audio, sr)

            # 결과 검증
            if f0 is None or len(f0) == 0:
                raise ValueError("F0 결과 없음")

            return f0, conf

        except RuntimeError as e:
            if attempt < max_retries - 1:
                logger.warning(f"시도 {attempt+1} 실패: {e}, 재시도...")
                continue
            else:
                logger.error(f"GPU 처리 실패, CPU로 재시도")
                service = AudioFeatureService(use_gpu=False)
                return service.extract_f0(audio, sr)

        except ValueError as e:
            logger.error(f"값 오류: {e}")
            raise

    return None, None
```

---

## 성능 메트릭

### 벤치마크 결과

#### 처리 속도 비교

```
단일 오디오 (1초, 16000 samples):
┌─────────────┬──────────────┬──────────────┐
│    항목     │     GPU      │     CPU      │
├─────────────┼──────────────┼──────────────┤
│ 처리 시간   │   1.76ms     │   200ms      │
│ 속도 향상   │      -       │   114배      │
└─────────────┴──────────────┴──────────────┘

배치 처리 (74,446개 윈도우):
┌─────────────┬──────────────┬──────────────┐
│    항목     │     GPU      │     CPU      │
├─────────────┼──────────────┼──────────────┤
│ 처리 시간   │   2분 11초   │   4.1시간    │
│ 속도 향상   │      -       │   113배      │
└─────────────┴──────────────┴──────────────┘
```

#### 정확도 메트릭

```
전체 데이터셋 (183개 파일):
┌──────────────────┬─────────────┐
│      지표        │    값       │
├──────────────────┼─────────────┤
│ 유효 F0 비율     │  99.0%      │
│ 평균 신뢰도      │  0.82       │
│ F0 범위          │  60-550Hz   │
│ 높은 신뢰도(>0.8)│  82%        │
└──────────────────┴─────────────┘
```

#### 메모리 사용량

```
GPU 메모리 (74,446개 윈도우 처리):
┌──────────────────┬─────────────┐
│      항목        │    사용량   │
├──────────────────┼─────────────┤
│ 배치 메모리      │   2GB       │
│ 메모리 누수      │   0건       │
│ 처리 후 정리     │  < 1MB      │
└──────────────────┴─────────────┘

배치 크기별 메모리:
100   윈도우 →  200MB
500   윈도우 →  900MB
1,000 윈도우 → 1.8GB
```

### 성능 최적화 가이드

#### 배치 크기 최적화

```python
# 최적 배치 크기 선택
import torch

def optimal_batch_size(sr=16000, gpu_memory_gb=8):
    """GPU 메모리에 따른 최적 배치 크기 계산"""

    # 경험적 공식: 약 24MB/윈도우
    bytes_per_window = 24 * 1024 * 1024
    gpu_memory_bytes = gpu_memory_gb * 1024 * 1024 * 1024

    optimal = int(gpu_memory_bytes * 0.8 / bytes_per_window)  # 80% 사용

    return max(100, min(optimal, 5000))  # 100-5000 범위

batch_size = optimal_batch_size(gpu_memory_gb=8)
print(f"권장 배치 크기: {batch_size}")
```

#### 메모리 모니터링

```python
import psutil
import torch

def monitor_memory(service, audio_windows):
    """메모리 사용량 모니터링"""

    # 처리 전
    torch.cuda.reset_peak_memory_stats()
    initial_gpu_mem = torch.cuda.memory_allocated() / 1e9
    initial_ram = psutil.Process().memory_info().rss / 1e9

    # 처리
    f0_batch, _ = service.gpu_backend.extract_f0_batch(audio_windows, sr)

    # 처리 후
    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    peak_ram = psutil.Process().memory_info().rss / 1e9

    print(f"GPU 메모리: {initial_gpu_mem:.2f}GB → {peak_gpu_mem:.2f}GB "
          f"(증가: {peak_gpu_mem - initial_gpu_mem:.2f}GB)")
    print(f"RAM: {initial_ram:.2f}GB → {peak_ram:.2f}GB "
          f"(증가: {peak_ram - initial_ram:.2f}GB)")

    return peak_gpu_mem - initial_gpu_mem
```

---

## 버전 호환성

| 패키지 | 최소 버전 | 권장 버전 | 테스트 버전 |
|--------|---------|---------|-----------|
| Python | 3.8 | 3.10+ | 3.11 |
| PyTorch | 1.12 | 2.0+ | 2.1 |
| torchcrepe | 0.0.12 | 0.0.12+ | 0.0.12 |
| librosa | 0.10 | 0.10+ | 0.10 |
| NumPy | 1.20 | 1.24+ | 1.24 |
| CUDA | 11.8 | 12.1+ | 12.1 |

---

## 참고 자료

### 관련 문서
- GPU_F0_EXTRACTION_GUIDE.md - 사용자 가이드
- VALIDATION_PHASE_*.md - 검증 결과

### 외부 참고
- [CREPE: A Convolutional Representation for Pitch Estimation](https://github.com/marl/crepe)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/)

### 디버깅 팁

```python
# API 호출 로깅
import logging
logging.basicConfig(level=logging.DEBUG)

# GPU 상태 확인
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# 성능 프로파일링
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

f0, _ = service.extract_f0(audio, sr)

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

**최종 업데이트**: 2026-01-10
**유지보수**: voice.man 개발팀
**지원**: voice.man@example.com

