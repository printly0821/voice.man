# SPEC-GPUAUDIO-001: GPU 가속 오디오 피처 추출

```yaml
id: SPEC-GPUAUDIO-001
version: 1.2.0
status: in_progress
created: 2026-01-10
updated: 2026-01-10
author: 지니
priority: high
lifecycle: spec-anchored
tags: [gpu, audio, performance, torchcrepe, nnAudio, cuda]
related_specs: [SPEC-FORENSIC-001]
```

---

## HISTORY

| Version | Date       | Author | Description                          |
|---------|------------|--------|--------------------------------------|
| 1.2.0   | 2026-01-10 | 지니   | Phase 2 완료: NNAudioProcessor 기반 GPU 스펙트로그램 구현 |
| 1.1.0   | 2026-01-10 | 지니   | Phase 1 완료: TorchCrepe 기반 GPU F0 추출 구현 |
| 1.0.0   | 2026-01-10 | 지니   | 초기 SPEC 작성                        |

---

## 1. Executive Summary

### 1.1 배경

현재 `AudioFeatureService`는 librosa 기반 CPU 처리로 인해 심각한 성능 병목이 발생하고 있다. 특히 `detect_emotional_escalation()` 메서드에서 2,520개 윈도우에 대한 F0 추출 시 약 500초가 소요되며, 이는 실시간 포렌식 분석에 부적합하다.

### 1.2 목표

- **librosa.pyin()을 torchcrepe로 교체**: GPU 배치 처리로 F0 추출 160x 성능 향상
- **librosa.stft/mel을 nnAudio로 교체**: GPU 스펙트로그램 생성으로 63-100x 성능 향상
- **전체 분석 시간**: 500초 → 5초 (100x 개선)

### 1.3 대상 환경

- GPU: NVIDIA GB10 (Blackwell Architecture, Compute Capability 12.1)
- CPU: ARM 20-Core
- RAM: 119GB
- CUDA: 12.x 호환

---

## 2. EARS Requirements

### 2.1 Ubiquitous Requirements (시스템 전역 요구사항)

> 시스템은 **항상** [동작]해야 한다

| ID | Requirement |
|----|-------------|
| REQ-U-001 | 시스템은 **항상** GPU 가용성을 확인하고 가용 시 CUDA 가속을 사용해야 한다 |
| REQ-U-002 | 시스템은 **항상** GPU 메모리 사용량을 모니터링하고 OOM 방지를 위해 배치 크기를 동적 조정해야 한다 |
| REQ-U-003 | 시스템은 **항상** 기존 librosa API와 호환되는 출력 형식을 유지해야 한다 |
| REQ-U-004 | 시스템은 **항상** CPU fallback을 제공하여 GPU 미사용 환경에서도 동작해야 한다 |

### 2.2 Event-Driven Requirements (이벤트 기반 요구사항)

> **WHEN** [이벤트] **THEN** [동작]

| ID | Trigger Event | Action |
|----|---------------|--------|
| REQ-E-001 | **WHEN** `extract_f0()` 호출 시 **THEN** torchcrepe를 사용하여 GPU 배치 처리로 F0 추출 |
| REQ-E-002 | **WHEN** 스펙트로그램 생성 요청 시 **THEN** nnAudio STFT/Mel 레이어를 통해 GPU 가속 처리 |
| REQ-E-003 | **WHEN** GPU 메모리 부족 발생 시 **THEN** 배치 크기를 50%로 감소시키고 재시도 |
| REQ-E-004 | **WHEN** 오디오 배열이 512 샘플 미만일 시 **THEN** 패딩을 적용하고 결과에서 trim |
| REQ-E-005 | **WHEN** `detect_emotional_escalation()` 호출 시 **THEN** 전체 윈도우에 대해 단일 배치 F0 추출 수행 |

### 2.3 State-Driven Requirements (상태 기반 요구사항)

> **IF** [조건] **THEN** [동작]

| ID | Condition | Action |
|----|-----------|--------|
| REQ-S-001 | **IF** CUDA 디바이스가 가용하면 **THEN** 모든 연산을 GPU에서 수행 |
| REQ-S-002 | **IF** 입력 오디오가 1시간 초과하면 **THEN** 청크 단위로 분할하여 순차 처리 |
| REQ-S-003 | **IF** GPU 메모리가 2GB 미만이면 **THEN** 배치 크기를 최소값(128)으로 제한 |
| REQ-S-004 | **IF** torchcrepe 모델이 로드되지 않았으면 **THEN** 첫 호출 시 lazy loading 수행 |
| REQ-S-005 | **IF** 다중 GPU 환경이면 **THEN** 첫 번째 가용 GPU 사용 (추후 분산 처리 지원 예정) |

### 2.4 Optional Requirements (선택적 요구사항)

> **가능하면** [동작] 제공

| ID | Feature | Condition |
|----|---------|-----------|
| REQ-O-001 | **가능하면** TensorRT 양자화를 통해 추가 50% 성능 향상 제공 | GPU 컴퓨트 >= 7.0 |
| REQ-O-002 | **가능하면** Mixed Precision (FP16) 연산을 통해 메모리 효율성 향상 | Ampere+ GPU |
| REQ-O-003 | **가능하면** 실시간 스트리밍 분석 모드 제공 | 오디오 스트림 입력 시 |
| REQ-O-004 | **가능하면** 다중 GPU 병렬 처리 지원 | multi-GPU 환경 |

### 2.5 Unwanted Behavior Requirements (비허용 동작)

> 시스템은 [동작]**하지 않아야 한다**

| ID | Prohibited Behavior | Reason |
|----|---------------------|--------|
| REQ-N-001 | 시스템은 GPU 메모리 전체를 사전 할당**하지 않아야 한다** | 다른 프로세스와의 공유 필요 |
| REQ-N-002 | 시스템은 CPU-GPU 간 불필요한 데이터 전송을 수행**하지 않아야 한다** | 전송 오버헤드 최소화 |
| REQ-N-003 | 시스템은 기존 AudioFeatureService API 시그니처를 변경**하지 않아야 한다** | 하위 호환성 유지 |
| REQ-N-004 | 시스템은 librosa 의존성을 완전히 제거**하지 않아야 한다** | CPU fallback 및 기타 기능 유지 |
| REQ-N-005 | 시스템은 동기화 없이 GPU 텐서를 반환**하지 않아야 한다** | 결과 일관성 보장 |

---

## 3. Technical Specifications

### 3.1 아키텍처 개요

```
AudioFeatureService (기존 API 유지)
    |
    +-- GPUAudioBackend (새로운 GPU 백엔드)
    |       |
    |       +-- TorchCrepeExtractor (F0 추출)
    |       |       +-- CREPE 모델 (full/tiny 선택)
    |       |       +-- 배치 처리 파이프라인
    |       |
    |       +-- NNAudioProcessor (스펙트로그램)
    |       |       +-- STFT Layer
    |       |       +-- MelSpectrogram Layer
    |       |       +-- CQT Layer (선택적)
    |       |
    |       +-- MemoryManager (GPU 메모리 관리)
    |               +-- 배치 크기 동적 조정
    |               +-- OOM 복구 전략
    |
    +-- CPUFallbackBackend (기존 librosa 기반)
```

### 3.2 핵심 컴포넌트

#### 3.2.1 TorchCrepeExtractor

```python
class TorchCrepeExtractor:
    """GPU 가속 F0 추출기 (torchcrepe 기반)"""

    def __init__(
        self,
        model: str = "full",  # "full" | "tiny"
        device: str = "cuda",
        batch_size: int = 2048,
    ): ...

    def extract_f0_batch(
        self,
        audio_windows: np.ndarray,  # Shape: (num_windows, window_samples)
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 F0 추출

        Returns:
            f0: Shape (num_windows,) - 각 윈도우의 F0 값
            confidence: Shape (num_windows,) - 각 F0의 신뢰도
        """
```

#### 3.2.2 NNAudioProcessor

```python
class NNAudioProcessor:
    """GPU 가속 스펙트로그램 생성기 (nnAudio 기반)"""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        device: str = "cuda",
    ): ...

    def compute_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """GPU STFT 계산"""

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """GPU Mel 스펙트로그램 계산"""
```

### 3.3 성능 목표

| Metric | Current (librosa) | Target (GPU) | Improvement |
|--------|-------------------|--------------|-------------|
| F0 추출 (42분 오디오) | ~500초 | ~3초 | 160x |
| Mel 스펙트로그램 | ~8초 | ~0.13초 | 63x |
| 전체 분석 파이프라인 | ~500초 | ~5초 | 100x |
| 메모리 사용량 | 2GB CPU | 4GB GPU | 허용 범위 |

### 3.4 의존성

```toml
# pyproject.toml 추가 의존성
[project.optional-dependencies]
gpu = [
    "torchcrepe>=0.0.22",
    "nnAudio>=0.3.2",
    "torch>=2.0.0",
]
```

---

## 4. Constraints

### 4.1 기술적 제약

| ID | Constraint | Impact |
|----|------------|--------|
| C-001 | CUDA 12.x 호환 필수 | Blackwell 아키텍처 지원 |
| C-002 | PyTorch 2.0+ 필수 | torchcrepe 및 nnAudio 의존성 |
| C-003 | GPU 메모리 최소 4GB | CREPE full 모델 로딩 |
| C-004 | ARM64 호환 휠 필요 | Jetson/ARM 플랫폼 지원 |

### 4.2 비기능적 제약

| ID | Constraint | Threshold |
|----|------------|-----------|
| NF-001 | 첫 호출 레이턴시 (모델 로딩) | < 5초 |
| NF-002 | 이후 호출 레이턴시 | < 0.1초/초 오디오 |
| NF-003 | GPU 메모리 피크 사용량 | < 6GB |
| NF-004 | CPU fallback 성능 저하 | 기존과 동일 |

---

## 5. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU 메모리 부족 | Medium | High | 동적 배치 크기 조정, 청크 처리 |
| CUDA 버전 비호환 | Low | High | 런타임 버전 체크, 명확한 에러 메시지 |
| torchcrepe 정확도 차이 | Low | Medium | librosa.pyin과 교차 검증 테스트 |
| ARM64 휠 미지원 | Medium | Medium | 소스 빌드 옵션 제공 |

---

## 6. References

### 6.1 기술 문서

- [torchcrepe GitHub](https://github.com/maxrmorrison/torchcrepe) - GPU F0 추출 라이브러리
- [nnAudio GitHub](https://github.com/KinWaiCheuk/nnAudio) - GPU 스펙트로그램 라이브러리
- CREPE: A Convolutional Representation for Pitch Estimation (ICASSP 2018)

### 6.2 관련 연구 (2025)

- CREPE Edge Optimization, ETRI Journal 2025
- Real-time Speech Emotion Recognition with GPU Acceleration, Nature Scientific Reports 2025

### 6.3 관련 SPEC

- [SPEC-FORENSIC-001](../SPEC-FORENSIC-001/spec.md) - 음성 포렌식 분석 시스템

---

## 7. Glossary

| Term | Definition |
|------|------------|
| F0 | Fundamental Frequency, 기본 주파수 (음높이) |
| CREPE | Convolutional Representation for Pitch Estimation |
| pyin | Probabilistic YIN, librosa의 F0 추출 알고리즘 |
| STFT | Short-Time Fourier Transform |
| Mel | Mel-frequency scale, 인간 청각 특성 반영 주파수 스케일 |
| OOM | Out of Memory |
| FP16 | Half-precision floating point (16-bit) |

---

## 8. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | 지니 | 2026-01-10 | |
| Reviewer | | | |
| Approver | | | |
