---
id: SPEC-WHISPERX-001
version: "1.0.0"
status: "planned"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
priority: "HIGH"
title: "WhisperX 통합 파이프라인 시스템"
related_specs:
  - SPEC-PARALLEL-001
  - SPEC-VOICE-001
tags:
  - WhisperX
  - GPU
  - WAV2VEC2
  - Pyannote
  - 화자분리
  - STT
  - 타임스탬프정렬
lifecycle: "spec-anchored"
---

# SPEC-WHISPERX-001: WhisperX 통합 파이프라인 시스템

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-08 | 지니 | 초안 작성 - WhisperX 통합 파이프라인 요구사항 정의 |

## CONTEXT

### 프로젝트 배경
- **프로젝트명**: voice.man - 음성 녹취 기반 증거 분석 시스템
- **선행 SPEC**: SPEC-PARALLEL-001 (GPU 기반 병렬처리 최적화) Phase 1-2 완료
- **현재 상태**: faster-whisper 기반 GPU STT 구현 완료, 개별 컴포넌트 독립 실행
- **핵심 목표**: WhisperX 통합으로 STT + 정렬 + 화자분리 end-to-end GPU 파이프라인 구현
- **성능 목표**: 183개 m4a 파일 처리 시간 3분 -> 1.2분 (2.5배 추가 향상)

### 기술적 맥락
SPEC-PARALLEL-001에서 정의한 Phase 3를 본 SPEC에서 상세화합니다:
- **Phase 3 목표**: WhisperX 통합 파이프라인으로 50배 성능 향상 달성
- **핵심 기술**: WhisperX = Whisper + WAV2VEC2 Alignment + Pyannote Diarization
- **차별점**: 개별 컴포넌트가 아닌 GPU에서 통합 실행되는 end-to-end 파이프라인

### 시스템 리소스
- **CPU**: 20코어 (Cortex-X925 + A725 구성)
- **메모리**: 119GB (사용 가능 114GB)
- **GPU**: NVIDIA GB10
  - Compute Capability: 12.1
  - CUDA Version: 13.0
  - 현재 활용률: Phase 2에서 70% 달성 (목표: 90%)

### 현재 구현 상태 (SPEC-PARALLEL-001 완료 기준)
1. **faster-whisper 래퍼**: `src/voice_man/models/whisper_model.py` (완료)
2. **GPU 모니터링**: `src/voice_man/services/gpu_monitor_service.py` (완료)
3. **배치 처리**: `src/voice_man/services/batch_service.py` (완료)
4. **메모리 관리**: `src/voice_man/services/memory_service.py` (완료)
5. **화자 분리**: `src/voice_man/services/diarization_service.py` (모의 구현, 실제 pyannote 통합 필요)

### 해결해야 할 문제
1. **파이프라인 단절**: STT와 화자분리가 별도 프로세스로 실행되어 CPU-GPU 전환 오버헤드 발생
2. **타임스탬프 부정확**: Whisper 기본 타임스탬프는 word-level 정확도 부족
3. **화자분리 모의 구현**: 현재 `diarization_service.py`는 실제 pyannote가 아닌 모의 데이터 반환
4. **순차 처리**: 각 단계가 순차적으로 실행되어 GPU 유휴 시간 발생

---

## EARS 요구사항

### 1. Ubiquitous Requirements (U) - 전역 요구사항

#### U1: WhisperX 파이프라인 일관성
**요구사항**: 시스템은 **항상** WhisperX 파이프라인의 모든 단계(전사, 정렬, 화자분리)를 동일한 GPU 컨텍스트에서 실행해야 한다.

**세부사항**:
- 단일 CUDA 디바이스에서 모든 모델 실행
- 모델 간 데이터 전송 시 GPU 메모리 내 유지
- CPU-GPU 간 불필요한 데이터 복사 최소화

**WHY**: GPU 컨텍스트 전환은 성능 저하의 주요 원인이며, 메모리 복사 오버헤드를 유발한다.

**IMPACT**: 컨텍스트 분리 시 파이프라인 성능 30-40% 저하.

---

#### U2: Word-level 타임스탬프 정확도
**요구사항**: 시스템은 **항상** word-level 타임스탬프 정확도를 100ms 이내로 보장해야 한다.

**세부사항**:
- WAV2VEC2 alignment 모델 사용
- 각 단어의 시작/종료 시간 제공
- 타임스탬프 정확도 검증 메트릭 수집

**WHY**: 법적 증거 분석에서 정확한 발화 시점 식별은 핵심 요구사항이다.

**IMPACT**: 타임스탬프 부정확 시 증거 분석 신뢰도 저하 및 법적 효력 감소.

---

#### U3: 화자 ID 일관성 유지
**요구사항**: 시스템은 **항상** 동일 화자에게 일관된 ID를 할당해야 한다.

**세부사항**:
- 단일 오디오 파일 내 동일 화자 = 동일 ID
- 화자 ID 형식: `SPEAKER_00`, `SPEAKER_01`, ...
- 화자별 발화 통계 자동 생성

**WHY**: 화자 일관성 없이는 대화 흐름 분석 및 증거 추적이 불가능하다.

**IMPACT**: 화자 ID 불일치 시 증거 분석 무효화.

---

### 2. Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: Hugging Face 토큰 검증
**요구사항**: **WHEN** 파이프라인이 초기화될 때 **THEN** 시스템은 Hugging Face 토큰 유효성을 검증해야 한다.

**세부사항**:
- 환경 변수 `HF_TOKEN` 존재 확인
- pyannote/speaker-diarization-3.1 모델 접근 권한 확인
- 토큰 만료 또는 권한 부족 시 명확한 오류 메시지

**WHY**: pyannote 모델은 Hugging Face 인증이 필수이며, 토큰 없이는 모델 로드 실패.

**IMPACT**: 토큰 미검증 시 런타임에 인증 오류 발생 및 처리 중단.

---

#### E2: 오디오 포맷 자동 변환
**요구사항**: **WHEN** m4a/mp3/wav 파일이 입력될 때 **THEN** 시스템은 WhisperX 호환 포맷으로 자동 변환해야 한다.

**세부사항**:
- 지원 입력 포맷: m4a, mp3, wav, flac, ogg
- 변환 출력: 16kHz mono WAV (WhisperX 최적)
- 임시 변환 파일은 처리 후 자동 삭제
- 원본 파일 무결성 유지 (SPEC-VOICE-001 연계)

**WHY**: WhisperX는 특정 오디오 포맷에서 최적 성능을 발휘한다.

**IMPACT**: 포맷 미변환 시 처리 오류 또는 성능 저하 발생.

---

#### E3: 파이프라인 단계별 진행률 업데이트
**요구사항**: **WHEN** 각 파이프라인 단계가 완료될 때 **THEN** 시스템은 진행률을 실시간 업데이트해야 한다.

**세부사항**:
- 단계별 진행률: 전사 0-40%, 정렬 40-70%, 화자분리 70-100%
- 현재 처리 중인 파일명 표시
- 예상 완료 시간 (ETA) 계산

**WHY**: 대량 파일 처리 시 사용자에게 진행 상황 피드백 필수.

**IMPACT**: 진행률 부재 시 사용자 경험 저하 및 시스템 응답 불확실성.

---

#### E4: 화자 수 자동 감지 또는 수동 지정
**요구사항**: **WHEN** 오디오 처리가 시작될 때 **THEN** 시스템은 화자 수를 자동 감지하거나 사용자 지정값을 사용해야 한다.

**세부사항**:
- 기본 모드: 자동 화자 수 감지 (`num_speakers=None`)
- 수동 모드: 사용자 지정 화자 수 (`num_speakers=2`)
- 최소 화자 수: 1, 최대 화자 수: 10
- 화자 수 감지 신뢰도 로깅

**WHY**: 증거 분석에서는 대화 참여자 수가 사전에 알려진 경우가 많다.

**IMPACT**: 잘못된 화자 수는 화자 분리 정확도를 크게 저하시킨다.

---

### 3. State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: 한국어 alignment 모델 사용
**요구사항**: **IF** 언어가 한국어(ko)로 설정되면 **THEN** 시스템은 한국어 최적화 alignment 모델을 로드해야 한다.

**세부사항**:
- 한국어 alignment 모델: `jonatasgrosman/wav2vec2-large-xlsr-53-korean`
- 모델 자동 다운로드 및 캐싱
- 언어별 모델 매핑 테이블 유지

**WHY**: 언어별 음소 체계가 다르므로 언어 최적화 모델이 정확도를 향상시킨다.

**IMPACT**: 범용 모델 사용 시 한국어 타임스탬프 정확도 20-30% 저하.

---

#### S2: GPU 메모리 70% 초과 시 모델 순차 로딩
**요구사항**: **IF** GPU 메모리 사용률이 70%를 초과하면 **THEN** 시스템은 모델을 순차적으로 로드/언로드해야 한다.

**세부사항**:
- 전체 모델 동시 로딩: Whisper(3GB) + WAV2VEC2(1.2GB) + Pyannote(1GB) = 약 5.2GB
- 메모리 제약 시: 모델 순차 로드 (처리 속도 약간 저하)
- 모델 언로드 후 `torch.cuda.empty_cache()` 호출

**WHY**: GB10 GPU 메모리는 제한적이며, OOM 방지가 필수.

**IMPACT**: 순차 로딩 시 약 20% 성능 저하 발생하나 안정성 확보.

---

#### S3: 긴 오디오 파일 청크 분할 처리
**요구사항**: **IF** 오디오 길이가 30분을 초과하면 **THEN** 시스템은 10분 청크로 분할하여 처리해야 한다.

**세부사항**:
- 청크 길이: 10분 (600초)
- 청크 간 오버랩: 30초 (경계 단어 누락 방지)
- 청크 결과 병합 시 중복 구간 제거

**WHY**: 장시간 오디오는 메모리 사용량 급증 및 처리 시간 예측 불가.

**IMPACT**: 분할 없이 처리 시 메모리 부족 및 처리 실패 위험.

---

### 4. Feature Requirements (F) - 기능 요구사항

#### F1: WhisperX 통합 파이프라인 클래스
**요구사항**: 시스템은 `WhisperXPipeline` 클래스를 구현하여 end-to-end 처리를 제공해야 한다.

**세부사항**:
- 위치: `src/voice_man/models/whisperx_pipeline.py`
- 메서드:
  - `__init__(model_size, device, language)`: 모델 초기화
  - `process(audio_path, num_speakers)`: 전체 파이프라인 실행
  - `transcribe(audio)`: STT 단계
  - `align(segments, audio)`: 타임스탬프 정렬 단계
  - `diarize(audio, segments)`: 화자 분리 단계
- 반환: 통합 결과 (텍스트, 세그먼트, 화자 정보)

**WHY**: 단일 인터페이스로 복잡한 파이프라인을 추상화하여 사용성 향상.

**IMPACT**: 파이프라인 미구현 시 각 단계 수동 호출 필요로 복잡도 증가.

---

#### F2: WAV2VEC2 기반 Word-level Alignment
**요구사항**: 시스템은 WAV2VEC2 모델을 사용하여 word-level 타임스탬프 정렬을 수행해야 한다.

**세부사항**:
- 라이브러리: `whisperx>=3.1.5`
- 한국어 모델: `jonatasgrosman/wav2vec2-large-xlsr-53-korean`
- 출력 포맷:
  ```python
  {
    "word": "안녕하세요",
    "start": 1.23,
    "end": 1.89,
    "score": 0.95  # alignment 신뢰도
  }
  ```

**WHY**: Whisper 기본 타임스탬프는 segment-level이며, 법적 증거 분석에는 word-level 필요.

**IMPACT**: Word-level 정렬 없이는 정확한 발화 시점 분석 불가능.

---

#### F3: Pyannote GPU 병렬 화자 분리
**요구사항**: 시스템은 `pyannote-audio`를 GPU에서 실행하여 화자 분리를 수행해야 한다.

**세부사항**:
- 라이브러리: `pyannote-audio>=3.1.1`
- 모델: `pyannote/speaker-diarization-3.1`
- GPU 가속: `device="cuda"`
- 출력 포맷:
  ```python
  [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5},
    {"speaker": "SPEAKER_01", "start": 2.5, "end": 5.0},
    ...
  ]
  ```

**WHY**: 화자 분리는 계산 집약적이며, GPU 가속으로 10배 이상 속도 향상 가능.

**IMPACT**: CPU 실행 시 화자 분리가 전체 파이프라인 병목 (처리 시간의 60% 이상).

---

#### F4: 화자별 발화 통계 및 분석
**요구사항**: 시스템은 화자별 발화 통계를 자동으로 생성해야 한다.

**세부사항**:
- 통계 항목:
  - 화자별 총 발화 시간 (초)
  - 화자별 발화 비율 (%)
  - 화자별 발화 횟수 (턴 수)
  - 화자 교대 빈도 (turns per minute)
- 출력 포맷: JSON 및 분석 리포트

**WHY**: 증거 분석에서 발화 패턴 분석은 대화 역학 이해에 필수적이다.

**IMPACT**: 통계 미제공 시 수동 분석 필요로 시간 소요 증가.

---

#### F5: 기존 서비스 인터페이스 호환
**요구사항**: 시스템은 기존 `diarization_service.py` 인터페이스와 호환되어야 한다.

**세부사항**:
- 기존 메서드 시그니처 유지:
  - `diarize_speakers(audio_path, num_speakers) -> DiarizationResult`
  - `merge_with_transcript(stt_segments, diarization_result)`
- 내부 구현만 WhisperX로 교체
- 기존 테스트 케이스 통과

**WHY**: 기존 코드베이스와의 호환성 유지로 마이그레이션 위험 최소화.

**IMPACT**: 인터페이스 변경 시 의존 코드 전면 수정 필요.

---

### 5. Unwanted Requirements (N) - 금지 요구사항

#### N1: 모델 동시 로딩으로 인한 OOM 금지
**요구사항**: 시스템은 **절대** 모든 모델을 동시에 로드하여 GPU OOM을 유발하지 않아야 한다.

**세부사항**:
- 모델 로딩 전 가용 GPU 메모리 확인
- 메모리 부족 시 순차 로딩 모드로 자동 전환
- OOM 발생 시 즉시 모델 언로드 및 재시도

**WHY**: OOM은 전체 프로세스를 중단시키며, 데이터 손실 위험이 있다.

**IMPACT**: OOM 발생 시 시스템 크래시 및 처리 중인 데이터 손실.

---

#### N2: Hugging Face 토큰 하드코딩 금지
**요구사항**: 시스템은 **절대** Hugging Face 토큰을 소스 코드에 하드코딩하지 않아야 한다.

**세부사항**:
- 토큰은 환경 변수 `HF_TOKEN`으로만 제공
- 설정 파일에도 토큰 저장 금지
- 토큰 로깅 금지 (마스킹 처리)

**WHY**: 토큰 노출은 보안 위반이며, 계정 악용 위험이 있다.

**IMPACT**: 토큰 노출 시 Hugging Face 계정 권한 탈취 가능.

---

#### N3: 임시 파일 미삭제 금지
**요구사항**: 시스템은 **절대** 임시 변환 파일을 처리 완료 후 삭제하지 않고 방치하지 않아야 한다.

**세부사항**:
- 오디오 변환 임시 파일: 처리 완료 후 즉시 삭제
- 중간 결과 파일: 최종 저장 후 삭제
- 비정상 종료 시에도 cleanup 보장 (try-finally 또는 context manager)

**WHY**: 임시 파일 누적은 디스크 공간 부족 및 보안 위험을 유발한다.

**IMPACT**: 대량 처리 시 수십 GB 임시 파일 누적 가능.

---

## TECHNICAL SPECIFICATIONS

### 라이브러리 버전

#### 필수 라이브러리 (신규 추가)
```python
# WhisperX 통합 파이프라인
whisperx>=3.1.5

# 화자 분리
pyannote-audio>=3.1.1

# WAV2VEC2 Alignment (whisperx 의존성으로 자동 설치)
transformers>=4.36.0

# Hugging Face Hub
huggingface-hub>=0.20.0
```

#### 기존 라이브러리 (SPEC-PARALLEL-001)
```python
# STT
faster-whisper>=1.0.3

# GPU 가속
torch>=2.5.0+cu121
torchaudio>=2.5.0+cu121

# 시스템 모니터링
psutil>=6.1.0
nvidia-ml-py>=12.560.30

# 오디오 처리
soundfile>=0.12.1
librosa>=0.10.2
ffmpeg-python>=0.2.0
```

---

### 성능 목표

#### 처리 시간 목표
- **입력**: 183개 m4a 파일 (평균 5분/파일)
- **Phase 2 기준**: 3분 (SPEC-PARALLEL-001 완료 시점)
- **Phase 3 목표**: 1.2분 (본 SPEC 목표)
- **성능 향상**: 2.5배 추가 향상 (Phase 2 대비)

#### GPU 활용률 목표
- **Phase 2 기준**: 70%
- **Phase 3 목표**: 90%
- **병렬 처리**: STT + Alignment + Diarization 파이프라인 최적화

#### 정확도 목표
- **타임스탬프 정확도**: word-level 100ms 이내
- **화자 분리 정확도**: DER (Diarization Error Rate) 10% 이하
- **STT 정확도**: WER 변화 < 1% (기존 대비)

---

### 파일 구조

```
voice.man/
├── src/
│   └── voice_man/
│       ├── models/
│       │   ├── whisper_model.py         # 기존 (유지)
│       │   └── whisperx_pipeline.py     # 신규 - WhisperX 통합 파이프라인
│       ├── services/
│       │   ├── diarization_service.py   # 수정 - pyannote 실제 통합
│       │   ├── alignment_service.py     # 신규 - WAV2VEC2 정렬 서비스
│       │   ├── audio_converter_service.py # 신규 - 오디오 포맷 변환
│       │   └── whisperx_service.py      # 신규 - WhisperX 서비스 레이어
│       └── config/
│           └── whisperx_config.py       # 신규 - WhisperX 설정
├── tests/
│   ├── test_whisperx_pipeline.py        # 신규 - 파이프라인 테스트
│   ├── test_alignment_service.py        # 신규 - 정렬 테스트
│   └── test_diarization_integration.py  # 신규 - 화자분리 통합 테스트
└── .moai/
    └── specs/
        └── SPEC-WHISPERX-001/           # 본 SPEC 문서
```

---

### 환경 변수

```bash
# Hugging Face 인증 (필수)
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# WhisperX 설정 (선택)
export WHISPERX_MODEL_SIZE="large-v3"
export WHISPERX_LANGUAGE="ko"
export WHISPERX_DEVICE="cuda"
export WHISPERX_COMPUTE_TYPE="float16"

# GPU 메모리 관리 (선택)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0"
```

---

## CONSTRAINTS

### 하드웨어 제약
- **GPU 메모리**: 전체 모델 동시 로드 시 약 5.2GB 필요
- **시스템 RAM**: 모델 캐싱 시 추가 8GB 필요
- **디스크**: 모델 다운로드 시 약 6GB 필요 (초기 1회)

### 소프트웨어 제약
- **Python 버전**: 3.10 이상 (whisperx 요구사항)
- **CUDA 호환성**: PyTorch 2.5.0은 CUDA 12.1-12.6 지원
- **Hugging Face**: pyannote 모델 사용 동의 필요

### 비즈니스 제약
- **처리 시간**: 1.5분 이내 (목표 1.2분)
- **정확도 유지**: 기존 대비 WER 변화 < 1%
- **원본 파일 보존**: 모든 처리는 비파괴적 (SPEC-VOICE-001 연계)

---

## ACCEPTANCE CRITERIA

### 기능적 요구사항
1. WhisperX 파이프라인 초기화 성공 (모든 모델 로드)
2. 183개 파일 1.5분 이내 처리 완료
3. 모든 파일에 word-level 타임스탬프 제공
4. 모든 파일에 화자 ID 할당
5. 화자별 발화 통계 자동 생성
6. 기존 diarization_service 인터페이스 호환

### 비기능적 요구사항
1. GPU 활용률 85% 이상
2. GPU 메모리 사용량 95% 미만 (OOM 방지)
3. 타임스탬프 정확도 100ms 이내
4. DER (Diarization Error Rate) 15% 이하
5. 테스트 커버리지 85% 이상
6. TRUST 5 품질 게이트 통과

---

## DEPENDENCIES

### 기술적 의존성
- **SPEC-PARALLEL-001**: GPU 기반 병렬처리 최적화 (Phase 2 완료 전제)
- **SPEC-VOICE-001**: 음성 분석 기본 시스템 (원본 무결성 요구사항)

### 외부 의존성
- **Hugging Face Hub**: pyannote, WAV2VEC2 모델 다운로드
- **Hugging Face Token**: 환경 변수로 제공 필수
- **인터넷 연결**: 초기 모델 다운로드 (약 6GB)

### 라이브러리 의존성
- whisperx>=3.1.5
- pyannote-audio>=3.1.1
- transformers>=4.36.0
- huggingface-hub>=0.20.0

---

## RISKS AND MITIGATION

### 고위험
1. **WhisperX 라이브러리 호환성 문제**
   - **확률**: 30%
   - **영향**: 파이프라인 통합 불가
   - **완화 전략**: Docker 컨테이너 환경 격리, 버전 고정

2. **Hugging Face 모델 접근 권한**
   - **확률**: 20%
   - **영향**: pyannote 모델 사용 불가
   - **완화 전략**: 토큰 사전 검증, 모델 사용 동의 체크리스트

### 중위험
1. **GPU 메모리 부족**
   - **확률**: 40%
   - **영향**: 동시 모델 로딩 불가
   - **완화 전략**: 순차 로딩 모드 구현, 동적 메모리 관리

2. **한국어 alignment 모델 정확도**
   - **확률**: 25%
   - **영향**: 타임스탬프 정확도 저하
   - **완화 전략**: 다중 모델 평가, fallback 모델 준비

### 저위험
1. **오디오 포맷 변환 실패**
   - **확률**: 10%
   - **영향**: 특정 파일 처리 실패
   - **완화 전략**: 포맷별 변환 로직, 재시도 메커니즘

---

## TRACEABILITY

### 관련 문서
- **SPEC-PARALLEL-001**: GPU 기반 병렬처리 최적화 (Phase 3 정의)
- **SPEC-VOICE-001**: 음성 분석 기본 시스템 (원본 무결성)
- **CLAUDE.md**: 프로젝트 실행 지침

### 구현 파일
- `src/voice_man/models/whisperx_pipeline.py` (신규)
- `src/voice_man/services/alignment_service.py` (신규)
- `src/voice_man/services/audio_converter_service.py` (신규)
- `src/voice_man/services/whisperx_service.py` (신규)
- `src/voice_man/services/diarization_service.py` (수정)

---

## REFERENCES

### 기술 문서
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [WhisperX Paper](https://arxiv.org/abs/2303.00747)
- [pyannote-audio Documentation](https://github.com/pyannote/pyannote-audio)
- [WAV2VEC2 Korean Model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-korean)
- [Hugging Face Token Guide](https://huggingface.co/docs/hub/security-tokens)

### 관련 표준
- **OWASP**: 보안 가이드라인 (토큰 관리)
- **PEP 8**: Python 코드 스타일
- **TRUST 5**: MoAI-ADK 품질 프레임워크

---

**문서 끝**
