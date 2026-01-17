# VoiceMan 포렌식 파이프라인 아키텍처

## 1. 개요

VoiceMan 포렌식 파이프라인은 오디오 파일의 음성을 텍스트로 변환하고, 범죄 패턴을 분석하여 포렌식 보고서를 생성하는 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VoiceMan Forensic Pipeline                          │
│                                                                             │
│  Input: 오디오 파일 (m4a, wav, mp3, etc.)                                   │
│  Output: 포렌식 분석 보고서 (JSON, HTML, PDF)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 파이프라인 스테이지

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Stage 1 │ -> │  Stage 2 │ -> │  Stage 3 │ -> │  Stage 4 │ -> │  Stage 5 │
│  Audio   │    │    STT   │    │  Diariz  │    │ Forensic │    │  Report  │
│  Input   │    │          │    │   & NER  │    │ Analysis │    │  Generate│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Audio    │    │ Whisper  │    │ pyannote │    │ Crime    │    │ JSON     │
│ Files    │    │ large-v3 │    │  3.3     │    │ Tagging  │    │ HTML     │
│          │    │          │    │          │    │ Emotion  │    │ PDF      │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Stage 1: Audio Input
- 입력: 오디오 파일 (m4a, wav, mp3, flac, ogg)
- 처리: 파일 검증, 형식 변환 (필요시)
- 출력: 표준화된 오디오 파일
- **서비스**: `AudioConverterService`

### Stage 2: Speech-to-Text (STT)
- 입력: 오디오 파일
- 처리: WhisperX (large-v3 모델)
  - 음성 인식 (ASR)
  - 텍스트 정렬 (Word-level timestamps)
- 출력: 트랜스크립션 (텍스트 + 타임스탬프)
- **서비스**: `WhisperXService`, `OptimizedWhisperXPipeline`
- **GPU 최적화**:
  - `DynamicBatchProcessor`: 배치 크기 4-32 동적 조정
  - `MultiGPUOrchestrator`: 다중 GPU 로드 밸런싱
  - `TranscriptionCache`: L1/L2 캐싱 (100MB + 디스크)
  - `RobustPipeline`: 재시도 + 서킷브레이커

### Stage 3: Diarization & NER
- 입력: 트랜스크립션
- 처리:
  - 화자 분리 (Speaker Diarization): pyannote.audio 3.3
  - 개체명 인식 (NER): KoBERT
- 출력: 화자별 텍스트 + 개체명 정보
- **서비스**: `DiarizationService`, `KoBERTModel`

### Stage 4: Forensic Analysis
- 입력: 화자별 텍스트 + 개체명
- 처리:
  - 범죄 태깅 (Crime Tagging): 키워드 매칭
  - 가스라이팅 감지 (Gaslighting Detection): 패턴 분석
  - 감정 분석 (Emotion Analysis): 규칙 기반
  - 스트레스 분석 (Stress Analysis): 음성 특징
  - 문맥 분석 (Context Analysis): 관계 파악
- 출력: 포렌식 점수 + 패턴 정보
- **서비스**:
  - `CrimeTaggingService`
  - `GaslightingService`
  - `EmotionAnalysisService`
  - `StressAnalysisService`
  - `ContextAnalysisService`
  - `ForensicScoringService`

### Stage 5: Report Generation
- 입력: 포렌식 분석 결과
- 처리:
  - JSON 데이터 생성
  - HTML 보고서 렌더링
  - PDF 변환
- 출력: 포렌식 보고서
- **서비스**:
  - `ComprehensiveReportService`
  - `ReportService`
  - `PDFService`
  - `ChartService`

## 3. 오케스트레이션 패턴

### 3.1 Sequential Pipeline (순차적)
```
File 1 -> STT -> Diariz -> Forensic -> Report
File 2 -> STT -> Diariz -> Forensic -> Report
File 3 -> STT -> Diariz -> Forensic -> Report
```
- 간단한 구현
- 전체 처리 시간: Σ(각 파일 처리 시간)
- **서비스**: `AnalysisPipelineService`

### 3.2 Producer-Consumer Pipeline (병렬)
```
       STT (Producer)          Forensic (Consumer)
          ┌───┐                    ┌───┐
File 1 -->│ Q │-------async-------->│   │--> Report 1
File 2 -->│ u │-------async-------->│ C │--> Report 2
File 3 -->│ e │-------async-------->│ o │--> Report 3
          └───┘                    └───┘
          (Queue: max 5)            (Consumer)
```
- STT와 Forensic 병렬 실행
- Backpressure로 큐 오버플로우 방지
- **서비스**: `PipelineOrchestrator`
- **효과**: 50% 이상 처리 시간 단축 (PR-007)

### 3.3 Dynamic Batch Processing (GPU 최적화)
```
        GPU 0              GPU 1
    ┌─────────┐        ┌─────────┐
    │ Batch 1 │        │ Batch 2 │
    │(4-8 files)       │(4-8 files)
    └─────────┘        └─────────┘
         │                  │
         └───────┬──────────┘
                 ▼
         Forensic Analysis
                 │
                 ▼
            Report Gen
```
- 다중 GPU 활용
- 동적 배치 크기 (4-32)
- **서비스**: `DynamicBatchProcessor`, `MultiGPUOrchestrator`

## 4. GPU 최적화 컴포넌트

### 4.1 DynamicBatchProcessor
```python
class DynamicBatchProcessor:
    """동적 배치 크기 조정"""

    # 배치 크기 범위: 4-32
    MIN_BATCH_SIZE = 4
    MAX_BATCH_SIZE = 32

    # GPU 메모리 기반 조정
    # - 119GB GPU: batch_size = 32
    # - 32GB GPU:  batch_size = 16
    # - 8GB GPU:   batch_size = 4
```

### 4.2 MultiGPUOrchestrator
```python
class MultiGPUOrchestrator:
    """다중 GPU 로드 밸런싱"""

    # 로드 밸런싱 전략
    STRATEGIES = ["round_robin", "least_loaded", "affinity"]

    # GPU 할당
    # - STT: GPU 0
    # - Diarization: GPU 1
    # - Forensic: CPU (현재)
```

### 4.3 TranscriptionCache
```python
class TranscriptionCache:
    """트랜스크립션 캐싱 (L1 + L2)"""

    # L1 캐시: 100MB (메모리)
    # L2 캐시: 무제한 (디스크: /tmp/whisperx_cache)

    # 캐시 키: file_hash + model_size + compute_type
    # 히트율 목표: 30% 이상 (중복 파일 처리 시)
```

### 4.4 RobustPipeline
```python
class RobustPipeline:
    """오류 내성 파이프라인"""

    # 재시도 설정
    MAX_RETRIES = 2

    # 서킷브레이커
    CIRCUIT_BREAKER_THRESHOLD = 5  # 연속 실패 5회 시 개방

    # 폴백
    # GPU 실패 -> CPU 재시도
    # large-v3 실패 -> medium 재시도
```

## 5. 성능 요구사항

### 5.1 처리량 (Throughput)
| 단계 | 목표 | 현재 (CPU) | 현재 (GPU) |
|------|------|-----------|-----------|
| STT (단일) | 0.5x 실시간 | 1.19x | 미측정 |
| STT (배치) | 0.3x 실시간 | - | - |
| 전체 파이프라인 | 0.8x 실시간 | - | - |

### 5.2 처리 시간 (Processing Time)
| 파일 길이 | 목표 | CPU (whisper-medium) |
|----------|------|---------------------|
| 1분 | 30초 | ~72초 |
| 5분 | 150초 | ~360초 |
| 10분 | 300초 | ~720초 |

### 5.3 정확도 (Accuracy)
| 단계 | 목표 | 현재 |
|------|------|------|
| STT WER | <10% | - |
| 화자 분리 | >90% | - |
| 범죄 태깅 | >85% | - |

## 6. 데이터 플로우

### 6.1 입력 데이터
```
audio_file.m4a (4.0s - 14.6s)
  - 샘플레이트: 44.1kHz 또는 16kHz
  - 채널: 모노 또는 스테레오
  - 코덱: AAC, MP3, FLAC, OGG
```

### 6.2 중간 데이터
```
{
  "transcription": {
    "text": "전체 텍스트",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "안녕하세요",
        "speaker": "SPEAKER_00"
      }
    ],
    "word_segments": [...]
  },
  "diarization": {
    "speakers": [
      {"id": "SPEAKER_00", "gender": "male", "segments": [...]}
    ]
  }
}
```

### 6.3 출력 데이터
```
{
  "forensic_score": 85.5,
  "crime_tags": [
    {"type": "협박", "confidence": 0.92, "evidence": [...]}
  ],
  "emotions": [
    {"emotion": "angry", "intensity": 0.8, "timeline": [...]}
  ],
  "report_url": "/results/forensic_result_20260113.html"
}
```

## 7. 현재 구현 상태

### 7.1 완료된 컴포넌트
- ✅ `WhisperXService`: Whisper large-v3 STT
- ✅ `DiarizationService`: pyannote.audio 3.3
- ✅ `KoBERTModel`: NER
- ✅ `CrimeTaggingService`: 범죄 태깅
- ✅ `GaslightingService`: 가스라이팅 감지
- ✅ `EmotionAnalysisService`: 감정 분석
- ✅ `ForensicScoringService`: 포렌식 점수
- ✅ `AnalysisPipelineService`: 순차적 파이프라인
- ✅ `PipelineOrchestrator`: Producer-Consumer 패턴

### 7.2 GPU 최적화 컴포넌트
- ✅ `DynamicBatchProcessor`: 배치 크기 4-32 동적 조정
- ✅ `MultiGPUOrchestrator`: 다중 GPU 관리
- ✅ `TranscriptionCache`: L1/L2 캐싱
- ✅ `RobustPipeline`: 오류 내성
- ✅ `OptimizedWhisperXPipeline`: GPU 최적화 STT

### 7.3 진행 중인 작업
- ⚠️ PyTorch CUDA 호환성 (PyTorch 2.11 nightly)
- ⚠️ pyannote.audio weights_only 문제
- ✅ CPU 벤치마크 완료 (1.19x 실시간)

## 8. 다음 단계

### 8.1 GPU 최적화 완료
1. PyTorch 안정 버전 사용 (2.5.1 또는 2.8.x)
2. pyannote.audio VAD 비활성화 또는 패치
3. GPU 벤치마크 실행
4. 성능 비교 분석

### 8.2 파이프라인 통합
1. GPU 최적화 컴포넌트를 `PipelineOrchestrator`에 통합
2. Producer-Consumer 패턴에 GPU 배치 처리 적용
3. Backpressure 최적화

### 8.3 포렌식 분석 고도화
1. NER 결과를 Forensic 분석에 통합
2. 멀티모달 분석 (오디오 + 텍스트)
3. 시간 기반 분석 (타임라인)

## 9. 참고 문서
- SPEC-PERFOPT-001: GPU 최적화
- SPEC-PARALLEL-001: 병렬 처리
- PR-007: 성능 목표
