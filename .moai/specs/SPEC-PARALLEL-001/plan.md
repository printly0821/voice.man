# SPEC-PARALLEL-001 구현 계획

## 개요

**목표**: 183개 m4a 파일 처리 시간을 60분 → 1.2분으로 단축 (50배 성능 향상)

**전략**: 3단계 점진적 최적화 접근
- Phase 1: CPU 병렬처리 최적화 (4배 향상)
- Phase 2: GPU 활성화 (20배 향상)
- Phase 3: 완전 파이프라인 통합 (50배 향상)

---

## Phase 1: 즉시 최적화 (4배 향상)

### 목표
- **처리 시간**: 60분 → 15분
- **변경 범위**: 기존 코드 최소 수정
- **위험도**: 낮음

### 구현 작업

#### 1.1 Worker 수 증가
**파일**: `scripts/process_audio_files.py`

**현재 코드**:
```python
max_workers = 4
```

**변경 코드**:
```python
import multiprocessing
max_workers = min(multiprocessing.cpu_count() - 2, 18)  # 20코어 중 18개 사용
```

**WHY**: CPU 코어를 최대한 활용하여 병렬 처리 성능 향상.

---

#### 1.2 배치 크기 증가
**파일**: `src/voice_analysis/services/batch_service.py`

**현재 코드**:
```python
batch_size = 5
```

**변경 코드**:
```python
batch_size = 15  # 메모리 여유 고려한 최적 배치 크기
```

**WHY**: 더 많은 파일을 동시 처리하여 I/O 오버헤드 감소.

---

#### 1.3 메모리 임계값 조정
**파일**: `src/voice_analysis/services/memory_service.py`

**현재 코드**:
```python
MEMORY_THRESHOLD_MB = 100  # 100MB
```

**변경 코드**:
```python
import psutil
available_memory = psutil.virtual_memory().available
MEMORY_THRESHOLD_MB = int(available_memory * 0.6 / (1024 ** 2))  # 가용 메모리의 60% (약 70GB)
```

**WHY**: 현재 설정은 119GB 시스템에 비해 과도하게 보수적.

---

#### 1.4 병렬 I/O 최적화
**파일**: `scripts/process_audio_files.py`

**추가 코드**:
```python
from concurrent.futures import ThreadPoolExecutor

def preload_audio_files(file_paths, batch_size=5):
    """오디오 파일을 미리 로드하여 I/O 대기 시간 감소"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(load_audio, file_paths[:batch_size]))
```

**WHY**: 파일 읽기와 처리를 오버랩하여 전체 처리 시간 단축.

---

### Phase 1 검증 기준
- ✅ 처리 시간 15분 이내
- ✅ CPU 활용률 80% 이상
- ✅ 메모리 사용량 95GB 미만
- ✅ 모든 파일 처리 성공률 100%

### Phase 1 예상 완료 시간
- **구현**: 1시간
- **테스트**: 30분
- **총**: 1.5시간

---

## Phase 2: GPU 활성화 (20배 향상)

### 목표
- **처리 시간**: 15분 → 3분
- **변경 범위**: faster-whisper 통합, GPU 디바이스 선택
- **위험도**: 중간

### 구현 작업

#### 2.1 faster-whisper 설치 및 통합
**파일**: `pyproject.toml` (또는 `requirements.txt`)

**추가 의존성**:
```toml
[tool.poetry.dependencies]
faster-whisper = ">=1.0.3"
torch = ">=2.5.0"
torchaudio = ">=2.5.0"
nvidia-ml-py = ">=12.560.30"
```

**WHY**: `faster-whisper`는 OpenAI Whisper 대비 4배 빠르고 GPU 지원.

---

#### 2.2 GPU 디바이스 선택 로직
**파일**: `src/voice_analysis/models/whisper_model.py` (신규)

**코드**:
```python
import torch
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

class GPUWhisperModel:
    def __init__(self, model_size="large-v3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info(f"Initializing Whisper model on {self.device} with {self.compute_type}")

        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=4,  # 멀티스레드 디코딩
        )

    def transcribe(self, audio_path, batch_size=16):
        """GPU 배치 추론으로 전사 수행"""
        segments, info = self.model.transcribe(
            audio_path,
            batch_size=batch_size if self.device == "cuda" else 1,
            language="ko",
            vad_filter=True,  # 무음 구간 필터링
        )
        return segments, info
```

**WHY**: GPU 가용성에 따라 자동으로 최적 설정 선택.

---

#### 2.3 analysis_pipeline_service.py 수정
**파일**: `src/voice_analysis/services/analysis_pipeline_service.py`

**현재 코드** (OpenAI Whisper 사용 추정):
```python
import whisper
model = whisper.load_model("large")
result = model.transcribe(audio_path)
```

**변경 코드**:
```python
from voice_analysis.models.whisper_model import GPUWhisperModel

model = GPUWhisperModel(model_size="large-v3")

def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path, batch_size=20)
    return {
        "text": " ".join([seg.text for seg in segments]),
        "segments": [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments],
        "language": info.language,
        "duration": info.duration,
    }
```

**WHY**: GPU 배치 추론으로 전사 속도 대폭 향상.

---

#### 2.4 GPU 메모리 모니터링
**파일**: `src/voice_analysis/services/memory_service.py`

**추가 코드**:
```python
import torch
import pynvml

class GPUMemoryMonitor:
    def __init__(self):
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 반환 (MB)"""
        if self.handle is None:
            return 0
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / (1024 ** 2)

    def should_reduce_batch_size(self):
        """GPU 메모리 사용률이 80% 초과하면 True 반환"""
        if self.handle is None:
            return False
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        usage_ratio = info.used / info.total
        return usage_ratio > 0.8
```

**WHY**: GPU OOM 오류 사전 방지 및 동적 배치 크기 조정 기반.

---

#### 2.5 동적 배치 크기 조정
**파일**: `src/voice_analysis/services/batch_service.py`

**추가 코드**:
```python
from voice_analysis.services.memory_service import GPUMemoryMonitor

gpu_monitor = GPUMemoryMonitor()
current_batch_size = 20

def adjust_batch_size():
    global current_batch_size
    if gpu_monitor.should_reduce_batch_size():
        current_batch_size = max(1, int(current_batch_size * 0.5))
        logger.warning(f"GPU memory high, reducing batch size to {current_batch_size}")
    elif gpu_monitor.get_gpu_memory_usage() < 50:
        current_batch_size = min(32, current_batch_size + 2)
        logger.info(f"GPU memory low, increasing batch size to {current_batch_size}")
```

**WHY**: GPU 메모리 상황에 따라 처리 속도 최적화.

---

### Phase 2 검증 기준
- ✅ 처리 시간 3분 이내
- ✅ GPU 활용률 70% 이상
- ✅ GPU 메모리 오류 0건
- ✅ WER (Word Error Rate) 변화 < 1%

### Phase 2 예상 완료 시간
- **구현**: 3시간
- **테스트**: 1시간
- **총**: 4시간

---

## Phase 3: 완전 파이프라인 통합 (50배 향상)

### 목표
- **처리 시간**: 3분 → 1.2분
- **변경 범위**: WhisperX 통합, pyannote 병렬화, VAD 전처리
- **위험도**: 높음

### 구현 작업

#### 3.1 WhisperX 설치 및 통합
**파일**: `pyproject.toml`

**추가 의존성**:
```toml
whisperx = ">=3.1.5"
pyannote-audio = ">=3.1.1"
```

**환경 변수**:
```bash
export HF_TOKEN="your_huggingface_token"
```

**WHY**: WhisperX는 Whisper + WAV2VEC2 + Pyannote을 GPU에서 통합 실행.

---

#### 3.2 WhisperX 파이프라인 구현
**파일**: `src/voice_analysis/models/whisperx_model.py` (신규)

**코드**:
```python
import whisperx
import torch
import logging

logger = logging.getLogger(__name__)

class WhisperXPipeline:
    def __init__(self, model_size="large-v3", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # 1. Whisper 모델 로드
        logger.info(f"Loading Whisper model ({model_size}) on {self.device}")
        self.whisper_model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=self.compute_type,
        )

        # 2. Alignment 모델 로드
        logger.info("Loading alignment model")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="ko",
            device=self.device,
        )

        # 3. Diarization 파이프라인 로드
        logger.info("Loading diarization pipeline")
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.getenv("HF_TOKEN"),
            device=self.device,
        )

    def process(self, audio_path, batch_size=16):
        """전체 파이프라인 실행: 전사 → 정렬 → 화자 분리"""
        import soundfile as sf

        # 오디오 로드
        audio, sr = sf.read(audio_path)

        # 1. Whisper 전사 (GPU 배치 추론)
        logger.info(f"Transcribing {audio_path}")
        result = self.whisper_model.transcribe(
            audio,
            batch_size=batch_size,
            language="ko",
        )

        # 2. WAV2VEC2 정렬 (타임스탬프 정확도 개선)
        logger.info("Aligning timestamps")
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
        )

        # 3. Pyannote 화자 분리
        logger.info("Diarizing speakers")
        diarize_segments = self.diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return result
```

**WHY**: 전체 파이프라인을 GPU에서 실행하여 CPU-GPU 전환 오버헤드 제거.

---

#### 3.3 화자 분리 병렬 처리
**파일**: `src/voice_analysis/services/batch_service.py`

**코드**:
```python
from concurrent.futures import ProcessPoolExecutor
from voice_analysis.models.whisperx_model import WhisperXPipeline

def process_batch_with_diarization(file_paths, batch_size=16):
    """배치 파일들을 병렬로 처리 (각 프로세스가 GPU 사용)"""
    pipeline = WhisperXPipeline(model_size="large-v3")

    results = []
    for file_path in file_paths:
        try:
            result = pipeline.process(file_path, batch_size=batch_size)
            results.append({"file": file_path, "result": result, "status": "success"})
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results.append({"file": file_path, "error": str(e), "status": "failed"})

    return results
```

**WHY**: WhisperX 파이프라인은 단일 파일 처리 최적화되어 있으므로 멀티프로세스로 병렬화.

---

#### 3.4 VAD 전처리 추가
**파일**: `src/voice_analysis/services/vad_service.py` (신규)

**코드**:
```python
import torch
from whisperx.vad import load_vad_model, merge_chunks

class VADPreprocessor:
    def __init__(self, device="cuda"):
        self.vad_model = load_vad_model(device=device)

    def remove_silence(self, audio, sr=16000):
        """무음 구간 제거 (GPU 가속)"""
        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr})
        audio_chunks = merge_chunks(audio, vad_segments)
        return audio_chunks
```

**통합**:
```python
# WhisperXPipeline.process() 메서드 내 추가
vad_preprocessor = VADPreprocessor(device=self.device)
audio = vad_preprocessor.remove_silence(audio, sr)
```

**WHY**: 무음 구간 제거로 처리할 데이터 양 감소 및 속도 향상.

---

#### 3.5 성능 리포트 생성
**파일**: `scripts/process_audio_files.py`

**추가 코드**:
```python
import time
import json

def generate_performance_report(results, total_time):
    """처리 완료 후 성능 리포트 생성"""
    total_files = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = total_files - successful

    avg_time_per_file = total_time / total_files if total_files > 0 else 0

    report = {
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "total_time_minutes": round(total_time / 60, 2),
        "avg_time_per_file_seconds": round(avg_time_per_file, 2),
        "gpu_utilization": get_avg_gpu_utilization(),
        "failed_files": [r["file"] for r in results if r["status"] == "failed"],
    }

    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Performance Report: {json.dumps(report, indent=2)}")
    return report
```

**WHY**: 성능 목표 달성 여부 검증 및 향후 최적화 방향 제시.

---

### Phase 3 검증 기준
- ✅ 처리 시간 1.5분 이내 (목표 1.2분)
- ✅ GPU 활용률 85% 이상
- ✅ 화자 분리 정확도 90% 이상
- ✅ 메모리 누수 0건
- ✅ 처리 성공률 100%

### Phase 3 예상 완료 시간
- **구현**: 6시간
- **테스트**: 2시간
- **총**: 8시간

---

## 의존성 및 순서

### Phase 간 의존성
- **Phase 1 → Phase 2**: CPU 최적화 후 GPU 통합
- **Phase 2 → Phase 3**: GPU 기반 전사 후 전체 파이프라인

### 병렬 작업 가능성
- Phase 1 구현 중 Phase 2 테스트 환경 구축 가능
- Phase 2 구현 중 Phase 3 WhisperX 설치 및 학습 가능

---

## 위험 분석 및 대응 전략

### 고위험 (Phase 3)
**위험**: WhisperX 통합 실패 (라이브러리 호환성 문제)
- **확률**: 30%
- **영향**: Phase 3 완전 실패, Phase 2로 폴백
- **대응**:
  1. Docker 컨테이너 사용하여 환경 격리
  2. 버전 고정 (`whisperx==3.1.5`)
  3. Phase 2 성능으로도 20배 향상 확보

### 중위험 (Phase 2)
**위험**: GPU 메모리 부족 (CUDA OOM)
- **확률**: 50%
- **영향**: 배치 크기 감소로 성능 저하
- **대응**:
  1. 동적 배치 크기 조정 로직
  2. `model_size="medium"` 사용 (정확도 약간 감소)
  3. CPU 폴백 모드 활성화

### 저위험 (Phase 1)
**위험**: CPU 병목 지속
- **확률**: 20%
- **영향**: 4배 향상 미달 (3배 정도)
- **대응**:
  1. I/O 병렬화 강화
  2. 파일 프리로딩
  3. Phase 2로 빠르게 전환

---

## 테스트 전략

### 단위 테스트 (각 Phase별)
- `test_batch_service.py`: 배치 크기 조정 로직
- `test_memory_service.py`: 메모리 모니터링
- `test_whisper_model.py`: GPU 디바이스 선택
- `test_whisperx_pipeline.py`: WhisperX 통합

### 통합 테스트
- **소규모 테스트**: 10개 파일로 전체 파이프라인 검증
- **중규모 테스트**: 50개 파일로 성능 측정
- **전체 테스트**: 183개 파일로 최종 검증

### 성능 테스트
- **처리 시간 측정**: 각 Phase별 목표 달성 여부
- **메모리 프로파일링**: 메모리 누수 확인
- **GPU 활용률**: nvidia-smi로 실시간 모니터링

---

## 롤백 계획

### Phase 3 실패 시
- Phase 2 (faster-whisper)로 롤백
- 성능: 20배 향상 유지
- 화자 분리: 별도 후처리로 수행

### Phase 2 실패 시
- Phase 1 (CPU 최적화)로 롤백
- 성능: 4배 향상 유지
- GPU 없이도 동작 보장

---

## 구현 우선순위

### 최우선 (P0)
- Phase 1.1: Worker 수 증가
- Phase 1.2: 배치 크기 증가
- Phase 2.1: faster-whisper 통합

### 우선 (P1)
- Phase 2.2: GPU 디바이스 선택
- Phase 2.4: GPU 메모리 모니터링
- Phase 3.2: WhisperX 통합

### 일반 (P2)
- Phase 1.4: 병렬 I/O 최적화
- Phase 3.4: VAD 전처리
- Phase 3.5: 성능 리포트

---

## 마일스톤

### Milestone 1: 기본 최적화 완료 (Phase 1)
- **목표**: 15분 이내 처리
- **검증**: 성능 테스트 통과
- **다음 단계**: Phase 2 시작

### Milestone 2: GPU 활성화 완료 (Phase 2)
- **목표**: 3분 이내 처리
- **검증**: GPU 활용률 70% 이상
- **다음 단계**: Phase 3 시작

### Milestone 3: 최종 목표 달성 (Phase 3)
- **목표**: 1.5분 이내 처리
- **검증**: 전체 테스트 및 품질 게이트 통과
- **다음 단계**: 프로덕션 배포

---

## 총 예상 소요 시간

- **Phase 1**: 1.5시간
- **Phase 2**: 4시간
- **Phase 3**: 8시간
- **테스트 및 통합**: 2시간
- **문서화**: 1시간

**총 예상 시간**: **16.5시간** (약 2-3일)

---

## 다음 단계

1. ✅ Phase 1 구현 시작 (`/moai:2-run SPEC-PARALLEL-001 --phase 1`)
2. ⏸ Phase 1 검증 후 Phase 2 진행
3. ⏸ Phase 2 검증 후 Phase 3 진행
4. ⏸ 최종 성능 검증 및 문서화
5. ⏸ 프로덕션 배포 및 모니터링

---

**문서 끝**
