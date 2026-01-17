# Optimization Review and CPU Parallel Processing Integration

**작성일**: 2026-01-15
**버전**: 1.0
**상태**: 검토 및 설계

---

## 1. 현재 최적화 스크립트 검토

### 1.1 구현된 최적화 기능

| 기능 | 상태 | 위치 |
|------|------|------|
| **배치 크기 증가** | ✅ 완료 | `batch_size=10` |
| **모델 상주 유지** | ✅ 완료 | `model_resident=True` |
| **GPU 메모리 모니터링** | ✅ 완료 | `MemoryMonitor.get_memory_stats()` |
| **동적 배치 크기** | ✅ 완료 | `_calculate_dynamic_batch_size()` |
| **병렬 처리** | ⚠️ 비활성화 | 경고 추가됨 |

### 1.2 발견된 문제점

#### 문제 1: Service Instance Lifecycle

**위치**: `run_optimized_batch.py:329-361`

```python
def _initialize_stt_service(self):
    if self._stt_service is not None:
        return  # 이미 초기화됨

    # Service 생성
    self._stt_service = WhisperXService(device=device, language="ko")
```

**문제점**:
- Service 초기화 실패 시 `_stt_service`가 여전히 `None`인데 재시도 불가
- GPU OOM 시 복구 메커니즘 없음

**영향**: 중간도 - 첫 초기화만 성공하면 문제 없음

**해결 방안**:
```python
def _initialize_stt_service(self):
    if self._stt_service is not None:
        return

    try:
        self._stt_service = WhisperXService(device=device, language="ko")
        # Warmup: test inference
        self._stt_service.transcribe_only("")  # or small dummy file
    except Exception as e:
        logger.error(f"Failed to initialize STT service: {e}")
        self._stt_service = None  # 명시적 초기화
        raise
```

---

#### 문제 2: GPU 메모리 모니터링 정확도

**위치**: `run_optimized_batch.py:154-178`

```python
if torch.cuda.is_available():
    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**2)
    gpu_reserved = torch.cuda.memory_reserved(0) / (1024**2)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
```

**문제점**:
- `memory_reserved`는 예약된 메모리, 실제 사용량과 다름
- 동적 배치 크기 결정에 부정확한 데이터 사용 가능

**영향**: 중간 - 동적 배치 크기가 보수적일 수 있음

**해결 방안**:
```python
# 더 정확한 GPU 메모리 추정
def get_gpu_memory_stats(self):
    try:
        import torch
        import pynvml

        # PyTorch memory info
        torch.cuda.set_device(0)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)

        # NVML 상세 정보
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return {
            "allocated_gb": allocated,
            "total_gb": info.total / (1024**3),
            "used_gb": info.used / (1024**3),
            "free_gb": info.free / (1024**3),
            "percent": (info.used / info.total) * 100,
        }
    except Exception:
        return {"percent": 0}  # Fallback
```

---

#### 문제 3: 동적 배치 크기 로직

**위치**: `run_optimized_batch.py:363-379`

```python
def _calculate_dynamic_batch_size(self) -> int:
    stats = self.memory_monitor.get_memory_stats()
    gpu_percent = stats.get("gpu_percent", 0.0)

    if gpu_percent < 50:
        return min(self.opt_config.max_batch_size, 32)
    elif gpu_percent < 70:
        return min(self.opt_config.max_batch_size, 16)
    # ...
```

**문제점**:
- GPU 메모리 사용량이 50% 미만일 때 최대 32 파일 배치
- WhisperX large-v3는 파일당 ~300-500MB GPU 메모리 사용
- 32 파일 × 500MB = 16GB 필요 (현재 GPU에서 불가능)

**영향**: 높음 - OOM 발생 가능

**해결 방안**:
```python
def _calculate_dynamic_batch_size(self) -> int:
    stats = self.memory_monitor.get_memory_stats()
    gpu_total_gb = stats.get("gpu_total_mb", 0) / 1024
    gpu_used_gb = stats.get("gpu_allocated_mb", 0) / 1024
    gpu_free_gb = gpu_total_gb - gpu_used_gb

    # WhisperX large-v3: ~400MB per file (conservative estimate)
    memory_per_file = 0.4  # GB
    safety_margin = 0.8  # 80% utilization max

    max_files_by_memory = int((gpu_free_gb * safety_margin) / memory_per_file)

    # Return min of:
    # 1. Memory-based limit
    # 2. Configured max
    # 3. Configured min (at least)
    return max(
        self.opt_config.min_batch_size,
        min(max_files_by_memory, self.opt_config.max_batch_size)
    )
```

---

#### 문제 4: Checkpoint State Store 접근

**위치**: `run_optimized_batch.py:579-587`

```python
existing_state = self.checkpoint_manager.state_store.get_file_state(
    str(audio_file)
)
if existing_state and existing_state.status == FileStatus.COMPLETED:
    logger.info(f"Skipping (already completed): {audio_file.name}")
    skipped += 1
    continue
```

**문제점**:
- 각 파일마다 DB 조회 (SQLite)
- 183 파일 × 2 조회 = 366 DB 쿼리 (초기화 시)

**영향**: 낮음 - 성능 저하 미미하지만 개선 여지 있음

**해결 방안**:
```python
# 초기화 시 한 번에 조회
completed_files = set()
if self.resume:
    for file in files:
        state = self.checkpoint_manager.state_store.get_file_state(str(file))
        if state and state.status == FileStatus.COMPLETED:
            completed_files.add(str(file))

# 처리 시 메모리 조회
if str(audio_file) in completed_files:
    logger.info(f"Skipping (already completed): {audio_file.name}")
    skipped += 1
    continue
```

---

### 1.3 누락된 최적화 기능

| 기능 | 현재 상태 | 예상 향상 |
|------|----------|----------|
| **파일 I/O 병렬화** | ❌ 미구현 | 2-3x |
| **오디오 전처리 병렬화** | ❌ 미구현 | 2-3x |
| **CPU 활용** | 20-40% | 60-80% |
| **배치 간 전처리 파이프라인** | ❌ 미구현 | 1.5x |

---

## 2. CPU 병렬 처리 통합 설계

### 2.1 기존 ARMCPUPipeline 활용

**이미 구현된 기능**:
- 20코어 ARM 탐지
- 병렬 파일 I/O (`load_parallel`)
- 병렬 전처리 (`preprocess_parallel`)
- CPU 모니터링

**파일 위치**: `src/voice_man/services/edgexpert/arm_cpu_pipeline.py`

### 2.2 통합 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                   OptimizedBatchProcessor                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              ARMCPUPipeline (기존)                    │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ Parallel File I/O (8x speedup)                   │ │ │
│  │  │   - 20-core ARM detection                         │ │ │
│  │  │   - ThreadPoolExecutor for I/O                    │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │                                                        │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ Parallel Audio Preprocessing                     │ │ │
│  │  │   - Format conversion (m4a → wav)                │ │ │
│  │  │   - Resampling (16kHz)                           │ │ │
│  │  │   - Normalization                                │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              GPU Processing (Sequential)              │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ STT Service (WhisperX on GPU)                    │ │ │
│  │  │ Forensic Service (Multi-model on GPU)           │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Data Flow:
  Raw Files → ARM CPU (Parallel I/O + Preprocessing)
              → Preprocessed Audio Files
              → GPU (Sequential: STT → Forensic)
              → Results
```

### 2.3 처리 파이프라인

```
Phase 1: Parallel I/O & Preprocessing (ARM CPU)
├── Batch 1 (10 files)
│   ├── File 1: m4a → wav (Thread 1)
│   ├── File 2: m4a → wav (Thread 2)
│   ├── ...
│   └── File 10: m4a → wav (Thread 10)
│
├── Batch 2 (10 files)
│   └── ... (parallel conversion)
│
└── Total: ~20-30초 (vs sequential ~2-3분)

Phase 2: GPU Processing (Sequential)
├── Preprocessed File 1 → STT → Forensic
├── Preprocessed File 2 → STT → Forensic
├── ...
└── Preprocessed File 10 → STT → Forensic
```

### 2.4 구현 상세

#### 2.4.1 통합된 OptimizedBatchProcessor

```python
class OptimizedBatchProcessor:
    """Optimized batch processor with CPU parallel preprocessing."""

    def __init__(
        self,
        audio_dir: Path = Path("ref/call"),
        batch_size: int = 10,
        enable_cpu_parallel: bool = True,  # NEW
        cpu_workers: int = None,  # NEW
        model_resident: bool = True,
        # ... other parameters
    ):
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.enable_cpu_parallel = enable_cpu_parallel
        self.cpu_workers = cpu_workers or min(10, batch_size)  # Use up to 10 workers
        self.model_resident = model_resident

        # Initialize ARM CPU pipeline for parallel preprocessing
        if self.enable_cpu_parallel:
            from voice_man.services.edgexpert.arm_cpu_pipeline import ARMCPUPipeline
            self.cpu_pipeline = ARMCPUPipeline()
            logger.info(f"ARM CPU pipeline initialized: {self.cpu_pipeline.total_cores} cores")
        else:
            self.cpu_pipeline = None

        # GPU services (lazy init)
        self._stt_service = None
        self._forensic_service = None
```

#### 2.4.2 병렬 전처리 함수

```python
def preprocess_audio_file(audio_file: Path) -> Path:
    """
    Preprocess a single audio file (m4a → wav, 16kHz).

    Args:
        audio_file: Path to input audio file

    Returns:
        Path to preprocessed wav file
    """
    try:
        from voice_man.services.audio_converter_service import AudioConverterService

        converter = AudioConverterService(target_sample_rate=16000, target_channels=1)

        # Convert to temp wav file
        temp_wav = converter.convert(str(audio_file))

        return Path(temp_wav)

    except Exception as e:
        logger.error(f"Failed to preprocess {audio_file.name}: {e}")
        raise
```

#### 2.4.3 배치 처리 with CPU Parallel

```python
async def process_batch(
    self,
    batch_number: int,
    files: List[Path],
) -> BatchResult:
    """Process a batch with CPU parallel preprocessing."""
    start_time = datetime.now(timezone.utc)

    # Phase 1: Parallel preprocessing (CPU)
    preprocessed_files = []
    if self.enable_cpu_parallel and len(files) > 1:
        logger.info(f"Phase 1: Parallel preprocessing {len(files)} files...")

        # Use ARMCPUPipeline for parallel I/O
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            futures = {
                executor.submit(preprocess_audio_file, f): f
                for f in files
            }

            for future in as_completed(futures):
                try:
                    preprocessed = future.result()
                    preprocessed_files.append(preprocessed)
                except Exception as e:
                    original_file = futures[future]
                    logger.error(f"Preprocessing failed for {original_file.name}: {e}")

        # Sort to maintain order
        preprocessed_files.sort()

        preprocess_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Phase 1 complete: {len(preprocessed_files)} files in {preprocess_time:.1f}s")

    else:
        # Sequential preprocessing (fallback)
        preprocessed_files = [preprocess_audio_file(f) for f in files]

    # Phase 2: GPU processing (sequential)
    logger.info(f"Phase 2: GPU processing {len(preprocessed_files)} preprocessed files...")

    # Initialize GPU services if needed
    if self._stt_service is None:
        self._initialize_stt_service()
    if self._forensic_service is None:
        self._initialize_forensic_service()

    # Process each file (sequential for GPU safety)
    successful = 0
    failed = 0
    results = []

    for i, (original_file, preprocessed_file) in enumerate(zip(files, preprocessed_files), 1):
        try:
            # STT
            logger.info(f"[{i}/{len(files)}] STT: {original_file.name}")
            stt_result = self._stt_service.transcribe_only(str(preprocessed_file))

            # Forensic
            logger.info(f"[{i}/{len(files)}] Forensic: {original_file.name}")
            forensic_result = self._forensic_service.analyze(
                transcript=stt_result["text"],
                audio_path=str(preprocessed_file),
                language="ko",
            )

            successful += 1
            results.append({
                "file": str(original_file),
                "stt": stt_result,
                "forensic": forensic_result,
            })

            # Cleanup preprocessed file
            preprocessed_file.unlink(missing_ok=True)

        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {original_file.name}: {e}")

    # Cleanup preprocessed files
    for f in preprocessed_files:
        f.unlink(missing_ok=True)

    # Lightweight cleanup (models kept resident)
    if not self.model_resident:
        MemoryMonitor.cleanup()

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    return BatchResult(
        batch_number=batch_number,
        total_files=len(files),
        successful=successful,
        failed=failed,
        skipped=0,
        retries=0,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        # ... memory stats
    )
```

---

## 3. 성능 예측

### 3.1 처리 시간 분해

| 단계 | 작업 | 현재 (순차) | CPU 병렬 | 개선 |
|------|------|------------|----------|------|
| **전처리** | m4a → wav 변환 (10파일) | ~120초 | ~20초 | 6x |
| **STT** | WhisperX 추론 (10파일) | ~600초 | ~600초 | 1x |
| **포렌식** | 분석 (10파일) | ~180초 | ~180초 | 1x |
| **총계** | | ~900초 (15분) | ~800초 (13.3분) | 1.1x |

### 3.2 전체 파이프라인 향상

```
기본 최적화만:
  6 파일/시간 → 24 파일/시간 (4x)

CPU 병렬 추가:
  24 파일/시간 → ~26 파일/시간 (4.3x)

이유: GPU 처리가 병목이므로 전처리 병렬화는
      전체 시간의 ~10%만 단축
```

### 3.3 확장 시나리오

**현재 하드웨어 (GPU 1개, CPU 20코어)**:
- 최적의 배치 크기: 10-15 파일
- CPU workers: 10 (전처리용)
- 예상 처리량: ~26 파일/시간

**GPU 2개 추가 시**:
- 배치 분리: 2개 GPU에 각 10파일
- 예상 처리량: ~52 파일/시간 (2x)

---

## 4. 구현 우선순위

### Phase 1: 문제 수정 (필수)

| 문제 | 우선순위 | 예상 시간 |
|------|----------|----------|
| Service lifecycle 개선 | 높음 | 30분 |
| GPU 메모리 정확도 | 높음 | 1시간 |
| 동적 배치 크기 로직 | 높음 | 1시간 |
| Checkpoint 캐싱 | 중간 | 30분 |

### Phase 2: CPU 병렬 처리 (선택)

| 작업 | 예상 시간 | 성능 향상 |
|------|----------|----------|
| ARMCPUPipeline 통합 | 2시간 | 1.1x |
| 병렬 전처리 구현 | 2시간 | 1.1x |
| 테스트 및 벤치마크 | 1시간 | - |

### Phase 3: 최종 최적화 (선택)

| 작업 | 예상 시간 | 성능 향상 |
|------|----------|----------|
| 배치 간 파이프라인 | 3시간 | 1.3x |
| GPU 메모리 최적화 | 2시간 | 1.2x |

---

## 5. 결론

### 5.1 현재 최적화 상태

**잘 구현된 기능**:
- ✅ 배치 크기 증가 (3 → 10)
- ✅ 모델 상주 유지
- ✅ GPU 메모리 모니터링

**개선이 필요한 기능**:
- ⚠️ Service lifecycle 안정성
- ⚠️ 동적 배치 크기 정확도
- ❌ CPU 병렬 처리 (미구현)

### 5.2 권장사항

**즉시 적용** (현재 하드웨어):
1. GPU 메모리 기반 정확한 동적 배치 크기
2. Service 안정성 개선
3. Checkpoint 캐싱

**선택 적용** (추가 성능 필요 시):
1. CPU 병렬 전처리 (ARMCPUPipeline 활용)
2. 배치 간 파이프라인 (전처리 → GPU → 다음 전처리)

**향후 확장** (Multi-GPU):
1. ProcessPoolExecutor 기반 GPU worker
2. Ray 기반 분산 처리

---

**문서 종료**
