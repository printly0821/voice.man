# SPEC-PERFOPT-001: Implementation Plan

## TAG BLOCK

```yaml
SPEC_ID: SPEC-PERFOPT-001
TITLE: Forensic Pipeline Performance Optimization
STATUS: In Progress
PRIORITY: High
CREATED: 2026-01-10
UPDATED: 2026-01-10
LIFECYCLE: spec-anchored
PHASE_1_STATUS: COMPLETED
PHASE_2_STATUS: COMPLETED
PHASE_3_STATUS: Planned
```

---

## 1. Implementation Overview

### 1.1 Objectives

Forensic 분석 파이프라인의 GPU 활용률을 최적화하여 183개 오디오 파일 처리 시간을 15시간 이상에서 4시간 미만으로 단축.

### 1.2 Key Deliverables

1. **Phase 1 (Quick Wins):** 메모리/GPU 기본 최적화로 2-3배 향상
2. **Phase 2 (Integration):** 통합 메모리/열 관리로 3-4배 향상
3. **Phase 3 (Advanced):** 파이프라이닝으로 4배+ 향상

### 1.3 Success Criteria

- 처리 시간: 15시간 → 4시간 미만
- GPU 활용률 (Forensic): 5% → 80%+
- 열 쓰로틀링: 0회
- OOM 발생: 0회

---

## 2. Phase 1: Quick Wins (Primary Goal) - COMPLETED

### 2.1 Milestone Summary

| Task | Files | Effort | Status |
|------|-------|--------|--------|
| 1.1 메모리 임계값 수정 | memory_service.py | Medium | COMPLETED |
| 1.2 SER GPU 추론 활성화 | ser_service.py | High | COMPLETED |
| 1.3 empty_cache 최적화 | e2e_test_service.py | Low | COMPLETED |
| 1.4 벤치마크 및 검증 | 테스트 스크립트 | Medium | COMPLETED |

### 2.1.1 Implementation Summary

**Completed: 2026-01-10**

**Commits:**
- `07d6349`: perf(memory): increase threshold to 30GB for forensic workloads
- `d4f681c`: perf(ser): add GPU-first detection and model caching
- `78969ba`: perf(e2e): add per-batch GPU cache cleanup

**Files Modified:**
- `src/voice_man/services/memory_service.py`: FORENSIC_MEMORY_THRESHOLD_MB = 30000
- `src/voice_man/services/forensic/ser_service.py`: GPU-first detection, class-level model caching, preload_models()
- `src/voice_man/services/e2e_test_service.py`: _cleanup_after_batch() per-batch GPU cleanup

**Tests Created:**
- `tests/unit/test_memory_service_perfopt.py` (10 tests)
- `tests/unit/test_ser_service_perfopt.py` (19 tests)
- `tests/unit/test_e2e_service_perfopt.py` (9 tests)

### 2.2 Task Details

#### Task 1.1: 메모리 임계값 수정 - COMPLETED

**목표:** 메모리 캐시 효율 300배 향상

**파일 수정:**
```
src/voice_man/services/memory_service.py
```

**변경 사항:**
- `FORENSIC_MEMORY_THRESHOLD_MB = 30000` 상수 추가
- `MemoryManager` 기본 임계값 30GB로 변경
- SPEC-PERFOPT-001 문서화 주석 추가

**검증:**
- 단위 테스트: 10개 테스트 통과
  - `test_forensic_memory_threshold_constant_exists`
  - `test_memory_manager_default_threshold_is_30gb`
  - `test_memory_manager_accepts_custom_threshold`
  - `test_memory_manager_docstring_reflects_new_threshold`
  - `test_should_collect_uses_correct_threshold`
  - `test_should_collect_triggers_above_threshold`
  - `test_get_memory_summary_shows_correct_threshold`
  - `test_explicit_100mb_threshold_still_works`
  - `test_explicit_threshold_overrides_default`
  - `test_usage_percentage_calculated_correctly`

---

#### Task 1.2: SER GPU 추론 활성화 - COMPLETED

**목표:** SER 추론 속도 10배 향상, 모델 로딩 오버헤드 제거

**파일 수정:**
```
src/voice_man/services/forensic/ser_service.py
```

**변경 사항:**

1. **GPU-first 디바이스 감지:** `_detect_optimal_device()` 메서드 추가
   - CUDA 사용 가능 시 자동으로 GPU 선택
   - CPU 폴백 지원 유지

2. **클래스 레벨 모델 캐싱:** `_model_cache` 클래스 속성 추가
   - 인스턴스 간 모델 공유
   - `_get_or_load_model()` 캐시 조회 메서드

3. **비동기 사전 로딩:** `preload_models()` async 메서드 추가
   - 로드 시간, 메모리 사용량 통계 반환
   - ensemble 설정에 따른 선택적 로딩

4. **캐시 관리:** `clear_model_cache()` 클래스 메서드 추가
   - GPU 메모리 해제
   - GC 트리거

**검증:**
- 단위 테스트: 19개 테스트 통과
  - TestSERModelCaching: 7개 테스트
  - TestGPUFirstDeviceDetection: 5개 테스트
  - TestPreloadModels: 6개 테스트
  - TestModelCacheIntegration: 1개 테스트

---

#### Task 1.3: torch.cuda.empty_cache() 최적화 - COMPLETED

**목표:** 불필요한 메모리 정리 오버헤드 제거

**파일 수정:**
```
src/voice_man/services/e2e_test_service.py
```

**변경 사항:**

1. **배치별 정리 메서드:** `_cleanup_after_batch()` async 메서드 추가
   - `gc.collect()` 호출
   - `torch.cuda.empty_cache()` 호출
   - `torch.cuda.ipc_collect()` 선택적 호출
   - 에러 핸들링 포함

2. **run() 메서드 통합:**
   - 각 배치 처리 완료 후 `_cleanup_after_batch()` 호출
   - SPEC-PERFOPT-001 주석 추가

**검증:**
- 단위 테스트: 9개 테스트 통과
  - TestPerBatchEmptyCacheOptimization: 7개 테스트
  - TestCleanupIntegration: 2개 테스트

---

#### Task 1.4: 벤치마크 및 검증 - COMPLETED

**목표:** Phase 1 최적화 효과 측정 및 검증

**테스트 파일:**
```
tests/unit/test_memory_service_perfopt.py (10 tests)
tests/unit/test_ser_service_perfopt.py (19 tests)
tests/unit/test_e2e_service_perfopt.py (9 tests)
```

**벤치마크 항목:**
1. 메모리 임계값 설정 검증
2. GPU-first 디바이스 감지 검증
3. 모델 캐싱 동작 검증
4. 배치별 GPU 캐시 정리 검증
5. 에러 핸들링 검증

**산출물:**
- 38개 단위 테스트 (TDD GREEN Phase 완료)
- SPEC 문서 업데이트

---

### 2.3 Phase 1 완료 기준 - ACHIEVED

| Metric | Baseline | Target | Actual | Status |
|--------|----------|--------|--------|--------|
| 메모리 임계값 | 100MB | 30GB | 30GB | PASS |
| GPU-first 감지 | CPU only | CUDA 우선 | CUDA 우선 | PASS |
| 모델 캐싱 | 없음 | 클래스 레벨 | 구현됨 | PASS |
| 배치별 GPU 정리 | 없음 | 배치당 1회 | 구현됨 | PASS |
| 단위 테스트 | 0개 | 38개 | 38개 | PASS |

**참고:** 실제 처리 시간 및 GPU 활용률 측정은 E2E 통합 테스트에서 수행 예정

---

## 3. Phase 2: Integration (Secondary Goal) - COMPLETED

### 3.1 Milestone Summary

| Task | Files | Effort | Status |
|------|-------|--------|--------|
| 2.1 ForensicMemoryManager 통합 | memory_manager.py (신규) | High | COMPLETED |
| 2.2 스테이지별 배치 설정 | batch_config.py (신규) | Medium | COMPLETED |
| 2.3 ThermalManager 통합 | thermal_manager.py (신규) | High | COMPLETED |
| 2.4 서비스 통합 및 테스트 | e2e_test_service.py, ser_service.py | Medium | COMPLETED |

### 3.1.1 Implementation Summary

**Completed: 2026-01-10**

**Commits:**
- `c28e343`: feat(forensic): add ForensicMemoryManager
- `13a2241`: feat(config): add BatchConfigManager
- `3785258`: feat(forensic): add ThermalManager
- `60d2ad9`: refactor(forensic): integrate Phase 2 managers

**Files Created:**
- `src/voice_man/services/forensic/memory_manager.py`: ForensicMemoryManager 클래스
- `src/voice_man/config/batch_config.py`: StageBatchConfig, BatchConfigManager 클래스
- `src/voice_man/services/forensic/thermal_manager.py`: ThermalManager 클래스

**Files Modified:**
- `src/voice_man/services/e2e_test_service.py`: 메모리/온도 관리자 통합
- `src/voice_man/services/forensic/ser_service.py`: 메모리 관리자 통합

**Tests Created/Updated:**
- 67개 테스트 통과 (86% 커버리지)

### 3.2 Task Details

#### Task 2.1: ForensicMemoryManager 통합 - COMPLETED

**목표:** 스테이지 간 메모리 충돌 방지, 통합 모니터링

**신규 파일:**
```
src/voice_man/services/forensic/memory_manager.py
```

**구현 내용:**

```python
class ForensicMemoryManager:
    """Forensic 파이프라인 통합 메모리 관리자."""

    # 스테이지별 메모리 할당량 (GB)
    STAGE_MEMORY_ALLOCATION = {
        "stt": 16.0,
        "alignment": 4.0,
        "diarization": 8.0,
        "ser": 10.0,
        "scoring": 2.0,
    }

    def __init__(self, total_memory_gb: float = 128.0):
        self.total_memory = total_memory_gb
        self.allocated = {}
        self._lock = threading.Lock()

    def allocate(self, stage: str) -> bool:
        """스테이지에 메모리 할당."""
        with self._lock:
            required = self.STAGE_MEMORY_ALLOCATION.get(stage, 4.0)
            available = self.get_available_memory()
            if available >= required:
                self.allocated[stage] = required
                return True
            return False

    def release(self, stage: str) -> None:
        """스테이지 메모리 해제."""
        with self._lock:
            if stage in self.allocated:
                del self.allocated[stage]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        """현재 메모리 상태 조회."""
        return {
            "total_gb": self.total_memory,
            "allocated_gb": sum(self.allocated.values()),
            "available_gb": self.get_available_memory(),
            "gpu_used_gb": self._get_gpu_memory_used(),
            "stages": dict(self.allocated),
        }
```

**SPEC-EDGEXPERT-001 통합:**
- `UnifiedMemoryManager` 클래스 재사용
- Zero-copy 메모리 접근 패턴 적용

---

#### Task 2.2: 스테이지별 배치 설정 - COMPLETED

**목표:** 각 스테이지에 최적화된 배치 사이즈 적용

**신규 파일:**
```
src/voice_man/config/batch_config.py
```

**구현 내용:**

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class StageBatchConfig:
    """스테이지별 배치 설정."""
    stage: str
    default_batch_size: int
    min_batch_size: int
    max_batch_size: int
    gpu_memory_per_item_mb: int

class BatchConfigManager:
    """동적 배치 사이즈 관리자."""

    STAGE_CONFIGS = {
        "stt": StageBatchConfig("stt", 8, 1, 32, 2000),
        "alignment": StageBatchConfig("alignment", 16, 4, 64, 500),
        "diarization": StageBatchConfig("diarization", 4, 1, 16, 4000),
        "ser": StageBatchConfig("ser", 8, 1, 32, 1500),
        "scoring": StageBatchConfig("scoring", 32, 8, 128, 100),
    }

    def get_optimal_batch_size(
        self,
        stage: str,
        available_memory_mb: int
    ) -> int:
        """가용 메모리 기반 최적 배치 사이즈 계산."""
        config = self.STAGE_CONFIGS.get(stage)
        if not config:
            return 8

        max_by_memory = available_memory_mb // config.gpu_memory_per_item_mb
        optimal = min(max_by_memory, config.max_batch_size)
        return max(optimal, config.min_batch_size)
```

---

#### Task 2.3: ThermalManager 통합 - COMPLETED

**목표:** GPU 온도 모니터링, 동적 쓰로틀링

**신규 파일:**
```
src/voice_man/services/forensic/thermal_manager.py
```

**구현 내용:**

```python
import threading
import time
from typing import Callable, Optional

class ThermalManager:
    """GPU 열 관리자."""

    # 온도 임계값 (°C)
    THROTTLE_START_TEMP = 80
    THROTTLE_STOP_TEMP = 70
    CRITICAL_TEMP = 85

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._throttle_active = False
        self._callbacks: List[Callable[[int, bool], None]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """모니터링 시작."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """모니터링 중지."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _monitor_loop(self) -> None:
        """온도 모니터링 루프."""
        while self._running:
            temp = self._get_gpu_temperature()
            self._handle_temperature(temp)
            time.sleep(self.poll_interval)

    def _get_gpu_temperature(self) -> int:
        """GPU 온도 조회."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            pynvml.nvmlShutdown()
            return temp
        except Exception:
            return 50  # 기본값

    def _handle_temperature(self, temp: int) -> None:
        """온도 기반 쓰로틀링 결정."""
        with self._lock:
            if temp >= self.CRITICAL_TEMP:
                # 긴급 중지
                self._notify_callbacks(temp, critical=True)
            elif temp >= self.THROTTLE_START_TEMP and not self._throttle_active:
                # 쓰로틀링 시작
                self._throttle_active = True
                self._notify_callbacks(temp, throttle=True)
            elif temp < self.THROTTLE_STOP_TEMP and self._throttle_active:
                # 쓰로틀링 해제
                self._throttle_active = False
                self._notify_callbacks(temp, throttle=False)

    def register_callback(
        self,
        callback: Callable[[int, bool], None]
    ) -> None:
        """온도 변화 콜백 등록."""
        self._callbacks.append(callback)

    @property
    def is_throttling(self) -> bool:
        """현재 쓰로틀링 상태."""
        return self._throttle_active
```

---

### 3.3 Phase 2 완료 기준 - ACHIEVED

| Metric | Phase 1 | Target | Actual | Status |
|--------|---------|--------|--------|--------|
| ForensicMemoryManager | 없음 | 구현 | 구현됨 | PASS |
| BatchConfigManager | 없음 | 구현 | 구현됨 | PASS |
| ThermalManager | 없음 | 구현 | 구현됨 | PASS |
| 서비스 통합 | 없음 | E2E + SER | 통합됨 | PASS |
| 단위 테스트 | 38개 | 60개+ | 67개 | PASS |
| 테스트 커버리지 | N/A | 80%+ | 86% | PASS |

**구현 상세:**
- **ForensicMemoryManager**: Stage별 GPU 메모리 할당 (STT:16GB, Alignment:4GB, Diarization:8GB, SER:10GB, Scoring:2GB)
- **BatchConfigManager**: StageBatchConfig 데이터클래스, 동적 배치 크기 계산
- **ThermalManager**: GPU 온도 임계값 관리, 히스테리시스 기반 스로틀링, 백그라운드 모니터링

**참고:** 실제 처리 시간 및 GPU 활용률 측정은 E2E 통합 테스트에서 수행 예정

---

## 4. Phase 3: Advanced (Optional Goal)

### 4.1 Milestone Summary

| Task | Files | Effort |
|------|-------|--------|
| 3.1 스테이지 파이프라이닝 | pipeline_orchestrator.py (신규) | High |
| 3.2 CUDA Graph 캐싱 (선택) | cuda_graph_cache.py (신규) | Medium |
| 3.3 ARM64 CPU 최적화 (선택) | arm_optimizer.py (신규) | Medium |

### 4.2 Task Details

#### Task 3.1: 스테이지 파이프라이닝

**목표:** STT + Forensic 스테이지 오버랩으로 50% 효율 향상

**신규 파일:**
```
src/voice_man/services/forensic/pipeline_orchestrator.py
```

**아키텍처:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ STT Stage   │───▶│ Result Queue│───▶│ Forensic    │
│ (Producer)  │    │ (Max 5)     │    │ (Consumer)  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                                      │
       │         Backpressure                 │
       │◀─────────────────────────────────────│
```

**구현 내용:**

```python
import asyncio
from typing import List, AsyncIterator

class PipelineOrchestrator:
    """STT-Forensic 파이프라인 오케스트레이터."""

    MAX_QUEUE_SIZE = 5

    def __init__(
        self,
        stt_service: WhisperXService,
        forensic_service: ForensicService,
    ):
        self.stt_service = stt_service
        self.forensic_service = forensic_service
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._stop_event = asyncio.Event()

    async def process_files(
        self,
        files: List[Path]
    ) -> AsyncIterator[ForensicResult]:
        """파이프라인 처리."""
        # Producer와 Consumer 동시 실행
        producer = asyncio.create_task(self._produce_stt_results(files))
        consumer = asyncio.create_task(self._consume_forensic())

        try:
            async for result in consumer:
                yield result
        finally:
            self._stop_event.set()
            await producer

    async def _produce_stt_results(self, files: List[Path]) -> None:
        """STT 결과 생성 (Producer)."""
        for file in files:
            if self._stop_event.is_set():
                break

            # Backpressure: 큐가 가득 차면 대기
            stt_result = await self.stt_service.process_audio(str(file))
            await self._queue.put((file, stt_result))

        # 종료 신호
        await self._queue.put(None)

    async def _consume_forensic(self) -> AsyncIterator[ForensicResult]:
        """Forensic 분석 수행 (Consumer)."""
        while True:
            item = await self._queue.get()
            if item is None:
                break

            file, stt_result = item
            forensic_result = await self.forensic_service.analyze(
                file, stt_result
            )
            yield forensic_result
```

---

### 4.3 Phase 3 완료 기준

| Metric | Phase 2 | Target | Measurement |
|--------|---------|--------|-------------|
| 처리 시간 (183파일) | 4-5 시간 | 3-4 시간 | E2E 테스트 |
| GPU 활용률 (Forensic) | > 80% | > 90% | nvidia-smi |
| 파이프라인 효율 | 순차 | 50% 오버랩 | 로그 분석 |

---

## 5. Technical Approach

### 5.1 Architecture Design

```
┌────────────────────────────────────────────────────────────────┐
│                     Forensic Pipeline                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Input     │───▶│    STT      │───▶│  Forensic   │        │
│  │   Files     │    │   Stage     │    │   Stage     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                            │                  │                │
│                            ▼                  ▼                │
│                    ┌─────────────────────────────────┐        │
│                    │   Unified Memory Manager        │        │
│                    │   (30GB threshold, zero-copy)   │        │
│                    └─────────────────────────────────┘        │
│                            │                  │                │
│                            ▼                  ▼                │
│                    ┌─────────────────────────────────┐        │
│                    │     Thermal Manager             │        │
│                    │   (80°C throttle, 85°C stop)    │        │
│                    └─────────────────────────────────┘        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
1. Input Files (183 M4A/WAV/MP3)
       │
       ▼
2. Audio Conversion (if needed)
       │
       ▼
3. STT Stage (WhisperX GPU)
   - Transcription
   - Alignment
   - Diarization
       │
       ▼
4. Forensic Stage (GPU Optimized)
   - SER Analysis (GPU cached)
   - Audio Features (GPU)
   - Cross Validation
   - Forensic Scoring
       │
       ▼
5. Report Generation (HTML/PDF)
```

### 5.3 Memory Management Strategy

```
Total Memory: 128GB (Unified)

Allocation:
├── STT Models:      20GB (WhisperX + Pyannote)
├── Forensic Models: 15GB (SER Primary + Secondary)
├── Cache Buffer:    30GB (Result cache, threshold)
├── OS/System:       10GB
└── Available:       53GB (Dynamic allocation)

Strategy:
1. Models loaded once at session start
2. Cache cleared only at batch boundaries
3. empty_cache() called once per batch
4. OOM triggers batch size reduction
```

### 5.4 Thermal Management Strategy

```
Temperature Zones:
├── Normal (< 70°C):  Full speed processing
├── Warning (70-80°C): Monitoring active
├── Throttle (80-85°C): 50% batch size, +2s delay
└── Critical (> 85°C): Processing halt

Hysteresis:
- Throttle ON:  80°C (rising)
- Throttle OFF: 70°C (falling)
- 10°C hysteresis prevents oscillation
```

---

## 6. Risks and Mitigation

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SER GPU 정확도 저하 | Low | High | CPU 폴백, 정확도 테스트 |
| Memory OOM | Medium | High | 동적 임계값, 배치 조정 |
| Thermal Throttling | Medium | Medium | 보수적 임계값, 쿨다운 |
| Pipeline Deadlock | Low | High | 타임아웃, 백프레셔 |

### 6.2 Contingency Plans

**Plan A (Primary):** 전체 Phase 구현으로 4배 향상
**Plan B (Fallback):** Phase 1만 구현으로 2-3배 향상
**Plan C (Minimal):** 메모리 임계값만 수정으로 1.5배 향상

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/unit/test_ser_gpu.py
def test_ser_gpu_inference():
    """GPU 추론 정확도 테스트."""

def test_ser_model_caching():
    """모델 캐싱 동작 테스트."""

def test_ser_memory_cleanup():
    """메모리 정리 테스트."""
```

### 7.2 Integration Tests

```python
# tests/integration/test_forensic_pipeline.py
def test_full_pipeline_gpu():
    """전체 파이프라인 GPU 테스트."""

def test_thermal_throttling():
    """열 쓰로틀링 동작 테스트."""

def test_memory_management():
    """메모리 관리 통합 테스트."""
```

### 7.3 Performance Tests

```python
# tests/performance/test_benchmark.py
def test_single_file_performance():
    """단일 파일 성능 테스트."""

def test_batch_performance():
    """배치 성능 테스트."""

def test_memory_efficiency():
    """메모리 효율 테스트."""
```

---

## 8. Documentation

### 8.1 Code Documentation

- 모든 신규 클래스/함수에 docstring 추가
- 복잡한 알고리즘에 주석 추가
- Type hints 완전 적용

### 8.2 User Documentation

- README.md 업데이트 (성능 최적화 옵션)
- 환경 변수 설정 가이드
- 트러블슈팅 가이드

### 8.3 Developer Documentation

- 아키텍처 다이어그램
- 데이터 흐름 문서
- API 레퍼런스

---

## 9. Dependencies

### 9.1 External Dependencies

```toml
# pyproject.toml 추가
nvidia-ml-py = ">=12.560.30"  # GPU 모니터링
```

### 9.2 Internal Dependencies

- SPEC-FORENSIC-001: SER 서비스 인터페이스
- SPEC-WHISPERX-001: WhisperX 파이프라인
- SPEC-EDGEXPERT-001: UnifiedMemoryManager, ThermalManager (재사용)

---

## 10. Rollback Plan

### 10.1 Phase 1 Rollback

```bash
# 메모리 임계값 원복
git checkout HEAD~1 -- src/voice_man/config/whisperx_config.py

# SER GPU 비활성화
export SER_DEVICE=cpu
```

### 10.2 Phase 2 Rollback

```bash
# 신규 파일 제거
rm -f src/voice_man/services/forensic/memory_manager.py
rm -f src/voice_man/services/forensic/thermal_manager.py
```

### 10.3 Feature Flags

```python
# 기능 플래그로 점진적 활성화
FEATURE_FLAGS = {
    "gpu_ser": True,           # Phase 1
    "unified_memory": False,    # Phase 2
    "thermal_manager": False,   # Phase 2
    "pipeline_overlap": False,  # Phase 3
}
```

---

## 11. Traceability

### Related Documents

- `SPEC-PERFOPT-001/spec.md`: 요구사항 명세
- `SPEC-PERFOPT-001/acceptance.md`: 인수 기준
- `SPEC-FORENSIC-001`: 포렌식 시스템 기반 SPEC
- `SPEC-EDGEXPERT-001`: EdgeXpert 최적화 SPEC

### Related Code

```
src/voice_man/
├── config/
│   ├── whisperx_config.py        # 수정
│   └── batch_config.py           # 신규 (Phase 2)
├── services/
│   └── forensic/
│       ├── ser_service.py        # 수정
│       ├── memory_manager.py     # 신규 (Phase 2)
│       ├── thermal_manager.py    # 신규 (Phase 2)
│       └── pipeline_orchestrator.py  # 신규 (Phase 3)
└── e2e_test_service.py           # 수정
```

---

**문서 끝**
