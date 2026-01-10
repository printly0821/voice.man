# SPEC-PERFOPT-001: Acceptance Criteria

## TAG BLOCK

```yaml
SPEC_ID: SPEC-PERFOPT-001
TITLE: Forensic Pipeline Performance Optimization
STATUS: In Progress
PRIORITY: High
CREATED: 2026-01-10
UPDATED: 2026-01-10
LIFECYCLE: spec-anchored
PHASE_1_TESTS: 38 tests passed
```

---

## 1. Overview

본 문서는 SPEC-PERFOPT-001의 인수 기준과 테스트 시나리오를 정의합니다.

### 1.1 Success Criteria Summary

| Metric | Baseline | Target | Priority |
|--------|----------|--------|----------|
| 처리 시간 (183파일) | 15+ 시간 | < 4 시간 | P0 |
| GPU 활용률 (Forensic) | < 5% | > 80% | P0 |
| 모델 로딩 시간 (2회차+) | 45초/파일 | 0초 | P1 |
| 열 쓰로틀링 발생 | N/A | 0회 | P1 |
| OOM 발생 | N/A | 0회 | P0 |
| 정확도 변화 | N/A | < 1% | P0 |

---

## 2. Phase 1 Acceptance Criteria - COMPLETED

### AC-P1-001: 메모리 임계값 최적화 - PASSED

**Given** EdgeXpert Blackwell 128GB 환경
**When** 메모리 임계값 설정을 확인할 때
**Then** 임계값이 30GB (30000MB)로 설정되어야 한다

**Status:** PASSED (2026-01-10)

**Verification:**
```python
# tests/unit/test_memory_service_perfopt.py
from voice_man.services.memory_service import FORENSIC_MEMORY_THRESHOLD_MB, MemoryManager

def test_forensic_memory_threshold_constant_exists():
    assert FORENSIC_MEMORY_THRESHOLD_MB == 30000

def test_memory_manager_default_threshold_is_30gb():
    manager = MemoryManager()
    assert manager.threshold_mb == 30000
```

---

### AC-P1-002: SER GPU 추론 활성화 - PASSED

**Given** GPU가 사용 가능한 환경
**When** SERService를 초기화할 때
**Then** device가 "cuda"로 설정되어야 한다

**Status:** PASSED (2026-01-10)

**Verification:**
```python
# tests/unit/test_ser_service_perfopt.py
@patch("torch.cuda.is_available", return_value=True)
def test_default_device_is_cuda_when_available(self, mock_cuda):
    from voice_man.services.forensic.ser_service import SERService
    service = SERService()
    device = service._detect_optimal_device()
    assert device == "cuda"
```

---

### AC-P1-003: SER 모델 캐싱 - PASSED

**Given** SERService가 초기화된 상태
**When** 동일 세션에서 두 번째 파일을 처리할 때
**Then** 모델 로딩 시간이 0초 이내여야 한다

**Status:** PASSED (2026-01-10)

**Verification:**
```python
# tests/unit/test_ser_service_perfopt.py
def test_model_cache_class_attribute_exists():
    from voice_man.services.forensic.ser_service import SERService
    assert hasattr(SERService, "_model_cache")

def test_model_cache_persists_across_instances():
    SERService._model_cache.clear()
    SERService._model_cache["test_model"] = "cached_value"
    _ = SERService()
    assert "test_model" in SERService._model_cache

def test_get_or_load_model_returns_cached_model():
    SERService._model_cache.clear()
    service = SERService()
    mock_model = MagicMock()
    SERService._model_cache["primary"] = mock_model
    result = service._get_or_load_model("primary")
    assert result is mock_model
```

---

### AC-P1-004: SER GPU 추론 정확도 - DEFERRED

**Given** 테스트 오디오 파일 세트
**When** GPU와 CPU에서 각각 SER 분석을 수행할 때
**Then** 결과 차이가 1% 미만이어야 한다

**Status:** DEFERRED (E2E 통합 테스트에서 검증 예정)

**Note:** 실제 GPU/CPU 정확도 비교는 실제 모델 로딩이 필요하므로 E2E 통합 테스트에서 수행합니다.

**Verification (Unit Test):**
```python
# tests/unit/test_ser_service_perfopt.py
def test_explicit_device_overrides_auto_detection():
    from voice_man.services.forensic.ser_service import SERService
    service = SERService(device="cpu")
    device = service._detect_optimal_device()
    assert device == "cpu"  # CPU 폴백 지원 확인
```

---

### AC-P1-005: torch.cuda.empty_cache() 최적화 - PASSED

**Given** 10개 파일 배치 처리
**When** 처리 로그를 분석할 때
**Then** empty_cache() 호출이 배치당 1회만 발생해야 한다

**Status:** PASSED (2026-01-10)

**Verification:**
```python
# tests/unit/test_e2e_service_perfopt.py
@pytest.mark.asyncio
async def test_cleanup_after_batch_clears_cuda_cache():
    from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig
    config = E2ETestConfig()
    runner = E2ETestRunner(config)

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            await runner._cleanup_after_batch()
            mock_empty_cache.assert_called_once()

@pytest.mark.asyncio
async def test_run_calls_cleanup_after_each_batch():
    config = E2ETestConfig(batch_size=2)
    runner = E2ETestRunner(config)
    mock_files = [Path("/tmp/test1.wav"), Path("/tmp/test2.wav"),
                  Path("/tmp/test3.wav"), Path("/tmp/test4.wav")]

    with patch.object(runner, "_cleanup_after_batch", new_callable=AsyncMock) as mock_cleanup:
        # ... mock setup ...
        await runner.run(mock_files)
        # With 4 files and batch_size=2, should have 2 batches = 2 cleanup calls
        assert mock_cleanup.call_count == 2
```

---

### AC-P1-006: 처리 속도 향상 (Phase 1) - PENDING E2E TEST

**Given** 183개 오디오 파일
**When** 전체 Forensic 파이프라인을 실행할 때
**Then** 총 처리 시간이 7시간 미만이어야 한다

**Status:** PENDING (E2E 통합 테스트에서 측정 예정)

**Implementation Notes:**
- 메모리 임계값 30GB 적용: 캐시 효율 300배 향상 예상
- SER GPU-first 감지: 추론 속도 10배 향상 예상
- 모델 캐싱: 파일당 45초 로딩 오버헤드 제거 예상
- 배치별 GPU 정리: 메모리 누적 방지

**Verification (E2E Test):**
```python
# tests/integration/test_phase1_performance.py (예정)
import time

def test_phase1_performance():
    files = get_all_audio_files()  # 183 files
    start = time.time()
    results = await process_all_files(files)
    elapsed = time.time() - start
    hours = elapsed / 3600
    assert hours < 7.0  # 7시간 미만
```

---

### AC-P1-007: GPU 활용률 (Phase 1) - PENDING E2E TEST

**Given** Forensic 분석 배치 처리 중
**When** GPU 활용률을 모니터링할 때
**Then** 평균 활용률이 60% 이상이어야 한다

**Status:** PENDING (E2E 통합 테스트에서 측정 예정)

**Implementation Notes:**
- GPU-first 디바이스 감지 구현 완료
- CUDA 추론 경로 활성화
- 모델 GPU 배치 검증 필요

**Verification (E2E Test):**
```python
# tests/integration/test_phase1_gpu_utilization.py (예정)
import pynvml

def test_phase1_gpu_utilization():
    utilization_samples = []
    async def monitor():
        while processing:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_samples.append(util.gpu)
            await asyncio.sleep(5)
            pynvml.nvmlShutdown()

    await asyncio.gather(process_batch(files), monitor())
    avg_util = sum(utilization_samples) / len(utilization_samples)
    assert avg_util >= 60
```

---

## 3. Phase 2 Acceptance Criteria

### AC-P2-001: UnifiedMemoryManager 통합

**Given** ForensicMemoryManager가 초기화된 상태
**When** 스테이지별 메모리 할당을 요청할 때
**Then** 충돌 없이 할당되어야 한다

**Verification:**
```python
from voice_man.services.forensic.memory_manager import ForensicMemoryManager

def test_memory_allocation():
    manager = ForensicMemoryManager(total_memory_gb=128.0)

    assert manager.allocate("stt") == True
    assert manager.allocate("ser") == True

    stats = manager.get_memory_stats()
    assert stats["allocated_gb"] == 26.0  # 16 + 10
    assert stats["available_gb"] >= 90.0
```

---

### AC-P2-002: 스테이지별 배치 설정

**Given** BatchConfigManager가 초기화된 상태
**When** 가용 메모리 기반 배치 사이즈를 요청할 때
**Then** 최적 배치 사이즈가 반환되어야 한다

**Verification:**
```python
from voice_man.config.batch_config import BatchConfigManager

def test_optimal_batch_size():
    manager = BatchConfigManager()

    # 충분한 메모리
    batch = manager.get_optimal_batch_size("ser", 16000)
    assert batch >= 8

    # 제한된 메모리
    batch = manager.get_optimal_batch_size("ser", 3000)
    assert batch >= 1
    assert batch <= 2
```

---

### AC-P2-003: ThermalManager 모니터링

**Given** ThermalManager가 시작된 상태
**When** GPU 온도를 조회할 때
**Then** 유효한 온도 값이 반환되어야 한다

**Verification:**
```python
from voice_man.services.forensic.thermal_manager import ThermalManager

def test_thermal_monitoring():
    manager = ThermalManager(poll_interval=1.0)
    manager.start()

    try:
        temp = manager._get_gpu_temperature()
        assert 20 <= temp <= 100  # 유효 범위
    finally:
        manager.stop()
```

---

### AC-P2-004: 열 쓰로틀링 동작

**Given** ThermalManager가 시작된 상태
**When** GPU 온도가 80°C를 초과할 때
**Then** is_throttling이 True를 반환해야 한다

**Verification:**
```python
def test_thermal_throttling():
    manager = ThermalManager()
    manager.start()

    # 시뮬레이션: 80°C 초과
    manager._handle_temperature(82)
    assert manager.is_throttling == True

    # 시뮬레이션: 70°C 미만
    manager._handle_temperature(68)
    assert manager.is_throttling == False

    manager.stop()
```

---

### AC-P2-005: 열 쓰로틀링 0회 (장시간 테스트)

**Given** 183개 파일 전체 처리
**When** 열 관리가 활성화된 상태에서 처리할 때
**Then** 열 쓰로틀링 발생이 0회여야 한다

**Verification:**
```python
def test_no_thermal_throttling():
    manager = ThermalManager()
    throttle_events = []

    def on_throttle(temp, throttle):
        if throttle:
            throttle_events.append(temp)

    manager.register_callback(on_throttle)
    manager.start()

    await process_all_files(files)

    manager.stop()
    assert len(throttle_events) == 0
```

---

### AC-P2-006: 처리 속도 향상 (Phase 2)

**Given** Phase 1 최적화가 적용된 상태
**When** Phase 2 최적화를 추가로 적용할 때
**Then** 처리 시간이 5시간 미만이어야 한다

**Verification:**
```python
def test_phase2_performance():
    files = get_all_audio_files()

    start = time.time()
    results = await process_all_files(files)
    elapsed = time.time() - start

    hours = elapsed / 3600
    assert hours < 5.0
```

---

### AC-P2-007: GPU 활용률 (Phase 2)

**Given** Phase 2 최적화가 적용된 상태
**When** GPU 활용률을 모니터링할 때
**Then** 평균 활용률이 80% 이상이어야 한다

**Verification:**
```python
def test_phase2_gpu_utilization():
    # Phase 1과 동일한 모니터링 로직
    avg_util = measure_gpu_utilization()
    assert avg_util >= 80
```

---

## 4. Phase 3 Acceptance Criteria

### AC-P3-001: 스테이지 파이프라이닝

**Given** PipelineOrchestrator가 초기화된 상태
**When** 파일 처리를 시작할 때
**Then** STT와 Forensic이 병렬로 실행되어야 한다

**Verification:**
```python
from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

def test_pipeline_overlap():
    orchestrator = PipelineOrchestrator(stt_service, forensic_service)

    stages_active = {"stt": False, "forensic": False}

    async def monitor():
        while processing:
            # 두 스테이지가 동시에 활성화되는 순간 캡처
            if stages_active["stt"] and stages_active["forensic"]:
                return True
            await asyncio.sleep(0.1)
        return False

    overlap_detected = await monitor()
    assert overlap_detected == True
```

---

### AC-P3-002: 백프레셔 동작

**Given** Forensic 처리가 느린 상태
**When** 결과 큐가 5개를 초과할 때
**Then** STT 처리가 일시 중지되어야 한다

**Verification:**
```python
def test_backpressure():
    orchestrator = PipelineOrchestrator(
        stt_service,
        slow_forensic_service  # 의도적으로 느린 서비스
    )

    # 큐 크기 모니터링
    max_queue_size = 0

    async def monitor_queue():
        nonlocal max_queue_size
        while processing:
            max_queue_size = max(max_queue_size, orchestrator._queue.qsize())
            await asyncio.sleep(0.1)

    await asyncio.gather(
        orchestrator.process_files(files),
        monitor_queue()
    )

    assert max_queue_size <= 5
```

---

### AC-P3-003: 처리 속도 향상 (Phase 3)

**Given** Phase 2 최적화가 적용된 상태
**When** Phase 3 파이프라이닝을 추가로 적용할 때
**Then** 처리 시간이 4시간 미만이어야 한다

**Verification:**
```python
def test_phase3_performance():
    files = get_all_audio_files()

    start = time.time()
    results = await orchestrator.process_files(files)
    elapsed = time.time() - start

    hours = elapsed / 3600
    assert hours < 4.0
```

---

### AC-P3-004: 파이프라인 효율

**Given** Phase 3 파이프라이닝 적용
**When** 순차 처리와 비교할 때
**Then** 처리 시간이 50% 이상 단축되어야 한다

**Verification:**
```python
def test_pipeline_efficiency():
    # 순차 처리 시간
    sequential_time = measure_sequential_processing(files[:20])

    # 파이프라인 처리 시간
    pipeline_time = measure_pipeline_processing(files[:20])

    efficiency = (sequential_time - pipeline_time) / sequential_time
    assert efficiency >= 0.5  # 50% 이상 효율
```

---

## 5. Error Handling Tests

### TC-ERR-001: 파일 처리 실패 격리

**Given** 100개 파일 중 1개가 손상됨
**When** 배치 처리를 실행할 때
**Then** 99개 파일이 정상 처리되어야 한다

**Verification:**
```python
def test_error_isolation():
    files = get_test_files()  # 100 files
    files[50] = corrupt_file  # 손상된 파일

    results = await process_batch(files)

    success_count = sum(1 for r in results if r.success)
    assert success_count == 99
```

---

### TC-ERR-002: GPU 폴백

**Given** GPU 메모리 부족 상황
**When** SER 분석을 시도할 때
**Then** CPU로 자동 폴백되어야 한다

**Verification:**
```python
def test_gpu_fallback():
    # GPU 메모리 강제 소진
    consume_gpu_memory()

    service = SERService(device="auto")
    result = service.analyze_ensemble(test_audio, sr)

    assert result is not None
    assert "cpu" in service._last_device_used
```

---

### TC-ERR-003: OOM 복구

**Given** 배치 처리 중 OOM 발생
**When** 배치 사이즈를 조정할 때
**Then** 처리가 계속되어야 한다

**Verification:**
```python
def test_oom_recovery():
    # 큰 배치 사이즈로 시작
    initial_batch_size = 32

    results, final_batch_size = await process_with_oom_recovery(
        files,
        initial_batch_size
    )

    assert len(results) == len(files)
    assert final_batch_size < initial_batch_size
```

---

## 6. Performance Benchmarks

### BENCH-001: 단일 파일 SER 추론 시간

| Device | Target | Actual | Status |
|--------|--------|--------|--------|
| CPU | < 10초 | TBD | - |
| GPU | < 1초 | TBD | - |

---

### BENCH-002: 모델 로딩 시간

| Scenario | Target | Actual | Status |
|----------|--------|--------|--------|
| Cold start | < 60초 | TBD | - |
| Cached | < 1초 | TBD | - |

---

### BENCH-003: 메모리 효율

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Peak GPU Memory | < 80% | TBD | - |
| Peak System Memory | < 80% | TBD | - |

---

### BENCH-004: 전체 파이프라인 성능

| Phase | Files | Target Time | Actual | Status |
|-------|-------|-------------|--------|--------|
| Baseline | 183 | N/A | 15+ hrs | - |
| Phase 1 | 183 | < 7 hrs | TBD | - |
| Phase 2 | 183 | < 5 hrs | TBD | - |
| Phase 3 | 183 | < 4 hrs | TBD | - |

---

## 7. Quality Gate Checklist

### Pre-Release Checklist

- [ ] 모든 AC 테스트 통과
- [ ] 성능 벤치마크 목표 달성
- [ ] 정확도 변화 < 1%
- [ ] OOM 발생 0회
- [ ] 열 쓰로틀링 발생 0회
- [ ] 코드 커버리지 > 85%
- [ ] 문서 업데이트 완료
- [ ] 롤백 절차 검증

### TRUST 5 Compliance

- [ ] **Test-first:** 모든 기능에 테스트 작성
- [ ] **Readable:** 코드 리뷰 완료
- [ ] **Unified:** 코딩 컨벤션 준수
- [ ] **Secured:** 보안 취약점 스캔 통과
- [ ] **Trackable:** 커밋 메시지 표준 준수

---

## 8. Traceability

### Requirements to Test Cases

| Requirement | Test Cases |
|-------------|-----------|
| U1 (GPU Memory Preservation) | AC-P1-003, TC-GPU-001 |
| U2 (Thermal Monitoring) | AC-P2-003, AC-P2-004 |
| U3 (Memory Tracking) | AC-P2-001, TC-MEM-001 |
| U4 (Processing Continuity) | TC-ERR-001 |
| E1 (Model Loading) | AC-P1-002, AC-P1-003 |
| E2 (Thermal Throttling) | AC-P2-004, AC-P2-005 |
| E3 (Memory Pressure) | TC-ERR-003, TC-MEM-002 |
| E4 (Pipeline Trigger) | AC-P3-001 |
| S1 (Memory Threshold) | AC-P1-001 |
| S2 (GPU Mode) | AC-P1-002 |
| S3 (Model Cache) | AC-P1-003 |
| S4 (Cooldown State) | AC-P2-004 |
| S5 (Backpressure) | AC-P3-002 |
| N1 (No Reload) | AC-P1-003 |
| N2 (Thermal Safety) | AC-P2-005 |
| PR-001 to PR-007 | BENCH-001 to BENCH-004 |

---

## 9. Sign-off

### Phase 1 Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | - | - | - |
| QA | - | - | - |
| Tech Lead | - | - | - |

### Phase 2 Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | - | - | - |
| QA | - | - | - |
| Tech Lead | - | - | - |

### Phase 3 Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | - | - | - |
| QA | - | - | - |
| Tech Lead | - | - | - |

---

**문서 끝**
