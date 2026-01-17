# Multi-GPU Batch-Level Parallelization Design Document

**작성일**: 2026-01-15
**버전**: 1.0
**상태**: 설계

---

## 1. 개요 (Overview)

### 1.1 목적

현재 단일 GPU 배치 처리를 다중 GPU 환경으로 확장하기 위한 상세 설계입니다. 배치 단위 병렬화를 통해 처리량을 선형적으로 확장합니다.

### 1.2 현재 상황

| 항목 | 현재 (단일 GPU) | 목표 (다중 GPU) |
|------|----------------|----------------|
| GPU 수 | 1 | N (2-8) |
| 배치 크기 | 10 | 10 per GPU |
| 처리량 | ~24 파일/시간 | ~24×N 파일/시간 |
| 병렬 방식 | 순차 | 배치 단위 병렬 |

### 1.3 핵심 원칙

1. **Batch-level Parallelism**: 각 배치는 독립적인 GPU에서 처리
2. **Resource Isolation**: GPU당 독립적인 service 인스턴스
3. **Fault Isolation**: 단일 GPU 장애가 다른 GPU에 영향 없음
4. **Centralized Orchestration**: 중앙 코디네이터가 배치 분배 및 결과 수집

---

## 2. 아키텍처 설계

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Multi-GPU Orchestrator                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │ Batch Queue   │  │ GPU Manager   │  │ Dispatcher    │  │ Aggregator│  │
│  │ (Priority)    │  │ (Monitor)     │  │ (Assign)      │  │ (Collect) │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │  GPU Worker 0     │ │  GPU Worker 1     │ │  GPU Worker N     │
        │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
        │  │STT Service  │  │ │  │STT Service  │  │ │  │STT Service  │  │
        │  │(WhisperX)   │  │ │  │(WhisperX)   │  │ │  │(WhisperX)   │  │
        │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
        │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
        │  │Forensic Svc │  │ │  │Forensic Svc │  │ │  │Forensic Svc │  │
        │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
        │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
        │  │Local State  │  │ │  │Local State  │  │ │  │Local State  │  │
        │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │     GPU 0         │ │     GPU 1         │ │     GPU N         │
        │  (CUDA:0)         │ │  (CUDA:1)         │ │  (CUDA:N)         │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
```

### 2.2 컴포넌트 정의

#### 2.2.1 Multi-GPU Orchestrator

**책임**:
- 배치 큐 관리
- GPU 상태 모니터링
- Worker에 배치 할당
- 결과 집계 및 체크포인트 저장

```python
class MultiGPUOrchestrator:
    """Coordinates batch processing across multiple GPUs."""

    def __init__(
        self,
        audio_files: List[Path],
        batch_size: int = 10,
        gpu_devices: List[int] = None,  # [0, 1, 2, ...]
        checkpoint_dir: str = "data/checkpoints",
    ):
        self.audio_files = audio_files
        self.batch_size = batch_size
        self.gpu_devices = gpu_devices or self._detect_gpus()

        # Worker processes (one per GPU)
        self.workers: Dict[int, GPUWorker] = {}

        # Batch queue
        self.batch_queue: asyncio.Queue = asyncio.Queue()

        # Results
        self.results: List[BatchResult] = []

        # Centralized checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        try:
            import torch
            return list(range(torch.cuda.device_count()))
        except Exception:
            return [0]  # Fallback to GPU 0
```

#### 2.2.2 GPU Worker (Process)

**책임**:
- 단일 GPU에서 배치 처리
- 독립적인 service 인스턴스
- Local state management
- 결과 반환

**중요**: Worker는 **별도 Process**로 실행 (ProcessPoolExecutor)

```python
def gpu_worker_process(
    worker_id: int,
    gpu_id: int,
    batch: List[Path],
    config: WorkerConfig,
) -> BatchResult:
    """
    GPU worker process (runs in separate process).

    Each worker has:
    - Isolated GPU context
    - Independent service instances
    - Local state management
    """
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize services (process-local)
    stt_service = WhisperXService(device="cuda", language="ko")
    forensic_service = ForensicScoringService(device="cuda")

    # Process batch sequentially (safe within single GPU)
    results = []
    for audio_file in batch:
        result = process_single_file(audio_file, stt_service, forensic_service)
        results.append(result)

    return BatchResult(
        worker_id=worker_id,
        gpu_id=gpu_id,
        results=results,
    )
```

#### 2.2.3 GPU Manager

**책임**:
- GPU 상태 모니터링
- Memory usage tracking
- Temperature monitoring
- Worker 할당 결정

```python
class GPUManager:
    """Manages GPU resource allocation and monitoring."""

    def __init__(self, gpu_devices: List[int]):
        self.gpu_devices = gpu_devices
        self.gpu_status: Dict[int, GPUStatus] = {}

    def get_available_gpus(self) -> List[int]:
        """Get GPUs available for new batch processing."""
        available = []
        for gpu_id in self.gpu_devices:
            status = self._get_gpu_status(gpu_id)
            if status.memory_percent < 80 and status.temperature < 80:
                available.append(gpu_id)
        return available

    def _get_gpu_status(self, gpu_id: int) -> GPUStatus:
        """Get current GPU status."""
        try:
            import torch
            import pynvml

            # PyTorch memory info
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)

            # NVML for detailed info
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvnl.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            return GPUStatus(
                gpu_id=gpu_id,
                memory_allocated_gb=allocated,
                memory_reserved_gb=reserved,
                memory_total_gb=info.total / (1024**3),
                memory_percent=(reserved / (info.total / (1024**3))) * 100,
                temperature=temp,
                utilization=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            )
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} status: {e}")
            return GPUStatus(gpu_id=gpu_id, memory_percent=0)
```

---

## 3. 데이터 흐름

### 3.1 배치 처리 시퀀스

```
1. Orchestrator 초기화
   ├─ GPU 감지
   ├─ Worker 프로세스 풀 생성
   └─ 배치 큐 구성

2. 배치 분할
   ├─ 전체 파일 리스트 → N개 배치
   └─ 각 배치 크기 = batch_size

3. GPU 할당
   ├─ GPU Manager로 사용 가능한 GPU 확인
   └─ Worker에 GPU 할당

4. 배치 처리 (병렬)
   GPU 0: ┌─────────────────────────────────────┐
          │ Batch 1 (10 files)                  │
          │ STT → Forensic → Save               │
          └─────────────────────────────────────┘
   GPU 1: ┌─────────────────────────────────────┐
          │ Batch 2 (10 files)                  │
          │ STT → Forensic → Save               │
          └─────────────────────────────────────┘
   GPU 2: ┌─────────────────────────────────────┐
          │ Batch 3 (10 files)                  │
          │ STT → Forensic → Save               │
          └─────────────────────────────────────┘

5. 결과 집계
   ├─ Worker로부터 BatchResult 수집
   ├─ 중앙 체크포인트 저장
   └─ 진행률 업데이트

6. 다음 라운드
   └─ 사용 가능한 GPU에 다음 배치 할당
```

### 3.2 장애 처리

```
Worker Failure (GPU 0 crash):
├─ Orchestrator가 감지 (timeout or exception)
├─ Failed batch를 큐에 재삽입
├─ GPU 0 상태 확인
│  ├─ OK → 같은 GPU에 재할당
│  └─ Error → GPU 제외하고 다른 GPU에 할당
└─ 체크포인트에 실패 기록
```

---

## 4. 구현 상세

### 4.1 Process-based Isolation

**이유**: Thread가 아닌 Process를 사용하는 이유

| 항목 | Thread | Process |
|------|--------|---------|
| **GPU Context** | 공유 (충돌) | 격리 (안전) |
| **Memory** | 공유 | 독립 |
| **CUDA Safety** | ❌ 불안정 | ✅ 안전 |
| **Overhead** | 낮음 | 높음 |
| **IPC** | Direct (race condition) | Queue (safe) |

**구현**: `concurrent.futures.ProcessPoolExecutor`

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class MultiGPUProcessor:
    def __init__(self, num_gpus: int = None):
        self.num_gpus = num_gpus or self._count_gpus()
        self.executor = ProcessPoolExecutor(max_workers=self.num_gpus)

        # Use multiprocessing Manager for shared state
        self.manager = mp.Manager()
        self.progress_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()

    async def process_batches_parallel(
        self,
        batches: List[List[Path]],
    ) -> List[BatchResult]:
        """Process batches in parallel across GPUs."""
        # Assign GPU to each batch
        gpu_assignments = []
        for i, batch in enumerate(batches):
            gpu_id = i % self.num_gpus  # Round-robin assignment
            gpu_assignments.append((gpu_id, batch))

        # Submit tasks to process pool
        loop = asyncio.get_event_loop()
        tasks = []
        for gpu_id, batch in gpu_assignments:
            task = loop.run_in_executor(
                self.executor,
                gpu_worker_process,
                gpu_id,
                batch,
                self._get_worker_config(),
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Worker failed: {result}")
                # Handle failure - resubmit batch
            else:
                batch_results.append(result)

        return batch_results
```

### 4.2 Inter-Process Communication (IPC)

**안전한 IPC 패턴**:

```python
# Shared state (managed by multiprocessing.Manager)
shared_state = {
    "progress": manager.dict(),      # {batch_id: progress}
    "results": manager.list(),       # List of results
    "errors": manager.queue(),       # Error queue
}

# Per-batch config (passed as argument, immutable)
batch_config = {
    "batch_id": "batch_1",
    "files": [...],
    "gpu_id": 0,
    "checkpoint_dir": "data/checkpoints",
}

# Worker returns result (pickled)
return BatchResult(...)
```

### 4.3 Checkpoint Management

**중앙화 vs 분산 체크포인트**:

```
┌─────────────────────────────────────────────────────────────┐
│              Centralized Checkpoint Store (SQLite)          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Worker 0 ──┐                                               │
│  Worker 1 ──┼──► Checkpoint Manager (Main Process)          │
│  Worker 2 ──┘       │                                        │
│                      ▼                                        │
│              ┌───────────────────┐                          │
│              │ WorkflowStateStore│                          │
│              │ (SQLite)          │                          │
│              └───────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**동기화 문제 해결**:

1. **Write Lock**: SQLite 내장 lock 사용
2. **Retry Logic**: Busy 시 재시도
3. **Local Buffer**: Worker가 결과를 로컬에 저장 후 주기적으로 flush

```python
class SafeCheckpointWriter:
    """Thread-safe and process-safe checkpoint writer."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = mp.Lock()  # Process-safe lock
        self.retry_count = 3
        self.retry_delay = 0.1  # 100ms

    def write_checkpoint(self, batch_id: str, data: dict):
        """Write checkpoint with retry logic."""
        for attempt in range(self.retry_count):
            try:
                with self.lock:
                    # SQLite write operation
                    self._write_to_db(batch_id, data)
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
```

---

## 5. 현재 코드에서 발생할 수 있는 문제

### 5.1 Global State 공유

**문제**:

```python
# 현재 코드 (run_optimized_batch.py)
class OptimizedBatchProcessor:
    def __init__(self):
        self._stt_service = None  # Instance variable
        self._forensic_service = None

# 복수 Process에서 실행 시:
# Process 1: self._stt_service = STTService() (GPU 0)
# Process 2: self._stt_service = STTService() (GPU 1)
# ❌ 각 Process가 독립적인 인스턴스를 가짐 → 동기화 문제
```

**해결**:

```python
# Service를 Process 내에서 초기화
def gpu_worker_process(gpu_id: int, batch: List[Path]):
    # 각 Process가 독립적으로 service 초기화
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stt_service = WhisperXService(device="cuda")  # Process-local
    # ...
```

### 5.2 CUDA Context 공유

**문제**:

```python
# 단일 Process 내에서 여러 GPU 사용 시
torch.cuda.set_device(0)  # Thread 1
stt.transcribe(file1)

torch.cuda.set_device(1)  # Thread 2 (동시 실행)
stt.transcribe(file2)

# ❌ CUDA context는 Process-wide, Thread 간 공유 불가
```

**해결**: Process 단위 GPU 할당

```python
# Process 0: GPU 0 전용
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
worker_0 = Process(target=gpu_worker, args=(0, batch_1))

# Process 1: GPU 1 전용
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
worker_1 = Process(target=gpu_worker, args=(1, batch_2))
```

### 5.3 Checkpoint Database Lock

**문제**:

```python
# 여러 Worker가 동시에 SQLite에 접근
Worker 0: BEGIN; INSERT INTO files ...; COMMIT
Worker 1: BEGIN; INSERT INTO files ...; COMMIT  # ← "database is locked"
Worker 2: BEGIN; INSERT INTO files ...; COMMIT  # ← "database is locked"
```

**해결 방안**:

**Option 1: SQLite with WAL mode**
```python
conn = sqlite3.connect("checkpoints.db")
conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
```

**Option 2: Client-Server DB (PostgreSQL)**
```python
# Better for multi-process
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/checkpoints")
```

**Option 3: Queue-based aggregation**
```python
# Workers write to queue, main process writes to DB
worker_0 → result_queue
worker_1 → result_queue
worker_2 → result_queue
           ↓
    main process → SQLite (single writer)
```

### 5.4 Memory Overhead

**문제**: 각 Process가 독립적으로 모델을 로드

```
Single Process:
  WhisperX: 3GB
  Forensic: 2GB
  Total: 5GB

Multi-Process (4 GPUs):
  Process 0: 5GB
  Process 1: 5GB
  Process 2: 5GB
  Process 3: 5GB
  Total: 20GB
```

**해결**:

1. **Shared Memory**: NVIDIA MPS (Multi-Process Service)
   - 단일 GPU memory space를 여러 Process가 공유
   - 제한적 지원 (CUDA only)

2. **Model Server**: TorchServe 또는 Triton
   - 모델을 별도 server에서 로드
   - Worker가 gRPC로 요청
   - Memory sharing 가능

3. **Gradient Checkpointing**: 모델을 여러 shard로 분리

---

## 6. 권장 구현 방안

### 6.1 Phase 1: Process-based Isolation (권장)

**아키텍처**:
```
Main Process (Orchestrator)
  ├── ProcessPoolExecutor(max_workers=num_gpus)
  │   ├── Process 0 (GPU 0) → Batch 1
  │   ├── Process 1 (GPU 1) → Batch 2
  │   └── Process 2 (GPU 2) → Batch 3
  │
  └── Result Queue (mp.Manager().Queue())
      └── Collect results from workers
```

**장점**:
- GPU isolation 보장
- CUDA safety 보장
- Fault isolation

**단점**:
- Memory overhead (각 Process에 모델 로드)
- IPC overhead

### 6.2 Phase 2: Ray-based Distributed Computing (고급)

**아키텍처**:

```python
import ray
from ray.util.actor_pool import ActorPool

# Ray initialization
ray.init(num_gpus=3)

@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, gpu_id: int):
        self.stt_service = WhisperXService(device="cuda")
        self.forensic_service = ForensicScoringService(device="cuda")

    def process_batch(self, batch: List[Path]) -> BatchResult:
        # Process batch
        return result

# Create workers
workers = [GPUWorker.remote(i) for i in range(3)]

# Process batches
pool = ActorPool(workers)
results = pool.map(lambda w, b: w.process_batch.remote(b), batches)
```

**장점**:
- GPU 자동 관리
- Fault tolerance
- Scaling
- Distributed state management

**단점**:
- 추가 dependency
- 복잡성

---

## 7. 구현 로드맵

### Phase 1: 기본 Multi-GPU 지원 (2주)

**목표**: ProcessPoolExecutor 기반 기본 구현

```python
# scripts/run_multi_gpu_batch.py

@click.option(
    "--num-gpus",
    "-g",
    default=None,
    type=int,
    help="Number of GPUs to use (default: auto-detect)",
)

@click.option(
    "--worker-batch-size",
    "-b",
    default=10,
    type=int,
    help="Batch size per GPU worker",
)
```

**Deliverables**:
1. `MultiGPUOrchestrator` class
2. `gpu_worker_process` function
3. Process-safe checkpoint writer
4. GPU status monitoring

### Phase 2: 고급 기능 (2주)

**목표**: 동적 배치 할당, GPU 모니터링

**Deliverables**:
1. GPU Manager (memory, temp monitoring)
2. Dynamic batch allocation
3. Worker failure recovery
4. Progress aggregation

### Phase 3: Ray Integration (선택, 3주)

**목표**: Ray 기반 분산 처리

**Deliverables**:
1. Ray actor-based GPU workers
2. Distributed state management
3. Auto-scaling

---

## 8. 테스트 계획

### 8.1 단위 테스트

```python
def test_gpu_worker_isolation():
    """Test that workers have independent GPU contexts."""
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for gpu_id in [0, 1]:
            future = executor.submit(
                get_gpu_device_id,
                gpu_id,
            )
            results.append(future.result())

    assert results[0] == 0  # Worker 0 uses GPU 0
    assert results[1] == 1  # Worker 1 uses GPU 1
```

### 8.2 통합 테스트

```python
def test_multi_gpu_batch_processing():
    """Test end-to-end multi-gpu processing."""
    orchestrator = MultiGPUOrchestrator(
        audio_files=test_files,
        num_gpus=2,
        batch_size=5,
    )

    results = await orchestrator.run()

    assert len(results) == 2  # 2 batches
    assert results[0].gpu_id == 0
    assert results[1].gpu_id == 1
    assert results[0].successful == 5
    assert results[1].successful == 5
```

---

## 9. 성능 예측

### 9.1 이론적 처리량

| GPU 수 | 배치/GPU | 총 배치 | 처리량 (파일/시간) | 확장 비율 |
|--------|----------|---------|-------------------|----------|
| 1 | 10 | 1 | ~24 | 1.0x (baseline) |
| 2 | 10 | 2 | ~48 | 2.0x |
| 4 | 10 | 4 | ~96 | 4.0x |
| 8 | 10 | 8 | ~192 | 8.0x |

### 9.2 실제 고려사항

**Overhead 요인**:
1. **Process spawning**: ~1-2초 per worker
2. **IPC communication**: ~10-50ms per batch
3. **Checkpoint aggregation**: ~100-500ms per batch
4. **GPU warm-up**: ~5-10초 초기화

**실제 확장률**:
- 2 GPUs: ~1.8x (90% 효율)
- 4 GPUs: ~3.4x (85% 효율)
- 8 GPUs: ~6.4x (80% 효율)

---

## 10. 결론

### 10.1 핵심 요약

1. **Process-based isolation**이 필수적입니다
   - Thread는 GPU safety 문제로 사용 불가
   - ProcessPoolExecutor가 적절한 도구

2. **중앙화된 orchestrator**가 필요합니다
   - Batch queue management
   - GPU resource allocation
   - Result aggregation

3. **Checkpoint 동기화**가 주요 과제입니다
   - SQLite with WAL mode 또는 PostgreSQL 권장
   - Queue-based aggregation이 안전한 대안

### 10.2 권장 구현

**최적 접근**: Phase 1 (ProcessPoolExecutor)로 시작

```python
# Simple, effective, production-ready
from concurrent.futures import ProcessPoolExecutor

orchestrator = MultiGPUOrchestrator(
    audio_files=files,
    batch_size=10,
    num_gpus=4,  # Auto-detect
)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = orchestrator.run_parallel(executor)
```

**향상**: 필요시 Phase 3 (Ray)로 이동

---

**문서 종료**
