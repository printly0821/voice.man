---
id: SPEC-E2ETEST-001
type: plan
version: "1.0.0"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
status: "planned"
related_spec: SPEC-E2ETEST-001
---

# SPEC-E2ETEST-001: 구현 계획

## TRACEABILITY
- **SPEC**: [SPEC-E2ETEST-001](./spec.md)
- **Acceptance**: [acceptance.md](./acceptance.md)

---

## 구현 개요

### 목적
183개 m4a 오디오 파일을 GPU 병렬 배치 처리로 WhisperX 전체 파이프라인(전사, 정렬, 화자 분리)을 실행하고, 성능 목표 달성 여부를 검증하는 E2E 통합 테스트를 구현합니다.

### 핵심 산출물
1. **E2E 테스트 스크립트**: `scripts/e2e_batch_test.py`
2. **E2E 테스트 서비스**: `src/voice_man/services/e2e_test_service.py`
3. **E2E 테스트 케이스**: `tests/e2e/test_full_batch_processing.py`
4. **종합 리포트**: `ref/call/reports/e2e_test_report.json`, `e2e_test_summary.md`

---

## 마일스톤

### Phase 1: 테스트 환경 및 스크립트 작성
**우선순위**: Primary Goal (최우선)

**작업 내용**:
1. E2E 테스트 디렉토리 구조 생성
   - `tests/e2e/` 디렉토리 생성
   - `ref/call/reports/results/` 디렉토리 생성

2. E2E 테스트 스크립트 작성 (`scripts/e2e_batch_test.py`)
   - 명령줄 인자 파싱 (argparse)
   - 파일 목록 수집 로직
   - BatchProcessor 초기화
   - 진행률 콜백 연결
   - 결과 저장 로직

3. E2E 테스트 서비스 작성 (`src/voice_man/services/e2e_test_service.py`)
   - `E2ETestRunner` 클래스 구현
   - 파일 목록 관리
   - 배치 처리 오케스트레이션
   - 결과 수집 및 집계
   - 리포트 생성 위임

**의존성**: SPEC-PARALLEL-001 (BatchProcessor), SPEC-WHISPERX-001 (WhisperXPipeline)

**검증 기준**:
- 스크립트 실행 가능 확인
- 단일 파일 테스트 성공
- 10개 파일 배치 테스트 성공

---

### Phase 2: 병렬 배치 처리 실행
**우선순위**: Secondary Goal

**작업 내용**:
1. BatchProcessor 통합
   - 기존 BatchProcessor 클래스 활용
   - GPU 배치 크기 설정 (기본값: 20)
   - 동적 배치 크기 조정 활성화

2. WhisperX 파이프라인 연결
   - WhisperXPipeline 인스턴스 생성
   - 오디오 포맷 변환 (m4a → WAV)
   - 전사, 정렬, 화자 분리 순차 실행

3. 진행률 콜백 구현
   - tqdm 통합
   - 콘솔 출력 (파일명, 진행률, ETA)
   - 로그 기록

4. 실패 처리 및 재시도
   - 재시도 큐 관리
   - 지수 백오프 (5초, 15초, 30초)
   - 최대 3회 재시도

**의존성**: Phase 1 완료

**검증 기준**:
- 50개 파일 배치 처리 성공
- GPU 메모리 95% 미만 유지
- 재시도 로직 동작 확인

---

### Phase 3: 결과 검증 및 리포트 생성
**우선순위**: Final Goal

**작업 내용**:
1. 결과 저장 구현
   - 파일별 JSON 결과 저장
   - 메타데이터 포함 (처리 시간, 상태)
   - 화자 정보 저장

2. 종합 리포트 생성
   - JSON 리포트 (`e2e_test_report.json`)
   - Markdown 요약 (`e2e_test_summary.md`)
   - 성능 통계 계산
   - 화자별 발화 통계

3. 실패 파일 처리
   - `failed_files.json` 생성
   - 실패 원인 상세 기록
   - 재처리 가이드 제공

4. 183개 전체 파일 실행
   - 전체 파일 배치 처리
   - 성능 목표 검증 (20분 이내)
   - 결과 정확성 검증

**의존성**: Phase 2 완료

**검증 기준**:
- 183개 파일 전체 처리 완료
- 처리 시간 20분 이내
- 종합 리포트 생성 확인
- 실패 파일 0개 (재시도 후)

---

## 기술적 접근 방법

### 아키텍처 설계

```
┌─────────────────────────────────────────────────────────────┐
│                 E2E Test Script (CLI)                       │
│                 scripts/e2e_batch_test.py                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 E2E Test Service                            │
│                 services/e2e_test_service.py                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  - FileCollector: 파일 목록 수집                       │ │
│  │  - BatchOrchestrator: 배치 처리 조율                   │ │
│  │  - ResultAggregator: 결과 집계                         │ │
│  │  - ReportGenerator: 리포트 생성 위임                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────────┐ ┌─────────────────────────────────────┐
│   BatchProcessor    │ │         WhisperXPipeline            │
│ (SPEC-PARALLEL-001) │ │        (SPEC-WHISPERX-001)          │
│  - GPU 배치 처리    │ │  - 전사 (STT)                       │
│  - 동적 배치 조정   │ │  - 정렬 (WAV2VEC2)                  │
│  - 메모리 모니터링  │ │  - 화자 분리 (Pyannote)             │
└─────────────────────┘ └─────────────────────────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Results Storage                         │
│                 ref/call/reports/                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  - e2e_test_report.json: 종합 리포트                   │ │
│  │  - e2e_test_summary.md: 요약 리포트                    │ │
│  │  - failed_files.json: 실패 파일 목록                   │ │
│  │  - results/*.json: 파일별 결과                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 클래스 설계

#### E2ETestRunner 클래스
```python
class E2ETestRunner:
    """E2E 통합 테스트 러너"""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        batch_size: int = 20,
        language: str = "ko",
        num_speakers: int | None = None
    ):
        """초기화"""

    def collect_files(self) -> list[Path]:
        """m4a 파일 목록 수집"""

    def run(
        self,
        progress_callback: Callable | None = None
    ) -> E2ETestResult:
        """전체 E2E 테스트 실행"""

    def generate_report(self, result: E2ETestResult) -> Path:
        """종합 리포트 생성"""
```

#### E2ETestResult 데이터 클래스
```python
@dataclass
class E2ETestResult:
    """E2E 테스트 결과"""
    total_files: int
    success_count: int
    failed_count: int
    total_duration_seconds: float
    avg_file_duration_seconds: float
    gpu_memory_peak_mb: float
    gpu_utilization_avg_percent: float
    file_results: list[FileProcessingResult]
    failed_files: list[FailedFile]
    speaker_statistics: dict[str, SpeakerStats]
```

### 진행률 콜백 인터페이스

```python
ProgressCallback = Callable[
    [int, int, float, str, str],  # current, total, elapsed, filename, status
    None
]

def default_progress_callback(
    current: int,
    total: int,
    elapsed_seconds: float,
    current_file: str,
    status: str
) -> None:
    """기본 진행률 콜백 (tqdm 통합)"""
    eta = (elapsed_seconds / current) * (total - current) if current > 0 else 0
    print(f"[{current}/{total}] {current_file} - {status} (ETA: {eta:.1f}s)")
```

---

## 리스크 대응 계획

### GPU 메모리 부족 대응
1. **모니터링**: 1초 간격 GPU 메모리 체크
2. **사전 조치**: 80% 도달 시 배치 크기 50% 감소
3. **캐시 정리**: 95% 도달 시 `torch.cuda.empty_cache()` 호출
4. **폴백**: 연속 OOM 시 순차 모델 로딩 모드 전환

### 처리 시간 초과 대응
1. **실시간 모니터링**: 파일당 처리 시간 추적
2. **조기 경고**: 평균 처리 시간 7초 초과 시 경고
3. **배치 크기 조정**: 속도 저하 시 배치 크기 증가 시도
4. **병목 분석**: 단계별 처리 시간 기록 및 분석

### 파일 처리 실패 대응
1. **재시도**: 지수 백오프로 최대 3회 재시도
2. **격리**: 반복 실패 파일은 별도 처리
3. **로깅**: 상세 오류 정보 기록
4. **리포트**: 실패 파일 목록 및 원인 문서화

---

## 테스트 전략

### 단위 테스트
- E2ETestRunner 클래스 메서드별 테스트
- 파일 수집 로직 테스트
- 리포트 생성 로직 테스트

### 통합 테스트
- 10개 파일 배치 처리 테스트
- 50개 파일 배치 처리 테스트
- 진행률 콜백 동작 테스트

### E2E 테스트
- 183개 전체 파일 처리 테스트
- 성능 목표 달성 검증
- 결과 정확성 검증

---

## 산출물 체크리스트

### 필수 산출물
- [ ] `scripts/e2e_batch_test.py` - E2E 테스트 스크립트
- [ ] `src/voice_man/services/e2e_test_service.py` - E2E 테스트 서비스
- [ ] `tests/e2e/test_full_batch_processing.py` - E2E 테스트 케이스
- [ ] `ref/call/reports/e2e_test_report.json` - 종합 리포트 (JSON)
- [ ] `ref/call/reports/e2e_test_summary.md` - 요약 리포트 (Markdown)

### 선택 산출물
- [ ] `ref/call/reports/failed_files.json` - 실패 파일 목록 (실패 시)
- [ ] `ref/call/reports/results/*.json` - 파일별 상세 결과

---

## 참조 문서

- [SPEC-E2ETEST-001 spec.md](./spec.md)
- [SPEC-E2ETEST-001 acceptance.md](./acceptance.md)
- [SPEC-PARALLEL-001](../SPEC-PARALLEL-001/spec.md)
- [SPEC-WHISPERX-001](../SPEC-WHISPERX-001/spec.md)

---

**문서 끝**
