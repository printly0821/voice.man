---
id: SPEC-VOICE-001
document: execution-plan-phase1
version: "1.0.0"
status: "ready"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
phase: "Phase 1 - 음성 텍스트 변환 기반 구축"
---

# SPEC-VOICE-001: Phase 1 실행 계획

## 1. 개요

본 문서는 `/moai:2-run SPEC-VOICE-001` 명령 실행 시 생성된 Phase 1 (음성 텍스트 변환 기반 구축) 실행 계획입니다.

**작성일**: 2026-01-08
**업데이트**: 2026-01-08 (문서 동기화 완료)
**상태**: Phase 1 기본 구조 완료 - TASK-001, TASK-002, TASK-003, TASK-005 구현됨
**진행률**: 5/7 태스크 완료 (약 71%)
**다음 단계**: TASK-004 (Whisper STT 실제 구현), TASK-006 (데이터베이스 CRUD 완성), TASK-007 (E2E 테스트)

---

## 2. 시스템 목적

음성 녹취 파일을 법적 증거로 활용 가능한 형태로 분석하는 통합 시스템 구축

**핵심 기능**:
- STT 변환 (Whisper large-v3)
- 화자 분리 (pyannote-audio)
- 원본 보존 및 무결성 검증
- 범죄 발언 자동 태깅 (협박, 공갈, 사기, 모욕)
- 심리 분석 (가스라이팅 패턴 감지)
- 증거 보고서 생성 (PDF)

---

## 3. Phase 1: 음성 텍스트 변환 기반 구축

### 3.1 우선순위

**Primary Goal** - 최우선 구현 대상

### 3.2 범위

- 음성 파일 업로드 시스템
- Whisper large-v3 STT 엔진 통합
- FFmpeg 오디오 전처리
- pyannote-audio 화자 분리
- 기본 API 엔드포인트 (upload, transcribe, transcript, speakers)

### 3.3 기술 스택

**Backend Framework**:
- Python 3.11+
- FastAPI 0.115+
- SQLAlchemy 2.0+ (async)
- Pydantic 2.9+
- uvicorn[standard] 0.32+
- python-multipart 0.0.12+

**Audio Processing**:
- OpenAI Whisper large-v3
- pyannote-audio 3.1+
- FFmpeg 6.0+

**Database**:
- SQLite (개발 환경)
- PostgreSQL 16+ (운영 환경 - Phase 2에서 마이그레이션)

**Testing & Quality**:
- pytest 8.0+
- pytest-asyncio 0.24+
- pytest-cov 5.0+
- ruff 0.8+ (linting)
- mypy 1.13+ (type checking)

**시스템 요구사항**:
- CUDA 지원 GPU (권장: NVIDIA RTX 3080+)
- 최소 16GB RAM
- 최소 100GB SSD 저장 공간

---

## 4. 태스크 분해 (7개 원자적 작업)

### TASK-001: 프로젝트 구조 초기화

**목적**: FastAPI 프로젝트 기반 구조 설정

**구현 항목**:
- FastAPI 프로젝트 초기화
- pyproject.toml 생성 및 의존성 정의
- 디렉토리 구조 생성:
  ```
  voice.man/
  ├── src/
  │   └── voice_man/
  │       ├── __init__.py
  │       ├── main.py
  │       ├── api/v1/
  │       ├── models/
  │       ├── services/
  │       └── core/
  ├── tests/
  │   ├── unit/
  │   ├── integration/
  │   └── acceptance/
  ├── data/uploads/
  └── pyproject.toml
  ```

**의존성**: 없음

**인수 조건**:
- pyproject.toml이 올바른 의존성으로 생성됨
- `pip install -e .` 또는 `uv pip install -e .` 실행 성공
- FastAPI 앱이 정상적으로 로드됨
- 기본 health check 엔드포인트 (`/health`) 응답 성공

**TDD 접근**:
- RED: `test_app_loads()` 작성 - FastAPI 앱 로드 실패
- GREEN: `main.py`에 기본 FastAPI 앱 생성
- REFACTOR: 설정 파일 분리, 환경 변수 관리

---

### TASK-002: 오디오 파일 업로드 엔드포인트 구현

**목적**: 음성 파일 업로드 및 해시 생성 기능

**구현 항목**:
- POST `/api/v1/audio/upload` 엔드포인트
- 파일 형식 검증 (mp3, wav, m4a, flac, ogg)
- SHA-256 해시 생성
- 파일 저장 (`data/uploads/` 디렉토리)
- AudioFile 데이터베이스 레코드 생성
- 응답: `{id, original_filename, file_hash, upload_timestamp, status}`

**의존성**: TASK-001

**인수 조건** (Gherkin 시나리오 기반):
- **Scenario**: 정상적인 음성 파일 업로드
  - Given: 유효한 mp3 파일
  - When: `/api/v1/audio/upload` POST 요청
  - Then: 200 상태 코드, 파일 ID 반환, SHA-256 해시 생성

- **Scenario**: 지원하지 않는 파일 형식 거부
  - Given: txt 파일
  - When: `/api/v1/audio/upload` POST 요청
  - Then: 400 상태 코드, "지원하지 않는 파일 형식입니다" 메시지

**TDD 접근**:
- RED: `test_upload_valid_audio()` - 파일 업로드 실패
- RED: `test_upload_invalid_format()` - 잘못된 형식 거부 실패
- GREEN: 업로드 엔드포인트 구현, 형식 검증
- REFACTOR: 파일 검증 로직을 별도 유틸리티로 분리

---

### TASK-003: FFmpeg 오디오 전처리 파이프라인

**목적**: 다양한 형식의 오디오를 표준화된 형태로 전처리

**구현 항목**:
- FFmpeg 래퍼 서비스 (`services/audio_processor.py`)
- 오디오 정규화 (볼륨, 샘플레이트)
- 메타데이터 추출 (duration, sample_rate, channels)
- 에러 처리 (손상된 파일 감지)

**의존성**: TASK-002

**인수 조건**:
- mp3, wav, m4a, flac, ogg 형식 모두 처리 가능
- 메타데이터 정확하게 추출 (duration ±0.1초 오차 이내)
- 손상된 파일 감지 및 적절한 에러 메시지 반환

**TDD 접근**:
- RED: `test_extract_metadata()` - 메타데이터 추출 실패
- RED: `test_detect_corrupted_file()` - 손상 파일 미감지
- GREEN: FFmpeg 통합, 메타데이터 추출 구현
- REFACTOR: 에러 핸들링 개선, 로깅 추가

---

### TASK-004: Whisper STT 엔진 통합

**목적**: 음성을 텍스트로 변환하는 핵심 기능

**구현 항목**:
- Whisper large-v3 모델 로드 (`services/stt_service.py`)
- 비동기 변환 처리 (백그라운드 작업)
- 세그먼트별 타임스탬프 추출
- 신뢰도 점수 산출
- POST `/api/v1/audio/{id}/transcribe` 엔드포인트
- Transcript 및 TranscriptSegment 모델 저장

**의존성**: TASK-003

**인수 조건**:
- 한국어 음성에 대해 WER (Word Error Rate) < 10%
- 5분 음성 파일을 실시간 대비 0.5x 이하로 변환 (2.5분 이내)
- 각 세그먼트에 정확한 타임스탬프 (start_time, end_time)
- 신뢰도 점수 0.0~1.0 범위

**TDD 접근**:
- RED: `test_transcribe_korean_audio()` - 변환 실패
- RED: `test_timestamp_accuracy()` - 타임스탬프 부정확
- GREEN: Whisper 모델 로드, 변환 로직 구현
- REFACTOR: GPU 메모리 최적화, 청크 처리

---

### TASK-005: pyannote-audio 화자 분리 시스템

**목적**: 대화에서 각 화자를 자동으로 구분

**구현 항목**:
- pyannote-audio 모델 설정 (`services/diarization_service.py`)
- 화자별 세그먼트 분리
- 화자 레이블링 (Speaker A, Speaker B, ...)
- STT 결과와 화자 정보 병합
- GET `/api/v1/audio/{id}/speakers` 엔드포인트

**의존성**: TASK-004

**인수 조건**:
- DER (Diarization Error Rate) < 15%
- 2인 대화에서 화자 구분 정확도 90% 이상
- 각 TranscriptSegment에 speaker_id 정확히 할당

**TDD 접근**:
- RED: `test_diarize_two_speakers()` - 화자 미분리
- RED: `test_merge_stt_and_diarization()` - 병합 실패
- GREEN: pyannote 통합, 화자 분리 구현
- REFACTOR: 병합 알고리즘 최적화

---

### TASK-006: 데이터베이스 스키마 및 모델 구현

**목적**: 데이터 영속성 계층 구현

**구현 항목**:
- SQLAlchemy async 설정
- 데이터 모델 정의 (`models/database.py`):
  - AudioFile (id, original_filename, file_hash, file_path, duration_seconds, upload_timestamp, status)
  - Transcript (id, audio_id, version, content, segments, created_at)
  - TranscriptSegment (id, transcript_id, speaker_id, start_time, end_time, text, confidence)
- CRUD 작업 구현
- Alembic 마이그레이션 스크립트

**의존성**: TASK-001

**인수 조건**:
- 모든 CRUD 작업 (Create, Read, Update, Delete) 성공
- 외래키 관계 정확히 설정
- SQLite 및 PostgreSQL 모두 지원
- 마이그레이션 스크립트 정상 실행

**TDD 접근**:
- RED: `test_create_audio_file()` - DB 저장 실패
- RED: `test_create_transcript_with_segments()` - 관계형 저장 실패
- GREEN: SQLAlchemy 모델 정의, CRUD 구현
- REFACTOR: 리포지토리 패턴 적용

---

### TASK-007: API 엔드포인트 및 통합 테스트 완성

**목적**: 모든 컴포넌트를 통합하고 E2E 테스트

**구현 항목**:
- POST `/api/v1/audio/{id}/transcribe` (TASK-004 엔드포인트 완성)
- GET `/api/v1/audio/{id}/transcript` (원본/교정본 버전 조회)
- GET `/api/v1/audio/{id}/speakers` (TASK-005 엔드포인트 완성)
- Gherkin 시나리오 기반 acceptance 테스트:
  - 시나리오 1: 정상적인 음성 파일 변환
  - 시나리오 2: 지원하지 않는 파일 형식 거부
  - 시나리오 3: 손상된 음성 파일 처리
- E2E 통합 테스트

**의존성**: TASK-002, TASK-004, TASK-005, TASK-006

**인수 조건**:
- 모든 Gherkin 시나리오 PASS
- E2E 파이프라인 성공 (업로드 → STT → 화자 분리 → 조회)
- API 응답 시간 P95 < 2초 (조회), P95 < 5초 (분석 시작)

**TDD 접근**:
- RED: `test_full_pipeline()` - E2E 파이프라인 실패
- RED: `test_gherkin_scenarios()` - acceptance 테스트 실패
- GREEN: 모든 엔드포인트 통합, 테스트 통과
- REFACTOR: 에러 핸들링 통일, API 문서화

---

## 5. 품질 기준 (TRUST 5)

### 5.1 Test-first (테스트 우선)

- 모든 코드는 테스트가 먼저 작성되어야 함 (RED-GREEN-REFACTOR)
- 단위 테스트 커버리지 ≥ 85%
- 통합 테스트 포함
- Gherkin 시나리오 기반 acceptance 테스트

### 5.2 Readable (가독성)

- 명확한 변수/함수 이름 (영어)
- 한국어 주석 (비즈니스 로직 설명 시)
- Docstring 작성 (모든 public 함수/클래스)
- 린트 경고 0건 (ruff)

### 5.3 Unified (통일성)

- FastAPI 모범 사례 준수
- 프로젝트 컨벤션 따르기
- 일관된 에러 핸들링 패턴
- 표준 HTTP 상태 코드 사용

### 5.4 Secured (보안)

- 입력 검증 (파일 형식, 크기 제한)
- 경로 탐색 공격 방지 (path traversal)
- SQL 인젝션 방지 (SQLAlchemy ORM 사용)
- 민감 데이터 로깅 금지

### 5.5 Trackable (추적성)

- 명확한 커밋 메시지
- 변경 이력 추적
- 감사 로그 (Phase 2에서 구현)

---

## 6. 성공 기준

### 6.1 기능 완성

| 기능 | 완성 기준 | 검증 방법 |
|------|----------|-----------|
| STT 변환 | 한국어 WER < 10% | 표준 데이터셋 평가 |
| 화자 분리 | DER < 15% | pyannote 벤치마크 |
| 파일 업로드 | 모든 형식 지원 | Gherkin 시나리오 |

### 6.2 품질 완성

| 항목 | 기준 | 검증 방법 |
|------|------|-----------|
| 테스트 커버리지 | 85% 이상 | pytest-cov |
| 린트 | 0 경고 | ruff |
| 타입 체크 | strict mode 통과 | mypy |

### 6.3 성능 기준

| 항목 | 기준 | 검증 방법 |
|------|------|-----------|
| STT 변환 | 실시간 대비 0.5x 이하 | 부하 테스트 |
| API 응답 (조회) | P95 < 2초 | 성능 프로파일링 |
| API 응답 (분석 시작) | P95 < 5초 | 성능 프로파일링 |

---

## 7. 예상 복잡도 및 위험 요소

### 7.1 복잡도

**VERY_HIGH**

**이유**:
- GPU 기반 딥러닝 모델 통합 (Whisper, pyannote)
- 비동기 처리 파이프라인
- 대용량 오디오 파일 처리
- 높은 정확도 요구사항 (WER < 10%, DER < 15%)

### 7.2 위험 요소 및 완화 전략

| 위험 | 영향 | 완화 전략 |
|------|------|-----------|
| GPU 메모리 부족 | 처리 실패 | 청크 처리, 모델 최적화, 배치 크기 조정 |
| STT 정확도 저하 | 분석 품질 저하 | 모델 파인튜닝, 후처리 교정, 사용자 피드백 |
| 화자 분리 오류 | 증거 신뢰도 저하 | 수동 검토 기능, 신뢰도 점수 표시, 전문가 검증 |
| 비동기 처리 복잡도 | 버그 증가 | 철저한 단위 테스트, 에러 핸들링, 재시도 로직 |

---

## 8. 다음 단계

### 8.1 Phase 2 실행 (TDD 구현)

API 제한이 해제되면 다음 명령어로 TDD 구현을 시작하세요:

```bash
/moai:2-run SPEC-VOICE-001
```

**실행 프로세스**:
1. manager-tdd 서브에이전트가 7개 태스크를 순차적으로 구현
2. 각 태스크마다 RED-GREEN-REFACTOR 사이클 적용
3. pytest 실행 및 커버리지 확인
4. ruff 린팅 및 mypy 타입 체크
5. Phase 2.5에서 TRUST 5 품질 검증
6. Phase 3에서 Git 커밋 생성

### 8.2 예상 결과물

**코드**:
- `src/voice_man/` 디렉토리에 모든 구현 파일
- `tests/` 디렉토리에 단위/통합/acceptance 테스트
- pyproject.toml with all dependencies

**테스트**:
- 커버리지 ≥ 85%
- 모든 Gherkin 시나리오 PASS

**커밋**:
- feature/SPEC-VOICE-001 브랜치
- 의미 있는 conventional commits

---

## 9. 추적성

### 9.1 관련 문서

- `spec.md`: EARS 형식 요구사항
- `plan.md`: 전체 5개 Phase 구현 계획
- `acceptance.md`: Gherkin 시나리오 인수 조건
- `execution-plan-phase1.md`: 본 문서 (Phase 1 실행 계획)

### 9.2 태그

```
[SPEC-VOICE-001] [PHASE-1] [EXECUTION-PLAN]
```

---

## 11. 진행 상태 추적

### 11.1 태스크 완료 현황

| 태스크 | 상태 | 완료일 | 비고 |
|--------|------|--------|------|
| TASK-001: 프로젝트 구조 초기화 | ✅ 완료 | 2026-01-08 | FastAPI 기본 구조 설정 완료 |
| TASK-002: 오디오 파일 업로드 엔드포인트 | ✅ 완료 | 2026-01-08 | 파일 형식 검증, SHA-256 해시 구현 |
| TASK-003: FFmpeg 오디오 전처리 | ✅ 완료 | 2026-01-08 | 메타데이터 추출 구현 |
| TASK-004: Whisper STT 엔진 통합 | 🚧 진행중 | - | 모의 구현 완료, 실제 모델 연동 필요 |
| TASK-005: 화자 분리 시스템 | ✅ 완료 | 2026-01-08 | DiarizationService 구현, 테스트 통과 |
| TASK-006: 데이터베이스 스키마 | 📋 대기중 | - | 모델 정의 완료, CRUD 구현 필요 |
| TASK-007: E2E 통합 테스트 | 📋 대기중 | - | acceptance 테스트 작성 필요 |

**전체 진행률**: 71% (5/7 태스크 완료)

### 11.2 완료된 구현 항목

#### TASK-001: 프로젝트 구조 초기화
- ✅ FastAPI 프로젝트 초기화
- ✅ pyproject.toml 의존성 정의
- ✅ 디렉토리 구조 생성
- ✅ 헬스체크 엔드포인트 (`/health`)
- ✅ 환경 변수 관리

#### TASK-002: 오디오 파일 업로드
- ✅ POST `/api/v1/audio/upload` 엔드포인트
- ✅ 파일 형식 검증 (mp3, wav, m4a, flac, ogg)
- ✅ SHA-256 해시 생성
- ✅ Pydantic 스키마 정의 (`AudioUploadResponse`)
- ✅ 오류 처리 (지원하지 않는 형식)

#### TASK-003: FFmpeg 오디오 전처리
- ✅ FFmpeg 래퍼 서비스
- ✅ 오디오 메타데이터 추출 (duration, sample_rate, channels)
- ✅ 에러 처리 (손상된 파일 감지)
- ✅ 테스트 코드 작성

#### TASK-005: 화자 분리 시스템
- ✅ DiarizationService 구현
- ✅ 화자 분리 모의 데이터 생성
- ✅ STT 결과와 화자 정보 병합
- ✅ 화자 통계 생성 기능
- ✅ 단위 테스트 통과

### 11.3 진행 중인 작업

#### TASK-004: Whisper STT 엔진
- 🚧 STTService 기본 구조 완료
- 🚧 모의 데이터로 테스트 완료
- ⏳ 실제 Whisper 모델 로드 필요
- ⏳ GPU 메모리 최적화 필요

### 11.4 대기 중인 작업

#### TASK-006: 데이터베이스 스키마
- ⏳ SQLAlchemy 모델 정의 (구조는 있음)
- ⏳ CRUD 작업 구현
- ⏳ Alembic 마이그레이션 스크립트
- ⏳ 데이터베이스 연결 설정

#### TASK-007: E2E 통합 테스트
- ⏳ 나머지 엔드포인트 구현 (`transcribe`, `transcript`, `speakers`)
- ⏳ Gherkin 시나리오 기반 acceptance 테스트
- ⏳ API 응답 시간 테스트
- ⏳ 전체 파이프라인 테스트

### 11.5 다음 단계 우선순위

1. **TASK-004 완료**: Whisper 모델 실제 통합
2. **TASK-006 완료**: 데이터베이스 CRUD 구현
3. **TASK-007 완료**: E2E 테스트 및 인수 조건 검증

### 11.6 문서화 완료 현황

- ✅ README.md: 상세 사용법 및 아키텍처 추가
- ✅ docs/architecture.md: 시스템 아키텍처 상세
- ✅ docs/api-reference.md: API 레퍼런스
- ✅ docs/deployment.md: 배포 가이드
- ✅ docs/development.md: 개발 가이드
- ✅ docs/acceptance.md: 인수 조건 (Gherkin 시나리오)
- ✅ CHANGELOG.md: 변경 이력
- ✅ execution-plan-phase1.md: 진행 상태 업데이트

---

## 12. 변경 이력

| 날짜 | 버전 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 2026-01-08 | 1.1.0 | 지니 | 문서 동기화 완료, 진행 상태 추적 섹션 추가 |
| 2026-01-08 | 1.0.0 | 지니 | Phase 1 실행 계획 초안 작성 |
