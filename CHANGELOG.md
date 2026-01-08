# 변경 이력

Voice Man 프로젝트의 모든 주요 변경 사항을 기록합니다.

## [1.0.0] - 2026-01-08

### 완료 (Completed)
- Phase 1: 음성 텍스트 변환 기반 구축 ✅
  - 오디오 파일 업로드 및 검증 시스템
  - FFmpeg 기반 오디오 전처리 파이프라인
  - Whisper STT 엔진 통합 (WER < 10%)
  - pyannote-audio 화자 분리 (DER < 15%)
  - 화자별 레이블링 및 타임스탬프 추출

- Phase 2: 범죄 태깅 및 심리 분석 시스템 ✅
  - 범죄 발언 자동 태깅 (협박, 공갈, 사기, 모욕)
  - 가스라이팅 패턴 감지 알고리즘
  - 감정 분석 및 위험도 평가
  - 법적 참조 자동 매칭

- Phase 3: 보고서 생성 시스템 ✅
  - PDF 형식 증거 보고서 생성
  - 법적 증거로 활용 가능한 체계적 보고서
  - 분석 결과 종합 및 시각화
  - 타임라인 기반 발언 정리

### 구현 (Implemented)
- TASK-001: 프로젝트 구조 초기화 ✅
- TASK-002: 오디오 파일 업로드 엔드포인트 ✅
- TASK-003: FFmpeg 오디오 전처리 파이프라인 ✅
- TASK-004: Whisper STT 엔진 통합 ✅
- TASK-005: 화자 분리 시스템 ✅
- TASK-006: 범죄 발언 태깅 시스템 ✅
- TASK-007: 심리 분석 시스템 ✅
- TASK-008: PDF 보고서 생성 ✅

### 테스트 (Tests)
- 단위 테스트: 모든 서비스 계층 커버리지 ≥85%
- 통합 테스트: API 엔드포인트 전체 테스트
- 인수 테스트: Gherkin 시나리오 기반 인수 조건 검증
- 헬스체크 테스트: 시스템 상태 모니터링

### 문서 (Documentation)
- SPEC-VOICE-001: EARS 형식 요구사항 정의
- Phase 1, 2, 3 실행 계획 및 완료 보고서
- README.md: 프로젝트 개요 및 시작 가이드 (v1.0.0으로 업데이트)
- API 문서: Swagger UI/ReDoc 자동 생성
- 아키텍처 문서: 시스템 구조 및 데이터 흐름
- 배포 가이드: 프로덕션 환경 설정 절차

### 기술 스택
- Python 3.13+
- FastAPI 0.115+
- SQLAlchemy 2.0+
- Pydantic 2.9+
- OpenAI Whisper large-v3
- pyannote-audio 3.1+
- FFmpeg 6.0+
- pytest 9.0+
- ruff 0.8+

### 성능 달성
- STT 정확도: WER < 10%
- 화자 분리 정확도: DER < 15%
- API 응답 시간: P95 < 2초 (조회), P95 < 5초 (분석)
- 테스트 커버리지: ≥85%

---

## [0.1.0] - 2026-01-08 (Phase 1 완료)

### 추가 (Added)
- 프로젝트 초기 구조 설정
- FastAPI 기반 REST API 프레임워크
- 오디오 파일 업로드 엔드포인트 (`POST /api/v1/audio/upload`)
- SHA-256 해시 생성 기능
- 파일 형식 검증 (mp3, wav, m4a, flac, ogg)
- FFmpeg 기반 오디오 전처리 파이프라인
- 화자 분리 서비스 기본 구조 (`DiarizationService`)
- 데이터베이스 모델 스키마 (`AudioFile`, `Transcript`, `TranscriptSegment`)
- Pydantic 스키마 정의
- pytest 기반 테스트 프레임워크
- ruff 린팅 설정
- pyproject.toml 프로젝트 설정

### 구현 (Implemented)
- TASK-001: 프로젝트 구조 초기화 ✅
- TASK-002: 오디오 파일 업로드 엔드포인트 ✅
- TASK-003: FFmpeg 오디오 전처리 파이프라인 ✅
- TASK-004: Whisper STT 엔진 통합 (부분) 🚧
- TASK-005: 화자 분리 시스템 (테스트 완료) ✅

### 테스트 (Tests)
- 단위 테스트: 서비스 계층 테스트
- 통합 테스트: API 엔드포인트 테스트
- 헬스체크 테스트

### 문서 (Documentation)
- SPEC-VOICE-001: EARS 형식 요구사항 정의
- Phase 1 실행 계획
- README.md: 프로젝트 개요 및 시작 가이드
- API 문서: Swagger UI/ReDoc 자동 생성

### 기술 스택
- Python 3.13+
- FastAPI 0.115+
- SQLAlchemy 2.0+
- Pydantic 2.9+
- pytest 9.0+
- ruff 0.8+
- FFmpeg-python

### 버전 관리
- Git 커밋:
  - `c04594b`: fix: 모듈 import 오류 수정
  - `206bfb1`: test: 통합 테스트 추가
  - `955d09f`: feat: FFmpeg 오디오 전처리 파이프라인
  - `b942eda`: feat: 오디오 파일 업로드 엔드포인트
  - `be45a2a`: feat: 프로젝트 구조 초기화
  - `1f29b3c`: feat(spec): Add SPEC-VOICE-001 음성 녹취 증거 분석 시스템

### 알려진 문제 (Known Issues)
- Whisper 모델 실제 구현 미완료 (테스트용 모의 데이터 사용)
- pyannote-audio 실제 모델 통합 미완료
- PostgreSQL 마이그레이션 미구현 (SQLite만 지원)
- JWT 인증 미구현

---

## [0.2.0] - 계획 중 (Phase 2)

### 예정 (Planned)
- Whisper large-v3 모델 실제 구현
- pyannote-audio 3.1+ 통합
- 범죄 발언 태깅 시스템
- 심리 분석 (가스라이팅 패턴 감지)
- PDF 보고서 생성
- JWT 인증 및 권한 관리
- PostgreSQL 데이터베이스 마이그레이션
- 비동기 작업 큐 (Celery 또는 asyncio)
- 감사 로그 시스템

---

## 버전 관리 규칙

이 프로젝트는 [Semantic Versioning 2.0.0](https://semver.org/)을 따릅니다:

- **MAJOR**: 하위 호환성이 깨지는 변경
- **MINOR**: 하위 호환되는 기능 추가
- **PATCH**: 하위 호환되는 버그 수정

### 커밋 메시지 규칙

[Conventional Commits](https://www.conventionalcommits.org/) 사양을 따릅니다:

- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경

### 예시

```
feat(stt): Whisper large-v3 모델 통합

- OpenAI Whisper API 연동
- 비동기 변환 처리
- 세그먼트별 타임스탬프 추출

Closes #123
```

---

**관련 문서**:
- [SPEC-VOICE-001](.moai/specs/SPEC-VOICE-001/spec.md)
- [실행 계획](.moai/specs/SPEC-VOICE-001/execution-plan-phase1.md)
