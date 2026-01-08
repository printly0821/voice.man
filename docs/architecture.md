# 시스템 아키텍처

Voice Man 시스템의 전체 아키텍처와 구성 요소를 설명합니다.

## 전체 시스템 구조

```mermaid
C4Context
    title Voice Man 시스템 컨텍스트
    Enterprise_Boundary(b0, "Voice Man 시스템") {
        Person(user, "사용자", "법적 증거가 필요한 피해자")
        System(web_app, "Web API", "FastAPI 기반 REST API")
        System(stt_engine, "STT 엔진", "Whisper large-v3")
        System(diarization, "화자 분리", "pyannote-audio")
        SystemDb(db, "데이터베이스", "SQLite/PostgreSQL")
    }
    System_Ext(external_law, "법률 시스템", "증거 보고서 제출")

    Rel(user, web_app, "업로드/조회", "HTTPS")
    Rel(web_app, stt_engine, "변환 요청", "API")
    Rel(web_app, diarization, "화자 분리", "API")
    Rel(web_app, db, "저장/조회", "SQL")
    Rel(web_app, external_law, "PDF 보고서", "제출")
```

## 핵심 컴포넌트

### 1. API 계층

FastAPI 기반의 REST API 계층입니다.

```mermaid
flowchart LR
    Client[클라이언트] --> Router[API Router]
    Router --> Upload[Upload Handler]
    Router --> Transcribe[Transcribe Handler]
    Router --> Query[Query Handler]

    Upload --> Validator[File Validator]
    Transcribe --> Queue[Task Queue]
    Query --> Database[(Database)]

    style Client fill:#e1f5ff
    style Router fill:#fff4e1
    style Database fill:#ffe1f5
```

**구성 요소**:
- **API Router**: 요청 라우팅 및 미들웨어
- **Handlers**: 각 엔드포인트별 비즈니스 로직
- **Validators**: 입력 데이터 검증
- **Response Models**: 표준화된 응답 형식

### 2. 서비스 계층

비즈니스 로직을 처리하는 서비스 계층입니다.

```mermaid
flowchart TD
    Audio[Audio Service] --> Preprocess[오디오 전처리]
    STT[STT Service] --> Whisper[Whisper 모델]
    Diarization[Diarization Service] --> Pyannote[pyannote 모델]
    Analysis[Analysis Service] --> Crime[범죄 태깅]
    Analysis --> Psychology[심리 분석]

    Preprocess --> FFmpeg[FFmpeg]
    STT --> Merge[병합 서비스]
    Diarization --> Merge
    Merge --> Report[보고서 서비스]

    style Audio fill:#e1f5ff
    style STT fill:#fff4e1
    style Diarization fill:#ffe1f5
    style Analysis fill:#f5e1ff
    style Report fill:#ffe1e1
```

**서비스 목록**:

#### 오디오 서비스 (AudioService)
- 파일 형식 검증
- SHA-256 해시 생성
- FFmpeg 전처리 연동
- 메타데이터 추출

#### STT 서비스 (STTService)
- Whisper 모델 로드
- 비동기 변환 처리
- 세그먼트별 타임스탬프 추출
- 신뢰도 점수 산출

#### 화자 분리 서비스 (DiarizationService)
- pyannote-audio 모델 로드
- 화자 구분 및 레이블링
- STT 결과와 병합
- 화자 통계 생성

#### 분석 서비스 (AnalysisService)
- 범죄 발언 태깅
- 가스라이팅 패턴 감지
- 감정 분석
- 위험도 평가

#### 보고서 서비스 (ReportService)
- PDF 보고서 생성
- 분석 결과 통합
- 템플릿 관리

### 3. 데이터 계층

데이터 영속성을 담당하는 계층입니다.

```mermaid
erDiagram
    AUDIO_FILE ||--o{ TRANSCRIPT : "has"
    TRANSCRIPT ||--o{ TRANSCRIPT_SEGMENT : "contains"
    TRANSCRIPT_SEGMENT }o--|| CRIME_TAG : "tagged_with"
    TRANSCRIPT_SEGMENT }o--|| EMOTION_ANALYSIS : "analyzed_with"

    AUDIO_FILE {
        uuid id PK
        string original_filename
        string file_hash "SHA-256"
        string file_path
        float duration_seconds
        datetime upload_timestamp
        string status
    }

    TRANSCRIPT {
        uuid id PK
        uuid audio_id FK
        string version "original/corrected"
        text content
        datetime created_at
    }

    TRANSCRIPT_SEGMENT {
        uuid id PK
        uuid transcript_id FK
        string speaker_id
        float start_time
        float end_time
        text text
        float confidence
    }

    CRIME_TAG {
        uuid id PK
        string type "협박/공갈/사기/모욕"
        float confidence
        json keywords
        string legal_reference
    }

    EMOTION_ANALYSIS {
        uuid id PK
        json emotion_scores
        string dominant_emotion
        float intensity
    }
```

### 4. 외부 연동

```mermaid
flowchart LR
    API[Voice Man API] --> Whisper[Whisper API]
    API --> Pyannote[pyannote-audio]
    API --> FFmpeg[FFmpeg]
    API --> LLM[LLM Service]

    Whisper --> GPU[GPU Server]
    Pyannote --> GPU
    LLM --> Claude[Claude API]

    style API fill:#e1f5ff
    style GPU fill:#ffe1e1
    style Claude fill:#ffe1f5
```

## 데이터 흐름

### 1. 파일 업로드 흐름

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Validator
    participant Storage
    participant Database

    Client->>API: POST /api/v1/audio/upload
    API->>Validator: 파일 형식 검증
    Validator->>API: 검증 결과

    alt 유효한 파일
        API->>Storage: 파일 저장
        Storage->>API: 저장 경로 반환
        API->>API: SHA-256 해시 생성
        API->>Database: AudioFile 레코드 생성
        Database->>API: 저장 완료
        API->>Client: 200 OK + 파일 ID
    else 잘못된 형식
        API->>Client: 400 Bad Request
    end
```

### 2. STT 변환 흐름

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Queue
    participant STT
    participant Database

    Client->>API: POST /api/v1/audio/{id}/transcribe
    API->>Queue: 변환 작업 추가
    Queue->>STT: 비동기 처리

    STT->>STT: Whisper 모델 로드
    STT->>STT: 오디오 변환
    STT->>Database: Transcript 저장
    STT->>API: 처리 완료 알림

    API->>Client: 202 Accepted
    Note over Client: 폴링 또는 WebSocket으로 결과 확인
```

### 3. 화자 분리 흐름

```mermaid
sequenceDiagram
    participant STT
    participant Diarization
    participant Merge
    participant Database

    STT->>Diarization: STT 완료 알림
    Diarization->>Diarization: pyannote 모델 로드
    Diarization->>Diarization: 화자 분리 실행

    Diarization->>Merge: 병합 요청
    Merge->>Merge: STT + 화자 정보 병합
    Merge->>Database: 업데이트된 세그먼트 저장
    Merge->>STT: 완료 알림
```

### 4. 분석 및 보고서 흐름

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Analysis
    participant Report
    participant Storage

    Client->>API: POST /api/v1/audio/{id}/report
    API->>Analysis: 분석 요청

    Analysis->>Analysis: 범죄 태깅
    Analysis->>Analysis: 심리 분석
    Analysis->>API: 분석 완료

    API->>Report: PDF 생성 요청
    Report->>Report: 템플릿 적용
    Report->>Storage: PDF 저장
    Storage->>Report: 파일 경로

    Report->>API: 생성 완료
    API->>Client: 200 OK + 다운로드 URL
```

## 비기능적 요구사항

### 성능

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| STT 변환 속도 | 실시간 대비 0.5x 이하 | 5분 파일 → 2.5분 이내 |
| API 응답 시간 (조회) | P95 < 2초 | 로그 분석 |
| API 응답 시간 (분석) | P95 < 5초 | 로그 분석 |
| 동시 처리 수 | 10개 파일 | 부하 테스트 |

### 보안

| 항목 | 구현 방법 |
|------|-----------|
| 원본 보존 | 불변(immutable) 저장소 |
| 데이터 암호화 | AES-256 at-rest |
| 전송 보안 | TLS 1.3 |
| 접근 제어 | JWT 인증 |
| 감사 로그 | 모든 요청/변경 기록 |

### 확장성

```mermaid
flowchart LR
    API[API Server] --> Queue[Task Queue]
    Queue --> Worker1[Worker 1]
    Queue --> Worker2[Worker 2]
    Queue --> WorkerN[Worker N]

    Worker1 --> GPU1[GPU 1]
    Worker2 --> GPU2[GPU 2]
    WorkerN --> GPUN[GPU N]

    Database[(Database)] --> Replica1[Replica 1]
    Database --> Replica2[Replica 2]

    style Queue fill:#fff4e1
    style GPU1 fill:#ffe1e1
    style GPU2 fill:#ffe1e1
    style GPUN fill:#ffe1e1
```

**확장 전략**:
- **수평 확장**: API 서버 다중화
- **비동기 처리**: Task Queue 기반 워커 확장
- **GPU 풀링**: GPU 리소스 공유
- **데이터베이스 복제**: 읽기 부하 분산

### 가용성

| 항목 | 목표 | 구현 방법 |
|------|------|-----------|
| 시스템 가용성 | 99.5% | 이중화 구성 |
| 데이터 내구성 | 99.99% | 다중 백업 |
| 재해 복구 | RPO < 1시간 | 백업 전략 |
| 장애 조치 | 자동 복구 | 헬스체크 및 재시작 |

## 기술 의사결정

### 왜 FastAPI인가?

1. **비동기 처리**: asyncio 기반 고성능
2. **자동 문서화**: OpenAPI/Swagger 자동 생성
3. **타입 검증**: Pydantic 기반 강력한 타입 체크
4. **현대적**: 최신 Python 기능 활용

### 왜 Whisper인가?

1. **정확도**: WER < 10% (한국어)
2. **오픈 소스**: 무료商用 사용 가능
3. **다국어**: 한국어/영어 동시 지원
4. **커뮤니티**: 활발한 개발 및 지원

### 왜 pyannote-audio인가?

1. **정확도**: DER < 15%
2. **사전 학습**: 바로 사용 가능한 모델
3. **Python 네이티브**: 쉬운 통합
4. **연구 기반**: 최신 연구 성과 반영

### 왜 SQLAlchemy인가?

1. **ORM 강력**: 복잡한 쿼리 지원
2. **비동기 지원**: async/await 패러다임
3. **데이터베이스 독립**: SQLite ↔ PostgreSQL 마이그레이션 용이
4. **마이그레이션**: Alembic과 연동

---

**관련 문서**:
- [API 레퍼런스](api-reference.md)
- [배포 가이드](deployment.md)
- [개발 가이드](development.md)
