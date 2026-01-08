# API 레퍼런스

Voice Man 시스템의 모든 API 엔드포인트에 대한 상세 문서입니다.

## 기본 정보

**Base URL**: `http://localhost:8000` (개발 환경)
**API Version**: v1
**Content-Type**: `application/json`
**Character Encoding**: UTF-8

## 인증

현재 Phase 1에서는 인증이 구현되어 있지 않습니다. Phase 2에서 JWT 인증이 추가될 예정입니다.

## 공통 응답 형식

### 성공 응답

```json
{
  "status": "success",
  "data": { ... }
}
```

### 에러 응답

```json
{
  "status": "error",
  "message": "에러 메시지",
  "detail": { ... }
}
```

## 엔드포인트

### 헬스체크

#### GET /health

시스템 상태를 확인합니다.

**요청**:
```http
GET /health HTTP/1.1
```

**응답** (200 OK):
```json
{
  "status": "healthy"
}
```

---

### 오디오 파일 업로드

#### POST /api/v1/audio/upload

오디오 파일을 업로드하고 해시를 생성합니다.

**요청**:
```http
POST /api/v1/audio/upload HTTP/1.1
Content-Type: multipart/form-data

file: <audio_file>
```

**파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file | File | Yes | 오디오 파일 (mp3, wav, m4a, flac, ogg) |

**성공 응답** (200 OK):
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "recording.mp3",
  "content_type": "audio/mpeg",
  "file_size": 1048576,
  "sha256_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
}
```

**에러 응답** (400 Bad Request):
```json
{
  "status": "error",
  "message": "지원하지 않는 파일 형식입니다"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/audio/upload"
files = {"file": open("recording.mp3", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

### STT 변환 시작

#### POST /api/v1/audio/{id}/transcribe

오디오 파일을 텍스트로 변환합니다.

**요청**:
```http
POST /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/transcribe HTTP/1.1
```

**경로 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| id | UUID | Yes | 오디오 파일 ID |

**성공 응답** (202 Accepted):
```json
{
  "status": "processing",
  "message": "변환 작업이 시작되었습니다"
}
```

**에러 응답** (404 Not Found):
```json
{
  "status": "error",
  "message": "파일을 찾을 수 없습니다"
}
```

**코드 예시**:
```python
import requests

file_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:8000/api/v1/audio/{file_id}/transcribe"
response = requests.post(url)
print(response.json())
```

---

### 변환 결과 조회

#### GET /api/v1/audio/{id}/transcript

변환된 텍스트를 조회합니다.

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/transcript?version=original HTTP/1.1
```

**경로 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| id | UUID | Yes | 오디오 파일 ID |

**쿼리 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| version | string | No | original 또는 corrected (기본값: original) |

**성공 응답** (200 OK):
```json
{
  "transcript_id": "660e8400-e29b-41d4-a716-446655440001",
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "version": "original",
  "content": "안녕하세요, 오늘 날씨가 정말 좋네요.",
  "segments": [
    {
      "segment_id": "770e8400-e29b-41d4-a716-446655440002",
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "안녕하세요,",
      "confidence": 0.98
    },
    {
      "segment_id": "880e8400-e29b-41d4-a716-446655440003",
      "speaker_id": "SPEAKER_00",
      "start_time": 2.5,
      "end_time": 5.2,
      "text": "오늘 날씨가 정말 좋네요.",
      "confidence": 0.95
    }
  ],
  "created_at": "2026-01-08T10:30:00Z"
}
```

**에러 응답** (404 Not Found):
```json
{
  "status": "error",
  "message": "변환 결과를 찾을 수 없습니다"
}
```

**코드 예시**:
```python
import requests

file_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:8000/api/v1/audio/{file_id}/transcript"
params = {"version": "original"}
response = requests.get(url, params=params)
print(response.json())
```

---

### 화자 분리 결과 조회

#### GET /api/v1/audio/{id}/speakers

화자 분리 결과를 조회합니다.

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/speakers HTTP/1.1
```

**경로 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| id | UUID | Yes | 오디오 파일 ID |

**성공 응답** (200 OK):
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_duration": 300.0,
  "num_speakers": 2,
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 150.0,
      "duration": 150.0,
      "confidence": 0.95
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 150.0,
      "end_time": 300.0,
      "duration": 150.0,
      "confidence": 0.92
    }
  ]
}
```

**에러 응답** (404 Not Found):
```json
{
  "status": "error",
  "message": "화자 분리 결과를 찾을 수 없습니다"
}
```

**코드 예시**:
```python
import requests

file_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:8000/api/v1/audio/{file_id}/speakers"
response = requests.get(url)
print(response.json())
```

---

### 범죄 발언 태깅 결과 (예정)

#### GET /api/v1/audio/{id}/analysis/crime

범죄 발언 태깅 결과를 조회합니다. (Phase 2에서 구현 예정)

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/analysis/crime HTTP/1.1
```

**예상 응답** (200 OK):
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_segments": 50,
  "tagged_segments": 5,
  "crime_tags": [
    {
      "segment_id": "770e8400-e29b-41d4-a716-446655440002",
      "type": "협박",
      "confidence": 0.92,
      "keywords": ["죽여버린다", "가만 안 둔다"],
      "legal_reference": "형법 제283조"
    }
  ]
}
```

---

### 심리 분석 결과 (예정)

#### GET /api/v1/audio/{id}/analysis/psychology

심리 분석 결과를 조회합니다. (Phase 2에서 구현 예정)

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/analysis/psychology HTTP/1.1
```

**예상 응답** (200 OK):
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "gaslighting_patterns": [
    {
      "type": "부정",
      "segments": ["segment_id_1", "segment_id_2"],
      "confidence": 0.85
    }
  ],
  "emotion_summary": {
    "anger": 0.6,
    "fear": 0.3,
    "sadness": 0.1
  },
  "risk_assessment": "고위험"
}
```

---

### 증거 보고서 생성 (예정)

#### POST /api/v1/audio/{id}/report

증거 보고서를 생성합니다. (Phase 2에서 구현 예정)

**요청**:
```http
POST /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/report HTTP/1.1
Content-Type: application/json

{
  "include_transcript": true,
  "include_speakers": true,
  "include_crime_tags": true,
  "include_psychology": true
}
```

**예상 응답** (200 OK):
```json
{
  "report_id": "990e8400-e29b-41d4-a716-446655440004",
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "report_url": "/api/v1/audio/550e8400-e29b-41d4-a716-446655440000/report",
  "created_at": "2026-01-08T10:35:00Z"
}
```

---

### 증거 보고서 다운로드 (예정)

#### GET /api/v1/audio/{id}/report

생성된 증거 보고서를 다운로드합니다. (Phase 2에서 구현 예정)

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/report HTTP/1.1
```

**예상 응답** (200 OK):
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="evidence_report_550e8400.pdf"

[PDF 바이너리 데이터]
```

---

## HTTP 상태 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 202 | 비동기 작업 수락 |
| 400 | 잘못된 요청 |
| 404 | 리소스를 찾을 수 없음 |
| 500 | 서버 내부 오류 |

## 에러 코드

| 코드 | 설명 |
|------|------|
| `UNSUPPORTED_FORMAT` | 지원하지 않는 파일 형식 |
| `FILE_NOT_FOUND` | 파일을 찾을 수 없음 |
| `TRANSCRIPTION_FAILED` | 변환 실패 |
| `DIARIZATION_FAILED` | 화자 분리 실패 |
| `INVALID_REQUEST` | 잘못된 요청 파라미터 |

## 속도 제한 (Rate Limiting)

현재 Phase 1에서는 속도 제한이 구현되어 있지 않습니다. Phase 2에서 다음 제한이 적용될 예정입니다:

- **익명 사용자**: 100 요청/시간
- **인증된 사용자**: 1000 요청/시간

## 버전 관리

API는 URL 경로 기반 버전 관리를 사용합니다:

- 현재 버전: `/api/v1/...`
- 다음 버전: `/api/v2/...` (하위 호환성이 깨지는 변경만)

---

**관련 문서**:
- [아키텍처](architecture.md)
- [배포 가이드](deployment.md)
- [개발 가이드](development.md)
