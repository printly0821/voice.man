# API 레퍼런스

Voice Man 시스템의 모든 API 엔드포인트에 대한 상세 문서입니다.

## 기본 정보

**Base URL**: `http://localhost:8000` (개발 환경)
**API Version**: v1
**Content-Type**: `application/json`
**Character Encoding**: UTF-8

## 인증

현재 v1.0.0에서는 인증이 구현되어 있지 않습니다. 향후 버전에서 JWT 인증이 추가될 예정입니다.

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

### 범죄 발언 태깅 결과

#### GET /api/v1/audio/{id}/analysis/crime

범죄 발언 태깅 결과를 조회합니다.

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/analysis/crime HTTP/1.1
```

**성공 응답** (200 OK):
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

### 심리 분석 결과

#### GET /api/v1/audio/{id}/analysis/psychology

심리 분석 결과를 조회합니다.

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/analysis/psychology HTTP/1.1
```

**성공 응답** (200 OK):
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

### 증거 보고서 생성

#### POST /api/v1/audio/{id}/report

증거 보고서를 생성합니다.

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

**성공 응답** (200 OK):
```json
{
  "report_id": "990e8400-e29b-41d4-a716-446655440004",
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "report_url": "/api/v1/audio/550e8400-e29b-41d4-a716-446655440000/report",
  "created_at": "2026-01-08T10:35:00Z"
}
```

---

### 증거 보고서 다운로드

#### GET /api/v1/audio/{id}/report

생성된 증거 보고서를 다운로드합니다.

**요청**:
```http
GET /api/v1/audio/550e8400-e29b-41d4-a716-446655440000/report HTTP/1.1
```

**성공 응답** (200 OK):
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="evidence_report_550e8400.pdf"

[PDF 바이너리 데이터]
```

---

## Forensic Evidence APIs

### 디지털 서명 생성

#### POST /api/v1/evidence/sign

파일 해시에 전자서명을 생성합니다.

**요청**:
```http
POST /api/v1/evidence/sign HTTP/1.1
Content-Type: application/json

{
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
  "private_key_path": "/secure/keys/forensic_private_key.pem"
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file_hash | string | Yes | SHA-256 파일 해시 (hex 문자열) |
| private_key_path | string | Yes | Private Key 파일 경로 |

**성공 응답** (200 OK):
```json
{
  "signature": "VGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHNpZ25hdHVyZQ==",
  "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  "metadata": {
    "algorithm": "RSA-2048-PSS-SHA256",
    "timestamp_iso8601": "2026-01-17T10:31:45+09:00",
    "signer": "forensic_analyzer_v1.0",
    "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
  }
}
```

**에러 응답** (400 Bad Request):
```json
{
  "status": "error",
  "message": "유효하지 않은 파일 해시입니다"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/sign"
data = {
    "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
    "private_key_path": "/secure/keys/forensic_private_key.pem"
}
response = requests.post(url, json=data)
print(response.json())
```

---

### 디지털 서명 검증

#### POST /api/v1/evidence/verify

전자서명을 검증합니다.

**요청**:
```http
POST /api/v1/evidence/verify HTTP/1.1
Content-Type: application/json

{
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
  "signature": "VGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHNpZ25hdHVyZQ==",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file_hash | string | Yes | SHA-256 파일 해시 (hex 문자열) |
| signature | string | Yes | Base64 인코딩된 서명 값 |
| public_key | string | Yes | PEM 형식 공개키 |

**성공 응답** (200 OK):
```json
{
  "is_valid": true,
  "metadata": {
    "algorithm": "RSA-2048-PSS-SHA256",
    "verification_timestamp": "2026-01-17T10:35:00+09:00"
  }
}
```

**검증 실패** (200 OK):
```json
{
  "is_valid": false,
  "reason": "서명이 파일 해시와 일치하지 않습니다"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/verify"
data = {
    "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
    "signature": "VGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHNpZ25hdHVyZQ==",
    "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}
response = requests.post(url, json=data)
print(response.json())
```

---

### 타임스탬프 생성

#### POST /api/v1/evidence/timestamp

RFC 3161 타임스탬프 토큰을 생성합니다.

**요청**:
```http
POST /api/v1/evidence/timestamp HTTP/1.1
Content-Type: application/json

{
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file_hash | string | Yes | SHA-256 파일 해시 (hex 문자열) |

**성공 응답** (200 OK):
```json
{
  "timestamp_token": "MIIFJDADCAQAw...",
  "timestamp_iso8601": "2026-01-17T10:31:55+09:00",
  "tsa_url": "https://freetsa.org/tsr",
  "hash_algorithm": "SHA-256",
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
  "is_rfc3161_compliant": true
}
```

**TSA 서비스 장애 시** (200 OK):
```json
{
  "timestamp_token": null,
  "timestamp_iso8601": "2026-01-17T10:31:55+09:00",
  "tsa_url": null,
  "hash_algorithm": "SHA-256",
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
  "is_rfc3161_compliant": false,
  "warning": "TSA 서비스 장애로 로컬 타임스탬프를 사용합니다"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/timestamp"
data = {
    "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
}
response = requests.post(url, json=data)
print(response.json())
```

---

### 감사 로그 조회

#### GET /api/v1/evidence/audit-log

증거 접근 이력을 조회합니다.

**요청**:
```http
GET /api/v1/evidence/audit-log?asset_uuid=550e8400-e29b-41d4-a716-446655440000&user_id=forensic_analyst_01 HTTP/1.1
```

**쿼리 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| asset_uuid | UUID | No | 특정 자산의 로그만 조회 |
| user_id | string | No | 특정 사용자의 로그만 조회 |
| event_type | string | No | 특정 이벤트 유형 (upload, access, analysis, report) |
| start_time | datetime | No | 시작 시간 (ISO 8601) |
| end_time | datetime | No | 종료 시간 (ISO 8601) |
| limit | integer | No | 최대 결과 수 (기본값: 100) |

**성공 응답** (200 OK):
```json
{
  "total_entries": 47,
  "entries": [
    {
      "entry_id": 1,
      "timestamp_iso8601": "2026-01-17T10:32:05+09:00",
      "event_type": "upload",
      "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "forensic_analyst_01",
      "action": "File uploaded and hash generated",
      "metadata": {
        "filename": "recording.mp3",
        "file_size": 1048576,
        "sha256_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
      }
    },
    {
      "entry_id": 2,
      "timestamp_iso8601": "2026-01-17T14:15:00+09:00",
      "event_type": "analysis",
      "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "forensic_analyst_01",
      "action": "Forensic analysis started",
      "metadata": {
        "analysis_type": "comprehensive_forensic",
        "workstation_id": "FORENSIC-WS-01"
      }
    }
  ]
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/audit-log"
params = {
    "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "event_type": "analysis",
    "limit": 50
}
response = requests.get(url, params=params)
print(response.json())
```

---

### 감사 로그 무결성 검증

#### POST /api/v1/evidence/audit-log/verify

감사 로그의 해시 체인 무결성을 검증합니다.

**요청**:
```http
POST /api/v1/evidence/audit-log/verify HTTP/1.1
Content-Type: application/json

{
  "log_file_path": "/var/log/forensic/audit.jsonl"
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| log_file_path | string | No | 감사 로그 파일 경로 (기본값: 시스템 기본 경로) |

**성공 응답** (200 OK):
```json
{
  "is_valid": true,
  "total_entries": 47,
  "verified_entries": 47,
  "invalid_entries": [],
  "hash_chain_valid": true,
  "verification_timestamp": "2026-01-18T09:00:00+09:00"
}
```

**변조 탐지** (200 OK):
```json
{
  "is_valid": false,
  "total_entries": 47,
  "verified_entries": 35,
  "invalid_entries": [36, 37, 38],
  "hash_chain_valid": false,
  "error_details": "Entry 36: Hash mismatch detected",
  "verification_timestamp": "2026-01-18T09:00:00+09:00"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/audit-log/verify"
data = {
    "log_file_path": "/var/log/forensic/audit.jsonl"
}
response = requests.post(url, json=data)
result = response.json()

if result["is_valid"]:
    print("✓ 감사 로그 무결성 검증 통과")
else:
    print(f"✗ 감사 로그 변조 탐지: {result['invalid_entries']}")
```

---

### Bootstrap 신뢰구간 계산

#### POST /api/v1/validation/bootstrap-ci

Bootstrap resampling을 사용하여 95% 신뢰구간을 계산합니다.

**요청**:
```http
POST /api/v1/validation/bootstrap-ci HTTP/1.1
Content-Type: application/json

{
  "data": [0.85, 0.72, 0.68, 0.90, 0.75, 0.82, 0.78, 0.88],
  "n_bootstrap": 10000,
  "confidence_level": 0.95,
  "method": "percentile",
  "random_seed": 42
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| data | array[float] | Yes | 분석 데이터 배열 |
| n_bootstrap | integer | No | Bootstrap 반복 횟수 (기본값: 10000) |
| confidence_level | float | No | 신뢰 수준 (기본값: 0.95) |
| method | string | No | 방법론: percentile 또는 bca (기본값: percentile) |
| random_seed | integer | No | 재현성을 위한 Random seed |

**성공 응답** (200 OK):
```json
{
  "mean": 0.7975,
  "median": 0.8,
  "lower_bound": 0.7125,
  "upper_bound": 0.8825,
  "confidence_level": 0.95,
  "method": "percentile",
  "n_bootstrap": 10000,
  "random_seed": 42
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/validation/bootstrap-ci"
data = {
    "data": [0.85, 0.72, 0.68, 0.90, 0.75, 0.82, 0.78, 0.88],
    "n_bootstrap": 10000,
    "confidence_level": 0.95,
    "method": "bca",
    "random_seed": 42
}
response = requests.post(url, json=data)
result = response.json()

print(f"평균: {result['mean']:.4f}")
print(f"95% 신뢰구간: [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")
```

---

### 성능 메트릭 계산

#### POST /api/v1/validation/performance-metrics

범죄 패턴 탐지 모듈의 성능 메트릭을 계산합니다.

**요청**:
```http
POST /api/v1/validation/performance-metrics HTTP/1.1
Content-Type: application/json

{
  "y_true": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
  "y_pred": [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
  "labels": ["gaslighting", "threat", "coercion"]
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| y_true | array[int] | Yes | Ground Truth 레이블 (0 또는 1) |
| y_pred | array[int] | Yes | 예측 레이블 (0 또는 1) |
| labels | array[string] | No | 클래스 레이블 이름 |

**성공 응답** (200 OK):
```json
{
  "precision": 0.8571,
  "recall": 0.8571,
  "f1_score": 0.8571,
  "accuracy": 0.8,
  "confusion_matrix": [
    [3, 1],
    [1, 5]
  ],
  "true_positives": 5,
  "true_negatives": 3,
  "false_positives": 1,
  "false_negatives": 1,
  "classification_report": {
    "0": {
      "precision": 0.75,
      "recall": 0.75,
      "f1_score": 0.75,
      "support": 4
    },
    "1": {
      "precision": 0.8333,
      "recall": 0.8333,
      "f1_score": 0.8333,
      "support": 6
    }
  }
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/validation/performance-metrics"
data = {
    "y_true": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    "y_pred": [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    "labels": ["gaslighting", "threat", "coercion"]
}
response = requests.post(url, json=data)
result = response.json()

print(f"Precision: {result['precision']:.4f}")
print(f"Recall: {result['recall']:.4f}")
print(f"F1 Score: {result['f1_score']:.4f}")
print(f"Accuracy: {result['accuracy']:.4f}")
```

---

### Chain of Custody 검증

#### POST /api/v1/evidence/chain-of-custody/verify

전체 Chain of Custody를 검증합니다.

**요청**:
```http
POST /api/v1/evidence/chain-of-custody/verify HTTP/1.1
Content-Type: application/json

{
  "asset_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

**요청 파라미터**:
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| asset_uuid | UUID | Yes | 검증할 자산의 UUID |

**성공 응답** (200 OK):
```json
{
  "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "all_passed": true,
  "checks": {
    "file_hash_integrity": {
      "passed": true,
      "initial_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
      "current_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
    },
    "digital_signature": {
      "passed": true,
      "signature_valid": true,
      "algorithm": "RSA-2048-PSS-SHA256"
    },
    "timestamp": {
      "passed": true,
      "is_rfc3161_compliant": true,
      "timestamp": "2026-01-17T10:31:55+09:00"
    },
    "audit_log": {
      "passed": true,
      "total_entries": 47,
      "hash_chain_valid": true
    }
  },
  "verification_timestamp": "2026-01-18T09:00:00+09:00"
}
```

**검증 실패** (200 OK):
```json
{
  "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "all_passed": false,
  "checks": {
    "file_hash_integrity": {
      "passed": false,
      "initial_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
      "current_hash": "different_hash_value",
      "error": "File hash mismatch - potential tampering detected"
    },
    "digital_signature": {
      "passed": true,
      "signature_valid": true
    },
    "timestamp": {
      "passed": true,
      "is_rfc3161_compliant": true
    },
    "audit_log": {
      "passed": true,
      "total_entries": 47,
      "hash_chain_valid": true
    }
  },
  "failed_checks": ["file_hash_integrity"],
  "verification_timestamp": "2026-01-18T09:00:00+09:00"
}
```

**코드 예시**:
```python
import requests

url = "http://localhost:8000/api/v1/evidence/chain-of-custody/verify"
data = {
    "asset_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
response = requests.post(url, json=data)
result = response.json()

if result["all_passed"]:
    print("✓ 모든 Chain of Custody 검증 통과 - 법정 제출 가능")
else:
    print(f"✗ 검증 실패: {result['failed_checks']}")
    for check_name, check_result in result["checks"].items():
        if not check_result["passed"]:
            print(f"  - {check_name}: {check_result.get('error', 'Failed')}")
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

현재 v1.0.0에서는 속도 제한이 구현되어 있지 않습니다. 향후 버전에서 다음 제한이 적용될 예정입니다:

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
