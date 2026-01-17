# Chain of Custody 운영 가이드

## 목차

1. [개요](#1-개요)
2. [증거 수집 단계](#2-증거-수집-단계)
3. [증거 보관 및 이동](#3-증거-보관-및-이동)
4. [증거 분석](#4-증거-분석)
5. [법정 제출](#5-법정-제출)
6. [문서화 및 기록](#6-문서화-및-기록)

---

## 1. 개요

### 1.1 Chain of Custody란?

Chain of Custody(증거 보관 연속성)는 증거가 수집된 시점부터 법정 제출까지 모든 단계에서 증거의 무결성과 진정성을 보장하는 절차입니다. 이는 법정에서 증거가 변조되지 않았음을 입증하는 핵심 요소입니다.

### 1.2 법적 요구사항

- **한국 형사소송법 Article 313(2)(3)**: 디지털 증거의 무결성 입증 요구
- **ISO/IEC 27037**: 디지털 증거 수집, 보존, 이동 가이드라인
- **NIST SP 800-86**: 디지털 포렌식 4단계 절차 (수집, 검사, 분석, 보고)

### 1.3 본 가이드의 범위

본 가이드는 voice.man 시스템을 사용하여 음성 증거를 처리하는 전 과정에서 Chain of Custody를 유지하는 방법을 설명합니다.

---

## 2. 증거 수집 단계

### 2.1 원본 증거 식별 및 봉인

#### 2.1.1 증거 식별

**필수 기록 사항:**
- 증거 파일명 (원본 그대로)
- 파일 크기 (바이트)
- 파일 포맷 (mp3, wav, m4a 등)
- 재생 시간 (초)
- 수집 일시 (ISO 8601 형식)
- 수집 장소
- 수집 담당자

**증거 식별 양식:**
```
증거번호: VOICE-2026-0117-001
파일명: recording_2026-01-15_143022.mp3
파일 크기: 1,048,576 bytes
파일 포맷: audio/mpeg
재생 시간: 180초 (3분)
수집 일시: 2026-01-17T10:30:45+09:00
수집 장소: 서울시 강남구 xx동 xx아파트 101호
수집 담당자: 홍길동 (경위, 배지번호 12345)
```

#### 2.1.2 원본 봉인

**물리적 봉인:**
- 원본 파일이 저장된 저장 매체(USB, SD카드 등) 봉인
- 봉인 스티커에 서명 및 날짜 기재
- 봉인 사진 촬영

**디지털 봉인:**
- 원본 파일을 읽기 전용 모드로 설정
- 파일 속성 변경 금지
- 별도 보안 저장소 이동

### 2.2 SHA-256 해시 생성

#### 2.2.1 해시 생성 절차

**voice.man 시스템 사용:**

```bash
# 파일 업로드 시 자동으로 SHA-256 해시 생성됨
curl -X POST http://localhost:8000/api/v1/audio/upload \
  -F "file=@recording.mp3"

# 응답 예시:
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "recording.mp3",
  "sha256_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
}
```

**수동 해시 생성 (검증용):**

```bash
# Linux/macOS
sha256sum recording.mp3

# Windows PowerShell
Get-FileHash recording.mp3 -Algorithm SHA256
```

#### 2.2.2 해시 값 기록

**해시 기록 양식:**
```
파일명: recording_2026-01-15_143022.mp3
SHA-256: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
생성 일시: 2026-01-17T10:31:12+09:00
생성 도구: voice.man v1.4.0
생성 담당자: 홍길동
```

**중요 사항:**
- 해시 값은 반드시 수기로 별도 기록
- 전산 기록과 수기 기록 모두 보관
- 해시 값 불일치 시 즉시 보고

### 2.3 디지털 서명 적용

#### 2.3.1 서명 생성

**voice.man 자동 서명:**

```python
from voice_man.forensics.evidence.digital_signature import DigitalSignatureService

# 서명 서비스 초기화
signature_service = DigitalSignatureService()

# 파일 해시에 서명 생성
file_hash = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
signature, public_key, metadata = signature_service.sign_hash(
    file_hash=file_hash,
    private_key_path="/secure/keys/forensic_private_key.pem"
)

# 서명 정보 출력
print(f"Signature: {signature}")
print(f"Public Key: {public_key}")
print(f"Timestamp: {metadata['timestamp_iso8601']}")
```

#### 2.3.2 서명 메타데이터 기록

**서명 기록 양식:**
```
파일명: recording_2026-01-15_143022.mp3
파일 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
서명 알고리즘: RSA-2048-PSS-SHA256
서명 값: [Base64 인코딩된 서명 값]
서명 일시: 2026-01-17T10:31:45+09:00
서명자: forensic_analyzer_v1.0
Public Key: [PEM 형식 공개키]
```

### 2.4 RFC 3161 타임스탬프 기록

#### 2.4.1 타임스탬프 발급

**voice.man 자동 타임스탬프:**

```python
from voice_man.forensics.evidence.timestamp_service import TimestampService

# 타임스탬프 서비스 초기화
timestamp_service = TimestampService(tsa_url="https://freetsa.org/tsr")

# 타임스탬프 토큰 생성
file_hash = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
timestamp_token = timestamp_service.generate_timestamp(file_hash)

# 타임스탬프 정보 출력
print(f"Timestamp Token: {timestamp_token['timestamp_token']}")
print(f"Timestamp (ISO 8601): {timestamp_token['timestamp_iso8601']}")
print(f"RFC 3161 Compliant: {timestamp_token['is_rfc3161_compliant']}")
```

#### 2.4.2 타임스탬프 기록

**타임스탬프 기록 양식:**
```
파일명: recording_2026-01-15_143022.mp3
파일 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
타임스탬프 토큰: [Base64 인코딩된 RFC 3161 토큰]
타임스탬프 일시: 2026-01-17T10:31:55+09:00
TSA URL: https://freetsa.org/tsr
RFC 3161 준수: Yes
```

**타임스탬프 실패 시:**
- 로컬 타임스탬프 사용 (NTP 동기화 필수)
- `is_rfc3161_compliant: false` 기록
- 감사 로그에 실패 원인 기록
- 보고서에 경고 메시지 포함

### 2.5 감사 로그 초기화

#### 2.5.1 감사 로그 시작

**voice.man 자동 로깅:**

```python
from voice_man.forensics.evidence.audit_logger import ImmutableAuditLogger

# 감사 로거 초기화
audit_logger = ImmutableAuditLogger(log_file_path="/var/log/forensic/audit.jsonl")

# 증거 업로드 이벤트 기록
audit_logger.log_event(
    event_type="upload",
    asset_uuid="550e8400-e29b-41d4-a716-446655440000",
    user_id="forensic_analyst_01",
    action="File uploaded and hash generated",
    metadata={
        "filename": "recording.mp3",
        "file_size": 1048576,
        "sha256_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
        "digital_signature": "[서명 값]",
        "timestamp_token": "[타임스탬프 토큰]"
    }
)
```

#### 2.5.2 감사 로그 확인

**로그 엔트리 예시:**
```json
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
  },
  "previous_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "current_hash": "d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5"
}
```

---

## 3. 증거 보관 및 이동

### 3.1 불변 감사 로그 기록

#### 3.1.1 모든 접근 기록

**기록 대상 이벤트:**
- 증거 파일 읽기
- 증거 파일 복사
- 증거 파일 이동
- 분석 시작/종료
- 보고서 생성
- 무결성 검증

**접근 로그 예시:**
```json
{
  "entry_id": 2,
  "timestamp_iso8601": "2026-01-17T11:15:30+09:00",
  "event_type": "access",
  "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "forensic_analyst_01",
  "action": "File accessed for analysis",
  "metadata": {
    "access_type": "read",
    "purpose": "forensic_analysis",
    "workstation_id": "FORENSIC-WS-01"
  },
  "previous_hash": "d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5",
  "current_hash": "e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5j6"
}
```

#### 3.1.2 로그 무결성 유지

**해시 체인 검증:**
- 매일 자동 무결성 검증
- 변조 탐지 시 즉시 경고
- 백업 로그와 비교 분석

### 3.2 이동 시 재서명

#### 3.2.1 증거 이동 절차

**이동 사유:**
- 분석 워크스테이션 이동
- 외부 전문가에게 제공
- 법정 제출

**이동 전 작업:**
1. 현재 해시 값 확인
2. 무결성 검증
3. 이동 이벤트 감사 로그 기록
4. 재서명 생성

**이동 후 작업:**
1. 해시 값 재확인
2. 서명 검증
3. 이동 완료 감사 로그 기록

#### 3.2.2 이동 기록

**이동 기록 양식:**
```
증거번호: VOICE-2026-0117-001
이동 일시: 2026-01-17T14:00:00+09:00
이동 전 위치: 서울 중앙 포렌식 센터 보관실
이동 후 위치: 분석실 워크스테이션 FORENSIC-WS-01
이동 담당자: 김철수 (분석관)
이동 사유: 포렌식 분석 수행
이동 전 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
이동 후 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
해시 일치 여부: 일치 (OK)
재서명 일시: 2026-01-17T14:05:00+09:00
```

### 3.3 해시 체인 검증

#### 3.3.1 정기 검증

**검증 주기:**
- 매일 1회 자동 검증
- 증거 이동 전/후 검증
- 보고서 생성 전 검증

**검증 스크립트:**
```python
from voice_man.forensics.evidence.audit_logger import ImmutableAuditLogger

# 감사 로거 초기화
audit_logger = ImmutableAuditLogger(log_file_path="/var/log/forensic/audit.jsonl")

# 해시 체인 검증
is_valid, invalid_entries = audit_logger.verify_chain()

if is_valid:
    print("✓ 감사 로그 무결성 검증 통과")
else:
    print(f"✗ 감사 로그 변조 탐지! 변조된 엔트리: {invalid_entries}")
    # 즉시 보고 및 조사
```

#### 3.3.2 검증 실패 처리

**검증 실패 시 조치:**
1. 모든 분석 작업 즉시 중단
2. 시스템 관리자 및 보안 담당자 긴급 알림
3. 변조 구간 식별 및 격리
4. 백업 로그와 비교 분석
5. 포렌식 조사 개시
6. 법률 자문 및 사법기관 보고

---

## 4. 증거 분석

### 4.1 사본으로 작업 (원본 보존)

#### 4.1.1 작업 사본 생성

**사본 생성 절차:**
```bash
# 원본 파일 읽기 전용 설정
chmod 444 recording_original.mp3

# 작업 사본 생성
cp recording_original.mp3 recording_working_copy.mp3

# 사본 해시 확인
sha256sum recording_working_copy.mp3
# 결과: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
```

**사본 생성 기록:**
```
원본 파일: recording_original.mp3
원본 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
사본 파일: recording_working_copy.mp3
사본 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
사본 생성 일시: 2026-01-17T14:10:00+09:00
사본 생성 담당자: 김철수 (분석관)
해시 일치 여부: 일치 (OK)
```

#### 4.1.2 원본 보존

**원본 보존 방법:**
- 원본 파일 읽기 전용 설정
- 별도 보안 저장소 보관
- 외부 매체(USB, 하드드라이브)에 백업
- 백업 매체 봉인 및 보관

### 4.2 모든 접근 감사 로그 기록

#### 4.2.1 분석 시작 로그

```python
audit_logger.log_event(
    event_type="analysis",
    asset_uuid="550e8400-e29b-41d4-a716-446655440000",
    user_id="forensic_analyst_01",
    action="Forensic analysis started",
    metadata={
        "analysis_type": "comprehensive_forensic",
        "workstation_id": "FORENSIC-WS-01",
        "tools": ["WhisperX", "KoBERT", "librosa", "parselmouth"],
        "start_timestamp": "2026-01-17T14:15:00+09:00"
    }
)
```

#### 4.2.2 분석 단계별 로그

**기록 대상 단계:**
- STT 변환 시작/완료
- 화자 분리 시작/완료
- 음성 특성 분석 시작/완료
- 범죄 언어 패턴 탐지 시작/완료
- 감정 분석 시작/완료
- 포렌식 스코어링 시작/완료

### 4.3 분석 결과 타임스탬프

#### 4.3.1 결과 생성 타임스탬프

**분석 완료 로그:**
```python
audit_logger.log_event(
    event_type="analysis",
    asset_uuid="550e8400-e29b-41d4-a716-446655440000",
    user_id="forensic_analyst_01",
    action="Forensic analysis completed",
    metadata={
        "end_timestamp": "2026-01-17T15:30:00+09:00",
        "analysis_duration_seconds": 4500,
        "result_file": "forensic_result_550e8400.json",
        "result_hash": "b6c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9f0"
    }
)
```

#### 4.3.2 결과 파일 서명

**결과 파일 보호:**
```python
# 분석 결과 JSON 파일 해시 생성
result_hash = calculate_sha256("forensic_result_550e8400.json")

# 결과 파일에 전자서명 적용
signature_service = DigitalSignatureService()
signature, public_key, metadata = signature_service.sign_hash(
    file_hash=result_hash,
    private_key_path="/secure/keys/forensic_private_key.pem"
)

# 결과 파일에 타임스탬프 발급
timestamp_service = TimestampService()
timestamp_token = timestamp_service.generate_timestamp(result_hash)
```

---

## 5. 법정 제출

### 5.1 증거 무결성 검증

#### 5.1.1 제출 전 전체 검증

**검증 체크리스트:**
- [ ] 원본 파일 해시 일치 확인
- [ ] 전자서명 검증 통과
- [ ] 타임스탬프 검증 통과
- [ ] 감사 로그 해시 체인 무결성 확인
- [ ] 분석 결과 파일 서명 검증

**검증 스크립트:**
```python
from voice_man.forensics.evidence.chain_of_custody import ChainOfCustodyVerifier

verifier = ChainOfCustodyVerifier()

# 전체 검증 실행
verification_result = verifier.verify_all(
    asset_uuid="550e8400-e29b-41d4-a716-446655440000"
)

if verification_result["all_passed"]:
    print("✓ 모든 검증 통과 - 법정 제출 가능")
else:
    print(f"✗ 검증 실패: {verification_result['failed_checks']}")
    # 법정 제출 불가
```

#### 5.1.2 검증 보고서 생성

**검증 보고서 내용:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증거 무결성 검증 보고서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

증거번호: VOICE-2026-0117-001
검증 일시: 2026-01-18T09:00:00+09:00
검증 담당자: 이영희 (수석 분석관)

[1] 원본 파일 해시 검증
  - 현재 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
  - 초기 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
  - 결과: ✓ 일치

[2] 전자서명 검증
  - 서명 알고리즘: RSA-2048-PSS-SHA256
  - Public Key: [PEM 형식 공개키]
  - 결과: ✓ 검증 통과

[3] 타임스탬프 검증
  - RFC 3161 준수: Yes
  - TSA: https://freetsa.org/tsr
  - 타임스탬프: 2026-01-17T10:31:55+09:00
  - 결과: ✓ 검증 통과

[4] 감사 로그 해시 체인 검증
  - 총 엔트리 수: 47
  - 변조 탐지: 없음
  - 결과: ✓ 무결성 확인

[5] 분석 결과 파일 검증
  - 결과 파일 해시: b6c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9f0
  - 서명 검증: ✓ 통과
  - 타임스탬프 검증: ✓ 통과

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
종합 결과: 모든 검증 통과 (법정 제출 가능)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

검증자 서명: _______________________
검토자 서명: _______________________
승인자 서명: _______________________
```

### 5.2 Chain of Custody 문서 준비

#### 5.2.1 증거 이력 요약

**증거 이력 문서:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증거 보관 연속성 (Chain of Custody) 기록
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

증거번호: VOICE-2026-0117-001
사건번호: 2026형단12345
증거 유형: 음성 녹취 파일

[증거 수집]
- 수집 일시: 2026-01-17T10:30:45+09:00
- 수집 장소: 서울시 강남구 xx동 xx아파트 101호
- 수집 담당자: 홍길동 (경위, 배지번호 12345)
- 초기 SHA-256: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e

[증거 이동 이력]
1. 2026-01-17 14:00 - 보관실 → 분석실 (김철수)
2. 2026-01-18 09:00 - 분석실 → 보관실 (김철수)
3. 2026-01-19 10:00 - 보관실 → 법정 (이영희)

[증거 접근 이력]
- 총 접근 횟수: 47회
- 주요 접근 이벤트:
  * 2026-01-17 14:15 - 포렌식 분석 시작 (김철수)
  * 2026-01-17 15:30 - 포렌식 분석 완료 (김철수)
  * 2026-01-18 09:00 - 무결성 검증 (이영희)

[증거 무결성 확인]
- 현재 SHA-256: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
- 초기 해시 일치: ✓ 확인
- 감사 로그 무결성: ✓ 확인
- 전자서명 검증: ✓ 통과
- 타임스탬프 검증: ✓ 통과

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

본 증거는 수집부터 법정 제출까지 모든 단계에서
무결성이 유지되었음을 확인합니다.

검증자: 이영희 (수석 분석관)
일시: 2026-01-19T10:00:00+09:00
서명: _______________________
```

### 5.3 전문가 증언 자료

#### 5.3.1 증언 준비 자료

**Q&A 형식 증언 자료:**

**Q1: 증거의 무결성은 어떻게 보장됩니까?**
A: 본 증거는 수집 즉시 SHA-256 해시를 생성하였고, RSA 2048-bit 전자서명과 RFC 3161 타임스탬프를 적용하였습니다. 모든 접근 이벤트는 변조 불가능한 append-only 감사 로그에 기록되었으며, 해시 체인으로 무결성을 검증하였습니다.

**Q2: 원본 파일이 변조되지 않았다는 것을 어떻게 증명할 수 있습니까?**
A: 초기 수집 시 생성된 SHA-256 해시 값과 현재 파일의 해시 값이 일치합니다. SHA-256은 NIST에서 승인한 암호학적 해시 함수로, 사실상 역산 및 충돌 공격이 불가능합니다. 해시 값이 일치한다는 것은 파일이 단 한 비트도 변경되지 않았음을 수학적으로 증명합니다.

**Q3: 타임스탬프의 신뢰성은 어떻게 보장됩니까?**
A: RFC 3161 표준을 준수하는 제3자 TSA(Time Stamping Authority)에서 발급한 타임스탬프 토큰을 사용하였습니다. 이는 국제적으로 인정받는 타임스탬프 프로토콜이며, TSA의 인증서 체인을 통해 신뢰성을 검증할 수 있습니다.

**Q4: 감사 로그가 변조되지 않았다는 것을 어떻게 확인합니까?**
A: 감사 로그는 append-only 구조로 각 엔트리가 이전 엔트리의 해시 값을 참조하는 해시 체인을 형성합니다. 단 하나의 엔트리라도 변조되면 해시 체인이 끊어지므로 즉시 탐지됩니다. 본 증거의 감사 로그는 총 47개 엔트리의 해시 체인 무결성을 검증하였습니다.

#### 5.3.2 시각 자료

**법정 제시용 다이어그램:**
- Chain of Custody 흐름도
- 해시 체인 구조 다이어그램
- 전자서명 검증 프로세스
- 타임스탬프 검증 절차

---

## 6. 문서화 및 기록

### 6.1 필수 기록 양식

#### 6.1.1 증거 수집 기록지

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증거 수집 기록지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

증거번호: VOICE-2026-0117-001
사건번호: 2026형단12345

[기본 정보]
수집 일시: 2026-01-17T10:30:45+09:00
수집 장소: 서울시 강남구 xx동 xx아파트 101호
수집 담당자: 홍길동 (경위, 배지번호 12345)
입회자: 박영희 (경사, 배지번호 67890)

[증거 파일 정보]
파일명: recording_2026-01-15_143022.mp3
파일 크기: 1,048,576 bytes (1.00 MB)
파일 포맷: audio/mpeg
재생 시간: 180초 (3분)
비트레이트: 128 kbps
샘플링 레이트: 44.1 kHz

[무결성 정보]
SHA-256 해시: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
해시 생성 일시: 2026-01-17T10:31:12+09:00
디지털 서명: [서명 값]
서명 일시: 2026-01-17T10:31:45+09:00
타임스탬프 토큰: [RFC 3161 토큰]
타임스탬프 일시: 2026-01-17T10:31:55+09:00

[물리적 봉인]
봉인 번호: SEAL-2026-0117-001
봉인 일시: 2026-01-17T10:35:00+09:00
봉인 담당자: 홍길동

수집 담당자 서명: _____________________ 일시: _____
입회자 서명: _____________________ 일시: _____
```

#### 6.1.2 증거 이동 기록지

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증거 이동 기록지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

증거번호: VOICE-2026-0117-001
이동 번호: TRANSFER-001

[이동 정보]
이동 일시: 2026-01-17T14:00:00+09:00
이동 전 위치: 서울 중앙 포렌식 센터 보관실 (A-101)
이동 후 위치: 분석실 워크스테이션 FORENSIC-WS-01
이동 담당자: 김철수 (분석관, 사원번호 A123)
이동 사유: 포렌식 분석 수행

[무결성 확인]
이동 전 SHA-256: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
이동 후 SHA-256: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
해시 일치 여부: ✓ 일치

[재서명]
재서명 일시: 2026-01-17T14:05:00+09:00
재서명 값: [새 서명 값]

이동 담당자 서명: _____________________ 일시: _____
확인자 서명: _____________________ 일시: _____
```

### 6.2 감사 로그 요약

#### 6.2.1 주요 이벤트 요약

**감사 로그 요약 예시:**
```
증거번호: VOICE-2026-0117-001
감사 로그 총 엔트리 수: 47
감사 로그 기간: 2026-01-17 10:32 ~ 2026-01-19 10:00

[주요 이벤트]
1. 업로드 (upload): 1회
   - 2026-01-17 10:32:05 - forensic_analyst_01

2. 접근 (access): 12회
   - 최초: 2026-01-17 14:15:00
   - 최근: 2026-01-19 09:55:00

3. 분석 (analysis): 5회
   - STT 변환: 2026-01-17 14:20:00 - 14:35:00
   - 화자 분리: 2026-01-17 14:36:00 - 14:50:00
   - 음성 특성 분석: 2026-01-17 14:51:00 - 15:10:00
   - 범죄 언어 탐지: 2026-01-17 15:11:00 - 15:20:00
   - 포렌식 스코어링: 2026-01-17 15:21:00 - 15:30:00

4. 보고서 생성 (report): 2회
   - HTML 보고서: 2026-01-18 09:30:00
   - PDF 보고서: 2026-01-18 09:45:00

5. 검증 (verification): 3회
   - 무결성 검증: 2026-01-18 09:00:00
   - 서명 검증: 2026-01-18 09:05:00
   - 해시 체인 검증: 2026-01-18 09:10:00

[해시 체인 상태]
초기 해시: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
최종 해시: x7y8z9a0b1c2d3e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4z5a6b7c8
무결성 검증: ✓ 통과
```

### 6.3 보관 기간 및 폐기 절차

#### 6.3.1 보관 기간

**법적 보관 기간:**
- 형사 사건: 공소시효 + 5년
- 민사 사건: 판결 확정 후 10년
- 행정 사건: 관련 법령에 따름

**본 시스템 권장 보관 기간:**
- 원본 증거 파일: 영구 보관
- 감사 로그: 영구 보관
- 분석 결과 파일: 10년
- 보고서: 10년

#### 6.3.2 폐기 절차

**폐기 승인 절차:**
1. 법적 보관 기간 만료 확인
2. 법률 자문 검토
3. 관련 부서 승인
4. 보관 책임자 최종 승인

**폐기 방법:**
- 디지털 파일: 3회 덮어쓰기 후 삭제 (DoD 5220.22-M 표준)
- 물리적 매체: 파쇄 또는 소각
- 백업: 모든 백업 동시 삭제

**폐기 기록:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증거 폐기 기록지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

증거번호: VOICE-2026-0117-001
폐기 일시: 2036-01-20T10:00:00+09:00
폐기 사유: 법적 보관 기간 만료 (10년 경과)

[폐기 승인]
법률 검토자: 박변호사 (서명: _____)
부서장 승인: 김부장 (서명: _____)
보관 책임자 승인: 이책임자 (서명: _____)

[폐기 방법]
디지털 파일: DoD 5220.22-M 3회 덮어쓰기
물리적 매체: 파쇄
백업: 모든 백업 동시 삭제

폐기 담당자 서명: _____________________ 일시: _____
확인자 서명: _____________________ 일시: _____
```

---

## 부록

### A. 관련 법령 및 표준

- 한국 형사소송법 Article 313(2)(3)
- ISO/IEC 27037:2012
- ISO/IEC 17025:2017
- NIST SP 800-86
- RFC 3161

### B. 용어 정의

- **Chain of Custody**: 증거 보관 연속성
- **SHA-256**: Secure Hash Algorithm 256-bit
- **Digital Signature**: 전자서명
- **RFC 3161**: 타임스탬프 프로토콜 표준
- **Append-only Log**: 추가만 가능한 로그
- **Hash Chain**: 해시 체인

### C. 문의처

**기술 지원:**
- 이메일: forensic-support@voice-man.kr
- 전화: 02-xxxx-xxxx

**법률 자문:**
- 이메일: legal@voice-man.kr
- 전화: 02-yyyy-yyyy

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2026-01-17
**작성자**: voice.man 개발팀
**검토자**: 포렌식 전문가
**승인자**: 법률 자문

**TAG**: [FORENSIC-EVIDENCE-001]
