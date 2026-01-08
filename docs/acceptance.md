# 인수 조건

Voice Man 시스템의 인수 조건을 Gherkin 시나리오로 정의합니다.

## 개요

이 문서는 SPEC-VOICE-001의 요구사항을 Gherkin (Given-When-Then) 형식의 인수 조건으로 정의합니다. 각 시나리오는 실제 사용자 관점에서의 기능 동작을 설명합니다.

## 시나리오 구조

```
Feature: [기능명]
  Scenario: [시나리오명]
    Given [사전 조건]
    When [행동]
    Then [예상 결과]
    And [추가 결과]
```

---

## Feature 1: 오디오 파일 업로드

### Scenario 1-1: 정상적인 mp3 파일 업로드

```gherkin
Scenario: 정상적인 mp3 파일 업로드
  Given 유효한 mp3 파일이 존재한다
    And 파일 크기가 500MB 이하이다
  When 사용자가 /api/v1/audio/upload로 파일을 POST한다
  Then 상태 코드 200이 반환된다
    And 응답에 파일 ID가 포함된다
    And 응답에 원본 파일명이 포함된다
    And 응답에 SHA-256 해시가 포함된다
    And 응답에 파일 크기가 포함된다
    And 파일이 data/uploads/ 디렉토리에 저장된다
    And 데이터베이스에 AudioFile 레코드가 생성된다
```

### Scenario 1-2: 지원하지 않는 파일 형식

```gherkin
Scenario: 지원하지 않는 파일 형식
  Given txt 파일이 존재한다
  When 사용자가 /api/v1/audio/upload로 파일을 POST한다
  Then 상태 코드 400이 반환된다
    And 에러 메시지 "지원하지 않는 파일 형식입니다"가 포함된다
    And 파일이 저장되지 않는다
    And 데이터베이스에 레코드가 생성되지 않는다
```

### Scenario 1-3: 파일 크기 초과

```gherkin
Scenario: 파일 크기 초과
  Given 600MB 크기의 mp3 파일이 존재한다
  When 사용자가 /api/v1/audio/upload로 파일을 POST한다
  Then 상태 코드 400이 반환된다
    And 에러 메시지 "파일 크기가 제한을 초과했습니다"가 포함된다
```

### Scenario 1-4: SHA-256 해시 무결성

```gherkin
Scenario: SHA-256 해시 무결성 검증
  Given 유효한 mp3 파일이 존재한다
  When 사용자가 파일을 업로드한다
  Then SHA-256 해시가 생성된다
    And 동일한 파일을 다시 업로드하면 동일한 해시가 생성된다
    And 파일 내용이 1비트라도 변경되면 해시가 완전히 달라진다
```

---

## Feature 2: 오디오 전처리

### Scenario 2-1: 다양한 형식의 오디오 전처리

```gherkin
Scenario: 다양한 형식의 오디오 전처리
  Given mp3 파일이 업로드되었다
    And wav 파일이 업로드되었다
    And m4a 파일이 업로드되었다
    And flac 파일이 업로드되었다
    And ogg 파일이 업로드되었다
  When 전처리가 실행된다
  Then 모든 파일이 성공적으로 처리된다
    And 각 파일의 메타데이터가 정확하게 추출된다
    And duration 초과 오차는 ±0.1초 이내이다
```

### Scenario 2-2: 손상된 오디오 파일 감지

```gherkin
Scenario: 손상된 오디오 파일 감지
  Given 헤더가 손상된 mp3 파일이 존재한다
  When 전처리가 실행된다
  Then 적절한 에러 메시지가 반환된다
    And 처리가 중단된다
```

### Scenario 2-3: 메타데이터 추출

```gherkin
Scenario: 오디오 메타데이터 추출
  Given 5분 길이의 스테레오 wav 파일이 존재한다
    And 샘플레이트는 44100Hz이다
  When 전처리가 실행된다
  Then 다음 메타데이터가 추출된다:
    | 필드 | 값 |
    | duration | 300.0 ± 0.1 |
    | sample_rate | 44100 |
    | channels | 2 |
    | format | wav |
```

---

## Feature 3: STT 변환

### Scenario 3-1: 한국어 음성 변환

```gherkin
Scenario: 한국어 음성 변환
  Given 3분 길이의 한국어 음성 파일이 존재한다
    And 화자는 2명이다
    And 음성 품질이 양호하다 (SNR > 15dB)
  When /api/v1/audio/{id}/transcribe로 POST 요청을 보낸다
  Then 상태 코드 202가 반환된다
    And 변환 작업이 백그라운드에서 시작된다
  And 변환이 완료되면:
    Then 전체 텍스트가 생성된다
    And 세그먼트별 타임스탬프가 생성된다
    And 각 세그먼트의 신뢰도 점수가 0.8 이상이다
    And WER (Word Error Rate)이 10% 미만이다
```

### Scenario 3-2: 영어 음성 변환

```gherkin
Scenario: 영어 음성 변환
  Given 2분 길이의 영어 음성 파일이 존재한다
  When STT 변환을 요청한다
  Then 변환이 성공적으로 완료된다
    And WER이 10% 미만이다
```

### Scenario 3-3: 타임스탬프 정확성

```gherkin
Scenario: 세그먼트 타임스탬프 정확성
  Given STT 변환이 완료되었다
  When 변환 결과를 조회한다
  Then 각 세그먼트의 타임스탬프가 정확하다
    And start_time과 end_time의 차이는 0.1초 이내 오차이다
    And 세그먼트 간 간격은 0.5초 이내이다
```

### Scenario 3-4: 변환 속도

```gherkin
Scenario: STT 변환 속도
  Given 5분 길이의 오디오 파일이 존재한다
  When STT 변환을 시작한다
  Then 변환은 2.5분 이내에 완료된다 (실시간 대비 0.5x)
```

---

## Feature 4: 화자 분리

### Scenario 4-1: 2인 대화 화자 분리

```gherkin
Scenario: 2인 대화 화자 분리
  Given 2명의 화자가 번갈아가며 대화하는 오디오가 존재한다
    And 전체 길이는 3분이다
  When 화자 분리를 실행한다
  Then 정확히 2명의 화자가 식별된다
    And 각 화자의 발언 구간이 정확히 분리된다
    And DER (Diarization Error Rate)이 15% 미만이다
    And 화자 구분 정확도는 90% 이상이다
```

### Scenario 4-2: 3인 이상 대화 화자 분리

```gherkin
Scenario: 3인 이상 대화 화자 분리
  Given 3명의 화자가 참여하는 회의录音이 존재한다
  When 화자 분리를 실행한다
  Then 3명의 화자가 모두 식별된다
    And 각 화자에게 고유한 레이블이 할당된다
```

### Scenario 4-3: STT와 화자 분리 병합

```gherkin
Scenario: STT 결과와 화자 분리 병합
  Given STT 변환이 완료되었다
    And 화자 분리가 완료되었다
  When 병합 작업이 실행된다
  Then 각 텍스트 세그먼트에 화자 ID가 할당된다
    And 할당된 화자는 해당 시간 구간에 가장 많이 발언한 화자이다
    And 모든 세그먼트에 화자 정보가 포함된다
```

### Scenario 4-4: 화자 통계 생성

```gherkin
Scenario: 화자 통계 생성
  Given 화자 분리가 완료되었다
  When 화자 통계를 생성한다
  Then 다음 정보가 포함된다:
    | 항목 | 설명 |
    | total_speakers | 총 화자 수 |
    | total_speech_duration | 총 발언 시간 |
    | speaker_details | 각 화자별 발언 시간 |
    | speaker_turns | 화자 교대 횟수 |
```

---

## Feature 5: 범죄 발언 태깅 (Phase 2)

### Scenario 5-1: 협박 발언 태깅

```gherkin
Scenario: 협박 발언 자동 태깅
  Given "죽여버린다"라는 발언이 포함된 대화가 존재한다
    And "가만 안 둔다"라는 발언도 포함되어 있다
  When 범죄 발언 분석을 실행한다
  Then 해당 발언에 "협박" 태그가 부착된다
    And 태그 신뢰도는 0.9 이상이다
    And 법적 참조 "형법 제283조"가 포함된다
```

### Scenario 5-2: 공갈 발언 태깅

```gherkin
Scenario: 공갈 발언 자동 태깅
  Given "돈 안 주면 유출한다"라는 발언이 존재한다
  When 범죄 발언 분석을 실행한다
  Then 해당 발언에 "공갈" 태그가 부착된다
    And 관련 키워드가 추출된다
```

### Scenario 5-3: 가스라이팅 패턴 감지

```gherkin
Scenario: 가스라이팅 패턴 감지
  Given 다음 발언들이 연속적으로 존재한다:
    | 발언 | 패턴 |
    | "그런 적 없어, 네가 착각하는 거야" | 부정 (Denial) |
    | "네가 예민하게 반응하는 거야" | 축소 (Minimizing) |
    | "네가 그렇게 했으니까 일이 이렇게 된 거야" | 전가 (Blame-shifting) |
  When 심리 분석을 실행한다
  Then 가스라이팅 패턴이 감지된다
    And 관련 발언들이 그룹화된다
    And 패턴 분석 결과가 제공된다
```

---

## Feature 6: 증거 보고서 생성 (Phase 2)

### Scenario 6-1: PDF 보고서 생성

```gherkin
Scenario: 증거 보고서 PDF 생성
  Given STT 변환이 완료되었다
    And 화자 분리가 완료되었다
    And 범죄 발언 태깅이 완료되었다
    And 심리 분석이 완료되었다
  When 보고서 생성을 요청한다
  Then PDF 보고서가 생성된다
    And 보고서에는 다음이 포함된다:
      | 섹션 | 내용 |
      | 개요 | 파일 정보, 분석 일자 |
      | 전체 대화 | 화자별 구분 텍스트 |
      | 타임라인 | 시간순 대 요약 |
      | 범죄 발언 | 태그된 발언 목록 |
      | 심리 분석 | 가스라이팅 패턴 등 |
      | 통계 | 화자별 발언 비율 등 |
```

### Scenario 6-2: 보고서 다운로드

```gherkin
Scenario: 생성된 보고서 다운로드
  Given 보고서가 생성되었다
  When 보고서 다운로드를 요청한다
  Then PDF 파일이 다운로드된다
    And Content-Type은 application/pdf이다
    And 파일명은 evidence_report_{file_id}.pdf 형식이다
```

---

## Feature 7: 데이터 무결성

### Scenario 7-1: 원본 파일 불변성

```gherkin
Scenario: 원본 파일 불변성 보장
  Given 오디오 파일이 업로드되었다
    And 파일의 SHA-256 해시가 생성되었다
  When 시스템이 해당 파일에 접근한다
  Then 원본 파일은 수정되지 않는다
    And 해시 값은 변경되지 않는다
    And 모든 분석은 원본을 기준으로 수행된다
```

### Scenario 7-2: 원본/교정본 분리

```gherkin
Scenario: 원본 텍스트와 교정본 분리
  Given STT 변환이 완료되었다
  When 변환 결과를 조회한다
  Then "original" 버전이 존재한다
    And "corrected" 버전이 생성될 수 있다
    And 두 버전은 별도로 저장된다
    And 원본은 수정되지 않는다
```

---

## Feature 8: 에러 핸들링

### Scenario 8-1: 존재하지 않는 파일 ID

```gherkin
Scenario: 존재하지 않는 파일 ID로 조회
  Given 존재하지 않는 파일 ID가 있다
  When /api/v1/audio/{id}/transcript로 GET 요청을 보낸다
  Then 상태 코드 404가 반환된다
    And 에러 메시지 "파일을 찾을 수 없습니다"가 포함된다
```

### Scenario 8-2: 진행 중인 작업 중복 요청

```gherkin
Scenario: 변환 진행 중 중복 요청
  Given 파일이 변환 중이다
    And 상태가 "processing"이다
  When 동일 파일에 대해 다시 변환을 요청한다
  Then 요청이 대기열에 추가된다
    And 또는 "이미 처리 중입니다" 메시지가 반환된다
```

### Scenario 8-3: 저장 공간 부족

```gherkin
Scenario: 저장 공간 부족 처리
  Given 디스크 사용량이 90% 이상이다
  When 새 파일이 업로드된다
  Then 관리자에게 경고가 발송된다
    And 업로드가 제한된다
    And 또는 "저장 공간이 부족합니다" 메시지가 반환된다
```

---

## 성능 인수 조건

### Scenario P-1: API 응답 시간

```gherkin
Scenario: 조회 API 응답 시간
  Given 100개의 변환된 파일이 존재한다
  When 각 파일의 변환 결과를 조회한다
  Then P95 응답 시간은 2초 미만이다
    And P99 응답 시간은 5초 미만이다
```

### Scenario P-2: 동시 처리

```gherkin
Scenario: 동시 파일 변환 처리
  Given 10개의 파일이 동시에 업로드된다
  When 모든 파일에 대해 변환을 요청한다
  Then 모든 변환이 성공적으로 완료된다
    And 처리 순서는 업로드 순서와 일치한다
    And 데이터 충돌이 발생하지 않는다
```

---

## 보안 인수 조건

### Scenario S-1: 인증 없이 접근 제한 (Phase 2)

```gherkin
Scenario: 인증 없이 접근 시도
  Given 사용자가 인증되지 않았다
  When 보호된 엔드포인트에 접근한다
  Then 상태 코드 401이 반환된다
    And "인증이 필요합니다" 메시지가 포함된다
```

### Scenario S-2: 민감 데이터 암호화

```gherkin
Scenario: 민감 데이터 암호화 저장
  Given 오디오 파일이 업로드되었다
    And 분석 결과가 생성되었다
  When 데이터베이스를 직접 조회한다
  Then 민정 데이터는 암호화되어 저장된다
    And AES-256 암호화가 사용된다
```

---

## 인수 테스트 실행

### pytest-bdd 설정

```bash
# 설치
pip install pytest-bdd

# 실행
pytest tests/acceptance/

# 특정 feature만 실행
pytest tests/acceptance/test_audio_upload.feature
```

### 예시 테스트 코드

```python
# tests/acceptance/test_upload.py
from pytest_bdd import given, when, then, scenario
import pytest

@pytest.fixture
def client():
    from voice_man.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)

@scenario("audio_upload.feature", "정상적인 mp3 파일 업로드")
def test_upload_valid_mp3():
    pass

@given("유효한 mp3 파일이 존재한다")
def valid_mp3_file(tmp_path):
    import wave
    audio_path = tmp_path / "test.mp3"
    # 테스트 오디오 파일 생성
    return audio_path

@when("사용자가 /api/v1/audio/upload로 파일을 POST한다")
def upload_file(client, valid_mp3_file):
    with open(valid_mp3_file, "rb") as f:
        client.response = client.post(
            "/api/v1/audio/upload",
            files={"file": f}
        )

@then("상태 코드 200이 반환된다")
def check_status_code():
    assert client.response.status_code == 200

@then("응답에 파일 ID가 포함된다")
def check_file_id():
    data = client.response.json()
    assert "file_id" in data
```

---

**관련 문서**:
- [아키텍처](architecture.md)
- [API 레퍼런스](api-reference.md)
- [개발 가이드](development.md)

**관련 SPEC**: [SPEC-VOICE-001](../.moai/specs/SPEC-VOICE-001/spec.md)
