---
id: SPEC-FORENSIC-001
document_type: "acceptance-criteria"
version: "1.0.0"
created: "2026-01-09"
updated: "2026-01-09"
author: "지니"
status: "planned"
---

# SPEC-FORENSIC-001: 인수 기준 (Acceptance Criteria)

## 개요

범죄 프로파일링 기반 음성 포렌식 분석 시스템의 인수 조건 및 Gherkin 형식의 테스트 시나리오입니다.

---

## 1. 음성 특성 분석 (Audio Feature Analysis)

### AC-1.1: 음량 분석 정상 동작

```gherkin
Feature: 음량 분석 (Volume Analysis)
  음성 세그먼트의 음량 특성을 정확하게 추출한다

  Background:
    Given WhisperX 파이프라인으로 처리된 오디오 파일이 존재한다
    And 해당 파일의 세그먼트 정보가 JSON 형식으로 저장되어 있다

  Scenario: 정상적인 음량 분석 수행
    Given 오디오 파일 "sample_001.m4a"가 로드되어 있다
    When 음량 분석을 요청한다
    Then RMS amplitude (dB) 값이 반환된다
    And Peak amplitude (dB) 값이 반환된다
    And Dynamic range (dB) 값이 반환된다
    And 음량 변화율 (dB/s) 값이 반환된다

  Scenario: 화자별 평균 음량 비교
    Given 2명의 화자가 식별된 오디오 파일이 있다
    When 화자별 음량 분석을 요청한다
    Then SPEAKER_00의 평균 음량이 계산된다
    And SPEAKER_01의 평균 음량이 계산된다
    And 두 화자 간 음량 차이가 dB 단위로 표시된다

  Scenario: 감정 격화 구간 자동 감지
    Given 음량 분석이 완료된 세그먼트 목록이 있다
    When 기준 음량 대비 150% 이상인 구간이 감지된다
    Then 해당 구간이 "감정 격화 구간"으로 태깅된다
    And 태그된 구간의 시작/종료 타임스탬프가 기록된다
```

### AC-1.2: 피치 분석 정상 동작

```gherkin
Feature: 피치 분석 (Pitch/F0 Analysis)
  음성의 기본 주파수와 변동성을 분석한다

  Scenario: 기본 주파수(F0) 추출
    Given 음성 세그먼트가 로드되어 있다
    When 피치 분석을 요청한다
    Then F0 평균값 (Hz)이 반환된다
    And F0 표준편차 (Hz)가 반환된다
    And F0 최소/최대값 (Hz)이 반환된다
    And 반환값은 75Hz ~ 600Hz 범위 내에 있다

  Scenario: Jitter (주파수 미세 변동) 계산
    Given 피치 분석이 완료된 세그먼트가 있다
    When Jitter 값을 계산한다
    Then Jitter 값이 백분율(%)로 반환된다
    And 정상 범위(< 1%)와 스트레스 범위(> 2%)가 표시된다

  Scenario: 피치 범위 계산 (반음 단위)
    Given F0 최소/최대값이 계산되어 있다
    When 피치 범위를 반음(semitone) 단위로 변환한다
    Then 피치 범위가 0 ~ 24 반음 사이의 값으로 반환된다
```

### AC-1.3: 말 속도 분석 정상 동작

```gherkin
Feature: 말 속도 분석 (Speech Rate Analysis)
  발화 속도와 휴지 패턴을 분석한다

  Scenario: WPM (Words Per Minute) 계산
    Given WhisperX word-level 타임스탬프가 있는 세그먼트가 있다
    When 말 속도 분석을 요청한다
    Then WPM 값이 계산된다
    And WPM 값은 60 ~ 300 범위 내에 있다

  Scenario: 휴지(Pause) 감지 및 통계
    Given 오디오 세그먼트가 로드되어 있다
    When 휴지 구간을 분석한다
    Then 휴지 횟수가 반환된다
    And 평균 휴지 시간 (ms)이 계산된다
    And 발화 비율 (speech_ratio)이 0~1 사이 값으로 반환된다

  Scenario: 말 속도 급변 감지
    Given 연속된 세그먼트의 말 속도 데이터가 있다
    When 말 속도가 기준 대비 30% 이상 변화한 구간이 감지된다
    Then 해당 구간이 "속도 변화 구간"으로 태깅된다
```

### AC-1.4: 스트레스 분석 정상 동작

```gherkin
Feature: 음성 스트레스 분석 (Voice Stress Analysis)
  음성 기반 스트레스 지표를 계산한다

  Scenario: Shimmer 및 HNR 계산
    Given Parselmouth가 설치되어 있다
    And 오디오 세그먼트가 로드되어 있다
    When 스트레스 분석을 요청한다
    Then Shimmer 값이 백분율(%)로 반환된다
    And HNR (Harmonic-to-Noise Ratio) 값이 dB 단위로 반환된다

  Scenario: 종합 스트레스 지수 산출
    Given Jitter, Shimmer, HNR, F0 표준편차가 계산되어 있다
    When 종합 스트레스 지수를 계산한다
    Then 0~100 범위의 스트레스 지수가 반환된다
    And 지수가 70 이상이면 "고스트레스"로 분류된다

  Scenario: 스트레스 지표 결정론적 재현
    Given 동일한 오디오 파일 "sample_001.m4a"가 있다
    When 스트레스 분석을 2회 수행한다
    Then 두 결과의 스트레스 지수가 동일하다
```

---

## 2. Speech Emotion Recognition (SER)

### AC-2.1: SER 모델 추론 정상 동작

```gherkin
Feature: Speech Emotion Recognition
  딥러닝 모델을 사용하여 음성 기반 감정을 인식한다

  Background:
    Given SER 모델이 GPU에 로드되어 있다
    And 오디오 파일이 16kHz mono WAV로 전처리되어 있다

  Scenario: 감정 분류 수행
    Given 오디오 세그먼트가 준비되어 있다
    When SER 추론을 요청한다
    Then 예측된 감정 레이블이 반환된다
    And 감정은 [joy, sadness, anger, fear, disgust, surprise, neutral] 중 하나이다
    And 신뢰도 점수 (0~1)가 함께 반환된다

  Scenario: 감정 분포 출력
    Given SER 추론이 완료되었다
    When 감정 분포를 요청한다
    Then 7가지 감정별 확률 분포가 반환된다
    And 모든 확률의 합은 1.0이다

  Scenario: 저신뢰도 결과 플래깅
    Given SER 추론 결과의 신뢰도가 0.6 미만이다
    When 결과를 저장한다
    Then "전문가 검토 필요" 플래그가 설정된다
```

### AC-2.2: 텍스트-음성 감정 교차 검증

```gherkin
Feature: 텍스트-음성 감정 교차 검증
  텍스트 기반 감정과 음성 기반 감정을 비교한다

  Scenario: 감정 일치 확인
    Given 텍스트 감정 분석 결과가 "anger"이다
    And 음성 감정 분석 결과가 "anger"이다
    When 교차 검증을 수행한다
    Then is_match가 True로 반환된다
    And combined_confidence가 계산된다

  Scenario: 감정 불일치 감지
    Given 텍스트 감정 분석 결과가 "neutral"이다
    And 음성 감정 분석 결과가 "fear"이다
    When 교차 검증을 수행한다
    Then is_match가 False로 반환된다
    And conflict_type이 "text_voice_mismatch"로 설정된다
    And "감정 불일치" 플래그가 설정된다
```

---

## 3. 가스라이팅 시계열 분석

### AC-3.1: 시계열 패턴 분석

```gherkin
Feature: 가스라이팅 시계열 분석
  6개월간의 녹취 데이터에서 가스라이팅 패턴 진행을 분석한다

  Background:
    Given 183개의 녹취 파일이 날짜순으로 정렬되어 있다
    And 각 파일에 가스라이팅 패턴 분석 결과가 존재한다

  Scenario: 패턴 빈도 추이 분석
    Given 6개월간의 가스라이팅 패턴 데이터가 있다
    When 시계열 분석을 요청한다
    Then 패턴 유형별(부정, 전가, 축소, 혼란) 월별 빈도가 계산된다
    And 전체 패턴 빈도의 시간적 추세가 반환된다

  Scenario: 에스컬레이션 감지
    Given 패턴 강도 데이터가 시간순으로 정렬되어 있다
    When 선형 회귀 분석을 수행한다
    Then 기울기(slope)가 계산된다
    And 기울기가 양수이면 "에스컬레이션 감지"로 분류된다
    And p-value가 0.05 미만이면 "통계적 유의성"이 확인된다

  Scenario: 연속 패턴 고위험 구간 식별
    Given 동일 화자에서 5분 이내 가스라이팅 패턴이 3회 이상 감지된다
    When 연속 패턴 분석을 수행한다
    Then 해당 구간이 "고위험 가스라이팅 구간"으로 분류된다
    And 알림이 생성된다
```

### AC-3.2: 피해자 감정 변화 추적

```gherkin
Feature: 피해자 감정 변화 추적
  시간에 따른 피해자의 감정 상태 변화를 추적한다

  Scenario: 감정 악화 추세 감지
    Given 피해자(SPEAKER_00)의 6개월간 감정 데이터가 있다
    When 부정 감정 빈도의 추세를 분석한다
    Then 부정 감정(anger, fear, sadness) 비율이 시간에 따라 증가하면
    And "심리적 피해 진행" 지표가 생성된다

  Scenario: 감정 변동성 분석
    Given 피해자의 감정 강도 데이터가 있다
    When 감정 변동성(volatility)을 계산한다
    Then 변동성 점수가 0~1 사이로 반환된다
    And 변동성이 0.7 이상이면 "불안정 상태"로 분류된다
```

---

## 4. 심리적 압박 지수

### AC-4.1: 종합 압박 지수 산출

```gherkin
Feature: 심리적 압박 지수 계산
  가스라이팅 패턴, 감정, 음성 특성을 종합한 압박 지수를 산출한다

  Scenario: 압박 지수 계산
    Given 가스라이팅 점수가 0.85이다
    And 부정 감정 비율이 0.65이다
    And 음성 스트레스 평균이 72이다
    And 가해자 발화 비율이 0.70이다
    When 심리 압박 지수를 계산한다
    Then 0~100 범위의 압박 지수가 반환된다
    And 구성 요소별 점수가 함께 제공된다

  Scenario: 위험도 레벨 분류
    Given 압박 지수가 계산되어 있다
    When 위험도 레벨을 결정한다
    Then 지수 0-25: "낮음"
    And 지수 26-50: "중간"
    And 지수 51-75: "높음"
    And 지수 76-100: "매우 높음"
```

---

## 5. 시각화 대시보드

### AC-5.1: 타임라인 시각화

```gherkin
Feature: 가스라이팅 타임라인 시각화
  가스라이팅 패턴의 시간적 진행을 시각화한다

  Scenario: 타임라인 차트 렌더링
    Given 시계열 가스라이팅 데이터가 있다
    When 타임라인 시각화를 요청한다
    Then Plotly 인터랙티브 차트가 생성된다
    And X축은 날짜, Y축은 패턴 강도를 표시한다
    And 패턴 유형별 색상이 구분된다

  Scenario: 인터랙티브 기능 동작
    Given 타임라인 차트가 렌더링되어 있다
    When 특정 구간을 줌인한다
    Then 해당 구간이 확대된다
    When 마커를 클릭한다
    Then 해당 발언의 상세 정보가 표시된다
```

### AC-5.2: Streamlit 대시보드

```gherkin
Feature: Streamlit 대시보드
  분석 결과를 통합 대시보드로 제공한다

  Scenario: 대시보드 로딩
    Given Streamlit 앱이 실행 중이다
    When 대시보드 URL에 접속한다
    Then 메인 페이지가 3초 이내에 로드된다
    And 파일 선택 사이드바가 표시된다
    And 타임라인 차트가 표시된다

  Scenario: 파일 필터링
    Given 183개 파일 목록이 표시되어 있다
    When 날짜 범위 필터를 적용한다
    Then 해당 기간의 파일만 표시된다
    And 차트가 필터링된 데이터로 업데이트된다
```

---

## 6. 포렌식 리포트 생성

### AC-6.1: PDF 리포트 생성

```gherkin
Feature: 법적 증거 리포트 생성
  분석 결과를 PDF 형식의 종합 리포트로 생성한다

  Background:
    Given 모든 분석(음성 특성, SER, 가스라이팅, 압박 지수)이 완료되어 있다

  Scenario: PDF 리포트 생성
    When 리포트 생성을 요청한다
    Then PDF 파일이 생성된다
    And 파일 크기는 10MB 이내이다
    And 한글 폰트가 정상적으로 렌더링된다

  Scenario: 리포트 섹션 완전성
    Given PDF 리포트가 생성되어 있다
    When 리포트 내용을 확인한다
    Then 요약 페이지가 포함되어 있다
    And 분석 방법론 섹션이 포함되어 있다
    And 시계열 분석 차트가 포함되어 있다
    And 증거 구간 목록이 포함되어 있다
    And 화자별 프로파일이 포함되어 있다
    And 제한사항 및 면책 섹션이 포함되어 있다

  Scenario: 증거 하이라이트 추출
    Given 고위험 가스라이팅 구간이 5개 감지되어 있다
    When 리포트를 생성한다
    Then "핵심 증거 구간" 섹션에 5개 구간이 나열된다
    And 각 구간의 타임스탬프, 발언 내용, 패턴 유형이 포함된다
```

---

## 7. 품질 및 성능 요구사항

### AC-7.1: 처리 성능

```gherkin
Feature: 성능 요구사항
  시스템의 처리 성능을 검증한다

  Scenario: 단일 파일 분석 시간
    Given 평균 5분 길이의 오디오 파일이 있다
    When 전체 포렌식 분석(음성 특성 + SER)을 수행한다
    Then 분석 시간은 파일 길이의 20% 이내이다 (5분 파일 -> 1분 이내)

  Scenario: 배치 처리 성능
    Given 183개 파일이 분석 대기열에 있다
    When 배치 포렌식 분석을 수행한다
    Then 전체 처리 시간은 30분 이내이다
    And GPU 활용률은 80% 이상이다
```

### AC-7.2: 데이터 무결성

```gherkin
Feature: 데이터 무결성
  원본 데이터의 무결성과 분석 추적성을 보장한다

  Scenario: 원본 파일 무변조 검증
    Given 분석 전 원본 파일의 SHA-256 해시가 "abc123"이다
    When 모든 분석이 완료된다
    Then 원본 파일의 SHA-256 해시가 여전히 "abc123"이다

  Scenario: 분석 결과 추적성
    Given 분석 결과가 저장되어 있다
    When 결과의 메타데이터를 확인한다
    Then 원본 파일 해시가 연결되어 있다
    And 분석 시점이 기록되어 있다
    And 사용 알고리즘 버전이 기록되어 있다

  Scenario: 결정론적 분석 재현
    Given 동일한 입력 파일과 설정이 있다
    When 분석을 2회 수행한다
    Then 두 결과의 수치가 완전히 동일하다
```

### AC-7.3: 테스트 커버리지

```gherkin
Feature: 테스트 커버리지
  코드 품질을 보장하기 위한 테스트 커버리지 검증

  Scenario: 단위 테스트 커버리지
    Given 모든 서비스 클래스가 구현되어 있다
    When pytest --cov를 실행한다
    Then 커버리지가 85% 이상이다

  Scenario: 통합 테스트 통과
    Given Phase 1-4 모든 기능이 구현되어 있다
    When 전체 파이프라인 통합 테스트를 실행한다
    Then 모든 테스트가 통과한다
```

---

## 8. 비기능 요구사항 검증

### AC-8.1: 보안 및 개인정보

```gherkin
Feature: 보안 및 개인정보 보호
  민감 데이터 보호 요구사항을 검증한다

  Scenario: 화자 ID 익명화
    Given 분석 결과에 화자 정보가 포함되어 있다
    When 결과를 저장한다
    Then 화자는 "SPEAKER_00", "SPEAKER_01" 형식으로 익명화된다
    And 실명 매핑은 별도 암호화 파일에 저장된다

  Scenario: 로그 마스킹
    Given 분석 중 로그가 생성된다
    When 로그 내용을 확인한다
    Then 개인 식별 정보가 마스킹 처리되어 있다
```

### AC-8.2: 오류 처리

```gherkin
Feature: 오류 처리
  시스템의 오류 복구 능력을 검증한다

  Scenario: GPU 메모리 부족 대응
    Given GPU 메모리 사용률이 90%를 초과한다
    When SER 추론을 요청한다
    Then 배치 크기가 자동으로 축소된다
    And 처리가 계속 진행된다

  Scenario: 손상된 오디오 파일 처리
    Given 손상된 오디오 파일이 입력된다
    When 분석을 시도한다
    Then 명확한 오류 메시지가 반환된다
    And 다른 파일의 처리는 영향받지 않는다
```

---

## 정의 완료 조건 (Definition of Done)

### Phase 1 완료 조건
- [ ] 음량/피치/속도/스트레스 분석 서비스 구현 완료
- [ ] 모든 AC-1.x 시나리오 통과
- [ ] 단위 테스트 커버리지 85% 이상
- [ ] 183개 파일 배치 처리 성공

### Phase 2 완료 조건
- [ ] SER 서비스 구현 및 GPU 추론 동작
- [ ] 텍스트-음성 감정 교차 검증 구현
- [ ] 가스라이팅 시계열 분석 구현
- [ ] 심리 압박 지수 계산 구현
- [ ] 모든 AC-2.x, AC-3.x, AC-4.x 시나리오 통과

### Phase 3 완료 조건
- [ ] 4가지 시각화 차트 구현
- [ ] Streamlit 대시보드 동작
- [ ] 모든 AC-5.x 시나리오 통과
- [ ] 차트 로딩 시간 3초 이내

### Phase 4 완료 조건
- [ ] PDF 리포트 생성 구현
- [ ] 모든 리포트 섹션 포함
- [ ] 한글 폰트 정상 렌더링
- [ ] 모든 AC-6.x 시나리오 통과

### 전체 완료 조건
- [ ] 모든 Phase 완료
- [ ] 전체 테스트 커버리지 85% 이상
- [ ] TRUST 5 품질 게이트 통과
- [ ] 문서화 완료 (README, API docs)
- [ ] 코드 리뷰 완료

---

## 테스트 데이터 요구사항

### 필수 테스트 데이터
1. **정상 음성 파일**: 잡음 없는 고품질 녹음 (3개 이상)
2. **저품질 음성 파일**: SNR 15dB 이하 (2개 이상)
3. **다화자 음성 파일**: 2인 이상 화자 (5개 이상)
4. **감정 표현 음성**: 분노, 공포, 슬픔 등 명확한 감정 (각 2개 이상)
5. **가스라이팅 패턴 포함**: 부정, 전가, 축소, 혼란 (각 3개 이상)

### 테스트 환경
- GPU: NVIDIA GB10 (또는 동급)
- RAM: 24GB 이상
- Python: 3.11+
- CUDA: 12.1+

---

## 참조

- SPEC-FORENSIC-001/spec.md: 상세 요구사항
- SPEC-FORENSIC-001/plan.md: 구현 계획
- SPEC-WHISPERX-001: WhisperX 파이프라인 (입력)
- SPEC-E2ETEST-001: E2E 테스트 참조

---

**문서 끝**
