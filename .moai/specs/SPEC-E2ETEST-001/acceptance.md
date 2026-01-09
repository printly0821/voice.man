---
id: SPEC-E2ETEST-001
type: acceptance
version: "1.0.0"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
status: "planned"
related_spec: SPEC-E2ETEST-001
---

# SPEC-E2ETEST-001: 인수 조건

## TRACEABILITY
- **SPEC**: [SPEC-E2ETEST-001](./spec.md)
- **Plan**: [plan.md](./plan.md)

---

## 인수 조건 개요

본 문서는 SPEC-E2ETEST-001 (WhisperX E2E 통합 테스트 - GPU 병렬 배치 처리)의 인수 조건을 정의합니다. 모든 시나리오는 Given-When-Then 형식으로 작성되었습니다.

---

## 시나리오 1: 전체 파일 배치 처리 성공

### AC-001: 183개 파일 전체 처리 완료

```gherkin
Feature: 183개 m4a 파일 전체 GPU 병렬 배치 처리

  Scenario: 전체 파일 처리 성공
    Given ref/call/ 디렉토리에 183개의 m4a 오디오 파일이 존재한다
    And GPU가 사용 가능한 상태이다
    And Hugging Face 토큰이 환경 변수로 설정되어 있다
    And WhisperX 파이프라인 모델이 로드되어 있다
    When E2E 테스트 스크립트를 실행한다
    Then 183개 파일 전체가 처리되어야 한다
    And 각 파일에 대해 전사(STT) 결과가 생성되어야 한다
    And 각 파일에 대해 word-level 타임스탬프가 생성되어야 한다
    And 각 파일에 대해 화자 분리 결과가 생성되어야 한다
    And 처리 결과가 ref/call/reports/ 디렉토리에 저장되어야 한다
```

### AC-002: 원본 파일 무결성 보장

```gherkin
Feature: 원본 파일 무결성 검증

  Scenario: 원본 파일이 수정되지 않음
    Given ref/call/ 디렉토리의 183개 m4a 파일에 대한 MD5 체크섬을 계산한다
    When E2E 테스트를 실행하고 완료된다
    Then 모든 원본 m4a 파일의 MD5 체크섬이 테스트 전과 동일해야 한다
    And 원본 파일의 수정 시간이 변경되지 않아야 한다
    And 원본 디렉토리에 새로운 파일이 생성되지 않아야 한다
```

---

## 시나리오 2: 성능 목표 달성

### AC-003: 처리 시간 목표 달성

```gherkin
Feature: 성능 목표 달성 검증

  Scenario: 20분 이내 전체 처리 완료
    Given 183개 m4a 파일이 처리 대기 중이다
    And GPU 배치 크기가 20으로 설정되어 있다
    When E2E 테스트를 시작하고 처리 시작 시간을 기록한다
    Then 전체 처리가 20분(1200초) 이내에 완료되어야 한다
    And 파일당 평균 처리 시간이 6초 이내여야 한다
    And 종합 리포트에 총 처리 시간이 기록되어야 한다

  Scenario: GPU 활용률 목표 달성
    Given E2E 테스트가 실행 중이다
    And GPU 모니터링이 활성화되어 있다
    When 배치 처리가 진행된다
    Then GPU 활용률이 평균 85% 이상이어야 한다
    And GPU 활용률이 종합 리포트에 기록되어야 한다
```

### AC-004: GPU 메모리 제한 준수

```gherkin
Feature: GPU 메모리 제한 준수

  Scenario: GPU 메모리 95% 미만 유지
    Given E2E 테스트가 실행 중이다
    And GPU 메모리 모니터링이 1초 간격으로 실행 중이다
    When 183개 파일이 순차적으로 배치 처리된다
    Then GPU 메모리 사용률이 95%를 초과하지 않아야 한다
    And GPU 메모리 최대 사용량이 종합 리포트에 기록되어야 한다

  Scenario: 메모리 부족 시 동적 배치 조정
    Given 초기 배치 크기가 20으로 설정되어 있다
    And GPU 메모리 사용률이 80%를 초과한다
    When 다음 배치 처리를 시작한다
    Then 배치 크기가 자동으로 50% 감소해야 한다
    And 배치 크기 조정 이력이 로그에 기록되어야 한다
```

---

## 시나리오 3: 결과 저장 및 리포트 생성

### AC-005: 종합 리포트 생성

```gherkin
Feature: 종합 리포트 생성

  Scenario: JSON 리포트 생성
    Given 183개 파일 처리가 완료되었다
    When 종합 리포트 생성 로직이 실행된다
    Then ref/call/reports/e2e_test_report.json 파일이 생성되어야 한다
    And 리포트에 테스트 메타데이터가 포함되어야 한다
      | 필드 | 설명 |
      | test_start_time | 테스트 시작 시간 (ISO 8601) |
      | test_end_time | 테스트 종료 시간 (ISO 8601) |
      | total_files | 총 파일 수 (183) |
      | success_count | 성공 파일 수 |
      | failed_count | 실패 파일 수 |
    And 리포트에 성능 통계가 포함되어야 한다
      | 필드 | 설명 |
      | total_duration_seconds | 총 처리 시간 (초) |
      | avg_file_duration_seconds | 파일당 평균 처리 시간 (초) |
      | gpu_memory_peak_mb | GPU 메모리 최대 사용량 (MB) |
      | gpu_utilization_avg_percent | GPU 평균 활용률 (%) |

  Scenario: Markdown 요약 리포트 생성
    Given JSON 리포트가 생성되었다
    When Markdown 요약 리포트 생성 로직이 실행된다
    Then ref/call/reports/e2e_test_summary.md 파일이 생성되어야 한다
    And 리포트에 처리 결과 요약이 포함되어야 한다
    And 리포트에 성능 통계 테이블이 포함되어야 한다
    And 리포트에 화자별 발화 통계가 포함되어야 한다
```

### AC-006: 파일별 결과 저장

```gherkin
Feature: 파일별 결과 저장

  Scenario: 개별 파일 결과 저장
    Given 파일 "통화 녹음 신기연_250603_201127.m4a"가 처리 완료되었다
    When 결과 저장 로직이 실행된다
    Then ref/call/reports/results/ 디렉토리에 결과 JSON이 저장되어야 한다
    And 결과 파일에 전사 텍스트가 포함되어야 한다
    And 결과 파일에 세그먼트 정보가 포함되어야 한다
      | 필드 | 설명 |
      | word | 단어 텍스트 |
      | start | 시작 시간 (초) |
      | end | 종료 시간 (초) |
      | speaker | 화자 ID |
    And 결과 파일에 화자 정보가 포함되어야 한다
    And 결과 파일에 처리 메타데이터가 포함되어야 한다
```

---

## 시나리오 4: 진행률 및 콜백

### AC-007: 진행률 콜백 동작

```gherkin
Feature: 진행률 콜백 시스템

  Scenario: 배치 완료 시 진행률 업데이트
    Given E2E 테스트가 진행률 콜백과 함께 실행 중이다
    And 첫 번째 배치 (20개 파일)가 처리 완료되었다
    When 진행률 콜백이 호출된다
    Then 콜백에 다음 정보가 전달되어야 한다
      | 파라미터 | 값 |
      | current | 20 |
      | total | 183 |
      | elapsed_seconds | (경과 시간) |
      | current_file | (현재 파일명) |
      | status | "completed" |
    And 콘솔에 진행률이 출력되어야 한다
    And 예상 완료 시간(ETA)이 계산되어야 한다

  Scenario: 진행률 로그 기록
    Given E2E 테스트가 실행 중이다
    When 각 배치가 완료된다
    Then 로그 파일에 배치 처리 통계가 기록되어야 한다
    And 로그에 배치 번호, 처리 시간, 파일 수가 포함되어야 한다
```

---

## 시나리오 5: 실패 처리 및 재시도

### AC-008: 실패 파일 재시도

```gherkin
Feature: 실패 파일 재시도 로직

  Scenario: 일시적 오류 발생 시 재시도
    Given 파일 처리 중 일시적 오류 (메모리 부족, 파일 접근 실패)가 발생했다
    When 재시도 로직이 실행된다
    Then 5초 후 첫 번째 재시도가 수행되어야 한다
    And 첫 번째 재시도 실패 시 15초 후 두 번째 재시도가 수행되어야 한다
    And 두 번째 재시도 실패 시 30초 후 세 번째 재시도가 수행되어야 한다
    And 모든 재시도 시도가 로그에 기록되어야 한다

  Scenario: 3회 재시도 후 최종 실패
    Given 파일 처리가 3회 연속 실패했다
    When 최종 실패 처리 로직이 실행된다
    Then 해당 파일이 failed_files.json에 추가되어야 한다
    And 실패 원인이 상세히 기록되어야 한다
    And 다음 파일 처리가 계속 진행되어야 한다
```

### AC-009: 실패 파일 리포트

```gherkin
Feature: 실패 파일 리포트

  Scenario: 실패 파일 목록 생성
    Given 일부 파일 처리가 최종 실패했다
    When 종합 리포트가 생성된다
    Then ref/call/reports/failed_files.json 파일이 생성되어야 한다
    And 각 실패 파일에 대해 다음 정보가 포함되어야 한다
      | 필드 | 설명 |
      | file_path | 파일 경로 |
      | error_message | 오류 메시지 |
      | retry_count | 재시도 횟수 |
      | last_error_time | 마지막 오류 시간 |
    And 종합 리포트에 실패 통계가 포함되어야 한다
```

---

## 시나리오 6: 화자 분석

### AC-010: 화자별 발화 통계

```gherkin
Feature: 화자별 발화 통계 생성

  Scenario: 화자 통계 계산
    Given 183개 파일 처리가 완료되었다
    And 각 파일에 화자 분리 결과가 있다
    When 화자 통계 계산 로직이 실행된다
    Then 화자별 다음 통계가 계산되어야 한다
      | 통계 항목 | 설명 |
      | total_speaking_time | 총 발화 시간 (초) |
      | speaking_percentage | 발화 비율 (%) |
      | turn_count | 발화 횟수 (턴 수) |
      | avg_turn_duration | 평균 턴 길이 (초) |
    And 통계가 종합 리포트에 포함되어야 한다

  Scenario: 주요 발화자 식별
    Given 화자 통계가 계산되었다
    When 주요 발화자 분석이 수행된다
    Then 발화 시간 기준 상위 화자가 식별되어야 한다
    And 화자별 발화 패턴이 요약되어야 한다
```

---

## 품질 게이트

### QG-001: TRUST 5 준수 검증

| 품질 항목 | 기준 | 검증 방법 |
|-----------|------|-----------|
| Test-first | 테스트 커버리지 85% 이상 | pytest --cov 실행 |
| Readable | ruff 린트 경고 0개 | ruff check 실행 |
| Unified | black 포맷팅 준수 | black --check 실행 |
| Secured | 하드코딩된 토큰 없음 | grep 검색 |
| Trackable | 모든 메트릭 로깅 | 로그 파일 검증 |

### QG-002: 성능 목표 검증

| 성능 항목 | 목표값 | 허용 범위 |
|-----------|--------|-----------|
| 총 처리 시간 | 20분 이내 | -10% (18분) ~ +5% (21분) |
| 파일당 평균 처리 시간 | 6초 이내 | +10% (6.6초) |
| GPU 메모리 사용률 | 95% 미만 | 필수 (초과 불가) |
| GPU 활용률 | 85% 이상 | -5% (80% 이상) |
| 성공률 | 100% | 재시도 후 기준 |

---

## Definition of Done

### 기능 완료 조건
- [ ] 183개 파일 전체 처리 성공 (100%)
- [ ] 전사, 정렬, 화자 분리 결과 생성
- [ ] 결과가 reports/ 디렉토리에 저장
- [ ] 종합 리포트 (JSON + Markdown) 생성
- [ ] 진행률 콜백 정상 동작

### 성능 완료 조건
- [ ] 총 처리 시간 20분 이내
- [ ] GPU 메모리 95% 미만 유지
- [ ] GPU 활용률 85% 이상

### 품질 완료 조건
- [ ] 원본 파일 무결성 100% 보장
- [ ] 테스트 커버리지 85% 이상
- [ ] ruff 린트 경고 0개
- [ ] 실패 파일 로깅 완료

---

## 검증 체크리스트

### 테스트 전 체크리스트
- [ ] GPU 가용성 확인 (`nvidia-smi`)
- [ ] Hugging Face 토큰 설정 확인 (`echo $HF_TOKEN`)
- [ ] 183개 m4a 파일 존재 확인 (`ls ref/call/*.m4a | wc -l`)
- [ ] 디스크 공간 확인 (최소 5GB 여유)
- [ ] 모델 캐시 확인 (~/.cache/huggingface/)

### 테스트 후 체크리스트
- [ ] 종합 리포트 생성 확인
- [ ] 성능 목표 달성 여부 확인
- [ ] 실패 파일 목록 확인
- [ ] 원본 파일 무결성 확인
- [ ] GPU 메모리 누수 확인

---

## 참조 문서

- [SPEC-E2ETEST-001 spec.md](./spec.md)
- [SPEC-E2ETEST-001 plan.md](./plan.md)
- [SPEC-PARALLEL-001](../SPEC-PARALLEL-001/spec.md)
- [SPEC-WHISPERX-001](../SPEC-WHISPERX-001/spec.md)

---

**문서 끝**
