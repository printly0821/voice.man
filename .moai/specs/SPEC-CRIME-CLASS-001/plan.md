---
id: SPEC-CRIME-CLASS-001
version: "1.0.0"
status: "planned"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
---

# SPEC-CRIME-CLASS-001: 구현 계획서

## 1. 개요

본 문서는 SPEC-CRIME-CLASS-001(범죄 유형 분류 시스템)의 구현 계획을 정의한다.

### 1.1 목표

- 11개 범죄 유형 멀티모달 분류 시스템 구축
- 법적 증거 매핑 및 검찰 제출 포맷 생성
- 기존 SPEC-FORENSIC-001 파이프라인과 통합

### 1.2 범위

- Phase 1: 범죄 언어 패턴 확장 및 분류기 기반 구축
- Phase 2: 심리 프로파일링 엔진 구현
- Phase 3: 법적 증거 매핑 서비스 구현
- Phase 4: 검찰 제출 포맷 및 통합

---

## 2. 마일스톤

### 2.1 Phase 1: 범죄 언어 패턴 확장 및 분류기 기반 (Primary Goal)

**목표**: 11개 죄명에 대한 언어 패턴 DB 확장 및 기본 분류기 구현

**작업 항목**:

1. **확장 범죄 언어 패턴 DB 구축**
   - 사기 패턴 정의 (30개 이상)
   - 공갈 패턴 정의 (25개 이상)
   - 강요 패턴 정의 (25개 이상)
   - 모욕 패턴 정의 (30개 이상)
   - 횡령 패턴 정의 (20개 이상)
   - 배임 패턴 정의 (20개 이상)
   - 조세포탈 패턴 정의 (15개 이상)
   - 가정폭력 패턴 정의 (25개 이상)
   - 스토킹 패턴 정의 (25개 이상)

2. **텍스트 기반 범죄 분류기 구현**
   - `extended_crime_patterns.py` 구현
   - 패턴 매칭 알고리즘 구현
   - 죄명별 점수 계산 로직

3. **음성 기반 범죄 지표 매핑**
   - 기존 `audio_feature_service.py` 결과 활용
   - 죄명별 음성 지표 매핑 테이블 정의
   - 음성 점수 변환 로직 구현

**산출물**:
- `src/voice_man/data/crime_classification/crime_patterns_extended.json`
- `src/voice_man/services/crime_classification/extended_crime_patterns.py`
- 단위 테스트: 패턴 매칭 정확도 95% 이상

---

### 2.2 Phase 2: 심리 프로파일링 엔진 (Secondary Goal)

**목표**: 다크 트라이어드 및 성격 특성 기반 심리 프로파일링 구현

**작업 항목**:

1. **심리 지표 정의**
   - 나르시시즘 지표 정의
   - 마키아벨리즘 지표 정의
   - 사이코패시 지표 정의
   - 애착 유형 지표 정의

2. **심리 프로파일러 구현**
   - `psychological_profiler.py` 구현
   - 텍스트 기반 심리 지표 추출
   - 음성 기반 심리 지표 추출
   - 다크 트라이어드 점수 계산

3. **범죄 성향 예측**
   - 심리 프로파일-범죄 유형 매핑
   - 성향 점수 계산 알고리즘

**산출물**:
- `src/voice_man/services/crime_classification/psychological_profiler.py`
- `src/voice_man/models/crime_classification/psychological_profile.py`
- 단위 테스트: 심리 프로파일 생성 검증

---

### 2.3 Phase 3: 법적 증거 매핑 서비스 (Secondary Goal)

**목표**: 형법 조문 매핑 및 구성요건 충족 자동 평가

**작업 항목**:

1. **한국 형법 DB 구축**
   - 형법 조문 JSON 구조화
   - 각 조문별 구성요건 정의
   - 판례 기반 증거 요건 정의

2. **증거 매핑 서비스 구현**
   - `legal_evidence_mapper.py` 구현
   - 구성요건 충족 평가 알고리즘
   - 충족률 계산 로직

3. **신뢰 구간 계산기 구현**
   - `confidence_calculator.py` 구현
   - 부트스트랩 방법론 적용
   - 95% 신뢰 구간 계산

**산출물**:
- `src/voice_man/data/crime_classification/korean_criminal_code.json`
- `src/voice_man/data/crime_classification/legal_requirements.json`
- `src/voice_man/services/crime_classification/legal_evidence_mapper.py`
- `src/voice_man/services/crime_classification/confidence_calculator.py`

---

### 2.4 Phase 4: 검찰 제출 포맷 및 통합 (Final Goal)

**목표**: 검찰 제출용 보고서 생성 및 전체 시스템 통합

**작업 항목**:

1. **멀티모달 앙상블 분류기 통합**
   - `multimodal_classifier.py` 구현
   - 3개 모달리티 가중 앙상블
   - 범죄 유형별 가중치 적용

2. **검찰 제출 포맷 생성기**
   - `prosecution_formatter.py` 구현
   - PDF 보고서 템플릿 설계
   - 죄명별 증거 요약 포맷

3. **API 엔드포인트 구현**
   - 분류 API 엔드포인트
   - 보고서 생성 API 엔드포인트
   - 화자별 조회 API 엔드포인트

4. **SPEC-FORENSIC-001 파이프라인 통합**
   - 포렌식 분석 완료 시 자동 분류 트리거
   - 결과 병합 및 저장

**산출물**:
- `src/voice_man/services/crime_classification/multimodal_classifier.py`
- `src/voice_man/services/crime_classification/prosecution_formatter.py`
- API 문서 및 통합 테스트

---

## 3. 기술적 접근

### 3.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Crime Classification Pipeline                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Text Analyzer  │    │  Audio Analyzer │    │ Psych Profiler│ │
│  │  (Extended      │    │  (Audio Feature │    │ (Dark Triad   │ │
│  │   Patterns)     │    │   + Stress)     │    │  + Traits)    │ │
│  └────────┬────────┘    └────────┬────────┘    └───────┬──────┘ │
│           │                      │                     │        │
│           └──────────────────────┼─────────────────────┘        │
│                                  ▼                              │
│                    ┌─────────────────────────┐                  │
│                    │  Multimodal Ensemble    │                  │
│                    │  Classifier             │                  │
│                    │  (Weighted Fusion)      │                  │
│                    └────────────┬────────────┘                  │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Crime Classification Result                  ││
│  │  - 11 Crime Types with Confidence                           ││
│  │  - Modality Contribution Scores                              ││
│  │  - Confidence Intervals                                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ Legal Evidence  │    │  Confidence     │    │ Prosecution  │ │
│  │ Mapper          │    │  Calculator     │    │ Formatter    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 핵심 알고리즘

#### 3.2.1 멀티모달 가중 앙상블

```python
def weighted_ensemble_classify(
    text_scores: Dict[CrimeType, float],
    audio_scores: Dict[CrimeType, float],
    psych_scores: Dict[CrimeType, float],
    weights: Dict[str, Dict[str, float]]
) -> Dict[CrimeType, float]:
    """
    가중 앙상블 분류

    각 범죄 유형에 최적화된 가중치 적용:
    - 텍스트 중심 범죄 (횡령, 배임, 조세포탈): text 50%
    - 음성 중심 범죄 (협박, 강요, 공갈): audio 40%
    - 심리 중심 범죄 (가스라이팅, 스토킹): psych 30%
    """
    results = {}
    for crime_type in CrimeType:
        w = weights[crime_type.value]
        results[crime_type] = (
            text_scores[crime_type] * w["text"] +
            audio_scores[crime_type] * w["audio"] +
            psych_scores[crime_type] * w["psych"]
        )
    return results
```

#### 3.2.2 신뢰 구간 계산 (부트스트랩)

```python
def bootstrap_confidence_interval(
    scores: List[float],
    n_iterations: int = 1000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    부트스트랩 방법론을 사용한 95% 신뢰 구간 계산
    """
    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return {
        "lower_95": lower,
        "upper_95": upper,
        "point_estimate": np.mean(scores)
    }
```

### 3.3 데이터 스키마

#### 3.3.1 범죄 패턴 JSON 스키마

```json
{
  "crime_type": "사기",
  "legal_code": "형법 제347조",
  "patterns": {
    "text_patterns": [
      {
        "pattern": "돈 빌려주면 갚을게",
        "weight": 0.8,
        "indicator": "기망행위"
      }
    ],
    "audio_indicators": [
      {
        "feature": "speech_rate",
        "threshold": ">180 WPM",
        "weight": 0.6
      }
    ],
    "psychological_indicators": [
      {
        "trait": "machiavellianism",
        "threshold": ">0.6",
        "weight": 0.7
      }
    ]
  }
}
```

#### 3.3.2 법적 구성요건 JSON 스키마

```json
{
  "crime_type": "사기",
  "legal_code": "형법 제347조",
  "requirements": [
    {
      "name": "기망행위",
      "description": "허위 사실을 진술하거나 사실을 은폐",
      "indicators": ["과장", "허위진술", "사실왜곡"],
      "required": true
    }
  ]
}
```

---

## 4. 의존성

### 4.1 내부 의존성

| 모듈 | 설명 | 상태 |
|------|------|------|
| SPEC-FORENSIC-001 | 포렌식 분석 파이프라인 | 완료 |
| crime_language_service.py | 범죄 언어 분석 | 완료 |
| audio_feature_service.py | 음성 특성 추출 | 완료 |
| stress_analysis_service.py | 스트레스 분석 | 완료 |
| ser_service.py | 감정 인식 | 완료 |
| cross_validation_service.py | 교차 검증 | 완료 |

### 4.2 외부 의존성

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| scikit-learn | 1.5+ | 앙상블 분류기 |
| numpy | 1.26+ | 수치 연산 |
| pandas | 2.2+ | 데이터 처리 |
| reportlab | 4.1+ | PDF 생성 |
| konlpy | 0.6+ | 한국어 형태소 분석 |

---

## 5. 리스크 및 대응

### 5.1 기술적 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 심리 프로파일링 정확도 | 분류 신뢰도 저하 | 학술 연구 기반 지표 활용 |
| 법적 구성요건 해석 | 오분류 위험 | 법률 전문가 검토 |
| 멀티모달 가중치 최적화 | 성능 저하 | 교차 검증 기반 튜닝 |

### 5.2 비즈니스 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 법적 책임 | 법적 분쟁 | 면책 조항 명시 |
| 오분류 | 증거 신뢰도 저하 | 전문가 검토 플래그 |
| 개인정보 노출 | 법적 제재 | 익명화 및 암호화 |

---

## 6. 테스트 전략

### 6.1 단위 테스트

- 패턴 매칭 정확도 테스트
- 심리 프로파일 생성 테스트
- 신뢰 구간 계산 테스트
- 가중 앙상블 테스트

### 6.2 통합 테스트

- 포렌식 파이프라인 연동 테스트
- 전체 분류 흐름 테스트
- 보고서 생성 테스트

### 6.3 검증 데이터셋

- 신기연 고소장 관련 녹취 파일
- 신동식 고소장 관련 녹취 파일
- 수동 레이블링된 테스트 세트

---

## 7. 추적성

### 7.1 SPEC 참조

| 요구사항 ID | 구현 위치 | 테스트 케이스 |
|-------------|-----------|---------------|
| F1 | multimodal_classifier.py | test_multimodal_classifier.py |
| F2 | extended_crime_patterns.py | test_crime_patterns.py |
| F3 | psychological_profiler.py | test_psychological.py |
| F4 | legal_evidence_mapper.py | test_legal_mapper.py |
| F5 | confidence_calculator.py | test_confidence.py |
| F6 | prosecution_formatter.py | test_formatter.py |

### 7.2 태그

```
[SPEC-CRIME-CLASS-001] [구현계획]
[Phase1] [Phase2] [Phase3] [Phase4]
[멀티모달] [심리프로파일링] [법적증거]
```

---

**문서 끝**
