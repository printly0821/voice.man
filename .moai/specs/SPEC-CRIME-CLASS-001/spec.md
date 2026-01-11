---
id: SPEC-CRIME-CLASS-001
version: "1.0.0"
status: "planned"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
title: "범죄 유형 분류 시스템"
related_specs:
  - SPEC-FORENSIC-001
  - SPEC-VOICE-001
  - SPEC-WHISPERX-001
tags:
  - 범죄분류
  - 멀티모달분석
  - 법적증거
  - 심리프로파일링
  - 한국형법
lifecycle: "spec-anchored"
---

# SPEC-CRIME-CLASS-001: 범죄 유형 분류 시스템

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-10 | 지니 | 초안 작성 - 멀티모달 범죄 유형 분류 시스템 요구사항 정의 |

---

## 1. 개요

### 1.1 목적

현재 SPEC-FORENSIC-001에서 구현된 5개 범죄 카테고리(가스라이팅, 협박, 강압, 기만, 감정조작)를 확장하여, 고소장에 명시된 11개 죄명(사기, 공갈, 강요, 협박, 모욕, 횡령, 조세포탈, 무등록영업, 배임)을 분류할 수 있는 멀티모달 범죄 유형 분류 시스템을 구축한다.

### 1.2 핵심 목표

1. **범죄 유형 정의**: 11개 죄명에 대한 텍스트/음성/심리 지표 정의
2. **멀티모달 융합**: 텍스트(30-40%), 음성(30-40%), 심리(20-30%) 점수 앙상블
3. **법적 증거 매핑**: 포렌식 결과를 특정 죄명에 매핑
4. **한국 법률 프레임워크**: 형법 조문 자동 참조 및 검찰 제출 포맷 지원

### 1.3 범위

- 기존 SPEC-FORENSIC-001 분석 결과 활용
- 11개 범죄 유형별 분류 모델 구축
- 고소장 2건(신기연, 신동식)의 죄명별 증거 매핑
- 법원 제출용 증거 요약 보고서 생성

### 1.4 분석 대상 고소장

#### 고소장 1: 신기연 (8개 죄명)
- 사기
- 공갈
- 강요
- 협박
- 모욕
- 횡령
- 조세포탈
- 무등록영업

#### 고소장 2: 신동식 (3개 죄명)
- 배임
- 사기
- 공갈

---

## 2. 범죄 유형 정의 (Crime Type Definitions)

### 2.1 범죄 유형별 지표 매트릭스

#### 2.1.1 가정폭력 (Domestic Violence)

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 협박, 통제, 고립, 경제적 통제, 행동 제한 |
| 음성 지표 | 고음, 공격적 톤, 급격한 음량 상승, 위협적 어조 |
| 심리 지표 | 권력 불균형, 지배 욕구, 분노 조절 장애 |

#### 2.1.2 스토킹 (Stalking)

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 집착, 감시, 위치 추적, 반복적 연락, 미행 |
| 음성 지표 | 반복 연락 패턴, 집착적 톤, 불안정한 음성 |
| 심리 지표 | 집착적 애착, 편집증적 사고, 경계선 인격 |

#### 2.1.3 협박 (Threat/Blackmail) - 형법 제283조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 직접 위협, 간접 위협, 조건부 위협, 해악 고지 |
| 음성 지표 | 위협적 음조, 낮고 단호한 톤, 의도적 강조 |
| 심리 지표 | 권력 과시, 공포 유발 의도, 지배 욕구 |

#### 2.1.4 가스라이팅 (Gaslighting)

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 현실 왜곡, 기억 부정, 감정 무효화, 자존감 공격 |
| 음성 지표 | 차분하고 조작적인 톤, 설득적 어조, 반복적 패턴 |
| 심리 지표 | 나르시시즘, 조작 성향, 공감 능력 결여 |

#### 2.1.5 사기 (Fraud) - 형법 제347조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 과장, 긴급성 강조, 허위 진술, 기만적 약속 |
| 음성 지표 | 설득적 톤, 자신감 과잉, 빠른 말속도 |
| 심리 지표 | 마키아벨리즘, 기만 성향, 양심 결여 |

#### 2.1.6 공갈 (Extortion) - 형법 제350조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 재산 요구, 협박 동반, 조건부 해악, 폭로 위협 |
| 음성 지표 | 단호한 톤, 압박적 어조, 강압적 말투 |
| 심리 지표 | 탐욕, 권력 지향, 타인 착취 성향 |

#### 2.1.7 강요 (Coercion) - 형법 제324조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 의무 없는 행위 요구, 폭행/협박 동반, 권리 침해 |
| 음성 지표 | 강압적 톤, 명령형 어조, 거부 불허 |
| 심리 지표 | 통제 욕구, 지배 성향, 경계 무시 |

#### 2.1.8 모욕 (Insult) - 형법 제311조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 경멸적 표현, 인격 비하, 공개적 모욕, 비속어 |
| 음성 지표 | 조롱하는 톤, 경멸적 어조, 비웃음 |
| 심리 지표 | 우월감, 타인 비하 성향, 자기애 |

#### 2.1.9 횡령 (Embezzlement) - 형법 제355조

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 금전 착복 언급, 회계 은폐, 용도 외 사용 |
| 음성 지표 | 회피적 톤, 모호한 설명, 방어적 어조 |
| 심리 지표 | 탐욕, 합리화 성향, 책임 회피 |

#### 2.1.10 배임 (Breach of Trust) - 형법 제355조 제2항

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 신임 위반, 자기 이익 추구, 의무 위배 |
| 음성 지표 | 정당화하는 톤, 변명적 어조, 책임 전가 |
| 심리 지표 | 자기합리화, 충성 결여, 기회주의 |

#### 2.1.11 조세포탈 (Tax Evasion)

| 지표 유형 | 세부 지표 |
|-----------|----------|
| 텍스트 지표 | 세금 회피 언급, 허위 신고, 소득 은닉 |
| 음성 지표 | 은밀한 톤, 낮은 음량, 주저하는 어조 |
| 심리 지표 | 탈법 성향, 책임 회피, 자기합리화 |

---

## 3. 환경 (Environment)

### 3.1 기술 스택

#### 3.1.1 멀티모달 분류 엔진

- **텍스트 분류**: 기존 crime_language_service.py 확장
- **음성 분류**: 기존 audio_feature_service.py, stress_analysis_service.py 확장
- **심리 분류**: 새 psychological_profiler_service.py 구축
- **앙상블**: weighted_ensemble_classifier.py 신규 구현

#### 3.1.2 법률 프레임워크

- **형법 조문 DB**: korean_criminal_code_db.py
- **판례 체크리스트**: precedent_evidence_checker.py
- **검찰 포맷**: prosecution_format_generator.py

### 3.2 시스템 요구사항

- Python 3.11+
- CUDA 12.1+ (GPU 가속)
- 기존 SPEC-FORENSIC-001 환경 전제

---

## 4. 가정 (Assumptions)

### 4.1 기술적 가정

1. SPEC-FORENSIC-001의 5개 카테고리 분석이 완료되어 있다
2. 183개 녹취 파일의 WhisperX 전사 결과가 존재한다
3. 기존 포렌식 스코어링 결과를 입력으로 활용할 수 있다

### 4.2 비즈니스 가정

1. 분석 대상은 2건의 고소장에 명시된 11개 죄명이다
2. 동일 발화가 복수의 죄명에 해당할 수 있다
3. 법적 증거로 활용하기 위한 신뢰도 기준이 필요하다

### 4.3 법적 가정

1. 형법 조문은 2026년 기준 대한민국 형법을 따른다
2. 증거 요건은 판례 기반 체크리스트로 검증한다
3. 분석 결과는 법적 조언이 아닌 보조 자료로 활용된다

---

## 5. 요구사항 (Requirements)

### 5.1 Ubiquitous Requirements (U) - 전역 요구사항

#### U1: 범죄 유형 분류 일관성
**요구사항**: 시스템은 **항상** 동일한 입력에 대해 동일한 범죄 유형 분류 결과를 생성해야 한다.

**세부사항**:
- 결정론적 분류 알고리즘 적용
- 난수 시드 고정
- 모델 버전 명시

**WHY**: 법적 증거로서 분류 결과의 재현 가능성이 필수적이다.
**IMPACT**: 비결정론적 분류는 증거 신뢰도를 저하시킨다.

---

#### U2: 멀티모달 데이터 추적성
**요구사항**: 시스템은 **항상** 각 분류 결과에 대해 텍스트/음성/심리 기여도를 명시해야 한다.

**세부사항**:
- 각 모달리티별 점수 기록
- 가중치 적용 내역 저장
- 최종 분류 근거 문서화

**WHY**: 분류 결과의 투명성과 검증 가능성을 보장해야 한다.
**IMPACT**: 기여도 미명시 시 분류 결과의 법적 신뢰도 저하.

---

#### U3: 형법 조문 참조 연결
**요구사항**: 시스템은 **항상** 각 범죄 유형에 해당하는 형법 조문을 자동으로 참조해야 한다.

**세부사항**:
- 범죄 유형별 형법 조문 매핑
- 구성요건 자동 체크
- 법률 개정 시 업데이트 메커니즘

**WHY**: 법적 정확성과 근거 제시가 필수적이다.
**IMPACT**: 형법 조문 미참조 시 증거 채택 거부 위험.

---

### 5.2 Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: 포렌식 분석 완료 시 범죄 분류 트리거
**요구사항**: **WHEN** SPEC-FORENSIC-001 분석이 완료되면 **THEN** 시스템은 범죄 유형 분류를 자동으로 시작해야 한다.

**세부사항**:
- 포렌식 결과(ForensicScoreResult)를 입력으로 수신
- 11개 범죄 유형에 대한 분류 수행
- 분류 진행률 실시간 업데이트

**WHY**: 파이프라인 연속성 보장으로 수동 개입 최소화.
**IMPACT**: 수동 트리거 시 처리 지연 및 오류 가능성 증가.

---

#### E2: 복수 죄명 해당 시 알림 생성
**요구사항**: **WHEN** 단일 발화가 2개 이상의 죄명에 해당하면 **THEN** 시스템은 "복수 죄명 해당" 알림을 생성해야 한다.

**세부사항**:
- 복수 분류 임계값: 신뢰도 0.5 이상
- 죄명 조합 기록
- 우선순위 표시 (중한 죄명 우선)

**WHY**: 경합범 또는 상상적 경합 가능성을 식별해야 한다.
**IMPACT**: 복수 죄명 미식별 시 고소장 기재 누락 위험.

---

#### E3: 법적 요건 충족 시 증거 요약 생성
**요구사항**: **WHEN** 특정 죄명의 구성요건이 모두 충족되면 **THEN** 시스템은 해당 죄명에 대한 법원 제출용 증거 요약을 자동 생성해야 한다.

**세부사항**:
- 형법 조문별 구성요건 체크리스트
- 충족 요건 목록화
- 증거 요약 PDF 생성

**WHY**: 법원 제출용 증거 자료 자동화.
**IMPACT**: 수동 요약 시 시간 소요 및 누락 위험.

---

### 5.3 State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: 신뢰도 기반 분류 검토
**요구사항**: **IF** 범죄 유형 분류 신뢰도가 0.7 미만이면 **THEN** 해당 분류에 "전문가 검토 필요" 플래그를 설정해야 한다.

**세부사항**:
- 신뢰도 임계값: 0.7
- 플래그 유형: 자동 분류 신뢰, 검토 권장, 검토 필수
- 저신뢰 구간 시각적 표시

**WHY**: 자동 분류의 한계를 명확히 표시하여 오분류 방지.
**IMPACT**: 저신뢰 결과의 무비판적 수용 시 법적 위험.

---

#### S2: 화자별 죄명 분류 집계
**요구사항**: **IF** 화자가 식별되면 **THEN** 시스템은 화자별로 해당 죄명을 집계해야 한다.

**세부사항**:
- 화자별 죄명 목록 생성
- 빈도 및 강도 통계
- 가해자/피해자 역할 구분

**WHY**: 고소장 작성 시 피고소인별 죄명 분류가 필요하다.
**IMPACT**: 화자 미구분 시 죄명 귀속 불명확.

---

### 5.4 Feature Requirements (F) - 기능 요구사항

#### F1: 멀티모달 융합 분류기 (Multimodal Fusion Classifier)
**요구사항**: 시스템은 텍스트, 음성, 심리 점수를 가중 앙상블하여 범죄 유형을 분류해야 한다.

**세부사항**:
- 텍스트 점수: 30-40%
- 음성 점수: 30-40%
- 심리 점수: 20-30%
- 범죄 유형별 가중치 조정 가능

**가중치 매트릭스**:
```python
CRIME_WEIGHTS = {
    "가정폭력": {"text": 0.35, "audio": 0.40, "psych": 0.25},
    "스토킹": {"text": 0.40, "audio": 0.30, "psych": 0.30},
    "협박": {"text": 0.35, "audio": 0.40, "psych": 0.25},
    "가스라이팅": {"text": 0.40, "audio": 0.30, "psych": 0.30},
    "사기": {"text": 0.40, "audio": 0.35, "psych": 0.25},
    "공갈": {"text": 0.35, "audio": 0.40, "psych": 0.25},
    "강요": {"text": 0.35, "audio": 0.40, "psych": 0.25},
    "모욕": {"text": 0.45, "audio": 0.35, "psych": 0.20},
    "횡령": {"text": 0.50, "audio": 0.25, "psych": 0.25},
    "배임": {"text": 0.50, "audio": 0.25, "psych": 0.25},
    "조세포탈": {"text": 0.55, "audio": 0.20, "psych": 0.25},
}
```

**출력**:
```python
{
    "crime_type": "사기",
    "confidence": 0.85,
    "modality_scores": {
        "text": 0.82,
        "audio": 0.78,
        "psychological": 0.75
    },
    "weighted_score": 0.80,
    "legal_reference": "형법 제347조"
}
```

**WHY**: 단일 모달리티로는 포착 불가능한 복합 범죄 패턴 감지.
**IMPACT**: 멀티모달 융합 없이는 분류 정확도 저하.

---

#### F2: 범죄 언어 패턴 확장 (Extended Crime Language Patterns)
**요구사항**: 시스템은 기존 5개 카테고리를 11개 죄명으로 확장한 언어 패턴 DB를 사용해야 한다.

**세부사항**:
- 각 죄명별 한국어 패턴 정의
- 정규표현식 기반 매칭
- 형태소 분석 연동

**패턴 예시**:
```python
FRAUD_PATTERNS_KO = [
    "돈 빌려주면 갚을게",
    "확실히 수익이 나",
    "원금 보장",
    "투자하면 두 배로",
    "이번만 믿어",
    "내가 책임질게",
    "절대 손해 안 봐",
]

EMBEZZLEMENT_PATTERNS_KO = [
    "회사 돈으로",
    "법인카드로",
    "경비 처리",
    "세금계산서 없이",
    "현금으로 받아",
    "장부에 안 올려",
]
```

**WHY**: 범죄 유형별 특화된 언어 패턴으로 분류 정확도 향상.
**IMPACT**: 일반 패턴으로는 특정 죄명 식별 불가.

---

#### F3: 심리 프로파일링 엔진 (Psychological Profiler)
**요구사항**: 시스템은 음성/텍스트 분석 결과를 기반으로 심리적 프로파일을 생성해야 한다.

**세부사항**:
- 다크 트라이어드 지표 (나르시시즘, 마키아벨리즘, 사이코패시)
- 애착 유형 분석
- 성격 특성 추정

**출력**:
```python
{
    "dark_triad_scores": {
        "narcissism": 0.72,
        "machiavellianism": 0.68,
        "psychopathy": 0.45
    },
    "attachment_style": "anxious_avoidant",
    "dominant_traits": ["manipulation", "lack_of_empathy", "grandiosity"],
    "crime_propensity": {
        "fraud": 0.80,
        "gaslighting": 0.75,
        "coercion": 0.65
    }
}
```

**WHY**: 심리적 특성이 범죄 유형 예측에 중요한 지표가 된다.
**IMPACT**: 심리 분석 없이는 동기 기반 분류 불가.

---

#### F4: 법적 증거 매핑 서비스 (Legal Evidence Mapper)
**요구사항**: 시스템은 포렌식 분석 결과를 특정 죄명의 구성요건에 매핑해야 한다.

**세부사항**:
- 형법 조문별 구성요건 정의
- 증거-요건 매칭 알고리즘
- 충족률 계산

**구성요건 예시 (사기죄)**:
```python
FRAUD_REQUIREMENTS = {
    "기망행위": {
        "description": "허위 사실을 진술하거나 사실을 은폐",
        "indicators": ["과장", "허위진술", "사실왜곡"],
        "required": True
    },
    "착오유발": {
        "description": "상대방의 착오를 유발",
        "indicators": ["오해유도", "정보은닉"],
        "required": True
    },
    "처분행위": {
        "description": "재산상 처분행위 유발",
        "indicators": ["금전요구", "투자권유", "계약체결"],
        "required": True
    },
    "재산상이익": {
        "description": "재산상 이익 취득",
        "indicators": ["금전수령", "이익실현"],
        "required": True
    },
    "인과관계": {
        "description": "기망-착오-처분-이익 간 인과관계",
        "indicators": ["연속성", "시간적근접"],
        "required": True
    }
}
```

**출력**:
```python
{
    "crime_type": "사기",
    "legal_code": "형법 제347조",
    "requirements_met": {
        "기망행위": {"satisfied": True, "evidence": ["발화 #45", "발화 #67"]},
        "착오유발": {"satisfied": True, "evidence": ["발화 #52"]},
        "처분행위": {"satisfied": True, "evidence": ["발화 #78"]},
        "재산상이익": {"satisfied": False, "evidence": []},
        "인과관계": {"satisfied": True, "evidence": ["시계열 분석"]}
    },
    "fulfillment_rate": 0.80,
    "legal_viability": "보통"
}
```

**WHY**: 법적 구성요건 충족 여부를 자동으로 평가.
**IMPACT**: 수동 검토 시 누락 및 오류 위험.

---

#### F5: 신뢰 구간 계산 (Confidence Interval Calculation)
**요구사항**: 시스템은 각 분류에 대해 95% 신뢰 구간을 제공해야 한다.

**세부사항**:
- 부트스트랩 방법론 적용
- 상한/하한 신뢰 구간
- 불확실성 시각화

**출력**:
```python
{
    "crime_type": "협박",
    "point_estimate": 0.78,
    "confidence_interval": {
        "lower_95": 0.72,
        "upper_95": 0.84
    },
    "standard_error": 0.03,
    "uncertainty_level": "낮음"
}
```

**WHY**: 분류 결과의 통계적 신뢰성을 법적 증거로 제시.
**IMPACT**: 신뢰 구간 미제공 시 결과 해석의 모호성.

---

#### F6: 검찰 제출 포맷 생성기 (Prosecution Format Generator)
**요구사항**: 시스템은 검찰 제출용 표준 포맷의 증거 보고서를 생성해야 한다.

**세부사항**:
- 표지, 목차, 본문, 부록 구성
- 죄명별 증거 요약
- 시계열 분석 포함

**보고서 구조**:
```
1. 표지
   - 사건번호
   - 피고소인 정보
   - 작성일

2. 분석 개요
   - 분석 대상
   - 분석 방법론
   - 사용 기술

3. 죄명별 분석 결과
   - 죄명 1: 사기
     - 구성요건 충족 여부
     - 증거 목록
     - 신뢰 구간
   - 죄명 2: 공갈
     - ...

4. 시계열 분석
   - 범죄 행위 타임라인
   - 에스컬레이션 패턴

5. 결론 및 의견
   - 종합 위험 평가
   - 권고사항

6. 부록
   - 원본 데이터 해시
   - 분석 로그
   - 용어 설명
```

**WHY**: 검찰 표준 형식 준수로 증거 채택률 향상.
**IMPACT**: 비표준 형식은 증거 검토 지연 또는 기각 위험.

---

### 5.5 Optional Requirements (O) - 선택적 요구사항

#### O1: 판례 기반 유사 사례 검색
**요구사항**: **가능하면** 유사한 판례를 검색하여 참조할 수 있다.

**WHY**: 판례 참조로 법적 논거 강화.
**IMPACT**: 미구현 시 판례 검색은 수동으로 수행.

---

#### O2: 실시간 분류 업데이트
**요구사항**: **가능하면** 새로운 증거 추가 시 실시간으로 분류를 업데이트할 수 있다.

**WHY**: 동적 증거 관리 지원.
**IMPACT**: 미구현 시 배치 처리만 가능.

---

### 5.6 Unwanted Requirements (N) - 금지 요구사항

#### N1: 법적 조언 제공 금지
**요구사항**: 시스템은 **절대** 법적 조언이나 법률 자문을 제공하지 않아야 한다.

**세부사항**:
- 모든 결과에 "법률 전문가 검토 필요" 명시
- "권고" 대신 "분석 결과" 표현 사용
- 법적 책임 면책 조항 포함

**WHY**: 법률 자문은 변호사만 제공할 수 있다.
**IMPACT**: 법적 조언 제공 시 변호사법 위반 위험.

---

#### N2: 개인정보 무단 노출 금지
**요구사항**: 시스템은 **절대** 분석 대상자의 개인정보를 암호화 없이 저장하거나 외부에 노출하지 않아야 한다.

**세부사항**:
- 화자 ID 익명화 (SPEAKER_00)
- 개인정보 마스킹
- 암호화 저장

**WHY**: 개인정보보호법 준수.
**IMPACT**: 개인정보 노출 시 법적 제재.

---

## 6. 명세 (Specifications)

### 6.1 데이터 모델

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

class CrimeType(Enum):
    """범죄 유형 열거형"""
    DOMESTIC_VIOLENCE = "가정폭력"
    STALKING = "스토킹"
    THREAT = "협박"
    GASLIGHTING = "가스라이팅"
    FRAUD = "사기"
    EXTORTION = "공갈"
    COERCION = "강요"
    INSULT = "모욕"
    EMBEZZLEMENT = "횡령"
    BREACH_OF_TRUST = "배임"
    TAX_EVASION = "조세포탈"

@dataclass
class ModalityScore:
    """모달리티별 점수"""
    text_score: float
    audio_score: float
    psychological_score: float

@dataclass
class CrimeClassification:
    """범죄 분류 결과"""
    crime_type: CrimeType
    confidence: float
    confidence_interval: Dict[str, float]  # {"lower_95": 0.7, "upper_95": 0.9}
    modality_scores: ModalityScore
    weighted_score: float
    legal_reference: str  # "형법 제347조"
    evidence_items: List[str]
    requires_review: bool

@dataclass
class LegalRequirement:
    """법적 구성요건"""
    name: str
    description: str
    indicators: List[str]
    satisfied: bool
    evidence: List[str]

@dataclass
class LegalEvidenceMapping:
    """법적 증거 매핑 결과"""
    crime_type: CrimeType
    legal_code: str
    requirements: List[LegalRequirement]
    fulfillment_rate: float
    legal_viability: str  # "높음", "보통", "낮음"

@dataclass
class PsychologicalProfile:
    """심리 프로파일"""
    dark_triad_scores: Dict[str, float]
    attachment_style: str
    dominant_traits: List[str]
    crime_propensity: Dict[str, float]

@dataclass
class CrimeClassificationResult:
    """범죄 분류 종합 결과"""
    analysis_id: str
    analyzed_at: datetime
    speaker_id: str
    classifications: List[CrimeClassification]
    legal_mappings: List[LegalEvidenceMapping]
    psychological_profile: Optional[PsychologicalProfile]
    summary: str
    recommendations: List[str]
```

### 6.2 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/crime-class/classify/{audio_id}` | POST | 범죄 유형 분류 시작 |
| `/api/v1/crime-class/result/{audio_id}` | GET | 분류 결과 조회 |
| `/api/v1/crime-class/evidence-map/{audio_id}` | GET | 법적 증거 매핑 조회 |
| `/api/v1/crime-class/psychological/{audio_id}` | GET | 심리 프로파일 조회 |
| `/api/v1/crime-class/report` | POST | 검찰 제출용 보고서 생성 |
| `/api/v1/crime-class/report/{report_id}` | GET | 보고서 다운로드 |
| `/api/v1/crime-class/speaker/{speaker_id}/crimes` | GET | 화자별 죄명 조회 |

### 6.3 가중치 앙상블 알고리즘

```python
def classify_crime(
    text_scores: Dict[CrimeType, float],
    audio_scores: Dict[CrimeType, float],
    psych_scores: Dict[CrimeType, float],
    weights: Dict[str, Dict[str, float]] = CRIME_WEIGHTS
) -> List[CrimeClassification]:
    """
    멀티모달 가중 앙상블 분류

    Args:
        text_scores: 텍스트 기반 각 범죄 유형 점수 (0-1)
        audio_scores: 음성 기반 각 범죄 유형 점수 (0-1)
        psych_scores: 심리 기반 각 범죄 유형 점수 (0-1)
        weights: 범죄 유형별 모달리티 가중치

    Returns:
        신뢰도 순 정렬된 범죄 분류 결과 목록
    """
    results = []

    for crime_type in CrimeType:
        crime_name = crime_type.value
        w = weights.get(crime_name, {"text": 0.4, "audio": 0.35, "psych": 0.25})

        weighted_score = (
            text_scores.get(crime_type, 0) * w["text"] +
            audio_scores.get(crime_type, 0) * w["audio"] +
            psych_scores.get(crime_type, 0) * w["psych"]
        )

        # 신뢰 구간 계산 (부트스트랩)
        ci = calculate_confidence_interval(
            text_scores.get(crime_type, 0),
            audio_scores.get(crime_type, 0),
            psych_scores.get(crime_type, 0),
            w
        )

        results.append(CrimeClassification(
            crime_type=crime_type,
            confidence=weighted_score,
            confidence_interval=ci,
            modality_scores=ModalityScore(
                text_score=text_scores.get(crime_type, 0),
                audio_score=audio_scores.get(crime_type, 0),
                psychological_score=psych_scores.get(crime_type, 0)
            ),
            weighted_score=weighted_score,
            legal_reference=get_legal_reference(crime_type),
            evidence_items=[],
            requires_review=weighted_score < 0.7
        ))

    # 신뢰도 순 정렬
    results.sort(key=lambda x: x.confidence, reverse=True)

    # 임계값 이상만 반환
    return [r for r in results if r.confidence >= 0.3]
```

---

## 7. 추적성 (Traceability)

### 7.1 관련 문서

- `SPEC-FORENSIC-001`: 범죄 프로파일링 기반 음성 포렌식 분석 시스템 (기반)
- `SPEC-VOICE-001`: 음성 녹취 기반 증거 분석 시스템 (입력)
- `SPEC-WHISPERX-001`: WhisperX 통합 파이프라인 (전사)
- `plan.md`: 구현 계획 및 마일스톤
- `acceptance.md`: 인수 기준 및 테스트 시나리오

### 7.2 태그

```
[SPEC-CRIME-CLASS-001]
[범죄분류] [멀티모달분석] [법적증거]
[심리프로파일링] [한국형법] [검찰포맷]
```

### 7.3 예상 구현 파일

```
src/voice_man/
├── services/
│   └── crime_classification/
│       ├── __init__.py
│       ├── multimodal_classifier.py     # F1 - 멀티모달 융합 분류기
│       ├── extended_crime_patterns.py   # F2 - 확장 범죄 언어 패턴
│       ├── psychological_profiler.py    # F3 - 심리 프로파일링
│       ├── legal_evidence_mapper.py     # F4 - 법적 증거 매핑
│       ├── confidence_calculator.py     # F5 - 신뢰 구간 계산
│       └── prosecution_formatter.py     # F6 - 검찰 제출 포맷
├── models/
│   └── crime_classification/
│       ├── __init__.py
│       ├── crime_types.py               # 범죄 유형 열거형
│       ├── classification_result.py     # 분류 결과 데이터 모델
│       ├── legal_requirements.py        # 법적 구성요건 모델
│       └── psychological_profile.py     # 심리 프로파일 모델
└── data/
    └── crime_classification/
        ├── korean_criminal_code.json    # 한국 형법 조문 DB
        ├── crime_patterns_extended.json # 확장 범죄 패턴 DB
        └── legal_requirements.json      # 법적 구성요건 DB
```

---

## 8. 참조 (References)

### 8.1 법률 조문

- 형법 제283조 (협박죄)
- 형법 제311조 (모욕죄)
- 형법 제324조 (강요죄)
- 형법 제347조 (사기죄)
- 형법 제350조 (공갈죄)
- 형법 제355조 (횡령, 배임죄)
- 조세범처벌법

### 8.2 학술 참조

- Paulhus, D. L., & Williams, K. M. (2002). "The Dark Triad of personality"
- Newman, M. L., et al. (2003). "Lying Words: Predicting Deception from Linguistic Styles"
- Pennebaker, J. W. (2011). "The Secret Life of Pronouns"

### 8.3 기술 참조

- SPEC-FORENSIC-001 구현 코드
- scikit-learn 앙상블 문서
- 한국어 형태소 분석기 (KoNLPy)

---

**문서 끝**
