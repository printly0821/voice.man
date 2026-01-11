---
id: SPEC-NLP-KOBERT-001
version: 1.0.0
status: Completed
created: 2026-01-10
updated: 2026-01-11
completed: 2026-01-11
author: workflow-spec
priority: High
related_specs:
  - SPEC-FORENSIC-001
tags:
  - NLP
  - KoBERT
  - 감정분석
  - 한국어
  - 머신러닝
---

# SPEC-NLP-KOBERT-001: 한국어 NLP 향상을 위한 KoBERT 통합

## 1. 개요

### 1.1 목적

현재 `CrimeLanguageAnalysisService`에서 사용 중인 키워드 매칭 기반 분석 방식을 KoBERT 기반 머신러닝 분석으로 교체하여 한국어 텍스트의 감정 및 범죄 언어 패턴 탐지 정확도를 향상시킨다.

### 1.2 배경

- **현재 시스템**: `crime_language_service.py`에서 `pattern_str in text` 방식의 단순 키워드 매칭 사용
- **추정 정확도**: 60-70% (컨텍스트 무시, 유사 표현 미탐지)
- **KoBERT 벤치마크**: AI Hub 한국어 감정 분류 데이터셋에서 90.50% 정확도 달성
- **개선 필요성**: 한국어 특수성(존댓말/반말, 가족/직장 위계 표현, 문화적 조작 표현) 반영 필요

### 1.3 범위

- KoBERT 모델 통합 및 Fine-tuning
- 7개 감정 카테고리 분류 시스템
- 기존 `CrimeLanguageAnalysisService`와의 통합
- 한국어 문화적 맥락 분석 기능

---

## 2. 환경 (Environment)

### 2.1 기술 스택

| 구성요소 | 버전 | 비고 |
|---------|------|------|
| Python | 3.10+ | 기존 프로젝트 호환 |
| PyTorch | 2.0+ | CUDA 지원 필수 |
| transformers | 4.36+ | HuggingFace |
| kobert-transformers | 0.5.1+ | SKTBrain KoBERT |
| sentencepiece | 0.1.99+ | 토크나이저 |

### 2.2 하드웨어 요구사항

| 리소스 | 최소 요구사항 | 권장 사항 |
|--------|-------------|----------|
| GPU VRAM | 4GB | 8GB |
| RAM | 16GB | 32GB |
| Storage | 5GB | 10GB |

### 2.3 데이터셋

- **학습 데이터**: AI Hub 한국어 감정 분류 데이터셋
- **감정 카테고리**: 7개 (공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오)
- **데이터 크기**: 약 150,000개 문장

---

## 3. 가정 (Assumptions)

### 3.1 기술적 가정

- [A-TECH-001] GPU가 사용 가능하며 CUDA가 설치되어 있다
- [A-TECH-002] HuggingFace 모델 허브 접근이 가능하다
- [A-TECH-003] 기존 `CrimeLanguagePatternDB`의 패턴 데이터를 학습 데이터 보강에 활용할 수 있다
- [A-TECH-004] 추론 시 배치 처리가 가능하다

### 3.2 비즈니스 가정

- [A-BIZ-001] 분석 정확도 향상이 키워드 매칭 대비 성능 저하(추론 시간)보다 우선시된다
- [A-BIZ-002] 모델 Fine-tuning에 필요한 레이블링된 데이터 확보가 가능하다
- [A-BIZ-003] 법적 증거로 활용 시 모델 설명 가능성(Explainability)이 요구될 수 있다

### 3.3 통합 가정

- [A-INT-001] 기존 `CrimeLanguageAnalysisService` 인터페이스를 유지하면서 내부 구현만 교체한다
- [A-INT-002] 하위 호환성을 위해 키워드 매칭 방식도 폴백(fallback)으로 유지한다

---

## 4. 요구사항 (Requirements)

### 4.1 기능 요구사항

#### 4.1.1 KoBERT 모델 통합

**[REQ-FUNC-001] 모델 로딩**
- **WHEN** 시스템이 초기화될 때 **THEN** KoBERT 모델과 토크나이저가 로드되어야 한다
- **IF** GPU가 사용 가능하면 **THEN** 모델은 GPU에 로드되어야 한다
- **IF** GPU가 사용 불가능하면 **THEN** 모델은 CPU에 로드되고 경고 로그가 출력되어야 한다

**[REQ-FUNC-002] 감정 분류**
- 시스템은 **항상** 입력 텍스트에 대해 7개 감정 카테고리 중 하나를 분류해야 한다
- 감정 카테고리: 공포(fear), 놀람(surprise), 분노(anger), 슬픔(sadness), 중립(neutral), 행복(happiness), 혐오(disgust)
- **WHEN** 감정이 분류될 때 **THEN** 각 카테고리별 확률(confidence score)이 함께 반환되어야 한다

**[REQ-FUNC-003] Fine-tuning 지원**
- **가능하면** 사용자 정의 데이터셋으로 모델 Fine-tuning을 제공한다
- **WHEN** Fine-tuning이 완료될 때 **THEN** 새로운 모델 체크포인트가 저장되어야 한다

#### 4.1.2 한국어 문화적 맥락 분석

**[REQ-FUNC-004] 존댓말/반말 분석**
- **WHEN** 텍스트가 입력될 때 **THEN** 존댓말/반말 사용 여부가 분석되어야 한다
- **IF** 동일 대화에서 존댓말/반말 전환이 감지되면 **THEN** 권력 관계 변화 지표로 기록해야 한다

**[REQ-FUNC-005] 위계 관계 패턴**
- **WHEN** 가족 관계 표현이 감지될 때 **THEN** 가족 위계 맥락이 분석에 반영되어야 한다
- **WHEN** 직장 관계 표현이 감지될 때 **THEN** 직장 위계 맥락이 분석에 반영되어야 한다

**[REQ-FUNC-006] 한국어 특수 조작 표현**
- 시스템은 **항상** 한국어 특유의 조작적 표현 패턴을 탐지해야 한다
- 예시: "네가 잘못했으니까", "다 널 위해서야", "그렇게 느끼는 게 이상한 거야"

#### 4.1.3 기존 시스템 통합

**[REQ-FUNC-007] 인터페이스 호환성**
- 시스템은 **항상** 기존 `CrimeLanguageAnalysisService`의 public API를 유지해야 한다
- 기존 메서드: `detect_gaslighting()`, `detect_threats()`, `detect_coercion()`, `analyze_deception()`, `analyze_comprehensive()`

**[REQ-FUNC-008] 하이브리드 분석 모드**
- **가능하면** 키워드 매칭과 ML 분석을 결합한 하이브리드 모드를 제공한다
- **IF** ML 모델 로드에 실패하면 **THEN** 키워드 매칭 모드로 자동 폴백되어야 한다

**[REQ-FUNC-009] 신뢰도 점수**
- **WHEN** 분석 결과가 반환될 때 **THEN** 각 탐지 결과에 신뢰도 점수(0.0-1.0)가 포함되어야 한다
- **IF** 신뢰도가 임계값(기본 0.7) 미만이면 **THEN** 결과에 불확실성 플래그가 표시되어야 한다

### 4.2 성능 요구사항

**[REQ-PERF-001] 추론 시간**
- **WHEN** 단일 문장(<100자)이 분석될 때 **THEN** 추론 시간이 100ms 미만이어야 한다
- **WHEN** 배치 분석(10문장)이 수행될 때 **THEN** 총 처리 시간이 500ms 미만이어야 한다

**[REQ-PERF-002] 정확도**
- 시스템은 **항상** 한국어 감정 분류에서 90% 이상의 정확도를 유지해야 한다
- 가스라이팅/협박/강압 탐지에서 85% 이상의 F1 점수를 달성해야 한다

**[REQ-PERF-003] 메모리 사용량**
- 시스템은 **항상** GPU VRAM 사용량을 4GB 이하로 유지해야 한다
- **IF** 메모리 부족이 감지되면 **THEN** 배치 크기를 자동으로 축소해야 한다

### 4.3 비기능 요구사항

**[REQ-NFUNC-001] 설명 가능성**
- **가능하면** 분류 결과에 대한 설명(attention 기반 핵심 토큰 하이라이트)을 제공한다

**[REQ-NFUNC-002] 로깅**
- 시스템은 **항상** 모델 추론 시간, 메모리 사용량, 오류를 구조화된 로그로 기록해야 한다

**[REQ-NFUNC-003] 모델 버전 관리**
- 시스템은 **항상** 사용 중인 모델 버전과 체크포인트 정보를 추적해야 한다

---

## 5. 명세 (Specifications)

### 5.1 모듈 구조

```
src/voice_man/services/nlp/
    __init__.py
    kobert_model.py          # KoBERT 모델 래퍼
    emotion_classifier.py    # 감정 분류기
    cultural_analyzer.py     # 문화적 맥락 분석기
    hybrid_analyzer.py       # 하이브리드 분석 통합
    training/
        __init__.py
        dataset.py           # 데이터셋 로더
        trainer.py           # Fine-tuning 트레이너
        evaluator.py         # 평가 도구
```

### 5.2 핵심 클래스 설계

#### 5.2.1 KoBERTEmotionClassifier

```python
class KoBERTEmotionClassifier:
    """KoBERT 기반 한국어 감정 분류기"""

    EMOTION_LABELS = [
        "fear",      # 공포
        "surprise",  # 놀람
        "anger",     # 분노
        "sadness",   # 슬픔
        "neutral",   # 중립
        "happiness", # 행복
        "disgust",   # 혐오
    ]

    def __init__(
        self,
        model_name: str = "skt/kobert-base-v1",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> None: ...

    def classify(
        self,
        text: str,
    ) -> EmotionResult: ...

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[EmotionResult]: ...
```

#### 5.2.2 KoreanCulturalAnalyzer

```python
class KoreanCulturalAnalyzer:
    """한국어 문화적 맥락 분석기"""

    def analyze_speech_level(
        self,
        text: str,
    ) -> SpeechLevelResult: ...

    def detect_hierarchy_context(
        self,
        text: str,
    ) -> HierarchyContext: ...

    def detect_manipulation_patterns(
        self,
        text: str,
    ) -> List[ManipulationPattern]: ...
```

#### 5.2.3 HybridCrimeLanguageAnalyzer

```python
class HybridCrimeLanguageAnalyzer:
    """하이브리드 범죄 언어 분석기 (ML + 키워드)"""

    def __init__(
        self,
        ml_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fallback_to_keyword: bool = True,
    ) -> None: ...

    # 기존 CrimeLanguageAnalysisService와 동일한 인터페이스
    def detect_gaslighting(
        self,
        text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[GaslightingMatch]: ...
```

### 5.3 데이터 모델

#### 5.3.1 EmotionResult

```python
@dataclass
class EmotionResult:
    """감정 분류 결과"""
    primary_emotion: str           # 주요 감정
    confidence: float              # 신뢰도 (0.0-1.0)
    emotion_scores: Dict[str, float]  # 전체 감정 확률 분포
    is_uncertain: bool             # 불확실성 플래그
    key_tokens: Optional[List[str]]   # 핵심 토큰 (설명 가능성)
```

#### 5.3.2 SpeechLevelResult

```python
@dataclass
class SpeechLevelResult:
    """존댓말/반말 분석 결과"""
    level: Literal["formal", "informal", "mixed"]
    formal_ratio: float
    informal_ratio: float
    level_transitions: List[LevelTransition]  # 레벨 전환 지점
```

### 5.4 설정 구조

```yaml
# config/kobert_config.yaml
model:
  name: "skt/kobert-base-v1"
  fine_tuned_path: null  # Fine-tuned 모델 경로 (선택)
  device: "auto"         # auto, cuda, cpu

inference:
  batch_size: 8
  max_length: 128
  confidence_threshold: 0.7

hybrid:
  ml_weight: 0.7
  keyword_weight: 0.3
  fallback_enabled: true

performance:
  max_gpu_memory_mb: 4096
  inference_timeout_ms: 100
```

---

## 6. 제약사항 (Constraints)

### 6.1 기술적 제약

- [C-TECH-001] KoBERT 모델 크기로 인해 GPU VRAM 4GB 이상 필요
- [C-TECH-002] 모델 로딩 시간 약 5-10초 소요 (콜드 스타트)
- [C-TECH-003] Fine-tuning 시 최소 1,000개 이상의 레이블링된 샘플 필요

### 6.2 호환성 제약

- [C-COMPAT-001] 기존 `CrimeLanguageAnalysisService` API 시그니처 변경 불가
- [C-COMPAT-002] Python 3.10+ 필수 (type hint 호환성)

### 6.3 보안 제약

- [C-SEC-001] 학습 데이터에 개인정보가 포함되어서는 안 됨
- [C-SEC-002] 모델 체크포인트는 암호화된 저장소에 보관

---

## 7. 추적성 (Traceability)

### 7.1 관련 SPEC

| SPEC ID | 관계 | 설명 |
|---------|------|------|
| SPEC-FORENSIC-001 | 확장 | 기존 법의학 분석 파이프라인의 NLP 기능 강화 |

### 7.2 영향받는 파일

| 파일 경로 | 변경 유형 | 설명 |
|----------|----------|------|
| `src/voice_man/services/forensic/crime_language_service.py` | 수정 | HybridAnalyzer 통합 |
| `src/voice_man/services/forensic/crime_language_pattern_db.py` | 유지 | 키워드 폴백용 유지 |
| `src/voice_man/services/nlp/` | 신규 | KoBERT 모듈 추가 |
| `config/kobert_config.yaml` | 신규 | 모델 설정 추가 |

### 7.3 의존성

- **외부 의존성**: HuggingFace transformers, PyTorch, kobert-transformers
- **내부 의존성**: `voice_man.models.forensic.crime_language`, `voice_man.services.forensic.crime_language_pattern_db`

---

## 8. 승인

| 역할 | 이름 | 날짜 | 서명 |
|-----|------|------|------|
| 작성자 | workflow-spec | 2026-01-10 | - |
| 검토자 | - | - | - |
| 승인자 | - | - | - |
