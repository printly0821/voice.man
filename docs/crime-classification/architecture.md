# 범죄 분류 시스템 아키텍처

## 시스템 구조

```mermaid
flowchart TD
    subgraph Input[입력 데이터]
        Text[전사 텍스트]
        Audio[오디오 특성]
        Psych[심리 프로파일]
    end

    subgraph Features[피쳐 추출]
        Pattern[범죄 패턴 매칭]
        AudioFeat[오디오 피쳐]
        DarkTriad[Dark Triad 분석]
    end

    subgraph Classifier[멀티모달 분류기]
        WeightedEnsemble[가중 앙상블]
        CrimeWeights[죄별 가중치]
    end

    subgraph Confidence[신뢰도 계산]
        Bootstrap[Bootstrap 리샘플링]
        Interval95[95% 신뢰구간]
    end

    subgraph Legal[법적 매핑]
        ReqMapper[요구사항 매핑]
        Evidence[증거 요약]
    end

    subgraph Output[출력]
        Crime[예측 죄명]
        Conf[신뢰도]
        CI[신뢰구간]
        LegalReq[법적 요구사항]
    end

    Text --> Pattern
    Audio --> AudioFeat
    Psych --> DarkTriad

    Pattern --> WeightedEnsemble
    AudioFeat --> WeightedEnsemble
    DarkTriad --> WeightedEnsemble

    WeightedEnsemble --> Bootstrap
    CrimeWeights --> WeightedEnsemble

    Bootstrap --> Interval95
    WeightedEnsemble --> ReqMapper

    Interval95 --> Conf
    WeightedEnsemble --> Crime
    ReqMapper --> LegalReq

    Interval95 --> CI

    style WeightedEnsemble fill:#e1f5ff
    style Bootstrap fill:#ffe1f5
    style ReqMapper fill:#f5e1ff
```

## 데이터 모델

### CrimeClassificationResult

```python
@dataclass
class CrimeClassificationResult:
    primary_crime: CrimeType  # 예측 죄명
    confidence: float  # 신뢰도 (0-1)
    modality_scores: Dict[CrimeType, ModalityScore]  # 모달별 점수
    confidence_interval: Dict[str, float]  # 95% 신뢰구간
    legal_requirements: Optional[LegalRequirement]  # 법적 요구사항
    psychological_profile: Optional[PsychologicalProfile]  # 심리 프로파일
```

## 서비스 상호작용

```mermaid
sequenceDiagram
    participant Client
    participant Classifier as MultimodalClassifier
    participant Pattern as ExtendedCrimePatterns
    participant Psych as PsychologicalProfiler
    participant Conf as ConfidenceCalculator
    participant Legal as LegalEvidenceMapper

    Client->>Classifier: classify(text, audio, psych)
    Classifier->>Pattern: match_and_calculate_score(text)
    Pattern-->>Classifier: text_score

    Classifier->>Psych: analyze_text(text)
    Psych-->>Classifier: psych_profile

    Classifier->>Conf: calculate_confidence_interval(scores)
    Conf-->>Classifier: confidence_interval

    Classifier->>Legal: get_legal_requirements(crime)
    Legal-->>Classifier: legal_reqs

    Classifier-->>Client: CrimeClassificationResult
```

## 파일 구조

```
src/voice_man/
├── models/crime_classification/
│   ├── crime_types.py           # 죄명 열거형
│   ├── classification_result.py # 결과 데이터 모델
│   ├── legal_requirements.py    # 법적 요구사항
│   └── psychological_profile.py # 심리 프로파일
│
└── services/crime_classification/
    ├── extended_crime_patterns.py      # 범죄 패턴 DB (240+ 패턴)
    ├── psychological_profiler.py       # Dark Triad 분석
    ├── confidence_calculator.py        # 신뢰구간 계산
    ├── legal_evidence_mapper.py        # 법적 매핑
    └── multimodal_classifier.py        # 멀티모달 분류
```

## 의존성

```mermaid
flowchart LR
    Multimodal[MultimodalClassifier] --> Patterns[ExtendedCrimePatterns]
    Multimodal --> Profiler[PsychologicalProfiler]
    Multimodal --> Calculator[ConfidenceCalculator]
    Multimodal --> Mapper[LegalEvidenceMapper]

    Patterns --> CrimeTypes[CrimeType Enum]
    Profiler --> PsychProfile[PsychologicalProfile]
    Calculator --> ClassificationResult[CrimeClassificationResult]
    Mapper --> LegalReq[LegalRequirement]

    style Multimodal fill:#ffd93d
    style Patterns fill:#6bcf7f
    style Profiler fill:#6bcf7f
    style Calculator fill:#6bcf7f
    style Mapper fill:#6bcf7f
```
