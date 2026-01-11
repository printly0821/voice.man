# 범죄 유형 분류 시스템

## 개요

Voice Man의 범죄 유형 분류 시스템은 음성 녹취 파일의 텍스트 증거를 기반으로 11개 한국 형법 죄명으로 자동 분류하는 멀티모달 머신러닝 시스템입니다.

## 지원 죄명 (11개)

| 죄명 | 법적 근거 | 주요 탐지 패턴 |
|------|----------|---------------|
| 강간죄 | 형법 제297조 | 성폭력, 성적 위협 |
| 강제추행죄 | 형법 제298조 | 성폭력, 추행 |
| 준강간 및 준강제추행죄 | 형법 제299조 | 심신미약, 위법약물 |
| 업무상위계등 | 형법 제356조 | 업무권한 남용, 배임 |
| 공갈죄 | 형법 제350조 | 협박, 금품 갈취 |
| 협박죄 | 형법 제283조 | 신체/생명/재산 위협 |
| 모욕죄 | 형법 제311조 | 인격 모독, 비하 |
| 명예훼손죄 | 형법 제307조 | 허위사실 유포 |
| 감금죄 | 형법 제276조 | 행동 제한, 구금 |
| 체포감금죄 | 형법 제277조 | 권한 없는 체포 |
| 특수공무원감금죄 | 형법 제278조 | 공무원 감금 |

## 핵심 서비스

### 1. ExtendedCrimePatternsService

240개 이상의 한국어 범죄 언어 패턴 데이터베이스를 제공합니다.

```python
from voice_man.services.crime_classification.extended_crime_patterns import ExtendedCrimePatternsService

service = ExtendedCrimePatternsService()
patterns = service.get_patterns("fraud")
score = service.match_and_calculate_score("돈 빌려줘, 이자는 갚지 않을게", "fraud")
```

### 2. PsychologicalProfilerService

Dark Triad (나르시시즘, 마키아벨리즘, 사이코패시) 심리 특성을 분석합니다.

```python
from voice_man.services.crime_classification.psychological_profiler import PsychologicalProfilerService

profiler = PsychologicalProfilerService()
profile = profiler.create_profile("내가 뭐라고 하면 다 맞아. 너는 틀렸어.")
```

### 3. ConfidenceCalculatorService

Bootstrap 방식으로 95% 신뢰구간을 계산합니다.

```python
from voice_man.services.crime_classification.confidence_calculator import ConfidenceCalculatorService

calculator = ConfidenceCalculatorService()
interval = calculator.calculate_confidence_interval([0.7, 0.8, 0.75], 0.95)
```

### 4. LegalEvidenceMapperService

한국 형법 요구사항을 분류 결과에 매핑합니다.

```python
from voice_man.services.crime_classification.legal_evidence_mapper import LegalEvidenceMapperService

mapper = LegalEvidenceMapperService()
requirements = mapper.get_legal_requirements("fraud")
```

### 5. MultimodalClassifierService

텍스트, 오디오, 심리 특성을 결합한 가중 앙상블 분류를 수행합니다.

```python
from voice_man.services.crime_classification.multimodal_classifier import MultimodalClassifierService

classifier = MultimodalClassifierService()
result = classifier.classify(
    text_transcript="협박 발언 내용...",
    audio_features={"stress_level": 0.8},
    psychological_profile={"narcissism": 0.7}
)
```

## 테스트 커버리지

- **총 테스트 수**: 176개
- **커버리지**: 98%
- **테스트 파일**: 7개

## 사용 예시

```python
from voice_man.services.crime_classification.multimodal_classifier import MultimodalClassifierService

classifier = MultimodalClassifierService()

# 단일 텍스트 분류
result = classifier.classify_text_only(
    "돈 빌려줄게? 갚을 생각 없는데. 꼼수 쓰지 마."
)

# 결과
print(f"예측 죄명: {result.primary_crime}")
print(f"신뢰도: {result.confidence:.2f}")
print(f"95% 신뢰구간: [{result.confidence_interval['lower']:.2f}, {result.confidence_interval['upper']:.2f}]")
```

## 성능 메트릭

| 메트릭 | 값 |
|--------|-----|
| 정확도 (Accuracy) | 85% 이상 |
| 5-fold 교차 검증 | 0.8 이상 |
| API 응답 시간 | 5초 이내 |
| Top-K 정확도 | K=1: 85%, K=3: 95% |
