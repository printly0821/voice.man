# 범죄 분류 API 레퍼런스

## ExtendedCrimePatternsService

### `match_and_calculate_score(text: str, crime_type: str) -> float`

텍스트와 범죄 유형을 받아 패턴 매칭 점수를 계산합니다.

**매개변수:**
- `text` (str): 분석할 텍스트
- `crime_type` (str): 범죄 유형 ("fraud", "threat", "coercion", etc.)

**반환값:**
- `float`: 0-1 범위의 매칭 점수

**예시:**
```python
from voice_man.services.crime_classification.extended_crime_patterns import ExtendedCrimePatternsService

service = ExtendedCrimePatternsService()
score = service.match_and_calculate_score(
    "돈 빌려줘, 이자는 갚지 않을게",
    "fraud"
)
print(f"사기 패턴 점수: {score:.2f}")
```

### `get_patterns(crime_type: str) -> List[str]`

특정 범죄 유형의 패턴 목록을 반환합니다.

**매개변수:**
- `crime_type` (str): 범죄 유형

**반환값:**
- `List[str]`: 패턴 문자열 목록

---

## PsychologicalProfilerService

### `create_profile(text: str, audio_features: Optional[Dict] = None, text_scores: Optional[Dict] = None) -> PsychologicalProfile`

텍스트를 분석하여 심리 프로파일을 생성합니다.

**매개변수:**
- `text` (str): 분석할 텍스트
- `audio_features` (Optional[Dict]): 오디오 특성
- `text_scores` (Optional[Dict]): 텍스트 점수

**반환값:**
- `PsychologicalProfile`: 심리 프로파일 객체
  - `narcissism` (float): 나르시시즘 점수 (0-1)
  - `machiavellianism` (float): 마키아벨리즘 점수 (0-1)
  - `psychopathy` (float): 사이코패시 점수 (0-1)
  - `attachment_style` (str): 애착 유형
  - `dominant_traits` (List[str]): 우세 특성
  - `crime_propensity` (Dict[str, float]): 범죄 성향

**예시:**
```python
from voice_man.services.crime_classification.psychological_profiler import PsychologicalProfilerService

profiler = PsychologicalProfilerService()
profile = profiler.create_profile("내가 뭐라고 하면 다 맞아. 너는 틀렸어.")
print(f"나르시시즘: {profile.narcissism:.2f}")
print(f"우세 특성: {profile.dominant_traits}")
```

---

## ConfidenceCalculatorService

### `calculate_confidence_interval(scores: List[float], confidence_level: float = 0.95) -> Dict[str, float]`

Bootstrap 방식으로 신뢰구간을 계산합니다.

**매개변수:**
- `scores` (List[float]): 점수 목록
- `confidence_level` (float): 신뢰수준 (기본값 0.95)

**반환값:**
- `Dict[str, float]`: 신뢰구간 정보
  - `lower` (float): 하한
  - `upper` (float): 상한
  - `point_estimate` (float): 점 추정

**예시:**
```python
from voice_man.services.crime_classification.confidence_calculator import ConfidenceCalculatorService

calculator = ConfidenceCalculatorService()
interval = calculator.calculate_confidence_interval([0.7, 0.8, 0.75], 0.95)
print(f"95% 신뢰구간: [{interval['lower']:.2f}, {interval['upper']:.2f}]")
```

---

## LegalEvidenceMapperService

### `get_legal_requirements(crime_type: str) -> LegalRequirement`

특정 범죄 유형의 법적 요구사항을 반환합니다.

**매개변수:**
- `crime_type` (str): 범죄 유형

**반환값:**
- `LegalRequirement`: 법적 요구사항 객체
  - `crime_type` (CrimeType): 죄명
  - `required_elements` (List[str]): 필요 요소
  - `evidence_standards` (List[str]): 증거 기준
  - `penalties` (List[str]): 형벌

**예시:**
```python
from voice_man.services.crime_classification.legal_evidence_mapper import LegalEvidenceMapperService

mapper = LegalEvidenceMapperService()
requirements = mapper.get_legal_requirements("fraud")
print(f"필요 요소: {requirements.required_elements}")
print(f"증거 기준: {requirements.evidence_standards}")
```

---

## MultimodalClassifierService

### `classify(text_transcript: str, audio_features: Optional[Dict] = None, psychological_profile: Optional[PsychologicalProfile] = None) -> CrimeClassificationResult`

멀티모달 데이터를 기반으로 범죄 유형을 분류합니다.

**매개변수:**
- `text_transcript` (str): 전사 텍스트
- `audio_features` (Optional[Dict]): 오디오 특성
- `psychological_profile` (Optional[PsychologicalProfile]): 심리 프로파일

**반환값:**
- `CrimeClassificationResult`: 분류 결과
  - `primary_crime` (CrimeType): 예측 죄명
  - `confidence` (float): 신뢰도 (0-1)
  - `modality_scores` (Dict): 모달별 점수
  - `confidence_interval` (Dict): 95% 신뢰구간
  - `legal_requirements` (LegalRequirement): 법적 요구사항
  - `psychological_profile` (PsychologicalProfile): 심리 프로파일

**예시:**
```python
from voice_man.services.crime_classification.multimodal_classifier import MultimodalClassifierService

classifier = MultimodalClassifierService()
result = classifier.classify(
    text_transcript="돈 빌려줄게? 갚을 생각 없는데.",
    audio_features={"stress_level": 0.8}
)
print(f"예측 죄명: {result.primary_crime}")
print(f"신뢰도: {result.confidence:.2f}")
print(f"95% 신뢰구간: [{result.confidence_interval['lower']:.2f}, {result.confidence_interval['upper']:.2f}]")
```

### `classify_text_only(text: str) -> CrimeClassificationResult`

텍스트만으로 범죄 유형을 분류합니다.

**매개변수:**
- `text` (str): 분석할 텍스트

**반환값:**
- `CrimeClassificationResult`: 분류 결과

---

## 열거형 (Enums)

### CrimeType

지원되는 11개 죄명:

```python
class CrimeType(str, Enum):
    RAPE = "rape"  # 강간죄
    FORCED_INDECENCY = "forced_indecency"  # 강제추행죄
    QUASI_RAPE_INDECENCY = "quasi_rape_indecency"  # 준강간 및 준강제추행죄
    BREACH_OF_TRUST = "breach_of_trust"  # 업무상위계등
    EXTORTION = "extortion"  # 공갈죄
    THREAT = "threat"  # 협박죄
    INSULT = "insult"  # 모욕죄
    DEFAMATION = "defamation"  # 명예훼손죄
    FALSE_IMPRISONMENT = "false_imprisonment"  # 감금죄
    ARREST_DETENTION = "arrest_detention"  # 체포감금죄
    SPECIAL_OFFICER_DETENTION = "special_officer_detention"  # 특수공무원감금죄
```
