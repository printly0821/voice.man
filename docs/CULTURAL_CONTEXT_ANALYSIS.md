# 문화적 맥락 분석 문서

## 개요

Voice Man 문화적 맥락 분석 서비스는 한국어 특유의 언어적, 문화적 특성을 분석하여 존댓말/반말 사용, 위계 관계, 조작적 표현 패턴을 탐지합니다.

---

## 기능 개요

### 1. 존댓말/반말 분석

한국어의 존댓말(높임법)과 반말(평대법) 사용 패턴을 분석합니다.

### 2. 위계 관계 탐지

텍스트에 포함된 가족, 직장, 사회적 관계 마커를 탐지합니다.

### 3. 조작 패턴 탐지

한국어 특유의 가스라이팅, 위협, 강압 표현 패턴을 탐지합니다.

---

## 존댓말/반말 분석

### 존댓말 패턴

| 패턴 | 설명 | 예시 |
|-----|------|------|
| 입니다체 | 격식체 공손 표현 | "입니다", "거든요" |
| 해요체 | 비격식체 공손 표현 | "세요", "군요", "나요" |
| 합쇼체 | 매우 격식 있는 표현 | "답", "입니다" |

### 반말 패턴

| 패턴 | 설명 | 예시 |
|-----|------|------|
| 해체 | 친근한 반말 표현 | "아", "어" |
| 해라체 | 평범한 반말 표현 | "야", "다", "니" |

### SpeechLevelResult

```python
@dataclass
class SpeechLevelResult:
    level: Literal["formal", "informal", "mixed"]
    formal_ratio: float
    informal_ratio: float
    level_transitions: List[LevelTransition]
```

**속성 설명**:

| 속성 | 타입 | 설명 |
|-----|------|------|
| `level` | str | 전체 speech level ("formal", "informal", "mixed") |
| `formal_ratio` | float | 존댓말 비율 (0.0-1.0) |
| `informal_ratio` | float | 반말 비율 (0.0-1.0) |
| `level_transitions` | List | 존댓말/반말 전환 지점 목록 |

### LevelTransition

```python
@dataclass
class LevelTransition:
    position: int
    from_level: str
    to_level: str
    context: str
```

---

## 위계 관계 탐지

### 위계 마커 카테고리

#### 가족 관계 (Kinship)

| 마커 | 관계 |
|-----|------|
| 할머니 | 조부모 |
| 할아버지 | 조부모 |
| 어머니 | 부모 |
| 아버지 | 부모 |
| 누나 | 누이 |
| 형 | 형 |
| 오빠 | 오빠 |
| 언니 | 언니 |
| 동생 | 동생 |
| 조카 | 조카 |
| 사촌 | 사촌 |
| 삼촌 | 삼촌 |
| 고모 | 고모 |
| 숙모 | 숙모 |

#### 직장 관계 (Job Titles)

| 마커 | 관계 |
|-----|------|
| 사장님 | 최고 경영자 |
| 부장님 | 부서장 |
| 과장님 | 과장 |
| 대리님 | 대리 |
| 사원님 | 사원 |
| 팀장님 | 팀장 |
| 실장님 | 실장 |
| 지점장님 | 지점장 |
| 원장님 | 원장 |
| 국장님 | 국장 |

#### 사회적 관계 (Social)

| 마커 | 관계 |
|-----|------|
| 선생님 | 교사 |
| 교수님 | 교수 |
| 변호사님 | 변호사 |
| 의사선생님 | 의사 |
| 고객님 | 고객 |
| 손님 | 손님 |
| 회원님 | 회원 |
| 회장님 | 회장 |
| 총리님 | 총리 |
| 대통령님 | 대통령 |

### HierarchyContext

```python
@dataclass
class HierarchyContext:
    has_family_markers: bool
    has_job_title_markers: bool
    has_social_markers: bool
    detected_relationships: List[str]
```

---

## 조작 패턴 탐지

### 가스라이팅 (Gaslighting) 패턴

심리적 조작을 위한 가스라이팅 표현 패턴입니다.

| 패턴 | 설명 | 예시 |
|-----|------|------|
| 현실 부정 | 피해자의 기억이나 인식 부정 | "네가 기억을 못하네", "내가 그런 적 없어" |
| 전가 | 책임을 피해자에게 전가 | "네가 잘못했으니까", "너 때문이야" |
| 축소/왜곡 | 피해자의 경험 과소평가 | "별거 아니야", "과민 반응이야" |
| 혼란 유발 | 모순된 말로 피해자 혼란 유도 | "아까 말했다고", "안 그랬어?" |
| 의심 조장 | 피해자의 판단력/기억력 의심 | "기억력이 나쁘네", "착각하는 거야" |
| 고립화 | 주변 지지체계로부터 격리 | "그들 말 믿지 마", "나만 믿어" |
| 자존감 공격 | 피해자의 자존감 약화 | "넌 못해", "넌 항상 틀려" |

### 위협 (Threat) 패턴

위협적 언어 패턴입니다.

| 패턴 | 설명 | 예시 |
|-----|------|------|
| 신체 위협 | 폭력행사 암시 또는 명시적 위협 | "큰일 날 거야" |
| 생명 위협 | 살해/상해 위협 | "후회하게 될 거야" |
| 법적 위협 | 고소/처벌 위협 | "신중하게 생각해" |
| 경제적 위협 | 경제적 손실/파산 위협 | "그러면 안돼" |

### 강압 (Coercion) 패턴

강압적 표현 패턴입니다.

| 패턴 | 설명 | 예시 |
|-----|------|------|
| 명령형 발화 | 강제적 지시나 요구 | "너라면 할 수 있잖아" |
| 선택지 제한 | 피해자의 선택권 박탈 | "나를 실망시키지 마" |
| 긴급성 강조 | 즉각적 순종 강요 | "너 때문에야", "너의 책임이야" |
| 죄책감 유발 | 죄책감을 이용한 강압 | "부탁할게" |

### ManipulationPattern

```python
@dataclass
class ManipulationPattern:
    pattern: str
    category: str
    confidence: float
    position: Optional[int]
```

---

## API 레퍼런스

### KoreanCulturalAnalyzer

문화적 맥락 분석기 클래스입니다.

#### 생성자

```python
KoreanCulturalAnalyzer() -> None
```

#### 메서드

### analyze_speech_level()

존댓말/반말 사용을 분석합니다.

```python
analyze_speech_level(text: str) -> SpeechLevelResult
```

**매개변수**:
- `text` (str): 분석할 텍스트

**반환값**:
- `SpeechLevelResult`: speech level 분석 결과

**예시**:
```python
analyzer = KoreanCulturalAnalyzer()
result = analyzer.analyze_speech_level("안녕하세요? 오늘 날씨 좋네요.")
print(result.level)  # "formal"
print(result.formal_ratio)  # 1.0
```

### detect_hierarchy_context()

위계 관계 마커를 탐지합니다.

```python
detect_hierarchy_context(text: str) -> HierarchyContext
```

**매개변수**:
- `text` (str): 분석할 텍스트

**반환값**:
- `HierarchyContext`: 위계 관계 탐지 결과

**예시**:
```python
result = analyzer.detect_hierarchy_context("어머니, 오늘 밥 드셨어요?")
print(result.has_family_markers)  # True
print(result.detected_relationships)  # ["family:어머니"]
```

### detect_manipulation_patterns()

조작적 표현 패턴을 탐지합니다.

```python
detect_manipulation_patterns(text: str) -> List[ManipulationPattern]
```

**매개변수**:
- `text` (str): 분석할 텍스트

**반환값**:
- `List[ManipulationPattern]`: 탐지된 조작 패턴 목록

**예시**:
```python
patterns = analyzer.detect_manipulation_patterns("네가 잘못했으니까 그래")
print(len(patterns))  # 1
print(patterns[0].category)  # "gaslighting"
```

### analyze_comprehensive()

종합 문화적 맥락 분석을 수행합니다.

```python
analyze_comprehensive(text: str) -> ComprehensiveAnalysisResult
```

**매개변수**:
- `text` (str): 분석할 텍스트

**반환값**:
- `ComprehensiveAnalysisResult`: 종합 분석 결과

---

## 사용 예시

### 예시 1: 존댓말/반말 분석

```python
from voice_man.services.nlp.cultural_analyzer import KoreanCulturalAnalyzer

analyzer = KoreanCulturalAnalyzer()

# 존댓말 텍스트
text = "선생님, 안녕하세요? 제가 이번 시험을 망쳤어요. 정말 속상하네요."
result = analyzer.analyze_speech_level(text)

print(f"전체 수준: {result.level}")
print(f"존댓말 비율: {result.formal_ratio:.2f}")
print(f"반말 비율: {result.informal_ratio:.2f}")

# 전환 지점 분석
if result.level_transitions:
    print(f"\n전환 지점: {len(result.level_transitions)}개")
    for t in result.level_transitions:
        print(f"  위치 {t.position}: {t.from_level} -> {t.to_level}")
        print(f"  문맥: ...{t.context}...")
```

### 예시 2: 위계 관계 탐지

```python
# 가족 관계 탐지
text1 = "어머니, 오늘 밥 드셨어요?"
result1 = analyzer.detect_hierarchy_context(text1)
print(f"가족 마커: {result1.has_family_markers}")
print(f"관계: {result1.detected_relationships}")

# 직장 관계 탐지
text2 = "부장님, 이 보고서 검토 부탁드립니다."
result2 = analyzer.detect_hierarchy_context(text2)
print(f"직장 마커: {result2.has_job_title_markers}")
print(f"관계: {result2.detected_relationships}")
```

### 예시 3: 가스라이팅 패턴 탐지

```python
text = "네가 기억을 못하네. 내가 그런 적 없다고. 네가 잘못 생각하고 있는 거야."
patterns = analyzer.detect_manipulation_patterns(text)

print(f"탐지된 패턴: {len(patterns)}개")
for p in patterns:
    print(f"\n카테고리: {p.category}")
    print(f"패턴: {p.pattern}")
    print(f"신뢰도: {p.confidence:.2f}")
    print(f"위치: {p.position}")
```

### 예시 4: 종합 분석

```python
text = "어머니, 네가 잘못했으니까 그래. 다 널 위해서야."

result = analyzer.analyze_comprehensive(text)

# Speech level
print(f"Speech Level: {result.speech_level.level}")

# Hierarchy
print(f"가족 마커: {result.hierarchy_context.has_family_markers}")

# Manipulation patterns
print(f"조작 패턴: {len(result.manipulation_patterns)}개")
for p in result.manipulation_patterns:
    print(f"  {p.category}: {p.pattern}")
```

---

## 응용 시나리오

### 시나리오 1: 가정 내 괴롭힘 탐지

```python
# 부모-자식 간 대화 분석
text = "너 그럴 수가 있니? 엄마가 얼마나 힘든데."
result = analyzer.analyze_comprehensive(text)

# 위계 마커 확인
if result.hierarchy_context.has_family_markers:
    print("가족 관계 확인")

# 조작 패턴 확인
for p in result.manipulation_patterns:
    if p.category == "coercion":
        print(f"강압 패턴 탐지: {p.pattern}")
```

### 시나리오 2: 직장 내 괴롭힘 탐지

```python
# 상사-부하 간 대화 분석
text = "대리님, 제가 일을 제대로 했는데요. 왜 그러시는 거예요?"
result = analyzer.analyze_comprehensive(text)

# 위계 관계 확인
if result.hierarchy_context.has_job_title_markers:
    print("직장 관계 확인")
```

---

## 관련 문서

- [KoBERT 통합 가이드](KOBERT_INTEGRATION_GUIDE.md)
- [감정 분석 API](EMOTION_ANALYSIS_API.md)
- [NLP 서비스 아키텍처](NLP_ARCHITECTURE.md)

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2026-01-11
**SPEC**: SPEC-NLP-KOBERT-001
