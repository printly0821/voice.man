# 감정 분석 API 문서

## 개요

Voice Man 감정 분석 API는 KoBERT 기반 한국어 감정 분류 서비스를 제공합니다. 7개 감정 카테고리에 대한 분류와 신뢰도 점수를 제공합니다.

---

## 감정 카테고리

| 감정 | 영어 | 설명 |
|-----|------|------|
| 행복 | happiness | 긍정적, 즐거운 감정 |
| 슬픔 | sadness | 부정적, 우울한 감정 |
| 분노 | anger | 공격적, 화난 감정 |
| 공포 | fear | 두려운, 불안한 감정 |
| 혐오 | disgust | 혐오스러운, 싫어하는 감정 |
| 놀람 | surprise | 놀란, 의외의 감정 |
| 중립 | neutral | 감정이 중립인 상태 |

---

## 데이터 모델

### EmotionResult

감정 분류 결과를 나타내는 데이터 모델입니다.

```python
@dataclass
class EmotionResult:
    primary_emotion: str      # 주요 감정
    confidence: float          # 신뢰도 (0.0-1.0)
    emotion_scores: Dict[str, float]  # 전체 감정 확률 분포
    is_uncertain: bool         # 불확실성 플래그
    key_tokens: Optional[List[str]]   # 핵심 토큰
```

**속성 설명**:

| 속성 | 타입 | 설명 |
|-----|------|------|
| `primary_emotion` | str | 가장 높은 확률의 감정 카테고리 |
| `confidence` | float | 주요 감정의 신뢰도 점수 (0.0-1.0) |
| `emotion_scores` | Dict[str, float] | 모든 감정의 확률 분포 |
| `is_uncertain` | bool | 신뢰도가 임계값 미만일 경우 True |
| `key_tokens` | Optional[List[str]] | 분류에 기여한 핵심 토큰 |

---

## API 레퍼런스

### KoBERTEmotionClassifier

감정 분류기 클래스입니다.

#### 생성자

```python
KoBERTEmotionClassifier(
    model_name: str = "skt/kobert-base-v1",
    device: str = "auto",
    confidence_threshold: float = 0.7,
    max_length: int = 128
) -> None
```

**매개변수**:

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model_name` | str | "skt/kobert-base-v1" | KoBERT 모델 이름 또는 경로 |
| `device` | str | "auto" | 실행 장치 ("auto", "cuda", "cpu") |
| `confidence_threshold` | float | 0.7 | 불확실성 플래그 임계값 |
| `max_length` | int | 128 | 최대 시퀀스 길이 |

#### 메서드

### classify()

단일 텍스트의 감정을 분류합니다.

```python
classify(text: str) -> EmotionResult
```

**매개변수**:
- `text` (str): 분석할 텍스트

**반환값**:
- `EmotionResult`: 감정 분류 결과

**예외**:
- `ValueError`: 텍스트가 비어있을 때
- `RuntimeError`: 모델이 로드되지 않았을 때

**예시**:
```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

classifier = KoBERTEmotionClassifier()
result = classifier.classify("오늘 정말 기뻐요!")

print(result.primary_emotion)  # "happiness"
print(result.confidence)       # 0.85
```

### classify_batch()

여러 텍스트의 감정을 배치로 분류합니다.

```python
classify_batch(texts: List[str], batch_size: int = 8) -> List[EmotionResult]
```

**매개변수**:
- `texts` (List[str]): 분석할 텍스트 목록
- `batch_size` (int): 배치 크기 (기본값: 8)

**반환값**:
- `List[EmotionResult]`: 감정 분류 결과 목록

**예외**:
- `ValueError`: 텍스트 목록이 비어있을 때
- `RuntimeError`: 모델이 로드되지 않았을 때

**예시**:
```python
texts = ["기분이 좋아요", "너무 슬퍼요", "화가 났어요"]
results = classifier.classify_batch(texts)

for result in results:
    print(f"{result.primary_emotion}: {result.confidence}")
```

---

## 사용 예시

### 예시 1: 기본 감정 분석

```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

# 분류기 초기화
classifier = KoBERTEmotionClassifier()

# 텍스트 분석
text = "오늘 날씨가 정말 좋아서 기분이 최고예요!"
result = classifier.classify(text)

# 결과 출력
print(f"주요 감정: {result.primary_emotion}")
print(f"신뢰도: {result.confidence:.2f}")
print(f"불확실: {result.is_uncertain}")
print(f"\n감정 점수:")
for emotion, score in result.emotion_scores.items():
    print(f"  {emotion}: {score:.3f}")
```

**출력**:
```
주요 감정: happiness
신뢰도: 0.85
불확실: False

감정 점수:
  happiness: 0.850
  sadness: 0.050
  anger: 0.020
  fear: 0.010
  disgust: 0.015
  surprise: 0.030
  neutral: 0.025
```

### 예시 2: 배치 처리

```python
# 대화 데이터셋 분석
conversations = [
    "정말 감사드려요!",
    "별로 안 좋아요.",
    "왜 그러세요!",
    "무서워요...",
    "정말 놀랐어요!",
    "그냥 그래요."
]

# 배치 분류
results = classifier.classify_batch(conversations, batch_size=6)

# 결과 요약
for text, result in zip(conversations, results):
    print(f"{text:20s} -> {result.primary_emotion:10s} ({result.confidence:.2f})")
```

### 예시 3: 불확실성 처리

```python
# 낮은 신뢰도 결과 처리
text = "그건 잘 모르겠어요."
result = classifier.classify(text)

if result.is_uncertain:
    print(f"분석이 불확실합니다 (신뢰도: {result.confidence:.2f})")
    print("상위 2개 감정:")
    sorted_emotions = sorted(
        result.emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:2]
    for emotion, score in sorted_emotions:
        print(f"  {emotion}: {score:.3f}")
else:
    print(f"확실한 감정: {result.primary_emotion}")
```

---

## 설정

### 신뢰도 임계값

분류 결과의 불확실성을 판단하는 임계값을 설정합니다.

```python
# 보수적 임계값 (높은 신뢰도 요구)
classifier = KoBERTEmotionClassifier(confidence_threshold=0.8)

# 관대한 임계값 (낮은 신뢰도 허용)
classifier = KoBERTEmotionClassifier(confidence_threshold=0.5)
```

### 장치 설정

GPU/CPU 장치를 명시적으로 설정합니다.

```python
# 자동 감지
classifier = KoBERTEmotionClassifier(device="auto")

# GPU 사용
classifier = KoBERTEmotionClassifier(device="cuda")

# CPU 사용
classifier = KoBERTEmotionClassifier(device="cpu")
```

---

## 성능

### 추론 시간

| 장치 | 단일 텍스트 | 배치 (8개) |
|-----|-----------|-----------|
| GPU | ~20-50ms | ~100-200ms |
| CPU | ~80-150ms | ~500-800ms |

### 정확도

- AI Hub 한국어 감정 분류 데이터셋: 90.50%
- 일반 대화 데이터: ~85-90%

---

## 트러블슈팅

### 문제: 항상 "neutral"로 분류됨

**원인**: 텍스트가 너무 짧거나 감정 표현이 명확하지 않음

**해결**:
```python
# 더 긴 텍스트 사용
text = "오늘 회의에서 정말 좋은 아이디어가 나와서 기뻤어요!"
```

### 문제: 신뢰도가 항상 낮음

**원인**: 임계값이 너무 높게 설정됨

**해결**:
```python
# 임계값 조정
classifier = KoBERTEmotionClassifier(confidence_threshold=0.5)
```

### 문제: GPU 메모리 부족

**원인**: 배치 크기가 너무 큼

**해결**:
```python
# 배치 크기 감소
results = classifier.classify_batch(texts, batch_size=4)
```

---

## 관련 문서

- [KoBERT 통합 가이드](KOBERT_INTEGRATION_GUIDE.md)
- [NLP 서비스 아키텍처](NLP_ARCHITECTURE.md)
- [문화적 맥락 분석 문서](CULTURAL_CONTEXT_ANALYSIS.md)

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2026-01-11
**SPEC**: SPEC-NLP-KOBERT-001
