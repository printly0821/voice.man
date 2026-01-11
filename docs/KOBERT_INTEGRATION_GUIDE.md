# KoBERT 통합 가이드

## 개요

본 가이드는 Voice Man 프로젝트에 KoBERT (Korean BERT) 모델을 통합하여 한국어 텍스트 분석 기능을 구현하는 방법을 설명합니다.

---

## 설치

### 1. 기본 의존성 설치

```bash
# 프로젝트 의존성 설치
pip install -e ".[dev]"
```

### 2. PyTorch 설치

**CPU 버전**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU 버전 (CUDA)**:
```bash
# CUDA 12.4 지원 버전
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Transformers 설치

```bash
pip install transformers>=4.36
```

---

## 빠른 시작

### KoBERT 모델 사용

```python
from voice_man.services.nlp.kobert_model import KoBERTModel

# 모델 초기화 (자동 GPU 감지)
model = KoBERTModel(device="auto")

# 텍스트 임베딩 추출
embeddings = model.get_embeddings("한국어 텍스트 분석")

# 배치 처리
results = model.encode_batch(["텍스트 1", "텍스트 2", "텍스트 3"])

# 모델 정보 확인
print(model.get_device_info())
print(model.get_model_info())
```

### 감정 분류

```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

# 분류기 초기화
classifier = KoBERTEmotionClassifier()

# 단일 텍스트 분류
result = classifier.classify("오늘 정말 기뻐요")
print(f"감정: {result.primary_emotion}")
print(f"신뢰도: {result.confidence}")
print(f"불확실: {result.is_uncertain}")

# 배치 분류
texts = ["기분이 좋아요", "너무 슬퍼요", "화가 났어요"]
results = classifier.classify_batch(texts)
for r in results:
    print(f"{r.primary_emotion}: {r.confidence}")
```

### 문화적 맥락 분석

```python
from voice_man.services.nlp.cultural_analyzer import KoreanCulturalAnalyzer

# 분석기 초기화
analyzer = KoreanCulturalAnalyzer()

# 존댓말/반말 분석
speech = analyzer.analyze_speech_level("안녕하세요? 오늘 날씨 좋네요.")
print(f"수준: {speech.level}")
print(f"존댓말 비율: {speech.formal_ratio}")

# 위계 관계 탐지
hierarchy = analyzer.detect_hierarchy_context("어머니, 오늘 밥 드셨어요?")
print(f"가족 마커: {hierarchy.has_family_markers}")
print(f"관계: {hierarchy.detected_relationships}")

# 조작 패턴 탐지
patterns = analyzer.detect_manipulation_patterns("네가 잘못했으니까 그래")
for p in patterns:
    print(f"{p.category}: {p.pattern} (신뢰도: {p.confidence})")
```

---

## 설정

### 설정 파일 위치

`src/voice_man/config/kobert_config.yaml`

### 설정 옵션

```yaml
model:
  name: "skt/kobert-base-v1"  # 모델 이름
  fine_tuned_path: null        # Fine-tuned 모델 경로
  device: "auto"               # 장치 (auto, cuda, cpu)

inference:
  batch_size: 8                # 배치 크기
  max_length: 128              # 최대 시퀀스 길이
  confidence_threshold: 0.7    # 신뢰도 임계값

emotion_classification:
  emotions:
    - happiness
    - sadness
    - anger
    - fear
    - disgust
    - surprise
    - neutral
```

---

## API 레퍼런스

### KoBERTModel

#### 생성자

```python
KoBERTModel(device="auto", model_name="skt/kobert-base-v1", max_length=128)
```

**매개변수**:
- `device` (str): 장치 타입 ("auto", "cuda", "cpu")
- `model_name` (str): 모델 이름 또는 경로
- `max_length` (int): 최대 시퀀스 길이

#### 메서드

**`encode(text: str)`**
- 단일 텍스트 인코딩
- 반환: 모델 출력 (last_hidden_state)

**`encode_batch(texts: List[str])`**
- 배치 텍스트 인코딩
- 반환: 모델 출력 목록

**`get_embeddings(text: str)`**
- 텍스트 임베딩 추출
- 반환: torch.Tensor

**`get_gpu_memory_info()`**
- GPU 메모리 정보 조회
- 반환: Dict[total, free] 또는 None

### KoBERTEmotionClassifier

#### 생성자

```python
KoBERTEmotionClassifier(
    model_name="skt/kobert-base-v1",
    device="auto",
    confidence_threshold=0.7,
    max_length=128
)
```

#### 메서드

**`classify(text: str) -> EmotionResult`**
- 단일 텍스트 감정 분류

**`classify_batch(texts: List[str], batch_size=8) -> List[EmotionResult]`**
- 배치 감정 분류

### KoreanCulturalAnalyzer

#### 생성자

```python
KoreanCulturalAnalyzer()
```

#### 메서드

**`analyze_speech_level(text: str) -> SpeechLevelResult`**
- 존댓말/반말 분석

**`detect_hierarchy_context(text: str) -> HierarchyContext`**
- 위계 관계 탐지

**`detect_manipulation_patterns(text: str) -> List[ManipulationPattern]`**
- 조작 패턴 탐지

---

## 예제

### 예제 1: 기본 감정 분석

```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

classifier = KoBERTEmotionClassifier()

text = "오늘 날씨가 정말 좋아서 기분이 최고예요!"
result = classifier.classify(text)

print(f"주요 감정: {result.primary_emotion}")
print(f"신뢰도: {result.confidence:.2f}")
print(f"감정 점수:")
for emotion, score in result.emotion_scores.items():
    print(f"  {emotion}: {score:.2f}")
```

### 예제 2: 존댓말/반말 전환 감지

```python
from voice_man.services.nlp.cultural_analyzer import KoreanCulturalAnalyzer

analyzer = KoreanCulturalAnalyzer()

text = "선생님, 안녕하세요? 제가 이번 시험을 망쳤어요. 정말 속상하네요."
result = analyzer.analyze_speech_level(text)

print(f"전체 수준: {result.level}")
print(f"존댓말 비율: {result.formal_ratio:.2f}")
print(f"전환 지점:")
for transition in result.level_transitions:
    print(f"  위치 {transition.position}: {transition.from_level} -> {transition.to_level}")
    print(f"  문맥: ...{transition.context}...")
```

### 예제 3: 가스라이팅 패턴 탐지

```python
from voice_man.services.nlp.cultural_analyzer import KoreanCulturalAnalyzer

analyzer = KoreanCulturalAnalyzer()

text = "네가 기억을 못하네. 내가 그런 적 없다고. 네가 잘못 생각하고 있는 거야."
patterns = analyzer.detect_manipulation_patterns(text)

print(f"탐지된 패턴: {len(patterns)}개")
for p in patterns:
    print(f"\n카테고리: {p.category}")
    print(f"패턴: {p.pattern}")
    print(f"신뢰도: {p.confidence:.2f}")
    print(f"위치: {p.position}")
```

---

## 트러블슈팅

### 문제: CUDA를 사용할 수 없습니다

```python
# 해결: CPU 명시적 지정
model = KoBERTModel(device="cpu")
```

### 문제: 메모리 부족 오류

```python
# 해결: 배치 크기 감소
results = classifier.classify_batch(texts, batch_size=4)

# 또는 캐시 정리
model.clear_cache()
```

### 문제: 모델 로딩이 너무 느림

```python
# 해결: 웜업 추론 실행
model = KoBERTModel()
model.warmup()
```

---

## 성능 팁

1. **GPU 사용**: CUDA 사용 가능 시 GPU로 10-50배 성능 향상
2. **배치 처리**: 배치 크기를 8-16으로 설정하여 효율적 처리
3. **메모리 관리**: 사용 후 캐시 정리로 메모리 확보
4. **싱글톤 패턴**: KoBERTModel 재사용으로 불필요한 로딩 방지

---

## 참고 자료

- [KoBERT GitHub](https://github.com/SKTBrain/KoBERT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SKT Aibrain KoBERT Model Card](https://huggingface.co/skt/kobert-base-v1)

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2026-01-11
**SPEC**: SPEC-NLP-KOBERT-001
