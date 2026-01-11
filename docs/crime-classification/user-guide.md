# 범죄 분류 사용자 가이드

## 빠른 시작

### 1. 텍스트 분류

가장 간단한 방법은 텍스트만으로 범죄 유형을 분류하는 것입니다.

```python
from voice_man.services.crime_classification.multimodal_classifier import MultimodalClassifierService

classifier = MultimodalClassifierService()
result = classifier.classify_text_only("돈 빌려줄게? 갚을 생각 없는데.")

print(f"예측 죄명: {result.primary_crime}")
print(f"신뢰도: {result.confidence:.2f}")
```

### 2. 오디오 특성 포함 분류

오디오 특성(스트레스 레벨, 음량 등)을 함께 제공하면 정확도가 향상됩니다.

```python
result = classifier.classify(
    text_transcript="돈 빌려줄게? 갚을 생각 없는데.",
    audio_features={
        "stress_level": 0.8,
        "volume_db": -10.5,
        "pitch_hz": 150
    }
)
```

### 3. 심리 프로파일 포함 분류

심리 프로파일을 함께 제공하면 가장 높은 정확도를 달성할 수 있습니다.

```python
from voice_man.services.crime_classification.psychological_profiler import PsychologicalProfilerService

profiler = PsychologicalProfilerService()
psych_profile = profiler.create_profile("내가 뭐라고 하면 다 맞아.")

result = classifier.classify(
    text_transcript="돈 빌려줄게? 갚을 생각 없는데.",
    psychological_profile=psych_profile
)
```

## 11개 죄명 분류 기준

### 금전 관련 범죄

#### 사기 (Fraud)
- **탐지 패턴**: 금전 요구, 갚지 않을 의사 표명, 기망
- **주요 키워드**: "돈 빌려줘", "갚을 생각 없어", "꼼수"
- **오디오 지표**: 높은 말속도, 낮은 음량 변화

#### 공갈 (Extortion)
- **탐지 패턴**: 협박에 의한 금품 갈취
- **주요 키워드**: "말 안 들으면 큰일난다", "신고해봐"
- **오디오 지표**: 높은 음량, 급격한 피치 변화

#### 업무상위계등 (Breach of Trust)
- **탐지 패턴**: 업무권한 남용, 배임
- **주요 키워드**: "회사 돈", "승인 없이", "유용"
- **오디오 지표**: 안정적인 음량, 낮은 감정 표현

### 폭력 관련 범죄

#### 강간죄 (Rape)
- **탐지 패턴**: 성폭력, 성적 위협
- **주요 키워드**: "성관계", "거부 못해"
- **오디오 지표**: 매우 높은 스트레스, 불규칙한 호흡

#### 강제추행죄 (Forced Indecency)
- **탐지 패턴**: 성폭력, 추행
- **주요 키워드**: "몸 만져", "거부 못해"
- **오디오 지표**: 높은 스트레스, 빠른 말속도

#### 준강간 및 준강제추행죄 (Quasi Rape/Indecency)
- **탐지 패턴**: 심신미약, 위법약물 상태에서의 성폭력
- **주요 키워드**: "약",술 먹고", "의식 없어"
- **오디오 지표**: 느린 말속도, 불명확한 발음

### 위협 관련 범죄

#### 협박죄 (Threat)
- **탐지 패턴**: 신체/생명/재산/관계 단절 위협
- **주요 키워드**: "죽여버린다", "신고해봐", "아무도 모를 거야"
- **오디오 지표**: 높은 음량, 급격한 피치 상승

#### 감금죄 (False Imprisonment)
- **탐지 패턴**: 행동 제한, 구금
- **주요 키워드**: "나가지 마", "잠가", "감시"
- **오디오 지표**: 낮은 음량, 느린 말속도

#### 체포감금죄 (Arrest Detention)
- **탐지 패턴**: 권한 없는 체포, 감금
- **주요 키워드**: "체포했다", "경찰이다", "데려간다"
- **오디오 지표**: 명령형 톤, 높은 음량

#### 특수공무원감금죄 (Special Officer Detention)
- **탐지 패턴**: 공무원 감금
- **주요 키워드**: "공무원이다", "가두", "출근 못하게"
- **오디오 지표**: 공격적인 톤, 불안정한 감정

### 명예 관련 범죄

#### 모욕죄 (Insult)
- **탐지 패턴**: 인격 모독, 비하
- **주요 키워드**: "바보", "쓰레기", "질 낮아"
- **오디오 지표**: 높은 음량, 빠른 말속도

#### 명예훼손죄 (Defamation)
- **탐지 패턴**: 허위사실 유포
- **주요 키워드**: "그 사람~했어", "사실이야", "알려줄게"
- **오디오 지표**: 중립적인 톤, 안정적인 말속도

## 신뢰도 점수 해석

| 점수 범위 | 해석 | 권장 사항 |
|-----------|------|-----------|
| 0.9 - 1.0 | 매우 높음 | 법적 증거로 사용 가능 |
| 0.7 - 0.9 | 높음 | 추가 검증 후 사용 |
| 0.5 - 0.7 | 중간 | 참고 자료로 활용 |
| 0.3 - 0.5 | 낮음 | 전문가 상담 권장 |
| 0.0 - 0.3 | 매우 낮음 | 신뢰하지 않음 |

## 95% 신뢰구간 해석

```python
result = classifier.classify_text_only("협박 텍스트...")

print(f"신뢰구간: [{result.confidence_interval['lower']:.2f}, {result.confidence_interval['upper']:.2f}]")
```

신뢰구간이 좁을수록 예측이 안정적입니다:
- `[0.75, 0.85]`: 안정적 예측 (구간 폭 0.10)
- `[0.50, 0.90]`: 불안정적 예측 (구간 폭 0.40)

## Dark Triad 심리 프로파일

### 나르시시즘 (Narcissism)
- **높은 점수**: 자기중심적 발언, 타인 무시
- **주요 패턴**: "내가", "내가 해결", "너는 틀렸어"

### 마키아벨리즘 (Machiavellianism)
- **높은 점수**: 조작적 발언, 기만
- **주요 패턴**: "속이다", "이용하다", "꼼수"

### 사이코패시 (Psychopathy)
- **높은 점수**: 공격적 발언, 죄책감 부재
- **주요 패턴**: "신경 안 써", "아무도 모른다", "귀찮게"

## 법적 증거로 사용하기

### 1. 증거 기준 확인

```python
from voice_man.services.crime_classification.legal_evidence_mapper import LegalEvidenceMapperService

mapper = LegalEvidenceMapperService()
requirements = mapper.get_legal_requirements("fraud")

print("필요 요소:")
for element in requirements.required_elements:
    print(f"- {element}")

print("\n증거 기준:")
for standard in requirements.evidence_standards:
    print(f"- {standard}")
```

### 2. 보고서 생성

분류 결과를 PDF 보고서로 생성할 수 있습니다:

```python
from voice_man.reports.pdf_generator import ForensicPDFGenerator

generator = ForensicPDFGenerator()
generator.generate_crime_classification_report(
    result=classification_result,
    output_path="crime_classification_report.pdf"
)
```

## 팁과 모범 사례

1. **최소 텍스트 길이**: 100자 이상 권장
2. **오디오 품질**: 16kHz 이상, 잡음 최소화
3. **여러 증거 결합**: 텍스트 + 오디오 + 심리 프로파일
4. **전문가 상담**: 높은 위험도 결과는 반드시 전문가와 상담

## 제한 사항

1. **법적 조언 아님**: 이 시스템은 법적 조언을 제공하지 않습니다
2. **거주지 법률 확인**: 한국 형법 기준이며, 다른 국가에는 적용되지 않을 수 있습니다
3. **오남용 금지**: 타인의 권리를 침해하는 목적으로 사용해서는 안 됩니다
