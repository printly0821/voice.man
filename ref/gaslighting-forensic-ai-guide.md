# 🔬 가스라이팅 탐지 AI & 범죄 프로파일링 포렌식 도구 종합 가이드

> **작성일**: 2025년 1월 10일  
> **목적**: 가스라이팅/심리조작 탐지 AI 시스템 및 범죄 프로파일링 포렌식 도구 개발을 위한 오픈소스, 학술 연구, 데이터셋 종합 정리

---

## 📑 목차

1. [가스라이팅 탐지 AI - 학술 연구 & 데이터셋](#1-가스라이팅-탐지-ai---학술-연구--데이터셋)
2. [음성 감정 인식 (Speech Emotion Recognition)](#2-음성-감정-인식-speech-emotion-recognition)
3. [한국어 NLP & 감정 분석](#3-한국어-nlp--감정-분석)
4. [포렌식 언어학 & 텍스트 분석](#4-포렌식-언어학--텍스트-분석)
5. [음성 포렌식 & 화자 인식](#5-음성-포렌식--화자-인식)
6. [심리 프로파일링 & Dark Triad 탐지](#6-심리-프로파일링--dark-triad-탐지)
7. [기만 탐지 (Deception Detection)](#7-기만-탐지-deception-detection)
8. [죄명별 범죄 분류](#8-죄명별-범죄-분류)
9. [한국 국내 자원](#9-한국-국내-자원)
10. [통합 포렌식 도구 & 플랫폼](#10-통합-포렌식-도구--플랫폼)
11. [시스템 구현 로드맵](#11-시스템-구현-로드맵)
12. [참고 자료 & 링크](#12-참고-자료--링크)

---

## 1. 가스라이팅 탐지 AI - 학술 연구 & 데이터셋

### 1.1 MentalManip (ACL 2024) ⭐ 핵심 자원

정신적 조작은 대인 대화에서 중요한 학대 형태로, 맥락 의존적이고 미묘한 특성으로 인해 식별이 어렵습니다.

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/audreycs/MentalManip |
| **논문** | ACL 2024 Long Papers |
| **데이터** | 4,000개 주석된 영화 대화 |
| **라이선스** | Apache-2.0 |
| **분석 차원** | 조작 기법 + 피해자 취약점 |

**디렉토리 구조:**
```
MentalManip/
├── README.md
├── mentalmanip_dataset/          # 최종 데이터셋
├── experiments/
│   ├── datasets/
│   ├── manipulation_detection/   # 조작 탐지 코드
│   └── technique_vulnerability/  # 기법/취약점 분류
└── statistic_analysis/
```

**설치:**
```bash
conda config --add channels conda-forge pytorch nvidia
pip install transformers datasets
```

---

### 1.2 DeepCoG Framework & Gaslighting Dataset

LLM의 가스라이팅 취약성을 연구한 프레임워크.

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/Maxwe11y/gaslightingLLM |
| **논문** | arXiv:2410.09181 |
| **HuggingFace** | `Maxwe11y/gaslighting` |
| **데이터 구성** | 훈련 1,752 / 검증 124 / 테스트 124 |

**데이터셋 특징:**
- 가스라이팅 대화: LLM이 사용자에게 자기 의심, 자기 비난, 혼란을 유발
- 안티-가스라이팅 대화: 지지적 언어로 안심과 격려 제공
- Apache-2.0 라이선스

**주요 발견:**
- 프롬프트/파인튜닝 공격으로 오픈소스 LLM을 가스라이터로 변환 가능
- 안티-가스라이팅 안전 정렬 전략으로 12.05% 안전성 향상
- 독성 탐지기로는 가스라이팅 탐지 불가 (미묘한 조작이라 독성 단어 없음)

---

### 1.3 MultiManip & SELF-PERCEPT (2025)

다중 참여자 대화에서의 조작 탐지 연구.

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/danushkhanna/self-percept |
| **데이터** | 220개 다중 턴, 다중 참여자 대화 |
| **출처** | 리얼리티 쇼 기반 |
| **조작 유형** | 11가지 구별되는 조작 기법 |

---

### 1.4 MentalMAC (2025 최신)

| 항목 | 내용 |
|------|------|
| **데이터셋** | ReaMent - 5,000개 실제 YouTube 대화 |
| **방법론** | EvoSA (진화적 연산 + 화행 이론) |
| **개선** | 64.2% 정확도, 53.7% 가중 F1 상대적 개선 |

---

### 1.5 Gaslighting R Package

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/Reilly-ConceptsCognitionLab/Gaslighting |
| **언어** | R |
| **라이선스** | GPL-3.0 |
| **용도** | 텍스트 샘플에서 가스라이팅 탐지 |

---

## 2. 음성 감정 인식 (Speech Emotion Recognition)

### 2.1 emotion2vec (ACL 2024) ⭐ 추천

음성 감정 인식을 위한 기반 모델 시리즈. 데이터 기반 방법을 통해 언어와 녹음 환경의 영향을 극복.

| 버전 | 파라미터 | 특징 |
|------|----------|------|
| emotion2vec+ seed | - | 학술 감정 데이터로 파인튜닝 |
| emotion2vec+ base | ~90M | 필터링된 대규모 의사 라벨 데이터 |
| emotion2vec+ large | ~300M | 40k 시간 음성 감정 데이터 학습 |

**GitHub:** https://github.com/ddlBoJack/emotion2vec

**사용 예시:**
```python
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_large")
rec_result = model.generate("audio.wav", extract_embedding=False)
# 9가지 감정: angry, disgust, fear, happy, neutral, sad, surprise, ...
```

**HuggingFace 모델:**
- `iic/emotion2vec_plus_seed`
- `iic/emotion2vec_plus_base`
- `iic/emotion2vec_plus_large`

---

### 2.2 SpeechBrain + wav2vec2

PyTorch 기반 올인원 음성 툴킷.

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/speechbrain/speechbrain |
| **HuggingFace** | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` |
| **데이터셋** | IEMOCAP |
| **지원 기능** | ASR, TTS, 화자 인식, 음성 향상, 분리, 감정 인식 |

**사용 예시:**
```python
from speechbrain.inference.interfaces import foreign_class

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)
out_prob, score, index, text_lab = classifier.classify_file("audio.wav")
print(text_lab)
```

---

### 2.3 기타 SER 오픈소스

| 프로젝트 | GitHub | 특징 |
|----------|--------|------|
| wav2vec2-ser | habla-liaa/ser-with-w2v2 | INTERSPEECH 2021 |
| soxan | m3hrdadfi/soxan | 다양한 SER 태스크 |
| FT-w2v2-ser | b04901014/FT-w2v2-ser | ICASSP 2022 |
| Speech-Emotion-Analyzer | MiteshPuthran/Speech-Emotion-Analyzer | CNN, RAVDESS |
| vera | GeorgiosIoannouCoder/vera | 7가지 감정 |

---

### 2.4 SER 데이터셋

| 데이터셋 | 파일 수 | 배우 | 감정 | 언어 |
|----------|---------|------|------|------|
| RAVDESS | 1,440 | 24 | 8 | 영어 |
| CREMA-D | 7,442 | 91 | 6 | 영어 |
| SAVEE | 480 | 4 | 7 | 영어 |
| TESS | 2,800 | 2 | 7 | 영어 |
| IEMOCAP | 10,039 | 10 | 9 | 영어 |
| EMO-DB | 535 | 10 | 7 | 독일어 |

---

## 3. 한국어 NLP & 감정 분석

### 3.1 KoBERT (SKT Brain)

한국어 BERT 모델. 기존 BERT의 한국어 성능 한계 극복.

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/SKTBrain/KoBERT |
| **라이선스** | Apache-2.0 |
| **학습 데이터** | 한국어 위키 5백만 문장, 54백만 단어 |
| **지원** | PyTorch, TensorFlow, ONNX, MXNet |
| **감정 분류 정확도** | 90.50% (어절 단위) |
| **감정 분류** | 7가지 (공포/놀람/분노/슬픔/중립/행복/혐오) |

**사용 예시:**
```python
from kobert import get_pytorch_kobert_model
model, vocab = get_pytorch_kobert_model()
```

---

### 3.2 한국어 감정 분석 데이터셋

| 데이터셋 | 출처 | 감정 분류 | 크기 |
|----------|------|----------|------|
| 한국어 감정 정보 단발성 대화 | AI Hub | 7가지 | 38,600 문장 |
| NSMC | Naver | 긍정/부정 | 200,000 |
| korean_hate_speech | GitHub | 혐오 댓글 | - |
| 국립국어원 감정 분석 | NIKL | 감정 라벨링 | - |

---

### 3.3 Korpora (한국어 말뭉치 통합)

다양한 한국어 데이터셋을 통합 제공하는 오픈소스 Python 패키지.

**GitHub:** https://github.com/ko-nlp/Korpora

```python
from Korpora import Korpora
corpus = Korpora.load("nsmc")  # 네이버 감성 영화 리뷰
```

**포함 데이터셋:**
- KcBERT 학습데이터
- 챗봇 문답 데이터
- 혐오댓글데이터
- 청와대 청원데이터
- NLI/STS 데이터
- 한국어 위키피디아
- 나무위키
- NER 데이터
- 모두의 말뭉치 시리즈

---

### 3.4 기타 한국어 NLP 도구

| 도구 | 용도 |
|------|------|
| KoNLPy | 형태소 분석, 토크나이징 |
| KoGPT2 | 한국어 디코더 언어모델 (40GB+ 텍스트 학습) |
| KcBERT | 댓글 기반 BERT |
| KLUE-BERT | Aspect-Based Sentiment Analysis |

---

## 4. 포렌식 언어학 & 텍스트 분석

### 4.1 포렌식 언어학 개념

법언어학자들이 증거를 제공하는 분야:
- **상표권 및 지적재산권 분쟁**
- **저자 식별**: 익명 텍스트의 작성자를 알려진 용의자의 글 샘플과 비교
- **포렌식 문체론**: 표절 사례 식별
- **음성 식별 (포렌식 음성학)**: 녹음된 음성의 신원 확인
- **담화 분석**: 발화 구조 분석, 범죄 공모 동의 여부 판단
- **언어 분석 (포렌식 방언학)**: 망명 신청자의 언어적 배경 추적

### 4.2 오픈소스 도구

| 프로젝트 | 기능 | 링크 |
|----------|------|------|
| **Fast Stylometry** | 저자의 문체적, 언어적 "지문"으로 텍스트 저자 식별 | fastdatascience.com |
| **stylo (R)** | 언어학/인문학 텍스트 분석용 그래픽 패키지 | CRAN |
| **SpeechToText** | 음성 활동 감지(VAD) + 자동 음성 인식(ASR), Autopsy 통합 | GitHub |

**Fast Stylometry 설치:**
```bash
pip install faststylometry
```

---

### 4.3 NLP 기반 포렌식 프레임워크

**핵심 논문:**
- "A psycholinguistic NLP framework for forensic text analysis of deception and emotion" (Frontiers AI, 2025)
- "Identifying Persons of Interest in Digital Forensics Using NLP-Based AI" (MDPI, 2024)
- "NLP-based digital forensic investigation platform for online communications" (ScienceDirect, 2021)

**분석 기법:**
- n-gram 분석
- 감정 분석 (Sentiment Analysis)
- 주관성 분석 (Subjectivity)
- 토픽 모델링 (LDA)
- 명명 개체 인식 (NER)
- 쌍별 상관관계 (Pairwise Correlation)
- 단어 벡터 코사인 거리 측정

---

## 5. 음성 포렌식 & 화자 인식

### 5.1 오픈소스 화자 인식 플랫폼

| 프로젝트 | 특징 | 라이선스 |
|----------|------|----------|
| **ALIZÉ** | 화자 인식 오픈소스 플랫폼, 가우시안 혼합 모델 기반, Linux/Windows/Mac/Android 지원 | LGPL |
| **SpeechBrain** | PyTorch 기반, ASR/TTS/화자인식/감정인식 통합 | Apache-2.0 |
| **SincNet** | 원시 오디오 샘플 효율적 처리 신경 아키텍처 | MIT |
| **SIDEKIT** | 화자/언어 인식 교육용 툴킷 | - |
| **OpenSpeaker** | 데이터 준비→모델 훈련→다중 플랫폼 배포 전체 과정 | - |

**ALIZÉ GitHub:** https://github.com/ALIZE-Speaker-Recognition

---

### 5.2 SpeechToText (디지털 포렌식용)

음성 활동 감지(VAD)와 자동 음성 인식(ASR) 모듈을 사용하여 음성 콘텐츠를 텍스트로 변환.

**특징:**
- Autopsy 포렌식 소프트웨어 모듈 제공
- 100% 음성 감지 정확도 (비암호화 오디오/비디오)
- WER 7.80% (LibriSpeech test-clean)
- 14개 Android 앱 음성 녹음 분석 지원

---

### 5.3 GitHub 토픽

- https://github.com/topics/speaker-recognition
- https://github.com/topics/speaker-identification
- https://github.com/topics/speaker-verification
- https://github.com/topics/speaker-diarization

---

## 6. 심리 프로파일링 & Dark Triad 탐지

### 6.1 Dark Triad (어두운 삼인방) 개념

Paulhus와 Williams(2002)가 정의한 세 가지 성격 특성:

| 특성 | 정의 | 행동 패턴 |
|------|------|----------|
| **나르시시즘** | 과대망상적 자기애, 우월감, 특권 의식 | 끊임없는 칭찬 추구, 대인관계 위협 |
| **마키아벨리즘** | 조작적, 냉소적, 기만적, 단기적 이익 추구 | 전략적 사고, 공감 부족 |
| **사이코패시** | 공감 부족, 충동성, 반사회적 행동 | 무자비함, 타인 감정 무시 |

**사회적 비용:**
- 사이코패시 개인의 범죄 활동으로 인한 사회적 금전적 손실: 약 5,000억 달러
- Dark Triad와 범죄 간 양의 상관관계 확인 (메타분석)

---

### 6.2 AI 기반 Dark Triad 탐지 연구

| 연구 | 정확도 | 모델 | 데이터 |
|------|--------|------|--------|
| Random Forest (마키아벨리즘) | 83-84% | BERT + ML | 텍스트 |
| Twitter 분석 | ~75% | CNN, NB, LR | 2,927명 |
| SD3 설문 기반 AI | ICC 최적화 | R 기반 | 210명 |
| BILSTM | - | Deep Neural Network | 트윗 |

**주요 논문:**
- "Identifying Human Dark Triad from Text Data Through Machine Learning Models" (IJRIAS, 2024)
- "Predicting Dark Triad Personality Traits from Twitter Usage" (Semantic Scholar)
- "The Dark Triad of personality and criminal and delinquent behavior: meta-analysis" (ScienceDirect, 2025)

---

### 6.3 측정 도구 & 데이터셋

| 자원 | 설명 | 문항 수 |
|------|------|---------|
| **SD3 (Short Dark Triad)** | 표준화된 Dark Triad 측정 도구 | 27 |
| **Dirty Dozen** | 간편 버전 | 12 |
| **MACH-IV** | 마키아벨리즘 측정 | 20 |
| **NPI** | 나르시시즘 측정 | 40 |
| **SRP-III** | 사이코패시 측정 | 64 |

**데이터셋:**
- Mendeley Dark Triad Data: doi:10.17632/87vx6jfnrp.1

---

## 7. 기만 탐지 (Deception Detection)

### 7.1 멀티모달 거짓말 탐지 시스템

**Liar AI 특징:**
- 다중 모달 데이터 융합: 얼굴 비디오(미세 표정, 시선), 오디오(음성 스트레스, 톤), 텍스트(언어적 단서), 생리적 센서 데이터(심박수, 피부 전도도)
- 설명 가능성: LIME/SHAP 기반 주의 시각화, 특징 중요도 점수
- 실시간 및 배치 처리 지원

**GitHub Gist:** https://gist.github.com/ruvnet/481d0f8c2190decead7b14164ae3323c

---

### 7.2 오픈소스 거짓말 탐지 프로젝트

| GitHub 프로젝트 | 특징 | 데이터셋 |
|-----------------|------|----------|
| **lie-detector** | CMU CDC, LSTM + GloVe | CMU Deceptive Speech |
| **Deception-Detection** | 언어적 단서, 재판 데이터 | Michigan + Opinion Spam |
| **TruthDetection** | Politifact, BERT/RoBERTa | Politifact |
| **rnn-lie-detector** | TensorFlow RNN, MFCC | CSC |
| **Lie_to_me** | 얼굴 + 음성 인식 | - |

**GitHub 토픽:** https://github.com/topics/lie-detector

---

### 7.3 LLM 기반 거짓말 탐지

**FLAN-T5 연구 (Scientific Reports, 2023):**
- 개인 의견, 자전적 기억, 미래 의도 데이터셋에서 테스트
- 10-fold 교차 검증
- 문체 분석으로 언어적 차이 파악

**데이터셋:**
- CMU Deceptive Speech Corpus
- Deceptive Opinion Spam Corpus v1.4
- Real life trial data (Michigan)

---

## 8. 죄명별 범죄 분류

### 8.1 지원 가능한 범죄 유형

**ENUSC 2023-24 (15개 유형):**
협박, 사이버 괴롭힘, 사이버 파괴, 사이버 해킹, 사기/속임, 스캠, 절도, 폭행, 차량 절도, 기습 절도, 차량 부품 절도, 폭력 강도, 가정 강도, 차량 기물 파손, 가정 기물 파손

**한국 경찰 데이터 (7개 범주):**
성범죄(성폭력, 성희롱, 데이트 폭력, 스토킹), 폭력 범죄(가정폭력, 아동학대, 공갈, 협박 등), 사기, 청소년 비행, 절도, 교통, 기타

---

### 8.2 가정폭력 & 대인 폭력 탐지

**BioBERT 기반 NLP 모델:**
- 폭력 존재 탐지
- 환자 상태: 가해자, 목격자, 피해자
- 폭력 유형: 가정, 신체, 성적

**텍스트 마이닝 학대 유형:**
- 신체적 학대
- 비신체적 학대 (욕설, 위협, 스토킹)
- ADVO 위반 (접근금지명령 위반, 괴롭힘)

---

### 8.3 범죄 분류 ML 모델 성능

| 연구 | 범죄 유형 | 모델 | 성능 |
|------|----------|------|------|
| ENUSC 2023-24 | 15개 | BERT, BOW | F1 향상 |
| 한국 경찰 데이터 | 7개 | Kernel SVM | 93.4% 정확도 |
| 터키 가정폭력 | 가정폭력 | SVM, LR | 100% |
| 사이버범죄 (인도) | Hinglish | HingBERT, HingRoBERTa | - |

---

### 8.4 죄명별 탐지 모듈 설계

| 죄명 | 텍스트 지표 | 음성 지표 | 심리 지표 |
|------|------------|----------|----------|
| **가정폭력** | 협박, 통제, 고립 표현 | 고음, 공격적 톤 | 권력 불균형 패턴 |
| **스토킹** | 집착, 감시, 위치 추적 언급 | 반복 연락 패턴 | 집착적 애착 |
| **협박** | 직/간접 위협, 조건부 표현 | 위협적 음조 | 권력 과시 |
| **가스라이팅** | 현실 왜곡, 기억 부정 | 차분하지만 조작적 | 나르시시즘 |
| **사기** | 과장, 긴급성 유발 | 설득적 톤 | 마키아벨리즘 |
| **아동학대** | 통제, 위협, 비하 | 감정적 변동 | 사이코패시 특성 |

---

## 9. 한국 국내 자원

### 9.1 공공 기관

| 기관 | 역할 | 링크 |
|------|------|------|
| **대검찰청 디지털포렌식연구소** | 디지털포렌식 연구·개발 (2013년 설립), 도구 검증, 기술 표준화 | spo.go.kr |
| **경찰청 디지털포렌식센터** | 전국 최대 규모 디지털 포렌식 | police.go.kr |
| **KISA (한국인터넷진흥원)** | 디지털 침해사고 대응, 피싱 대응 | kisa.or.kr |
| **보호나라 (KrCERT/CC)** | 보안 취약점 정보 포털, 침해사고 대응 | boho.or.kr |

---

### 9.2 한국어 참고 도서

**『오픈소스 툴을 이용한 디지털 포렌식』**
- 저자: Cory Altheide, Harlan Carvey
- 출판사: 비제이퍼블릭
- 내용: 포렌식 파일 시스템 분석, 윈도우즈/맥/리눅스 아티팩트 분석, 커맨드라인/그래픽 기반 도구 사용법

---

### 9.3 한국 관련 연구

**한국 경찰 통화 데이터 연구 (Asian Journal of Criminology, 2024):**
- 2020년 9개월, 1,500만 통화 로그
- Kernel SVM 사용, 93.4% 정확도
- 사이버범죄 4.6% 추정

---

## 10. 통합 포렌식 도구 & 플랫폼

### 10.1 오픈소스 포렌식 툴킷

| 도구 | 용도 | 링크 |
|------|------|------|
| **Autopsy** | 통합 디지털 포렌식 플랫폼 | sleuthkit.org |
| **The Sleuth Kit** | 디스크 이미지 분석 | sleuthkit.org |
| **Volatility** | 메모리 포렌식 | volatilityfoundation.org |
| **Wireshark** | 네트워크 패킷 분석 | wireshark.org |
| **DeepSound** | 오디오 스테가노그래피 분석 | - |
| **CAINE** | 포렌식 전용 Linux 배포판 | caine-live.net |

**GitHub:** https://github.com/karthik997/Forensic_Toolkit

---

### 10.2 상용 도구 (참고용)

| 도구 | 특징 |
|------|------|
| **OpenText Forensic (EnCase)** | 36,000+ 디바이스 프로필, AI 기반 CSAM/무기 탐지 |
| **FTK (Forensic Toolkit)** | 데이터 수집, 복호화, 분석 |
| **Cellebrite** | 모바일 포렌식 특화 |

---

## 11. 시스템 구현 로드맵

### 11.1 Phase 1: 텍스트 분석 (한국어 특화)

**목표:** 한국어 가스라이팅/심리조작 패턴 탐지

**기술 스택:**
- NLP: KoBERT, KLUE-BERT
- 데이터: MentalManip (영어) + 한국어 감정 데이터셋
- 프레임워크: Hugging Face Transformers, PyTorch

**구현 내용:**
- 한국어 가스라이팅 표현 패턴 코퍼스 구축
- 존댓말/반말 권력 관계 분석
- 한국 문화적 맥락 반영 (가족/직장 내 위계)

---

### 11.2 Phase 2: 음성 분석

**목표:** 음성에서 감정 및 조작 신호 탐지

**기술 스택:**
- SER 모델: emotion2vec, SpeechBrain
- 특징 추출: LibROSA (MFCC, 톤, 피치, 속도)
- STT: OpenAI Whisper

**데이터셋:**
- RAVDESS, IEMOCAP (영어)
- 한국어 음성 데이터 (AI Hub)

---

### 11.3 Phase 3: 심리 프로파일링

**목표:** Dark Triad 특성 및 기만 패턴 탐지

**기술 스택:**
- Dark Triad 분류기: BERT + Random Forest
- 기만 탐지: LSTM, Transformer
- 설명 가능 AI: SHAP, LIME

---

### 11.4 Phase 4: 통합 시스템

**목표:** 멀티모달 융합 및 죄명 분류

**아키텍처:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    범죄 프로파일링 AI 포렌식 시스템                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   텍스트 분석    │  │   음성 분석     │  │  심리 프로파일   │         │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤         │
│  │ • KoBERT/KLUE   │  │ • emotion2vec   │  │ • Dark Triad    │         │
│  │ • 가스라이팅 탐지│  │ • SpeechBrain   │  │ • 기만 탐지     │         │
│  │ • 위협 분류     │  │ • ALIZÉ 화자ID  │  │ • 조작 패턴     │         │
│  │ • 조작 기법     │  │ • Whisper STT   │  │ • 취약점 분석   │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                  │
│           └────────────────────┼────────────────────┘                  │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      멀티모달 융합 레이어                         │   │
│  │  • Late Fusion (각 모달리티 점수 통합)                           │   │
│  │  • Attention 기반 동적 가중치                                    │   │
│  │  • 시간 경과 패턴 추적 (에스컬레이션)                            │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      죄명 분류 & 위험도 산출                      │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  가정폭력 │ 스토킹 │ 협박 │ 성희롱 │ 사기 │ 아동학대 │ ...      │   │
│  │  ───────────────────────────────────────────────────────────────│   │
│  │  위험도: ████████░░ 80%  │  신뢰도: 92%  │  긴급도: HIGH        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  출력: 포렌식 리포트 │ 법적 증거 포맷 │ 훈련 피드백 │ 실시간 알림 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 11.5 Phase 5: 훈련 시스템 (차별화)

**목표:** 가스라이팅 대응 훈련 시뮬레이터

**구현:**
- AI 가스라이터 시뮬레이션: DeepCoG 방법론 활용
- 실시간 피드백: 위험 패턴 감지 시 교육적 알림
- 대응 훈련: 건강한 경계 설정 대화법, 상황별 스크립트

---

### 11.6 기술 스택 요약

```python
# 핵심 프레임워크
frameworks = {
    "NLP": ["transformers", "KoBERT", "spaCy", "KLUE-BERT"],
    "음성": ["SpeechBrain", "emotion2vec", "librosa", "whisper"],
    "ML": ["PyTorch", "scikit-learn", "XGBoost"],
    "포렌식": ["Autopsy", "The Sleuth Kit"],
    "설명가능AI": ["SHAP", "LIME", "attention visualization"],
    "배포": ["FastAPI", "Docker", "WebSocket", "React Native"]
}

# 데이터셋
datasets = {
    "가스라이팅": "Maxwe11y/gaslighting (HuggingFace)",
    "심리조작": "MentalManip (GitHub: audreycs/MentalManip)",
    "Dark Triad": "SD3 기반 Twitter 데이터",
    "기만탐지": "CMU Deceptive Speech Corpus",
    "한국어 감정": "AI Hub 감정 대화 데이터셋",
    "범죄분류": "한국 경찰 통화 로그 (학술용)"
}
```

---

## 12. 참고 자료 & 링크

### 12.1 핵심 GitHub 리포지토리

| 분류 | 프로젝트 | URL |
|------|----------|-----|
| 심리조작 | MentalManip | https://github.com/audreycs/MentalManip |
| 가스라이팅 | gaslightingLLM | https://github.com/Maxwe11y/gaslightingLLM |
| 음성감정 | emotion2vec | https://github.com/ddlBoJack/emotion2vec |
| 음성처리 | SpeechBrain | https://github.com/speechbrain/speechbrain |
| 화자인식 | ALIZE | https://github.com/ALIZE-Speaker-Recognition |
| 한국어NLP | KoBERT | https://github.com/SKTBrain/KoBERT |
| 한국어말뭉치 | Korpora | https://github.com/ko-nlp/Korpora |
| 포렌식툴킷 | Forensic_Toolkit | https://github.com/karthik997/Forensic_Toolkit |

---

### 12.2 HuggingFace 모델 & 데이터셋

| 분류 | 이름 | URL |
|------|------|-----|
| 데이터셋 | Gaslighting | huggingface.co/datasets/Maxwe11y/gaslighting |
| 모델 | emotion2vec+ | huggingface.co/iic |
| 모델 | SpeechBrain SER | huggingface.co/speechbrain |

---

### 12.3 학술 검색 키워드

**영어:**
- "Forensic linguistics NLP"
- "Deception detection machine learning"
- "Dark Triad personality text analysis"
- "Domestic violence NLP classification"
- "Criminal profiling AI"
- "Mental manipulation detection"
- "Gaslighting NLP"

**한국어:**
- "디지털 포렌식 NLP"
- "범죄 프로파일링 AI"
- "가정폭력 텍스트 분석"
- "화자 인식 범죄수사"
- "가스라이팅 탐지"

---

### 12.4 주요 학회 & 저널

| 분야 | 학회/저널 |
|------|----------|
| 디지털 포렌식 | DFRWS, ACSAC |
| NLP | ACL, EMNLP, NAACL |
| 범죄학 | Journal of Criminal Justice, Criminology |
| 심리학 | Journal of Personality, Personality and Individual Differences |
| AI | Frontiers in AI, Scientific Reports |

---

### 12.5 온라인 자원

| 자원 | URL |
|------|-----|
| hatespeechdata.com | 63개 혐오발언 데이터셋 카탈로그 |
| AI Hub | aihub.or.kr (한국어 감정 데이터) |
| NIST 포렌식 도구 검증 | tsapps.nist.gov |
| Forensic-Proof | forensic-proof.com |

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-01-10 | 1.0 | 초기 버전 작성 |

---

## 📜 라이선스

이 문서는 연구 및 개발 참고용으로 작성되었습니다. 각 도구 및 데이터셋의 라이선스는 해당 프로젝트의 라이선스를 따릅니다.

---

*이 가이드는 Printly AI 팀의 가스라이팅 탐지 및 범죄 프로파일링 포렌식 시스템 개발 프로젝트를 위해 작성되었습니다.*
