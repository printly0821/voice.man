---
id: SPEC-FORENSIC-001
version: "1.0.0"
status: "in_progress"
created: "2026-01-09"
updated: "2026-01-09"
author: "지니"
priority: "HIGH"
title: "범죄 프로파일링 기반 음성 포렌식 분석 시스템"
related_specs:
  - SPEC-VOICE-001
  - SPEC-WHISPERX-001
  - SPEC-E2ETEST-001
tags:
  - 음성포렌식
  - 범죄프로파일링
  - SER
  - 스트레스분석
  - 가스라이팅
  - 시각화
  - 법적증거
lifecycle: "spec-anchored"
---

# SPEC-FORENSIC-001: 범죄 프로파일링 기반 음성 포렌식 분석 시스템

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-09 | 지니 | 초안 작성 - 범죄 프로파일링 기반 음성 포렌식 분석 시스템 요구사항 정의 |

---

## 1. 개요

### 1.1 목적

국내외 범죄 프로파일링 기법과 음성 포렌식 학술 연구를 기반으로 한 고급 음성 분석 시스템을 구축한다. 기존 SPEC-VOICE-001의 텍스트 기반 분석을 넘어, 음성의 물리적 특성(피치, 음량, 말 속도, 스트레스)과 심리 분석을 통합하여 법적 증거로서의 신뢰도를 높인다.

### 1.2 핵심 목표

1. **음성 특성 분석 엔진**: 데시벨, 피치, 말 속도, 음성 스트레스 분석
2. **Speech Emotion Recognition (SER)**: 딥러닝 기반 음성 감정 인식
3. **시계열 시각화**: 가스라이팅 진행 과정 타임라인 시각화
4. **종합 포렌식 리포트**: 법적 증거 요건을 충족하는 전문 보고서

### 1.3 범위

- 기존 WhisperX 파이프라인(SPEC-WHISPERX-001) 결과물 활용
- 음성 특성 추출 및 분석 레이어 추가
- 심리 분석 고도화 (텍스트 + 음성 특성 통합)
- 시각화 대시보드 및 리포트 생성기

---

## 2. 학술 및 연구 기반 (Academic & Research Foundation)

### 2.1 음성 포렌식 학술 연구 참조

#### 2.1.1 음성 스트레스 분석 (Voice Stress Analysis)

**주요 연구:**
- Lippold, G. (1971). "Psychological Stress Evaluation (PSE)"
- Hopkins, C. S., et al. (2005). "Evaluation of Voice Stress Analysis Technology"
- Damphousse, K. R., et al. (2007). "Assessing the Validity of Voice Stress Analysis Tools"

**핵심 원리:**
- 스트레스 상황에서 발성 근육(후두 근육)의 미세 진동(micro-tremor) 변화 감지
- 기본 주파수(F0)의 변동성(pitch variability) 증가
- 포먼트 주파수(F1, F2) 변화 패턴 분석

**기술적 구현:**
- Fundamental Frequency (F0) 추출: 100-300Hz 대역
- Jitter (주파수 미세 변동): 정상 < 1%, 스트레스 > 2%
- Shimmer (진폭 미세 변동): 정상 < 3%, 스트레스 > 5%
- Harmonic-to-Noise Ratio (HNR): 스트레스 시 감소

#### 2.1.2 Speech Emotion Recognition (SER)

**주요 연구:**
- Schuller, B. W. (2018). "Speech Emotion Recognition: Two Decades in a Nutshell"
- Trigeorgis, G., et al. (2016). "Adieu Features? End-to-End Speech Emotion Recognition"
- Zhao, J., et al. (2019). "Speech Emotion Recognition Using Deep 1D & 2D CNN LSTM"

**데이터셋:**
- RAVDESS (Ryerson Audio-Visual Database): 영어 감정 음성
- SAVEE (Surrey Audio-Visual Expressed Emotion): 영국 영어
- EMO-DB (Berlin Emotional Speech Database): 독일어
- AIHUB 한국어 감정 음성 데이터셋: 한국어 특화

**모델 아키텍처:**
- Wav2Vec 2.0 기반 특성 추출 + Transformer 분류기
- CNN-LSTM 하이브리드 모델
- HuBERT (Hidden-Unit BERT) 기반 자기지도학습 모델

#### 2.1.3 거짓말 탐지 보조 지표 (Deception Detection Indicators)

**주요 연구:**
- Vrij, A. (2008). "Detecting Lies and Deceit: Pitfalls and Opportunities"
- DePaulo, B. M., et al. (2003). "Cues to Deception"
- Newman, M. L., et al. (2003). "Lying Words: Predicting Deception from Linguistic Styles"

**음성적 지표:**
- 말 속도 변화: 거짓 진술 시 일반적으로 느려짐
- 발화 지연(Hesitation): 응답 전 지연 시간 증가
- 필러(Filler words) 빈도: "음", "어", "그러니까" 증가
- 피치 상승: 스트레스로 인한 성대 긴장

**언어적 지표:**
- 1인칭 대명사 사용 감소 ("나"보다 수동태 선호)
- 구체적 세부사항 부족
- 부정적 감정 단어 증가
- 배타적 단어 사용 ("~는 아니고", "~만")

### 2.2 범죄 프로파일링 기법 참조

#### 2.2.1 FBI 행동분석단 (Behavioral Analysis Unit) 기법

**언어 분석 영역:**
- 위협 평가(Threat Assessment): 위협의 구체성, 실행 가능성
- 언어 패턴 분석: 반복적 표현, 에스컬레이션 패턴
- 심리적 압박 지표: 통제 시도, 고립화 전략

**가스라이팅 프로파일링:**
- 점진적 현실 왜곡(Progressive Reality Distortion)
- 자존감 약화 전략(Self-Esteem Degradation)
- 의존성 형성(Dependency Creation)
- 고립화(Isolation Tactics)

#### 2.2.2 한국 경찰청 과학수사 기법

**음성 증거 분석 표준:**
- 화자 식별(Speaker Identification): 성문 분석
- 오디오 인증(Audio Authentication): 편집 여부 검증
- 음성 향상(Audio Enhancement): 배경 잡음 제거

**법적 증거 요건:**
- 원본 무결성 증명 (Hash 검증)
- 연속성 입증 (Chain of Custody)
- 분석 방법론 문서화

### 2.3 오픈소스 및 기술 커뮤니티 참조

#### 2.3.1 음성 분석 라이브러리

| 라이브러리 | 용도 | 라이선스 |
|-----------|------|----------|
| **librosa** | 음성 특성 추출 (MFCC, chroma, spectral) | ISC |
| **pyAudioAnalysis** | 음성 분류 및 세그멘테이션 | Apache 2.0 |
| **OpenSMILE** | 음향 특성 추출 (eGeMAPS) | 학술/상업 |
| **Parselmouth (Praat)** | 음성학 분석 (피치, 포먼트) | GPL |
| **PEFT/LoRA** | 효율적 모델 미세조정 | Apache 2.0 |

#### 2.3.2 Speech Emotion Recognition 모델

| 모델 | 특징 | Hugging Face |
|------|------|--------------|
| **wav2vec2-large-emotion** | 영어 SER, 7 감정 | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition |
| **HuBERT-large-emotion** | 다국어 감정 인식 | facebook/hubert-large-ls960-ft |
| **speechbrain-emotion** | End-to-end SER | speechbrain/emotion-recognition-wav2vec2-IEMOCAP |
| **KoBERT-emotion** | 한국어 텍스트 감정 | skt/kobert-base-v1 |

#### 2.3.3 포렌식 커뮤니티 도구

| 도구 | 용도 |
|------|------|
| **Audacity** | 오디오 파형 분석 및 편집 검증 |
| **Praat** | 음성학적 분석 (피치, 강도, 스펙트로그램) |
| **FFmpeg** | 오디오 포맷 변환 및 메타데이터 추출 |
| **ExifTool** | 오디오 메타데이터 분석 |

---

## 3. 환경 (Environment)

### 3.1 기술 스택

#### 3.1.1 음성 분석 엔진

- **librosa 0.10.2+**: 음향 특성 추출 (MFCC, chroma, spectral contrast)
- **parselmouth 0.4.3+**: Praat 기반 음성학 분석 (F0, formants, jitter, shimmer)
- **pyAudioAnalysis 0.3.14+**: 음성 분류 및 세그멘테이션

#### 3.1.2 Speech Emotion Recognition

- **speechbrain 1.0+**: End-to-end SER 파이프라인
- **transformers 4.40+**: Wav2Vec2/HuBERT 모델
- **torch 2.5+**: GPU 가속 추론

#### 3.1.3 시각화 및 리포팅

- **plotly 5.18+**: 인터랙티브 시각화
- **matplotlib 3.8+**: 정적 차트
- **pandas 2.2+**: 시계열 데이터 처리
- **reportlab 4.1+**: PDF 리포트 생성

#### 3.1.4 대시보드 (선택적)

- **streamlit 1.40+**: 분석 대시보드
- **gradio 4.44+**: 인터랙티브 데모 (대안)

### 3.2 시스템 요구사항

- Python 3.11+
- CUDA 12.1+ (GPU 가속)
- 최소 24GB RAM (SER 모델 로딩)
- 최소 8GB VRAM (GPU 모델 추론)

---

## 4. 가정 (Assumptions)

### 4.1 기술적 가정

1. SPEC-WHISPERX-001의 WhisperX 파이프라인이 완료되어 word-level 타임스탬프가 제공된다
2. 183개 녹취 파일의 STT 및 화자 분리 결과가 저장되어 있다
3. GPU 환경에서 SER 모델 추론이 가능하다 (NVIDIA GB10)
4. 음성 파일의 품질이 분석에 적합하다 (SNR > 15dB)

### 4.2 비즈니스 가정

1. 분석 대상은 형사고소에 필요한 증거 자료이다
2. 신기연, 신동식 발화자의 가스라이팅 패턴을 시간순으로 증명해야 한다
3. 6개월간의 심리적 압박을 입증하는 것이 핵심 목표이다
4. 분석 결과는 법률 전문가 검토를 위한 보조 자료로 활용된다

### 4.3 분석적 가정

1. 음성 스트레스 분석은 보조 지표로 활용되며, 단독으로 법적 증거력을 갖지 않는다
2. SER 모델의 정확도는 참고 자료 수준이며, 전문가 해석이 필요하다
3. 가스라이팅 패턴 분석은 시간적 진행 과정의 가시화에 초점을 둔다

---

## 5. 요구사항 (Requirements)

### 5.1 Ubiquitous Requirements (U) - 전역 요구사항

#### U1: 음성 특성 데이터 무결성
**요구사항**: 시스템은 **항상** 추출된 음성 특성 데이터를 원본 오디오 파일과 연결하여 추적 가능하게 저장해야 한다.

**세부사항**:
- 원본 파일 해시(SHA-256)와 연결된 분석 결과
- 분석 시점, 사용 알고리즘 버전 기록
- 결과 재현 가능성 보장

**WHY**: 법적 증거로서 분석 결과의 신뢰성을 입증하려면 원본과의 연결성이 필수적이다.

**IMPACT**: 추적성 미비 시 분석 결과의 법적 증거 능력 상실.

---

#### U2: 분석 결과 일관성
**요구사항**: 시스템은 **항상** 동일한 입력에 대해 동일한 분석 결과를 생성해야 한다 (결정론적 분석).

**세부사항**:
- 난수 시드 고정
- 모델 버전 명시
- 재현 가능한 환경 설정

**WHY**: 법적 검증 과정에서 분석 재현이 요구될 수 있다.

**IMPACT**: 비결정론적 결과는 증거 신뢰도를 저하시킨다.

---

#### U3: 화자별 분석 격리
**요구사항**: 시스템은 **항상** 각 화자의 음성 특성과 심리 분석을 독립적으로 수행해야 한다.

**세부사항**:
- 화자별 음성 프로파일 분리
- 화자 간 비교 분석 지원
- 가해자/피해자 구분 가능

**WHY**: 가스라이팅 분석에서 가해자와 피해자의 발화 패턴 구분이 핵심이다.

**IMPACT**: 화자 혼동 시 가스라이팅 역학 분석 불가능.

---

### 5.2 Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: 음성 파일 분석 트리거
**요구사항**: **WHEN** WhisperX 파이프라인 처리가 완료되면 **THEN** 시스템은 음성 특성 분석을 자동으로 시작해야 한다.

**세부사항**:
- WhisperX 결과(segments, speakers)를 입력으로 수신
- 각 세그먼트에 대해 음향 특성 추출
- 분석 진행률 실시간 업데이트

**WHY**: 파이프라인 연속성 보장으로 수동 개입 최소화.

**IMPACT**: 수동 트리거 시 처리 지연 및 오류 가능성 증가.

---

#### E2: 감정 격화 구간 감지
**요구사항**: **WHEN** 음량이 기준치 대비 150% 이상 증가하거나 피치가 급격히 변화하면 **THEN** 시스템은 해당 구간을 "감정 격화 구간"으로 자동 태깅해야 한다.

**세부사항**:
- 음량(RMS): 기준치 대비 1.5배 이상
- 피치(F0): 50Hz 이상 급격한 변화
- 말 속도: 기준 대비 30% 이상 변화

**WHY**: 감정적 고조 구간은 가스라이팅 분석에서 핵심 증거가 될 수 있다.

**IMPACT**: 중요 구간 누락 시 핵심 증거 발굴 실패.

---

#### E3: 가스라이팅 패턴 연속 감지
**요구사항**: **WHEN** 동일 화자에서 가스라이팅 패턴이 3회 이상 연속 감지되면 **THEN** 시스템은 "고위험 가스라이팅 구간"으로 분류하고 알림을 생성해야 한다.

**세부사항**:
- 연속 발화 기준: 5분 이내
- 패턴 유형: 부정, 전가, 축소, 혼란
- 알림 생성 및 우선순위 표시

**WHY**: 집중적인 심리 조작 구간 식별로 가스라이팅 의도 입증 강화.

**IMPACT**: 패턴 연속성 미감지 시 조직적 가스라이팅 증명 약화.

---

#### E4: 시계열 이상 감지
**요구사항**: **WHEN** 피해자의 감정 상태가 시간에 따라 지속적으로 악화되는 패턴이 감지되면 **THEN** 시스템은 "심리적 피해 진행" 지표를 생성해야 한다.

**세부사항**:
- 분석 기간: 녹취 전체 기간 (6개월)
- 지표: 부정 감정 빈도, 스트레스 지수 증가 추세
- 통계적 유의성 검증

**WHY**: 장기간에 걸친 심리적 압박의 누적 효과를 입증해야 한다.

**IMPACT**: 시계열 분석 부재 시 "일회성 갈등"으로 축소될 위험.

---

### 5.3 State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: GPU 메모리 기반 모델 전략
**요구사항**: **IF** GPU 메모리 사용률이 80%를 초과하면 **THEN** 시스템은 SER 모델을 배치 분할 처리 모드로 전환해야 한다.

**세부사항**:
- 배치 크기 동적 조정 (32 -> 16 -> 8)
- 모델 언로드 후 재로드 최소화
- 메모리 부족 시 CPU 폴백

**WHY**: OOM 방지 및 안정적인 대량 파일 처리.

**IMPACT**: 메모리 관리 미비 시 처리 중단 및 데이터 손실.

---

#### S2: 분석 정확도 기반 검토 플래그
**요구사항**: **IF** SER 모델의 감정 분류 신뢰도가 0.6 미만이면 **THEN** 해당 세그먼트에 "전문가 검토 필요" 플래그를 설정해야 한다.

**세부사항**:
- 신뢰도 임계값: 0.6
- 플래그 유형: 자동 분석 신뢰, 검토 권장, 검토 필수
- 불확실 구간 시각적 표시

**WHY**: 자동 분석의 한계를 명확히 표시하여 오해석 방지.

**IMPACT**: 저신뢰 결과의 무비판적 수용 시 분석 신뢰도 저하.

---

#### S3: 파일 처리 상태 관리
**요구사항**: **IF** 파일이 "분석 대기" 상태이면 **THEN** 대기열 순서에 따라 처리를 진행하고, "분석 중" 상태의 파일은 중복 처리를 방지해야 한다.

**세부사항**:
- 상태: 대기, 분석 중, 완료, 오류
- 대기열 우선순위 지원
- 중복 처리 방지 락(lock)

**WHY**: 대량 파일 처리 시 리소스 효율성과 일관성 보장.

**IMPACT**: 상태 관리 미비 시 중복 처리 및 리소스 낭비.

---

### 5.4 Feature Requirements (F) - 기능 요구사항

#### F1: 음량 분석 (Volume Analysis)
**요구사항**: 시스템은 각 발화 세그먼트의 음량 특성을 분석해야 한다.

**세부사항**:
- RMS (Root Mean Square) amplitude 계산
- Peak amplitude 감지
- 음량 변화율 (dB/초)
- 화자별 평균 음량 비교

**출력**:
```python
{
    "rms_db": -20.5,
    "peak_db": -12.3,
    "dynamic_range_db": 8.2,
    "volume_change_rate": 2.1  # dB/s
}
```

**WHY**: 음량은 감정 강도와 지배력의 직접적 지표이다.

**IMPACT**: 음량 분석 없이는 "고압적 발언" 객관화 불가능.

---

#### F2: 피치 분석 (Pitch/F0 Analysis)
**요구사항**: 시스템은 발화의 기본 주파수(F0)와 변동성을 분석해야 한다.

**세부사항**:
- F0 평균, 최소, 최대, 표준편차
- Jitter (주파수 미세 변동)
- Pitch contour (피치 곡선)
- 성별/연령 보정

**출력**:
```python
{
    "f0_mean_hz": 150.2,
    "f0_std_hz": 25.3,
    "f0_min_hz": 100.5,
    "f0_max_hz": 220.1,
    "jitter_percent": 1.2,
    "pitch_range_semitones": 12.5
}
```

**WHY**: 피치 변동은 감정 상태와 스트레스의 핵심 지표이다.

**IMPACT**: 피치 분석 없이는 음성 기반 감정 인식 신뢰도 저하.

---

#### F3: 말 속도 분석 (Speech Rate Analysis)
**요구사항**: 시스템은 발화 속도와 변화 패턴을 분석해야 한다.

**세부사항**:
- WPM (Words Per Minute)
- Syllables Per Second
- 발화 지속 시간 vs 무음 구간 비율
- 말더듬/반복 감지

**출력**:
```python
{
    "wpm": 145.5,
    "syllables_per_second": 4.2,
    "speech_ratio": 0.75,  # 발화 시간 / 전체 시간
    "pause_count": 3,
    "avg_pause_duration_ms": 450
}
```

**WHY**: 말 속도 변화는 스트레스, 망설임, 거짓 진술의 보조 지표이다.

**IMPACT**: 속도 분석 없이는 "발화 망설임" 패턴 감지 불가.

---

#### F4: 음성 스트레스 분석 (Voice Stress Analysis)
**요구사항**: 시스템은 음성 기반 스트레스 지표를 계산해야 한다.

**세부사항**:
- Shimmer (진폭 미세 변동)
- HNR (Harmonic-to-Noise Ratio)
- Formant stability (F1, F2)
- 종합 스트레스 지수 (0-100)

**출력**:
```python
{
    "shimmer_percent": 3.5,
    "hnr_db": 18.2,
    "f1_stability": 0.85,
    "f2_stability": 0.82,
    "stress_index": 65  # 0-100 (높을수록 스트레스)
}
```

**WHY**: 학술 연구에 기반한 스트레스 지표는 심리적 압박 증명에 활용된다.

**IMPACT**: 주관적 판단 대신 객관적 수치 제공으로 증거력 강화.

---

#### F5: Speech Emotion Recognition (SER)
**요구사항**: 시스템은 딥러닝 모델을 사용하여 음성 기반 감정을 인식해야 한다.

**세부사항**:
- 감정 분류: 기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 중립
- 신뢰도 점수 제공
- 텍스트 감정과 음성 감정 교차 검증
- 화자별 감정 프로파일

**출력**:
```python
{
    "predicted_emotion": "anger",
    "confidence": 0.85,
    "emotion_distribution": {
        "anger": 0.85,
        "fear": 0.08,
        "neutral": 0.05,
        "sadness": 0.02
    },
    "text_emotion_match": True
}
```

**WHY**: 텍스트 분석만으로 파악 불가능한 음성적 감정 특성 포착.

**IMPACT**: 텍스트-음성 감정 불일치 감지로 진정성 평가 가능.

---

#### F6: 가스라이팅 시계열 분석
**요구사항**: 시스템은 6개월간의 녹취 데이터에서 가스라이팅 패턴의 시간적 진행을 분석해야 한다.

**세부사항**:
- 파일별 날짜/시간 순서 정렬
- 패턴 유형별 빈도 추이
- 강도 에스컬레이션 감지
- 피해자 반응 변화 추적

**분석 항목**:
- 부정(Denial) 빈도 변화
- 전가(Blame-shifting) 강도 추이
- 축소(Minimizing) 패턴 변화
- 혼란(Confusion) 유발 빈도
- 피해자 감정 상태 변화

**WHY**: 가스라이팅은 일회성이 아닌 지속적 패턴으로 증명해야 한다.

**IMPACT**: 시계열 분석 없이는 "장기간 심리 조작" 입증 불가.

---

#### F7: 심리적 압박 강도 측정
**요구사항**: 시스템은 종합적인 심리적 압박 강도 지수를 산출해야 한다.

**세부사항**:
- 가스라이팅 패턴 빈도 + 강도
- 부정 감정 발화 빈도
- 음성 스트레스 지수
- 발화 비율 (가해자 vs 피해자)

**출력**:
```python
{
    "pressure_index": 78,  # 0-100
    "components": {
        "gaslighting_score": 0.85,
        "negative_emotion_ratio": 0.65,
        "voice_stress_avg": 72,
        "dominance_ratio": 0.7  # 가해자 발화 비율
    },
    "risk_level": "매우 높음"
}
```

**WHY**: 개별 지표를 종합하여 전체적인 심리 압박 상황 평가.

**IMPACT**: 복합적 심리 조작의 총체적 영향 입증.

---

#### F8: 시각화 대시보드
**요구사항**: 시스템은 분석 결과를 시각화하는 인터랙티브 대시보드를 제공해야 한다.

**세부사항**:
1. **시계열 타임라인**: 가스라이팅 패턴 진행 과정
2. **감정 변화 그래프**: 화자별 감정 추이
3. **음량/피치 히트맵**: 시간대별 음성 특성
4. **화자 프로파일**: 화자별 종합 분석
5. **위험 구간 하이라이트**: 핵심 증거 구간 표시

**기술**:
- Plotly 인터랙티브 차트
- Streamlit 대시보드 프레임워크
- 필터링 및 드릴다운 지원

**WHY**: 복잡한 분석 결과를 직관적으로 이해할 수 있어야 한다.

**IMPACT**: 시각화 없이는 패턴 인식 및 증거 설명 어려움.

---

#### F9: 법적 증거 리포트 생성
**요구사항**: 시스템은 법적 제출용 종합 리포트를 PDF 형식으로 생성해야 한다.

**세부사항**:
1. **요약 페이지**: 핵심 발견 사항
2. **분석 방법론**: 사용 알고리즘 및 학술 참조
3. **시계열 분석**: 가스라이팅 진행 과정
4. **증거 구간 목록**: 타임스탬프 및 내용
5. **화자별 프로파일**: 심리 분석 결과
6. **통계 요약**: 수치화된 분석 결과
7. **부록**: 원본 데이터 참조

**WHY**: 법정 제출을 위한 표준화된 형식 필요.

**IMPACT**: 비표준 형식은 증거 채택 거부 위험.

---

### 5.5 Optional Requirements (O) - 선택적 요구사항

#### O1: 실시간 분석 스트리밍
**요구사항**: **가능하면** 실시간 녹취 중 분석 결과를 스트리밍 제공할 수 있다.

**WHY**: 향후 실시간 모니터링 기능 확장 가능성.

**IMPACT**: 미구현 시 배치 처리만 가능하나 핵심 기능에 영향 없음.

---

#### O2: 다화자 비교 분석
**요구사항**: **가능하면** 3인 이상 화자 간 역학 관계 분석을 지원할 수 있다.

**WHY**: 조직적 가스라이팅 패턴 분석에 유용.

**IMPACT**: 미구현 시 2인 대화 분석에 집중.

---

#### O3: 외부 전문가 협업 기능
**요구사항**: **가능하면** 분석 결과를 외부 전문가와 공유하고 코멘트를 수집할 수 있다.

**WHY**: 법률/심리 전문가 협업 강화.

**IMPACT**: 미구현 시 오프라인 협업.

---

### 5.6 Unwanted Requirements (N) - 금지 요구사항

#### N1: 원본 오디오 변조 금지
**요구사항**: 시스템은 **절대** 원본 오디오 파일을 수정, 변조, 삭제하지 않아야 한다.

**세부사항**:
- 분석용 임시 파일만 생성
- 원본은 읽기 전용 접근
- 모든 변환은 복사본에서 수행

**WHY**: 원본 무결성은 법적 증거력의 기본 요건.

**IMPACT**: 원본 변조 시 전체 분석 결과의 법적 효력 상실.

---

#### N2: 과학적 근거 없는 주장 금지
**요구사항**: 시스템은 **절대** 학술적 근거가 없는 분석 결과를 확정적으로 제시하지 않아야 한다.

**세부사항**:
- 모든 분석 결과에 신뢰도/한계 명시
- "거짓말 탐지" 대신 "스트레스 지표" 표현 사용
- 분석은 "보조 자료"로 명시

**WHY**: 과학적 검증이 불완전한 기술의 과대 해석 방지.

**IMPACT**: 과장된 주장은 법정에서 신뢰도 저하 및 기각 위험.

---

#### N3: 개인정보 무단 노출 금지
**요구사항**: 시스템은 **절대** 분석 대상자의 개인정보를 암호화 없이 저장하거나 로깅하지 않아야 한다.

**세부사항**:
- 화자 ID는 익명화 처리 (SPEAKER_00)
- 실명은 별도 매핑 테이블에 암호화 저장
- 로그에 민감 정보 마스킹

**WHY**: 개인정보보호법 준수 및 증거 보호.

**IMPACT**: 개인정보 노출 시 법적 제재 및 증거 무효화 위험.

---

## 6. 명세 (Specifications)

### 6.1 데이터 모델

```python
# 음성 특성 분석 결과
class AudioFeatureAnalysis:
    segment_id: UUID
    audio_file_id: UUID
    start_time: float
    end_time: float

    # 음량 분석
    volume: VolumeFeatures
    # 피치 분석
    pitch: PitchFeatures
    # 말 속도 분석
    speech_rate: SpeechRateFeatures
    # 스트레스 분석
    stress: StressFeatures
    # SER 결과
    emotion: EmotionRecognition

    created_at: datetime
    analysis_version: str

class VolumeFeatures:
    rms_db: float
    peak_db: float
    dynamic_range_db: float
    volume_change_rate: float  # dB/s

class PitchFeatures:
    f0_mean_hz: float
    f0_std_hz: float
    f0_min_hz: float
    f0_max_hz: float
    jitter_percent: float
    pitch_range_semitones: float

class SpeechRateFeatures:
    wpm: float
    syllables_per_second: float
    speech_ratio: float
    pause_count: int
    avg_pause_duration_ms: float

class StressFeatures:
    shimmer_percent: float
    hnr_db: float
    f1_stability: float
    f2_stability: float
    stress_index: int  # 0-100

class EmotionRecognition:
    predicted_emotion: str
    confidence: float
    emotion_distribution: Dict[str, float]
    text_emotion_match: bool

# 가스라이팅 시계열 분석
class GaslightingTimeline:
    file_id: UUID
    recording_date: datetime
    patterns: List[GaslightingPattern]
    pattern_frequency: Dict[str, int]
    intensity_score: float
    cumulative_pressure: float

# 종합 포렌식 리포트
class ForensicReport:
    report_id: UUID
    generated_at: datetime

    # 요약
    executive_summary: str
    key_findings: List[str]
    risk_assessment: str

    # 분석 결과
    timeline_analysis: TimelineAnalysis
    speaker_profiles: List[SpeakerProfile]
    evidence_highlights: List[EvidenceHighlight]

    # 통계
    statistics: AnalysisStatistics

    # 메타데이터
    analysis_methodology: str
    limitations: List[str]
    references: List[str]
```

### 6.2 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/forensic/analyze/{audio_id}` | POST | 음성 특성 분석 시작 |
| `/api/v1/forensic/features/{audio_id}` | GET | 음성 특성 결과 조회 |
| `/api/v1/forensic/emotion/{audio_id}` | GET | SER 분석 결과 조회 |
| `/api/v1/forensic/stress/{audio_id}` | GET | 스트레스 분석 결과 조회 |
| `/api/v1/forensic/timeline` | GET | 가스라이팅 시계열 분석 |
| `/api/v1/forensic/pressure-index` | GET | 심리 압박 지수 조회 |
| `/api/v1/forensic/report` | POST | 종합 리포트 생성 |
| `/api/v1/forensic/report/{report_id}` | GET | 리포트 다운로드 |
| `/api/v1/forensic/dashboard` | GET | 대시보드 데이터 조회 |

### 6.3 시각화 명세

#### 6.3.1 가스라이팅 타임라인 (Timeline Visualization)

**구성 요소:**
- X축: 시간 (녹취 날짜)
- Y축: 패턴 강도 (0-100)
- 색상 코드: 패턴 유형별 구분
  - 부정(Denial): 빨강
  - 전가(Blame-shifting): 주황
  - 축소(Minimizing): 노랑
  - 혼란(Confusion): 보라
- 마커: 개별 패턴 발생 지점
- 추세선: 강도 에스컬레이션 표시

#### 6.3.2 감정 변화 그래프 (Emotion Flow Chart)

**구성 요소:**
- Sankey Diagram: 감정 전이 패턴
- Area Chart: 시간별 감정 분포
- 화자별 분리 표시
- 인터랙티브 필터링

#### 6.3.3 음량/피치 히트맵 (Audio Feature Heatmap)

**구성 요소:**
- X축: 시간 (세그먼트)
- Y축: 음성 특성 유형
- 색상: 강도 (낮음-파랑, 높음-빨강)
- 화자별 분리 뷰

#### 6.3.4 화자 프로파일 (Speaker Profile Card)

**구성 요소:**
- 레이더 차트: 감정 분포
- 바 차트: 가스라이팅 패턴 빈도
- 통계 요약: 평균 음량, 피치, 발화 비율
- 위험도 게이지

---

## 7. 추적성 (Traceability)

### 7.1 관련 문서

- `SPEC-VOICE-001`: 음성 녹취 기반 증거 분석 시스템 (기반)
- `SPEC-WHISPERX-001`: WhisperX 통합 파이프라인 (입력)
- `SPEC-E2ETEST-001`: E2E 통합 테스트 (검증)
- `plan.md`: 구현 계획 및 마일스톤
- `acceptance.md`: 인수 기준 및 테스트 시나리오

### 7.2 태그

```
[SPEC-FORENSIC-001]
[음성포렌식] [SER] [스트레스분석] [가스라이팅]
[시각화] [법적증거] [심리분석] [범죄프로파일링]
```

### 7.3 구현 파일 (예정)

```
src/voice_man/
├── services/
│   ├── forensic/
│   │   ├── __init__.py
│   │   ├── audio_feature_service.py      # F1, F2, F3 - 음량/피치/속도
│   │   ├── stress_analysis_service.py    # F4 - 스트레스 분석
│   │   ├── ser_service.py                # F5 - Speech Emotion Recognition
│   │   ├── timeline_analysis_service.py  # F6 - 가스라이팅 시계열
│   │   ├── pressure_index_service.py     # F7 - 심리 압박 지수
│   │   └── forensic_report_service.py    # F9 - 리포트 생성
│   └── visualization/
│       ├── __init__.py
│       ├── timeline_visualizer.py        # F8 - 타임라인
│       ├── emotion_chart.py              # F8 - 감정 차트
│       ├── heatmap_generator.py          # F8 - 히트맵
│       └── dashboard_service.py          # F8 - 대시보드
├── models/
│   └── forensic/
│       ├── audio_features.py
│       ├── stress_analysis.py
│       ├── emotion_recognition.py
│       └── forensic_report.py
└── config/
    └── forensic_config.py
```

---

## 8. 참조 (References)

### 8.1 학술 논문

1. Schuller, B. W. (2018). "Speech Emotion Recognition: Two Decades in a Nutshell"
2. Trigeorgis, G., et al. (2016). "Adieu Features? End-to-End Speech Emotion Recognition"
3. Zhao, J., et al. (2019). "Speech Emotion Recognition Using Deep 1D & 2D CNN LSTM"
4. Vrij, A. (2008). "Detecting Lies and Deceit: Pitfalls and Opportunities"
5. DePaulo, B. M., et al. (2003). "Cues to Deception"
6. Hopkins, C. S., et al. (2005). "Evaluation of Voice Stress Analysis Technology"

### 8.2 기술 문서

- [librosa Documentation](https://librosa.org/doc/latest/)
- [Parselmouth (Praat)](https://parselmouth.readthedocs.io/)
- [SpeechBrain](https://speechbrain.github.io/)
- [Hugging Face Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)

### 8.3 법적 참조

- 형법 제283조 (협박죄)
- 형법 제350조 (공갈죄)
- 형법 제311조 (모욕죄)
- 개인정보보호법
- 경찰청 과학수사 음성 분석 가이드라인

---

**문서 끝**
