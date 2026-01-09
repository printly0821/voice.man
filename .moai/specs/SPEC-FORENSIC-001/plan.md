---
id: SPEC-FORENSIC-001
document_type: "implementation-plan"
version: "1.0.0"
created: "2026-01-09"
updated: "2026-01-09"
author: "지니"
status: "planned"
---

# SPEC-FORENSIC-001: 구현 계획서

## 개요

범죄 프로파일링 기반 음성 포렌식 분석 시스템의 구현 계획서입니다. 4개의 주요 Phase로 구성되며, 각 Phase는 독립적으로 테스트 가능한 단위로 설계되었습니다.

---

## Phase 1: 음성 특성 분석 엔진 (Audio Feature Analysis Engine)

### 목표
음량, 피치, 말 속도, 스트레스 지표를 추출하는 기반 엔진 구현

### 구현 항목

#### P1-1: 음량 분석 서비스 (Volume Analysis)
**파일**: `src/voice_man/services/forensic/audio_feature_service.py`

**구현 내용**:
- RMS (Root Mean Square) amplitude 계산
- Peak amplitude 감지
- 동적 범위(dynamic range) 계산
- 음량 변화율(dB/s) 추적
- 화자별 평균 음량 비교

**라이브러리**:
```python
librosa>=0.10.2
numpy>=1.26.0
```

**기술적 접근**:
```python
# 핵심 알고리즘
rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
```

---

#### P1-2: 피치 분석 서비스 (Pitch/F0 Analysis)
**파일**: `src/voice_man/services/forensic/audio_feature_service.py`

**구현 내용**:
- 기본 주파수(F0) 추출 (PYIN 알고리즘)
- F0 통계: 평균, 표준편차, 최소, 최대
- Jitter 계산 (주파수 미세 변동)
- 피치 범위 (반음 단위)
- 피치 contour 시각화 데이터

**라이브러리**:
```python
librosa>=0.10.2  # PYIN
parselmouth>=0.4.3  # Praat 기반 정밀 분석
```

**기술적 접근**:
```python
# librosa PYIN
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=75, fmax=600, sr=sr
)

# parselmouth (Praat) for precision
import parselmouth
sound = parselmouth.Sound(audio_path)
pitch = sound.to_pitch()
f0_values = pitch.selected_array['frequency']
```

---

#### P1-3: 말 속도 분석 서비스 (Speech Rate Analysis)
**파일**: `src/voice_man/services/forensic/audio_feature_service.py`

**구현 내용**:
- WPM (Words Per Minute) 계산
- Syllables Per Second 추정
- 발화/무음 비율 계산
- 휴지(pause) 감지 및 통계
- 말더듬/반복 패턴 감지

**기술적 접근**:
```python
# WhisperX 결과 활용
word_count = len(segments)
total_duration = end_time - start_time
wpm = (word_count / total_duration) * 60

# 무음 구간 감지
silent_intervals = librosa.effects.split(y, top_db=30)
```

---

#### P1-4: 스트레스 분석 서비스 (Voice Stress Analysis)
**파일**: `src/voice_man/services/forensic/stress_analysis_service.py`

**구현 내용**:
- Shimmer 계산 (진폭 미세 변동)
- HNR (Harmonic-to-Noise Ratio) 계산
- Formant stability (F1, F2) 분석
- 종합 스트레스 지수 산출 (0-100)

**라이브러리**:
```python
parselmouth>=0.4.3  # Praat 기반
```

**기술적 접근**:
```python
# parselmouth를 통한 Praat 기능 활용
sound = parselmouth.Sound(audio_path)
point_process = parselmouth.praat.call(
    sound, "To PointProcess (periodic, cc)", 75, 600
)
shimmer = parselmouth.praat.call(
    [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
)

# HNR 계산
harmonicity = sound.to_harmonicity()
hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

# 스트레스 지수 산출 (가중 평균)
stress_index = (
    jitter_weight * normalize(jitter) +
    shimmer_weight * normalize(shimmer) +
    hnr_weight * (1 - normalize(hnr)) +  # HNR은 낮을수록 스트레스
    f0_var_weight * normalize(f0_std)
) * 100
```

---

### Phase 1 산출물

| 파일 | 설명 |
|------|------|
| `audio_feature_service.py` | 음량/피치/속도 분석 통합 서비스 |
| `stress_analysis_service.py` | 스트레스 분석 서비스 |
| `models/forensic/audio_features.py` | 데이터 모델 |
| `tests/test_audio_features.py` | 단위 테스트 |

### Phase 1 검증 기준

- [ ] 183개 파일에 대해 음향 특성 추출 성공
- [ ] 처리 시간: 파일당 평균 2초 이내
- [ ] 테스트 커버리지: 85% 이상
- [ ] 결과 재현성: 동일 입력 동일 출력

---

## Phase 2: 심리 분석 고도화 (Advanced Psychological Analysis)

### 목표
Speech Emotion Recognition (SER) 통합 및 텍스트-음성 감정 교차 검증

### 구현 항목

#### P2-1: Speech Emotion Recognition 서비스
**파일**: `src/voice_man/services/forensic/ser_service.py`

**구현 내용**:
- Wav2Vec2 기반 SER 모델 통합
- 7가지 감정 분류 (기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 중립)
- 신뢰도 점수 및 분포 출력
- GPU 가속 배치 추론
- 메모리 최적화 (청크 처리)

**모델 선택**:
```python
# 주요 후보 모델
models = {
    "english": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "korean": "kresnik/wav2vec2-large-xlsr-korean",  # 한국어 특화
    "multilingual": "facebook/wav2vec2-large-xlsr-53"
}
```

**기술적 접근**:
```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

class SERService:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def predict(self, audio: np.ndarray, sr: int = 16000) -> EmotionRecognition:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        predicted_id = probs.argmax(-1).item()
        confidence = probs[0, predicted_id].item()

        return EmotionRecognition(
            predicted_emotion=self.id2label[predicted_id],
            confidence=confidence,
            emotion_distribution=dict(zip(self.labels, probs[0].tolist()))
        )
```

---

#### P2-2: 텍스트-음성 감정 교차 검증
**파일**: `src/voice_man/services/forensic/ser_service.py`

**구현 내용**:
- 기존 EmotionAnalysisService (텍스트) 결과와 SER 결과 비교
- 불일치 감지 및 플래그 설정
- 종합 감정 판단 로직

**기술적 접근**:
```python
def cross_validate_emotion(
    text_emotion: EmotionAnalysis,
    voice_emotion: EmotionRecognition
) -> CrossValidationResult:
    match = text_emotion.primary_emotion.value == voice_emotion.predicted_emotion

    if not match:
        # 불일치 시 추가 분석
        conflict_type = determine_conflict_type(text_emotion, voice_emotion)

    return CrossValidationResult(
        text_emotion=text_emotion,
        voice_emotion=voice_emotion,
        is_match=match,
        conflict_type=conflict_type if not match else None,
        combined_confidence=(text_emotion.confidence + voice_emotion.confidence) / 2
    )
```

---

#### P2-3: 가스라이팅 시계열 분석 고도화
**파일**: `src/voice_man/services/forensic/timeline_analysis_service.py`

**구현 내용**:
- 파일별 날짜 순서 정렬
- 패턴 빈도 및 강도 추이 분석
- 에스컬레이션 감지 알고리즘
- 피해자 감정 상태 변화 추적
- 통계적 유의성 검증

**기술적 접근**:
```python
class TimelineAnalysisService:
    def analyze_escalation(
        self, timeline_data: List[GaslightingTimeline]
    ) -> EscalationAnalysis:
        # 시간순 정렬
        sorted_data = sorted(timeline_data, key=lambda x: x.recording_date)

        # 패턴 강도 추세 분석 (선형 회귀)
        dates = [(d.recording_date - sorted_data[0].recording_date).days
                 for d in sorted_data]
        intensities = [d.intensity_score for d in sorted_data]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            dates, intensities
        )

        return EscalationAnalysis(
            trend_slope=slope,
            is_escalating=slope > 0.01,  # 양의 기울기
            statistical_significance=p_value < 0.05,
            r_squared=r_value ** 2
        )
```

---

#### P2-4: 심리적 압박 지수 서비스
**파일**: `src/voice_man/services/forensic/pressure_index_service.py`

**구현 내용**:
- 가스라이팅 패턴 점수
- 부정 감정 비율
- 음성 스트레스 평균
- 발화 지배력 비율 (가해자 vs 피해자)
- 종합 압박 지수 산출 (0-100)

**기술적 접근**:
```python
class PressureIndexService:
    def calculate_pressure_index(
        self,
        gaslighting_score: float,
        negative_emotion_ratio: float,
        voice_stress_avg: float,
        dominance_ratio: float
    ) -> PressureIndex:
        # 가중 평균 계산
        weights = {
            "gaslighting": 0.35,
            "negative_emotion": 0.25,
            "voice_stress": 0.20,
            "dominance": 0.20
        }

        pressure_index = (
            weights["gaslighting"] * gaslighting_score * 100 +
            weights["negative_emotion"] * negative_emotion_ratio * 100 +
            weights["voice_stress"] * voice_stress_avg +
            weights["dominance"] * dominance_ratio * 100
        )

        risk_level = self._determine_risk_level(pressure_index)

        return PressureIndex(
            value=pressure_index,
            risk_level=risk_level,
            components={...}
        )
```

---

### Phase 2 산출물

| 파일 | 설명 |
|------|------|
| `ser_service.py` | Speech Emotion Recognition 서비스 |
| `timeline_analysis_service.py` | 가스라이팅 시계열 분석 |
| `pressure_index_service.py` | 심리 압박 지수 계산 |
| `models/forensic/emotion_recognition.py` | SER 데이터 모델 |
| `tests/test_ser_service.py` | SER 테스트 |
| `tests/test_timeline_analysis.py` | 시계열 분석 테스트 |

### Phase 2 검증 기준

- [ ] SER 모델 로드 및 추론 성공
- [ ] 감정 분류 정확도: 영어 데이터셋 기준 70% 이상
- [ ] GPU 메모리 사용: 8GB 이내
- [ ] 텍스트-음성 감정 교차 검증 동작
- [ ] 시계열 에스컬레이션 감지 동작

---

## Phase 3: 시각화 대시보드 (Visualization Dashboard)

### 목표
분석 결과를 직관적으로 이해할 수 있는 인터랙티브 시각화 구현

### 구현 항목

#### P3-1: 가스라이팅 타임라인 시각화
**파일**: `src/voice_man/services/visualization/timeline_visualizer.py`

**구현 내용**:
- X축: 녹취 날짜
- Y축: 패턴 강도
- 패턴 유형별 색상 코드
- 마커: 개별 이벤트
- 추세선: 에스컬레이션 표시
- 인터랙티브 줌/필터

**기술적 접근**:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TimelineVisualizer:
    PATTERN_COLORS = {
        "denial": "#FF4444",       # 빨강
        "blame_shifting": "#FF8C00", # 주황
        "minimizing": "#FFD700",    # 노랑
        "confusion": "#9932CC"      # 보라
    }

    def create_timeline(self, data: List[TimelineData]) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        for pattern_type, color in self.PATTERN_COLORS.items():
            pattern_data = [d for d in data if d.pattern_type == pattern_type]
            fig.add_trace(
                go.Scatter(
                    x=[d.date for d in pattern_data],
                    y=[d.intensity for d in pattern_data],
                    mode='markers+lines',
                    name=pattern_type,
                    marker=dict(color=color, size=10),
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )

        # 추세선 추가
        fig.add_trace(
            go.Scatter(
                x=dates, y=trendline,
                mode='lines',
                name='Escalation Trend',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )

        return fig
```

---

#### P3-2: 감정 변화 그래프
**파일**: `src/voice_man/services/visualization/emotion_chart.py`

**구현 내용**:
- Sankey Diagram: 감정 전이 패턴
- Stacked Area Chart: 시간별 감정 분포
- 화자별 분리 뷰
- 인터랙티브 필터링

---

#### P3-3: 음성 특성 히트맵
**파일**: `src/voice_man/services/visualization/heatmap_generator.py`

**구현 내용**:
- 세그먼트 vs 특성 유형 히트맵
- 색상: 강도 (낮음-파랑, 높음-빨강)
- 화자별 분리
- 클릭 시 상세 정보 표시

---

#### P3-4: Streamlit 대시보드 통합
**파일**: `src/voice_man/services/visualization/dashboard_service.py`

**구현 내용**:
- 메인 대시보드 레이아웃
- 파일 선택 및 필터링
- 실시간 분석 결과 표시
- PDF 리포트 다운로드 버튼

**기술적 접근**:
```python
import streamlit as st

def main():
    st.set_page_config(page_title="Voice Forensic Dashboard", layout="wide")

    st.title("음성 포렌식 분석 대시보드")

    # 사이드바: 파일 선택
    with st.sidebar:
        selected_files = st.multiselect("분석 파일 선택", available_files)
        date_range = st.date_input("날짜 범위")

    # 메인 영역: 타임라인
    st.subheader("가스라이팅 패턴 타임라인")
    timeline_fig = timeline_visualizer.create_timeline(data)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # 2열 레이아웃: 감정 + 음성 특성
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("감정 변화 추이")
        emotion_fig = emotion_chart.create_flow(data)
        st.plotly_chart(emotion_fig)

    with col2:
        st.subheader("음성 특성 히트맵")
        heatmap_fig = heatmap_generator.create_heatmap(data)
        st.plotly_chart(heatmap_fig)

    # 화자 프로파일
    st.subheader("화자 프로파일")
    for speaker in speakers:
        profile_card = create_speaker_profile_card(speaker)
        st.write(profile_card)
```

---

### Phase 3 산출물

| 파일 | 설명 |
|------|------|
| `timeline_visualizer.py` | 타임라인 시각화 |
| `emotion_chart.py` | 감정 변화 차트 |
| `heatmap_generator.py` | 히트맵 생성기 |
| `dashboard_service.py` | Streamlit 대시보드 |
| `dashboard_app.py` | 대시보드 실행 엔트리포인트 |

### Phase 3 검증 기준

- [ ] 4가지 시각화 차트 정상 렌더링
- [ ] 인터랙티브 기능 동작 (줌, 필터, 클릭)
- [ ] Streamlit 대시보드 실행 성공
- [ ] 반응 속도: 차트 로딩 3초 이내

---

## Phase 4: 포렌식 리포트 생성 (Forensic Report Generation)

### 목표
법적 제출용 종합 PDF 리포트 생성

### 구현 항목

#### P4-1: 리포트 템플릿 설계
**파일**: `src/voice_man/services/forensic/forensic_report_service.py`

**리포트 구조**:
1. **표지**: 제목, 분석 기간, 생성 일시
2. **요약**: 핵심 발견 사항 (1-2페이지)
3. **분석 방법론**: 사용 알고리즘 및 학술 참조
4. **시계열 분석**: 가스라이팅 진행 과정 (차트 포함)
5. **증거 구간 목록**: 핵심 발언 타임스탬프 및 내용
6. **화자별 프로파일**: 심리 분석 결과
7. **통계 요약**: 수치화된 분석 결과
8. **제한사항 및 면책**: 분석 한계 명시
9. **부록**: 원본 데이터 참조, 용어 정의

---

#### P4-2: PDF 생성 엔진
**파일**: `src/voice_man/services/forensic/forensic_report_service.py`

**라이브러리**:
```python
reportlab>=4.1.0
weasyprint>=61.0  # HTML to PDF (대안)
```

**기술적 접근**:
```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table

class ForensicReportService:
    def generate_report(self, analysis_data: ForensicAnalysisData) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # 표지
        story.append(self._create_cover_page(analysis_data))

        # 요약
        story.append(self._create_executive_summary(analysis_data))

        # 방법론
        story.append(self._create_methodology_section())

        # 시계열 분석 (차트 이미지 포함)
        timeline_image = self._render_timeline_chart(analysis_data)
        story.append(Image(timeline_image, width=500, height=300))

        # 증거 구간 표
        evidence_table = self._create_evidence_table(analysis_data)
        story.append(evidence_table)

        # ... (나머지 섹션)

        doc.build(story)
        return buffer.getvalue()
```

---

#### P4-3: 증거 하이라이트 추출
**파일**: `src/voice_man/services/forensic/forensic_report_service.py`

**구현 내용**:
- 가스라이팅 고위험 구간 자동 추출
- 감정 격화 구간 식별
- 범죄 키워드 발언 목록
- 시간순 정렬 및 우선순위 표시

---

#### P4-4: 분석 제한사항 문서화
**파일**: `src/voice_man/services/forensic/forensic_report_service.py`

**포함 내용**:
```markdown
## 분석 제한사항

1. **음성 스트레스 분석**: 본 분석에서 사용된 음성 스트레스 지표는
   학술 연구에 기반하나, 법적 증거로서의 효력은 제한적입니다.

2. **감정 인식 정확도**: Speech Emotion Recognition 모델의 정확도는
   약 70-80% 수준이며, 전문가 검토를 권장합니다.

3. **가스라이팅 패턴**: 자동 감지된 패턴은 참고 자료이며,
   최종 판단은 심리 전문가의 해석이 필요합니다.

4. **원본 무결성**: 본 리포트는 제공된 원본 파일의 무결성을
   전제로 하며, 원본 검증은 별도 진행이 필요합니다.
```

---

### Phase 4 산출물

| 파일 | 설명 |
|------|------|
| `forensic_report_service.py` | 리포트 생성 서비스 |
| `report_templates/` | 리포트 템플릿 디렉토리 |
| `models/forensic/forensic_report.py` | 리포트 데이터 모델 |
| `tests/test_forensic_report.py` | 리포트 생성 테스트 |

### Phase 4 검증 기준

- [ ] PDF 리포트 정상 생성
- [ ] 모든 섹션 포함 확인
- [ ] 차트 이미지 정상 렌더링
- [ ] 한글 폰트 정상 출력
- [ ] 파일 크기: 10MB 이내

---

## 기술 스택 종합

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| librosa | 0.10.2+ | 음향 특성 추출 |
| parselmouth | 0.4.3+ | Praat 기반 음성학 분석 |
| speechbrain | 1.0+ | SER 파이프라인 |
| transformers | 4.40+ | Wav2Vec2/HuBERT 모델 |
| torch | 2.5+ | GPU 가속 |
| plotly | 5.18+ | 인터랙티브 시각화 |
| streamlit | 1.40+ | 대시보드 |
| reportlab | 4.1+ | PDF 생성 |
| pandas | 2.2+ | 데이터 처리 |

### 의존성 설치

```bash
# Phase 1: 음성 분석
pip install librosa>=0.10.2 parselmouth>=0.4.3 numpy>=1.26.0

# Phase 2: SER
pip install speechbrain>=1.0 transformers>=4.40.0 torch>=2.5.0

# Phase 3: 시각화
pip install plotly>=5.18.0 streamlit>=1.40.0 matplotlib>=3.8.0

# Phase 4: 리포트
pip install reportlab>=4.1.0 weasyprint>=61.0
```

---

## 우선순위 기반 마일스톤

### Primary Goal (1순위)
- Phase 1 전체: 음성 특성 분석 기반
- P2-1: SER 통합
- P2-3: 가스라이팅 시계열 분석

### Secondary Goal (2순위)
- P2-2: 텍스트-음성 감정 교차 검증
- P2-4: 심리 압박 지수
- P3-1: 타임라인 시각화
- P4-1, P4-2: 리포트 생성

### Final Goal (3순위)
- P3-2, P3-3: 감정/히트맵 차트
- P3-4: Streamlit 대시보드
- P4-3, P4-4: 증거 하이라이트 및 제한사항

### Optional Goal (4순위)
- 실시간 분석 스트리밍
- 다화자 비교 분석
- 외부 전문가 협업 기능

---

## 리스크 및 대응 전략

### 고위험

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|-----------|
| SER 모델 한국어 정확도 저하 | 40% | 높음 | 다중 모델 앙상블, 영어 모델 + 번역 fallback |
| GPU 메모리 부족 | 30% | 중간 | 배치 분할, 모델 순차 로딩, CPU 폴백 |

### 중위험

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|-----------|
| Parselmouth 설치 이슈 | 25% | 중간 | Docker 환경 격리, conda 패키지 |
| 시각화 성능 저하 | 20% | 낮음 | 데이터 샘플링, 캐싱, lazy loading |

---

## 테스트 전략

### 단위 테스트
- 각 서비스 클래스별 pytest 테스트
- 목표 커버리지: 85%

### 통합 테스트
- WhisperX 결과 -> 포렌식 분석 파이프라인
- 전체 183개 파일 배치 처리

### 회귀 테스트
- 기존 SPEC-VOICE-001 기능 유지 확인
- SPEC-WHISPERX-001 출력 호환성

---

## 참조

- SPEC-FORENSIC-001/spec.md: 상세 요구사항
- SPEC-FORENSIC-001/acceptance.md: 인수 기준
- SPEC-WHISPERX-001: WhisperX 파이프라인 (입력)
- SPEC-VOICE-001: 기반 시스템

---

**문서 끝**
