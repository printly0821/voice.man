---
id: SPEC-E2ETEST-001
version: "1.0.0"
status: "planned"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
priority: "HIGH"
title: "WhisperX E2E 통합 테스트 - GPU 병렬 배치 처리"
related_specs:
  - SPEC-VOICE-001
  - SPEC-PARALLEL-001
  - SPEC-WHISPERX-001
tags:
  - E2E테스트
  - GPU
  - 병렬처리
  - WhisperX
  - 배치처리
  - 성능검증
lifecycle: "spec-first"
---

# SPEC-E2ETEST-001: WhisperX E2E 통합 테스트 - GPU 병렬 배치 처리

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-08 | 지니 | 초안 작성 - E2E 통합 테스트 요구사항 정의 |

## CONTEXT

### 프로젝트 배경
- **프로젝트명**: Voice Man - 음성 녹취 기반 증거 분석 시스템
- **선행 SPEC**:
  - SPEC-VOICE-001: 음성 분석 기본 시스템 (완료)
  - SPEC-PARALLEL-001: GPU 기반 병렬처리 최적화 (완료)
  - SPEC-WHISPERX-001: WhisperX 통합 파이프라인 (완료)
- **핵심 목표**: 183개 m4a 오디오 파일 전체를 GPU 병렬 배치 처리로 E2E 테스트 수행
- **테스트 데이터**: `ref/call/` 디렉토리 내 183개 m4a 파일 (총 1.5GB)
- **주요 발화자**: 신기연, 김경민 등

### 테스트 목적
1. **기능 검증**: SPEC-WHISPERX-001 파이프라인의 실제 대용량 데이터 처리 능력 확인
2. **성능 검증**: SPEC-PARALLEL-001의 GPU 병렬 배치 처리 성능 목표 달성 확인
3. **안정성 검증**: 183개 파일 연속 처리 시 메모리 누수 및 GPU OOM 미발생 확인
4. **품질 검증**: 전사, 정렬, 화자 분리 결과의 품질 검증

### 시스템 리소스
- **CPU**: 20코어 (Cortex-X925 + A725 구성)
- **메모리**: 119GB (사용 가능 114GB)
- **GPU**: NVIDIA GB10
  - Compute Capability: 12.1
  - CUDA Version: 13.0
  - 목표 활용률: 90% 이상

### 테스트 대상 파일 현황
- **총 파일 수**: 183개
- **파일 포맷**: m4a
- **위치**: `ref/call/`
- **총 용량**: 약 1.5GB
- **주요 발화자**: 신기연, 김경민(리버스마운틴대표) 등

---

## EARS 요구사항

### 1. Ubiquitous Requirements (U) - 전역 요구사항

#### U1: GPU 병렬 배치 처리 필수
**요구사항**: 시스템은 **항상** SPEC-PARALLEL-001의 BatchProcessor를 활용하여 GPU 병렬 배치 처리를 수행해야 한다.

**세부사항**:
- BatchProcessor 클래스 활용
- GPU 배치 크기: 15-20 (최대 32)
- 동적 배치 크기 조정 활성화
- 183개 파일 전체 병렬 처리

**WHY**: GPU 병렬 배치 처리는 성능 목표 달성의 핵심 요소이다.

**IMPACT**: 순차 처리 시 목표 시간(20분) 초과 불가피.

---

#### U2: 전체 파이프라인 실행 필수
**요구사항**: 시스템은 **항상** 전체 WhisperX 파이프라인(전사, 정렬, 화자 분리)을 실행해야 한다.

**세부사항**:
1. 오디오 포맷 변환 (m4a → 16kHz mono WAV)
2. WhisperX STT (faster-whisper large-v3)
3. WAV2VEC2 타임스탬프 정렬
4. pyannote-audio 화자 분리
5. 결과 저장 및 리포트 생성

**WHY**: 부분 파이프라인 실행은 실제 운영 환경을 반영하지 못한다.

**IMPACT**: 부분 테스트는 통합 문제 발견 불가능.

---

#### U3: 성능 메트릭 수집 필수
**요구사항**: 시스템은 **항상** 처리 전 과정에서 성능 메트릭을 수집해야 한다.

**세부사항**:
- 배치별 처리 시간 (초 단위)
- GPU 메모리 사용량 (MB 단위)
- GPU 활용률 (%)
- 파일별 처리 시간
- 총 처리 시간
- 실패 파일 목록

**WHY**: 성능 메트릭은 목표 달성 검증 및 병목 지점 식별에 필수적이다.

**IMPACT**: 메트릭 부재 시 성능 검증 불가능.

---

#### U4: 원본 파일 무결성 보장
**요구사항**: 시스템은 **항상** 원본 오디오 파일의 무결성을 보장해야 한다 (SPEC-VOICE-001 연계).

**세부사항**:
- 원본 파일은 읽기 전용으로 처리
- 모든 변환 및 분석 결과는 별도 디렉토리에 저장
- 테스트 전후 원본 파일 MD5 체크섬 비교

**WHY**: 원본 파일은 법적 증거로서 무결성 보장 필수.

**IMPACT**: 원본 손상 시 증거 효력 상실.

---

### 2. Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: 파일 처리 완료 시 결과 저장
**요구사항**: **WHEN** 개별 파일 처리가 완료될 때 **THEN** 시스템은 결과를 즉시 저장해야 한다.

**세부사항**:
- 결과 저장 위치: `ref/call/reports/` 디렉토리
- 파일별 JSON 결과: 전사 텍스트, 세그먼트, 화자 정보
- 처리 메타데이터: 처리 시간, 사용 메모리, 성공/실패 상태

**WHY**: 즉시 저장은 중간 장애 시 데이터 손실 방지.

**IMPACT**: 지연 저장 시 시스템 크래시로 전체 결과 손실 위험.

---

#### E2: 배치 완료 시 진행률 콜백
**요구사항**: **WHEN** 배치 처리가 완료될 때 **THEN** 시스템은 진행률 콜백을 호출해야 한다.

**세부사항**:
- 콜백 정보: 완료된 파일 수, 총 파일 수, 경과 시간, ETA
- 콘솔 출력: 진행 바 및 상태 메시지
- 로그 기록: 배치별 처리 통계

**WHY**: 대량 파일 처리 시 진행 상황 모니터링 필수.

**IMPACT**: 진행률 미제공 시 사용자 경험 저하 및 시스템 응답 불확실성.

---

#### E3: 전체 처리 완료 시 리포트 생성
**요구사항**: **WHEN** 모든 파일 처리가 완료될 때 **THEN** 시스템은 종합 리포트를 생성해야 한다.

**세부사항**:
- 리포트 내용:
  - 총 처리 시간 (분:초 형식)
  - 파일당 평균 처리 시간
  - GPU 사용 통계 (최대/평균 메모리, 활용률)
  - 성공/실패 파일 통계
  - 화자별 발화 통계 요약
- 리포트 포맷: JSON + Markdown 요약

**WHY**: 종합 리포트는 E2E 테스트 결과 검증의 핵심 산출물.

**IMPACT**: 리포트 부재 시 테스트 결과 분석 불가능.

---

#### E4: GPU 메모리 부족 시 동적 배치 조정
**요구사항**: **WHEN** GPU 메모리가 부족할 때 **THEN** 시스템은 배치 크기를 자동으로 조정해야 한다.

**세부사항**:
- GPU 메모리 80% 초과 시 배치 크기 50% 감소
- GPU 메모리 50% 이하 시 배치 크기 증가 (최대 32)
- CUDA OOM 발생 시 즉시 배치 크기 감소 및 재시도
- 3회 연속 실패 시 해당 파일 스킵 및 로깅

**WHY**: 동적 조정은 다양한 GPU 환경에서 안정적 처리 보장.

**IMPACT**: 고정 배치 크기는 OOM 또는 GPU 미활용 유발.

---

### 3. State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: GPU 메모리 상태 모니터링
**요구사항**: **IF** GPU 메모리 사용률이 95%를 초과하면 **THEN** 시스템은 현재 배치 처리를 일시 중단해야 한다.

**세부사항**:
- 1초 간격 GPU 메모리 사용량 체크
- 95% 초과 시 `torch.cuda.empty_cache()` 호출
- 메모리 해제 후 처리 재개
- 지속적 초과 시 배치 크기 영구 감소

**WHY**: GPU 메모리 95% 이상 사용은 OOM 직전 상태.

**IMPACT**: 모니터링 실패 시 시스템 크래시 및 처리 중단.

---

#### S2: 실패 파일 재시도 큐 관리
**요구사항**: **IF** 파일 처리가 실패하면 **THEN** 시스템은 해당 파일을 재시도 큐에 추가해야 한다.

**세부사항**:
- 최대 3회 재시도 허용
- 재시도 간격: 5초, 15초, 30초 (지수 백오프)
- 3회 실패 시 `failed_files.json`에 기록
- 실패 원인 상세 로깅

**WHY**: 일시적 오류(메모리, 파일 접근)로 인한 처리 실패 복구.

**IMPACT**: 재시도 없이 즉시 실패 처리 시 성공률 저하.

---

#### S3: 긴 오디오 파일 청크 분할
**요구사항**: **IF** 오디오 길이가 30분을 초과하면 **THEN** 시스템은 10분 청크로 분할하여 처리해야 한다.

**세부사항**:
- 청크 길이: 10분 (600초)
- 청크 간 오버랩: 30초 (경계 단어 누락 방지)
- 청크 결과 병합 시 중복 구간 제거
- 화자 ID 일관성 유지

**WHY**: 장시간 오디오는 메모리 급증 및 처리 시간 예측 불가.

**IMPACT**: 분할 없이 처리 시 OOM 및 처리 실패 위험.

---

### 4. Feature Requirements (F) - 기능 요구사항

#### F1: 배치 처리 테스트 스크립트
**요구사항**: 시스템은 183개 파일을 GPU 병렬 배치로 처리하는 테스트 스크립트를 제공해야 한다.

**세부사항**:
- 스크립트 위치: `scripts/e2e_batch_test.py`
- 입력: `ref/call/` 디렉토리 경로
- 출력: `ref/call/reports/` 디렉토리에 결과 저장
- 설정 옵션:
  - `--batch-size`: 배치 크기 (기본값: 20)
  - `--num-speakers`: 화자 수 (기본값: 자동 감지)
  - `--language`: 언어 코드 (기본값: ko)
  - `--output-dir`: 결과 저장 경로

**WHY**: 표준화된 테스트 스크립트는 재현 가능한 테스트 실행 보장.

**IMPACT**: 스크립트 미제공 시 테스트 절차 불일치 및 재현 불가.

---

#### F2: 진행률 콜백 시스템
**요구사항**: 시스템은 처리 진행률을 실시간으로 보고하는 콜백 시스템을 제공해야 한다.

**세부사항**:
- 콜백 인터페이스:
  ```python
  def progress_callback(
      current: int,
      total: int,
      elapsed_seconds: float,
      current_file: str,
      status: str
  ) -> None
  ```
- 진행률 정보: 파일 수, 시간, 현재 파일명, 상태
- tqdm 통합 지원

**WHY**: 대량 파일 처리 시 진행 상황 피드백 필수.

**IMPACT**: 콜백 미제공 시 사용자 경험 저하.

---

#### F3: 종합 결과 리포트 생성
**요구사항**: 시스템은 전체 E2E 테스트 결과를 종합 리포트로 생성해야 한다.

**세부사항**:
- 리포트 파일: `ref/call/reports/e2e_test_report.json`, `e2e_test_summary.md`
- 리포트 내용:
  - 테스트 메타데이터 (시작/종료 시간, 환경 정보)
  - 성능 통계 (총 시간, 평균 시간, 처리량)
  - GPU 통계 (메모리 사용, 활용률)
  - 파일별 처리 결과 (성공/실패, 처리 시간)
  - 화자 분석 요약 (발화자별 통계)

**WHY**: 종합 리포트는 테스트 결과 검증 및 문서화의 핵심.

**IMPACT**: 리포트 미생성 시 테스트 결과 분석 및 공유 불가능.

---

#### F4: 오디오 포맷 변환 서비스
**요구사항**: 시스템은 m4a 파일을 WhisperX 호환 포맷(16kHz mono WAV)으로 변환해야 한다.

**세부사항**:
- 지원 입력 포맷: m4a, mp3, wav, flac, ogg
- 출력 포맷: 16kHz mono WAV
- 변환 도구: ffmpeg 또는 pydub
- 임시 파일 자동 삭제

**WHY**: WhisperX는 16kHz mono WAV에서 최적 성능 발휘.

**IMPACT**: 포맷 미변환 시 처리 오류 또는 성능 저하.

---

### 5. Unwanted Requirements (N) - 금지 요구사항

#### N1: GPU 메모리 초과 사용 금지
**요구사항**: 시스템은 **절대** GPU 메모리 사용률 95%를 초과하지 않아야 한다.

**세부사항**:
- 실시간 GPU 메모리 모니터링
- 95% 도달 전 사전 조치 (배치 크기 감소, 캐시 정리)
- OOM 발생 시 즉시 복구 로직 실행

**WHY**: GPU OOM은 전체 프로세스를 중단시킨다.

**IMPACT**: OOM 발생 시 처리 중인 모든 데이터 손실.

---

#### N2: 실패 파일 무시 금지
**요구사항**: 시스템은 **절대** 처리 실패 파일을 로깅 없이 무시하지 않아야 한다.

**세부사항**:
- 모든 실패 파일을 `failed_files.json`에 기록
- 실패 원인 상세 로깅 (오류 메시지, 스택 트레이스)
- 최종 리포트에 실패 통계 포함

**WHY**: 실패 파일 무시는 데이터 손실 및 분석 불완전성 유발.

**IMPACT**: 실패 파일 미기록 시 문제 원인 파악 불가능.

---

#### N3: 원본 파일 수정 금지
**요구사항**: 시스템은 **절대** `ref/call/` 디렉토리의 원본 오디오 파일을 수정하지 않아야 한다.

**세부사항**:
- 원본 파일은 읽기 전용 접근
- 모든 변환 및 결과는 `reports/` 또는 임시 디렉토리에 저장
- 테스트 전후 원본 파일 해시 검증

**WHY**: 원본 파일은 법적 증거로서 무결성 보장 필수.

**IMPACT**: 원본 수정 시 증거 효력 상실 및 재분석 불가능.

---

## TECHNICAL SPECIFICATIONS

### 성능 목표

#### 처리 시간 목표
- **입력**: 183개 m4a 파일 (총 1.5GB)
- **목표 처리 시간**: 20분 이내
- **단일 파일 평균**: 6초 이내 (6초 x 183 = 18.3분)
- **GPU 활용률**: 85% 이상

#### GPU 리소스 제약
- **GPU 메모리 사용률**: 최대 95%
- **배치 크기**: 15-20 (최대 32)
- **동적 조정**: GPU 메모리 상황에 따라 자동 조정

### 라이브러리 요구사항

```python
# E2E 테스트 필수 라이브러리
whisperx>=3.1.5
faster-whisper>=1.0.3
pyannote-audio>=3.1.1
torch>=2.5.0+cu121
torchaudio>=2.5.0+cu121

# 오디오 처리
soundfile>=0.12.1
librosa>=0.10.2
ffmpeg-python>=0.2.0

# 시스템 모니터링
psutil>=6.1.0
nvidia-ml-py>=12.560.30
tqdm>=4.67.1

# 테스트 및 리포팅
pytest>=8.0
pytest-asyncio>=0.24
```

### 환경 변수

```bash
# Hugging Face 인증 (필수)
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# GPU 설정
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# WhisperX 설정
export WHISPERX_MODEL_SIZE="large-v3"
export WHISPERX_LANGUAGE="ko"
export WHISPERX_DEVICE="cuda"
export WHISPERX_COMPUTE_TYPE="float16"
```

### 파일 구조

```
voice.man/
├── ref/
│   └── call/
│       ├── *.m4a                          # 원본 오디오 파일 (183개)
│       └── reports/                        # 결과 저장 디렉토리
│           ├── e2e_test_report.json       # 종합 리포트 (JSON)
│           ├── e2e_test_summary.md        # 요약 리포트 (Markdown)
│           ├── failed_files.json          # 실패 파일 목록
│           └── results/                   # 파일별 결과
│               ├── file_001_result.json
│               └── ...
├── scripts/
│   └── e2e_batch_test.py                  # E2E 테스트 스크립트
├── src/
│   └── voice_man/
│       ├── models/
│       │   └── whisperx_pipeline.py       # WhisperX 파이프라인 (기존)
│       └── services/
│           ├── batch_service.py           # 배치 처리 서비스 (기존)
│           ├── e2e_test_service.py        # E2E 테스트 서비스 (신규)
│           └── report_service.py          # 리포트 생성 서비스 (기존)
└── tests/
    └── e2e/
        └── test_full_batch_processing.py  # E2E 테스트 케이스
```

---

## CONSTRAINTS

### 하드웨어 제약
- **GPU 메모리**: 최대 16GB (GB10)
- **시스템 RAM**: 최대 119GB
- **디스크**: 임시 파일용 추가 2GB 필요

### 소프트웨어 제약
- **Python 버전**: 3.10 이상
- **CUDA 호환성**: PyTorch 2.5.0은 CUDA 12.1-12.6 지원
- **Hugging Face**: pyannote 모델 사용 동의 및 토큰 필수

### 테스트 데이터 제약
- **파일 개수**: 183개 고정
- **파일 포맷**: m4a
- **언어**: 한국어 (ko)
- **화자**: 주로 2인 대화 (신기연, 김경민 등)

---

## ACCEPTANCE CRITERIA

### 기능적 요구사항
1. 183개 m4a 파일 전체 처리 성공 (100% 완료)
2. 모든 파일에 대해 전사, 정렬, 화자 분리 수행
3. 처리 결과 `reports/` 디렉토리에 저장
4. 종합 리포트 생성 (JSON + Markdown)
5. 진행률 콜백 정상 작동

### 비기능적 요구사항
1. 총 처리 시간 20분 이내
2. 단일 파일 평균 처리 시간 6초 이내
3. GPU 메모리 사용률 95% 미만 유지
4. GPU 활용률 85% 이상
5. 원본 파일 무결성 100% 보장
6. 실패 파일 0개 (재시도 후 기준)

### 리포트 검증
1. 성능 통계 포함 (처리 시간, GPU 사용량)
2. 화자별 발화 통계 포함
3. 실패 파일 목록 포함 (있는 경우)
4. 테스트 환경 정보 포함

---

## DEPENDENCIES

### 기술적 의존성
- **SPEC-VOICE-001**: 원본 파일 무결성 요구사항
- **SPEC-PARALLEL-001**: BatchProcessor 클래스, GPU 모니터링 서비스
- **SPEC-WHISPERX-001**: WhisperXPipeline 클래스, 화자 분리 서비스

### 외부 의존성
- **Hugging Face Hub**: pyannote, WAV2VEC2 모델
- **Hugging Face Token**: 환경 변수로 제공 필수
- **ffmpeg**: 오디오 포맷 변환

---

## RISKS AND MITIGATION

### 고위험
1. **GPU 메모리 부족 (183개 연속 처리)**
   - **확률**: 40%
   - **영향**: 처리 중단, 결과 손실
   - **완화 전략**: 동적 배치 크기 조정, 주기적 캐시 정리, 순차 모델 로딩

2. **처리 시간 목표 미달**
   - **확률**: 30%
   - **영향**: 성능 검증 실패
   - **완화 전략**: 배치 크기 최적화, GPU 활용률 모니터링, 병목 지점 분석

### 중위험
1. **특정 파일 처리 실패**
   - **확률**: 20%
   - **영향**: 불완전한 결과
   - **완화 전략**: 재시도 로직, 상세 오류 로깅, 실패 파일 별도 처리

2. **임시 파일 누적**
   - **확률**: 25%
   - **영향**: 디스크 공간 부족
   - **완화 전략**: 처리 완료 후 즉시 삭제, context manager 사용

---

## TRACEABILITY

### 관련 문서
- **SPEC-VOICE-001**: 음성 분석 기본 시스템
- **SPEC-PARALLEL-001**: GPU 기반 병렬처리 최적화
- **SPEC-WHISPERX-001**: WhisperX 통합 파이프라인

### 구현 파일
- `scripts/e2e_batch_test.py` (신규)
- `src/voice_man/services/e2e_test_service.py` (신규)
- `tests/e2e/test_full_batch_processing.py` (신규)

---

## REFERENCES

### 기술 문서
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [pyannote-audio Documentation](https://github.com/pyannote/pyannote-audio)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 관련 표준
- **TRUST 5**: MoAI-ADK 품질 프레임워크
- **PEP 8**: Python 코드 스타일

---

**문서 끝**
