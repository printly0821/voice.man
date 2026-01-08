---
id: SPEC-PARALLEL-001
version: "1.0.0"
status: "draft"
created: "2026-01-08"
updated: "2026-01-08"
author: "지니"
priority: "HIGH"
title: "GPU 기반 병렬처리 최적화 시스템"
related_specs:
  - SPEC-VOICE-001
tags:
  - GPU
  - 병렬처리
  - 성능최적화
  - Whisper
  - WhisperX
lifecycle: "spec-anchored"
---

# SPEC-PARALLEL-001: GPU 기반 병렬처리 최적화 시스템

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-08 | 지니 | 초안 작성 - GPU 병렬처리 최적화 요구사항 정의 |

## CONTEXT

### 프로젝트 배경
- **프로젝트명**: voice.man - 음성 녹취 기반 증거 분석 시스템
- **현재 상태**: SPEC-VOICE-001 음성 분석 기본 시스템 구현 완료
- **핵심 문제**: 183개 m4a 파일 처리에 60분 소요 → 실용성 저하
- **목표**: 처리 시간을 1.2분으로 단축 (50배 성능 향상)

### 시스템 리소스
- **CPU**: 20코어 (Cortex-X925 + A725 구성)
- **메모리**: 119GB (사용 가능 114GB)
- **GPU**: NVIDIA GB10
  - Compute Capability: 12.1
  - CUDA Version: 13.0
  - 현재 활용률: 0% (Whisper가 CPU에서 실행 중)

### 현재 병목 지점 분석
1. **Worker 수 부족**: 4개 → 16-18개로 증가 필요
2. **배치 크기 최적화**: 5개 → 15-20개로 증가 필요
3. **GPU 미활용**: Whisper가 CPU에서 실행
4. **메모리 임계값 부적절**: 100MB → 70GB로 증가 필요

### 관련 SPEC
- **SPEC-VOICE-001**: 음성 분석 기본 시스템 (의존성)
  - 원본 파일 무결성 보장 요구사항 상속
  - 기존 분석 파이프라인 인터페이스 호환성 유지

---

## EARS 요구사항

### 1. Ubiquitous Requirements (U) - 전역 요구사항

#### U1: 성능 메트릭 로깅
**요구사항**: 시스템은 **항상** 모든 배치 처리 작업에 대한 성능 메트릭을 로깅해야 한다.

**세부사항**:
- 배치별 처리 시간 (초 단위)
- GPU 메모리 사용량 (MB 단위)
- CPU 활용률 (백분율)
- 처리된 파일 수 및 실패 건수
- 평균 처리 속도 (파일/초)

**WHY**: 성능 메트릭 로깅은 병목 지점 식별과 최적화 효과 측정의 기반이 된다.

**IMPACT**: 메트릭 부재 시 성능 저하 원인 파악 불가능 및 최적화 검증 실패.

---

#### U2: GPU 메모리 모니터링
**요구사항**: 시스템은 **항상** GPU 메모리 사용량을 실시간으로 모니터링해야 한다.

**세부사항**:
- 1초 간격 메모리 사용량 체크
- 메모리 사용률 80% 초과 시 경고 로그 생성
- 메모리 사용률 95% 초과 시 자동 배치 크기 감소

**WHY**: GPU 메모리 초과는 CUDA Out of Memory 오류로 이어져 전체 프로세스를 중단시킨다.

**IMPACT**: 메모리 모니터링 실패 시 시스템 크래시 및 처리 중단 발생.

---

#### U3: 원본 파일 무결성 보장
**요구사항**: 시스템은 **항상** 원본 오디오 파일의 무결성을 보장해야 한다 (SPEC-VOICE-001 연계).

**세부사항**:
- 원본 파일은 읽기 전용으로 처리
- 모든 분석 결과는 별도 디렉토리에 저장 (`.cache/`, `analysis-results/`)
- 파일 처리 전후 MD5 체크섬 비교 수행

**WHY**: 원본 파일 손상 시 증거 능력 상실 및 재분석 불가능.

**IMPACT**: 무결성 검증 실패 시 법적 증거 효력 상실.

---

### 2. Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: 배치 시작 시 GPU 가용성 확인
**요구사항**: **WHEN** 배치 처리가 시작될 때 **THEN** 시스템은 GPU 가용성을 확인해야 한다.

**세부사항**:
- `torch.cuda.is_available()` 호출로 GPU 감지
- GPU 사용 불가 시 CPU 폴백 모드로 전환
- GPU 감지 실패 시 로그 경고 및 성능 저하 안내

**WHY**: GPU 미감지 상태에서 GPU 전용 코드 실행 시 런타임 오류 발생.

**IMPACT**: GPU 가용성 미확인 시 처리 실패 및 사용자 혼란.

---

#### E2: GPU 메모리 부족 시 배치 크기 조정
**요구사항**: **WHEN** GPU 메모리가 부족할 때 **THEN** 시스템은 배치 크기를 자동으로 조정해야 한다.

**세부사항**:
- CUDA Out of Memory 오류 감지
- 배치 크기를 50% 감소 (최소 1까지)
- 감소된 배치 크기로 재시도 수행
- 3회 연속 실패 시 CPU 폴백

**WHY**: 고정 배치 크기는 다양한 GPU 환경에서 메모리 오류를 유발한다.

**IMPACT**: 동적 조정 실패 시 처리 중단 및 사용자 개입 필요.

---

#### E3: 처리 완료 시 성능 리포트 생성
**요구사항**: **WHEN** 모든 파일 처리가 완료될 때 **THEN** 시스템은 성능 리포트를 생성해야 한다.

**세부사항**:
- 총 처리 시간 (분:초 형식)
- 파일당 평균 처리 시간
- GPU vs CPU 처리 비율
- 메모리 사용 통계 (최대/평균)
- 실패 파일 목록 및 오류 원인

**WHY**: 성능 리포트는 최적화 효과 검증 및 향후 개선 방향 제시에 필수적이다.

**IMPACT**: 리포트 부재 시 성능 개선 검증 불가능.

---

### 3. State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: GPU 사용 중일 때 CPU 폴백 활성화
**요구사항**: **IF** GPU가 다른 프로세스에 의해 사용 중이면 **THEN** 시스템은 CPU 폴백 모드를 활성화해야 한다.

**세부사항**:
- `nvidia-smi` 명령으로 GPU 프로세스 확인
- GPU 메모리 90% 이상 사용 중이면 CPU 모드 전환
- 1분마다 GPU 가용성 재확인 및 자동 복귀

**WHY**: GPU 독점 사용 불가 시 대기로 인한 시간 낭비 방지.

**IMPACT**: 폴백 로직 부재 시 무한 대기 또는 처리 실패.

---

#### S2: 메모리 사용률 80% 초과 시 GC 트리거
**요구사항**: **IF** 시스템 메모리 사용률이 80%를 초과하면 **THEN** 명시적 가비지 컬렉션을 트리거해야 한다.

**세부사항**:
- `psutil.virtual_memory().percent` 모니터링
- 80% 초과 시 `gc.collect()` 및 `torch.cuda.empty_cache()` 호출
- GC 수행 후 메모리 회수량 로깅

**WHY**: 메모리 누수 또는 과다 사용 시 시스템 불안정 방지.

**IMPACT**: GC 트리거 실패 시 메모리 부족 오류 및 시스템 크래시.

---

#### S3: 배치 실패 시 재시도 큐 추가
**요구사항**: **IF** 배치 처리가 실패하면 **THEN** 해당 파일들을 재시도 큐에 추가해야 한다.

**세부사항**:
- 최대 3회 재시도 허용
- 재시도 간격: 5초, 15초, 30초 (지수 백오프)
- 3회 실패 시 `failed_files.json`에 기록

**WHY**: 일시적 오류(네트워크, 메모리)로 인한 처리 실패 복구.

**IMPACT**: 재시도 로직 부재 시 일시적 오류로도 영구 실패 처리.

---

### 4. Feature Requirements (F) - 기능 요구사항

#### F1: faster-whisper 기반 STT 변환
**요구사항**: 시스템은 `faster-whisper` 라이브러리를 사용하여 음성-텍스트 변환을 수행해야 한다.

**세부사항**:
- `faster-whisper>=1.0.3` 버전 사용
- GPU 모드: `device="cuda"`, `compute_type="float16"`
- CPU 폴백: `device="cpu"`, `compute_type="int8"`
- 모델: `large-v3` (정확도 우선) 또는 `medium` (속도 우선)

**WHY**: `faster-whisper`는 OpenAI Whisper 대비 4배 빠르고 메모리 효율적이다.

**IMPACT**: 라이브러리 미사용 시 목표 성능 달성 불가능.

---

#### F2: GPU 기반 배치 추론
**요구사항**: 시스템은 GPU에서 배치 추론을 수행하여 처리 속도를 극대화해야 한다.

**세부사항**:
- 배치 크기: 15-20 (GPU 메모리 상황에 따라 동적 조정)
- 최대 배치 크기: 32 (GB10 GPU 제약)
- VAD (Voice Activity Detection) 전처리로 무음 구간 제거

**WHY**: 배치 추론은 GPU 병렬 처리 능력을 최대한 활용한다.

**IMPACT**: 배치 미사용 시 GPU 활용률 저하 및 성능 미달.

---

#### F3: WhisperX 통합 파이프라인
**요구사항**: 시스템은 `WhisperX` 통합 파이프라인을 사용하여 전사, 정렬, 화자 분리를 수행해야 한다.

**세부사항**:
- `whisperx>=3.1.5` 버전 사용
- 파이프라인 단계:
  1. Whisper 전사 (GPU 배치 추론)
  2. WAV2VEC2 정렬 (타임스탬프 정확도 개선)
  3. Pyannote 화자 분리 (병렬 처리)

**WHY**: WhisperX는 전체 파이프라인을 GPU에서 최적화하여 end-to-end 성능을 극대화한다.

**IMPACT**: 통합 파이프라인 미사용 시 단계별 CPU-GPU 전환으로 오버헤드 증가.

---

#### F4: Pyannote 병렬 화자 분리
**요구사항**: 시스템은 `pyannote-audio`를 사용하여 화자 분리를 병렬로 수행해야 한다.

**세부사항**:
- `pyannote-audio>=3.1.1` 버전 사용
- Hugging Face 토큰 인증 필요 (환경 변수)
- GPU 메모리 허용 시 다중 파일 동시 처리

**WHY**: 화자 분리는 계산 집약적이며 병렬 처리로 속도 향상 가능.

**IMPACT**: 순차 처리 시 전체 파이프라인 병목 지점 형성.

---

#### F5: 동적 배치 크기 조정
**요구사항**: 시스템은 GPU 메모리 상황에 따라 배치 크기를 동적으로 조정해야 한다.

**세부사항**:
- 초기 배치 크기: 20
- GPU 메모리 80% 이상 사용 시: 배치 크기 감소
- GPU 메모리 50% 이하 사용 시: 배치 크기 증가 (최대 32)
- 조정 로직은 5개 배치마다 실행

**WHY**: 고정 배치 크기는 다양한 GPU 환경에서 비효율적이다.

**IMPACT**: 동적 조정 실패 시 메모리 오류 또는 GPU 미활용.

---

### 5. Optional/Unwanted Requirements (O/N) - 선택/금지 요구사항

#### O1: Multi-GPU 지원 (선택 사항)
**요구사항**: **가능하면** 시스템은 다중 GPU 환경을 지원해야 한다.

**세부사항**:
- `torch.nn.DataParallel` 또는 `torch.distributed` 활용
- GPU 간 배치 분산 처리
- 단일 GPU 환경에서도 정상 동작 보장

**WHY**: Multi-GPU는 대규모 처리에서 선형 성능 향상 제공.

**IMPACT**: 미구현 시 단일 GPU 제약으로 확장성 제한.

---

#### O2: 분산 처리 확장 (Celery) (선택 사항)
**요구사항**: **가능하면** 시스템은 Celery를 사용한 분산 처리를 지원해야 한다.

**세부사항**:
- Redis 또는 RabbitMQ 메시지 브로커
- Worker 노드별 GPU 할당
- 작업 큐 우선순위 관리

**WHY**: 분산 처리는 여러 서버에 워크로드를 분산하여 처리량을 증가시킨다.

**IMPACT**: 미구현 시 단일 서버 성능에 제약.

---

#### N1: 원본 파일 직접 수정 금지
**요구사항**: 시스템은 **절대** 원본 오디오 파일을 직접 수정하지 않아야 한다.

**세부사항**:
- 모든 변환 및 분석 결과는 별도 디렉토리에 저장
- 원본 파일은 읽기 전용 권한으로 접근
- 파일 이동 또는 삭제 금지

**WHY**: 원본 파일은 법적 증거로서 무결성 보장 필수.

**IMPACT**: 원본 수정 시 증거 효력 상실 및 법적 문제 발생.

---

#### N2: 무제한 메모리 할당 금지
**요구사항**: 시스템은 **절대** 무제한 메모리 할당을 허용하지 않아야 한다.

**세부사항**:
- 최대 메모리 사용량: 시스템 RAM의 80% (약 95GB)
- GPU 메모리: 100% 사용 금지 (최대 95%)
- 메모리 임계값 초과 시 처리 중단 및 오류 로그

**WHY**: 메모리 초과 사용은 시스템 불안정 및 다른 프로세스 영향.

**IMPACT**: 무제한 할당 시 시스템 크래시 및 데이터 손실.

---

## TECHNICAL SPECIFICATIONS

### 라이브러리 버전

#### 필수 라이브러리
```python
# STT 및 음성 처리
faster-whisper>=1.0.3
whisperx>=3.1.5
pyannote-audio>=3.1.1

# GPU 가속
torch>=2.5.0+cu121
torchaudio>=2.5.0+cu121

# 병렬 처리
concurrent.futures (Python 표준 라이브러리)

# 시스템 모니터링
psutil>=6.1.0
nvidia-ml-py>=12.560.30
```

#### 의존성 라이브러리
```python
# 오디오 처리
soundfile>=0.12.1
librosa>=0.10.2

# 데이터 처리
numpy>=1.26.4
pandas>=2.2.3

# 유틸리티
tqdm>=4.67.1
```

---

### 성능 목표

#### Phase 1: 즉시 최적화 (4배 향상)
- **목표 처리 시간**: 15분 (현재 60분 대비)
- **Worker 수**: 16-18
- **배치 크기**: 15
- **메모리 임계값**: 70GB
- **CPU 활용률**: 80%

#### Phase 2: GPU 활성화 (20배 향상)
- **목표 처리 시간**: 3분
- **GPU 활용률**: 70%
- **faster-whisper 통합**: float16 precision
- **배치 크기**: 20
- **디바이스**: CUDA (GB10)

#### Phase 3: 완전 파이프라인 (50배 향상)
- **목표 처리 시간**: 1.2분
- **GPU 활용률**: 90%
- **WhisperX 통합**: 전체 파이프라인 GPU 최적화
- **화자 분리**: 병렬 처리
- **VAD 전처리**: 무음 구간 제거
- **배치 크기**: 32 (동적 조정)

---

### CUDA 및 GPU 요구사항

#### 최소 요구사항
- **CUDA Version**: 12.3 이상
- **cuDNN Version**: 9.0 이상
- **GPU Compute Capability**: 12.1 (GB10)
- **GPU 메모리**: 최소 8GB (권장 16GB)

#### 환경 변수
```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="12.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

### 파일 구조

```
voice.man/
├── scripts/
│   └── process_audio_files.py          # 메인 처리 스크립트 (수정 필요)
├── src/
│   └── voice_analysis/
│       ├── services/
│       │   ├── batch_service.py         # 배치 처리 로직 (수정 필요)
│       │   ├── memory_service.py        # 메모리 관리 (수정 필요)
│       │   └── analysis_pipeline_service.py  # GPU 통합 (수정 필요)
│       └── models/
│           └── whisper_model.py         # faster-whisper 래퍼 (신규)
├── tests/
│   └── test_parallel_processing.py      # 병렬 처리 테스트 (신규)
└── .moai/
    └── specs/
        └── SPEC-PARALLEL-001/           # 본 SPEC 문서
```

---

## CONSTRAINTS

### 하드웨어 제약
- **GPU 메모리**: 최대 16GB (GB10)
- **시스템 RAM**: 최대 119GB (사용 가능 114GB)
- **디스크 I/O**: SSD 권장 (대용량 오디오 파일 읽기)

### 소프트웨어 제약
- **Python 버전**: 3.10 이상
- **CUDA 호환성**: PyTorch 2.5.0은 CUDA 12.1-12.6 지원
- **Hugging Face 토큰**: pyannote 모델 접근용 (환경 변수)

### 비즈니스 제약
- **원본 파일 보존**: 모든 처리는 비파괴적
- **처리 시간**: 1.5분 이내 (목표 1.2분)
- **정확도 유지**: WER (Word Error Rate) 변화 < 1%

---

## ACCEPTANCE CRITERIA

### 기능적 요구사항
1. ✅ 183개 파일을 1.5분 이내에 처리 완료
2. ✅ GPU 활용률 85% 이상 유지
3. ✅ 모든 파일에 대해 STT, 정렬, 화자 분리 수행
4. ✅ 원본 파일 무결성 100% 보장
5. ✅ 처리 완료 후 성능 리포트 자동 생성

### 비기능적 요구사항
1. ✅ 메모리 사용량 시스템 RAM의 80% 미만
2. ✅ GPU 메모리 오류 발생 시 자동 복구 (배치 크기 조정)
3. ✅ CPU 폴백 모드 정상 동작
4. ✅ 테스트 커버리지 85% 이상
5. ✅ TRUST 5 품질 게이트 통과

---

## DEPENDENCIES

### 기술적 의존성
- **SPEC-VOICE-001**: 기존 음성 분석 파이프라인 인터페이스
- **CUDA Toolkit**: GPU 연산 필수
- **Hugging Face Hub**: pyannote 모델 다운로드

### 외부 의존성
- **Hugging Face Token**: 환경 변수로 제공 필요
- **인터넷 연결**: 초기 모델 다운로드 (약 3GB)

---

## RISKS AND MITIGATION

### 고위험
1. **GPU 메모리 부족**
   - **완화 전략**: 동적 배치 크기 조정, CPU 폴백

2. **CUDA 호환성 문제**
   - **완화 전략**: Docker 컨테이너 사용, 버전 고정

### 중위험
1. **처리 속도 목표 미달**
   - **완화 전략**: Phase별 점진적 최적화, 성능 프로파일링

2. **모델 다운로드 실패**
   - **완화 전략**: 로컬 모델 캐시, 재시도 로직

---

## TRACEABILITY

### 관련 문서
- **SPEC-VOICE-001**: 음성 분석 기본 시스템
- **CLAUDE.md**: 프로젝트 실행 지침
- **README.md**: 프로젝트 개요 및 설치 가이드

### 구현 파일
- `scripts/process_audio_files.py`
- `src/voice_analysis/services/batch_service.py`
- `src/voice_analysis/services/memory_service.py`
- `src/voice_analysis/services/analysis_pipeline_service.py`

---

## REFERENCES

### 기술 문서
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX Paper](https://arxiv.org/abs/2303.00747)
- [pyannote-audio Documentation](https://github.com/pyannote/pyannote-audio)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 관련 표준
- **OWASP**: 보안 가이드라인
- **PEP 8**: Python 코드 스타일
- **TRUST 5**: MoAI-ADK 품질 프레임워크

---

**문서 끝**
