---
id: SPEC-PERFOPT-001
version: "1.3.0"
status: "COMPLETED"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
title: "Forensic Pipeline Performance Optimization"
phase_status:
  phase_1: "COMPLETED"
  phase_2: "COMPLETED"
  phase_3: "COMPLETED"
related_specs:
  - SPEC-FORENSIC-001
  - SPEC-GPUOPT-001
  - SPEC-EDGEXPERT-001
  - SPEC-PARALLEL-001
tags:
  - forensic
  - GPU-optimization
  - performance
  - SER
  - memory-management
  - thermal-management
  - pipeline-optimization
lifecycle: "spec-anchored"
---

# SPEC-PERFOPT-001: Forensic Pipeline Performance Optimization

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-10 | 지니 | 초안 작성 - Forensic 파이프라인 성능 최적화 요구사항 정의 |
| 1.1.0 | 2026-01-10 | 지니 | Phase 1 구현 완료 - 메모리 임계값 30GB, SER GPU-first 감지, 모델 캐싱, 배치별 GPU 캐시 정리 |
| 1.2.0 | 2026-01-10 | 지니 | Phase 2 구현 완료 - ForensicMemoryManager, BatchConfigManager, ThermalManager 통합 (67개 테스트, 86% 커버리지) |
| 1.3.0 | 2026-01-10 | 지니 | Phase 3 구현 완료 - PipelineOrchestrator 프로듀서-컨슈머 패턴 (34개 테스트, 89% 커버리지) - SPEC 완료 |

---

## 1. Executive Summary

### 1.1 Overview

Forensic 분석 파이프라인의 GPU 활용률을 최적화하여 183개 오디오 파일 처리 시간을 15시간 이상에서 4시간 미만으로 단축하는 성능 최적화 시스템 구현. EdgeXpert Blackwell 128GB 환경에 특화된 통합 메모리 관리, 열 관리, 스테이지 파이프라이닝을 적용.

### 1.2 Business Value

**Performance Improvements:**
- Phase 1 (Quick Wins): 2-3배 처리 속도 향상
- Phase 2 (Integration): 3-4배 처리 속도 향상 (누적)
- Phase 3 (Advanced): 4배 이상 처리 속도 향상 (누적)

**Cost Efficiency:**
- GPU idle time 90% 감소 (forensic 단계)
- 메모리 캐시 효율 30배 증가 (100MB -> 30GB)
- 열 쓰로틀링 발생 0% 유지

**Technical Excellence:**
- SER 모델 캐싱으로 로딩 시간 45초/파일 → 0초
- STT + Forensic 스테이지 오버랩으로 파이프라인 효율 50% 향상
- ARM64 CPU 최적화로 I/O 병목 해소

### 1.3 Current Problems (탐색 결과)

**Problem 1: GPU Idle During Forensic (40-50% 파이프라인 시간)**
- 현상: WhisperX STT 완료 후 Forensic 분석 시 GPU 미사용
- 영향: 전체 처리 시간의 40-50%가 CPU-bound 작업에 소모
- 근본 원인: SER 모델이 GPU로 로드되지 않음, 순차적 파일 처리

**Problem 2: Memory Cache Conflicts**
- 현상: 메모리 임계값 100MB 설정으로 인한 과도한 캐시 무효화
- 영향: 128GB 통합 메모리 중 0.08%만 활용
- 근본 원인: 일반 시스템 기준 설정, EdgeXpert 환경 미최적화

**Problem 3: Thermal Throttling**
- 현상: 장시간 처리 시 열 관리 부재로 성능 저하
- 영향: 연속 처리 시 예측 불가능한 처리 시간
- 근본 원인: 능동적 열 관리 시스템 미구현

**Problem 4: Sequential Stage Execution**
- 현상: STT → Forensic 순차 실행, 파이프라인 오버랩 없음
- 영향: GPU/CPU 리소스 번갈아 유휴 상태
- 근본 원인: 스테이지 간 의존성 미분석, 병렬화 미적용

**Problem 5: SER Model Loading Overhead**
- 현상: 파일당 SER 모델 로딩 45초 소요
- 영향: 183개 파일 기준 약 2.3시간 순수 로딩 시간
- 근본 원인: 모델 캐싱 미구현, 파일별 모델 재로딩

---

## 2. Scope Definition

### 2.1 In-Scope (본 SPEC 범위)

**Forensic Layer Optimization:**
- SER 모델 GPU 가속화 및 캐싱
- 텍스트-음성 교차검증 GPU 가속화
- 포렌식 스코어링 연산 최적화

**Unified Memory Management:**
- 메모리 임계값 30GB로 상향 조정
- 스테이지별 메모리 프로파일링
- torch.cuda.empty_cache() 전략적 호출

**Thermal-Aware Scheduling:**
- GPU 온도 모니터링 통합
- 동적 쓰로틀링 정책
- 열 기반 배치 사이즈 조정

**Stage Pipelining:**
- STT + Forensic 스테이지 오버랩
- 비동기 파일 처리 큐
- 프로듀서-컨슈머 패턴 구현

### 2.2 Out-of-Scope (타 SPEC 범위)

**SPEC-GPUOPT-001 범위:**
- WhisperX STT 파이프라인 GPU 최적화
- torch.compile() 적용
- Faster-Whisper 마이그레이션
- TensorRT 최적화

**SPEC-EDGEXPERT-001 범위:**
- 통합 메모리 Zero-copy 구현
- CUDA Stream 병렬 처리
- FP4/Sparse 연산 최적화
- ARM CPU 병렬 I/O

**SPEC-FORENSIC-001 범위 (완료):**
- 범죄 언어 패턴 DB
- SER 서비스 기본 구현
- 포렌식 리포트 생성

### 2.3 Dependencies

**Upstream (의존성):**
- SPEC-FORENSIC-001: SER 서비스, 포렌식 스코어링 인터페이스
- SPEC-WHISPERX-001: WhisperX 파이프라인 결과물

**Downstream (영향):**
- SPEC-GPUOPT-001: 통합 메모리 관리자 공유
- SPEC-EDGEXPERT-001: 열 관리 시스템 공유

---

## 3. Environment

### 3.1 Target Hardware (EdgeXpert Blackwell)

**GPU:**
- 아키텍처: NVIDIA Grace Blackwell
- 메모리: 128GB LPDDR5x 통합 메모리
- 대역폭: 273 GB/s
- Tensor Cores: 1000 AI FLOPS @ FP4/Sparse

**CPU:**
- 아키텍처: 20코어 ARM (Cortex-X925 + A725)
- 최대 클럭: 3.8GHz (X925), 3.3GHz (A725)

**Thermal Constraints:**
- 최대 온도: 85°C
- 쓰로틀링 시작: 80°C
- 폼팩터: 151 x 151 x 52mm 미니PC

### 3.2 Software Stack

**Core Dependencies:**
- Python: >=3.12,<3.13 (ARM64 호환)
- PyTorch: >=2.5.0
- CUDA: >=12.6 (Grace Blackwell 지원)
- WhisperX: >=3.1.6
- SpeechBrain: >=1.0

**Forensic Services:**
- ser_service.py: SER 감정 인식 서비스
- audio_feature_service.py: 음성 특성 분석
- forensic_scoring_service.py: 포렌식 스코어링

### 3.3 Workload Characteristics

**Input:**
- 파일 수: 183개 오디오 파일
- 평균 길이: 약 5-30분/파일
- 형식: M4A, WAV, MP3

**Current Performance (Baseline):**
- 총 처리 시간: 15+ 시간
- STT 단계: 5-6시간
- Forensic 단계: 9-10시간
- GPU 활용률 (Forensic): < 5%

---

## 4. Assumptions

### 4.1 Technical Assumptions

**T1 - SER Model GPU Compatibility:**
- **가정:** wav2vec2, SpeechBrain SER 모델이 GPU에서 정상 동작
- **신뢰도:** High
- **근거:** 현재 CPU에서 정상 동작, GPU 이식 표준 절차
- **위험 시:** CPU 폴백 유지
- **검증 방법:** Phase 1에서 단일 파일 GPU 추론 테스트

**T2 - Memory Threshold Scaling:**
- **가정:** 메모리 임계값 30GB로 상향해도 OOM 발생 없음
- **신뢰도:** Medium
- **근거:** 128GB 통합 메모리, 단일 모델 최대 10GB
- **위험 시:** 동적 임계값 조정 로직 적용
- **검증 방법:** Phase 1에서 메모리 프로파일링

**T3 - Thermal Monitoring API:**
- **가정:** nvidia-smi 또는 pynvml로 GPU 온도 실시간 조회 가능
- **신뢰도:** High
- **근거:** NVIDIA 표준 API
- **위험 시:** 주기적 폴링으로 대체
- **검증 방법:** 초기화 시 API 가용성 확인

### 4.2 Business Assumptions

**B1 - Processing Time Target:**
- **가정:** 4시간 목표 처리 시간이 비즈니스 요구사항 충족
- **신뢰도:** High
- **근거:** 현재 15시간 대비 4배 이상 개선
- **위험 시:** Phase 3 추가 최적화로 대응
- **검증 방법:** 사용자 피드백

**B2 - Accuracy Preservation:**
- **가정:** 성능 최적화가 분석 정확도에 영향 없음
- **신뢰도:** High
- **근거:** 동일 모델, 동일 알고리즘, 배치 처리만 변경
- **위험 시:** 정확도 검증 테스트 추가
- **검증 방법:** 기존 결과와 교차 검증

---

## 5. EARS Requirements

### 5.1 Ubiquitous Requirements (항상 활성화)

**U1 - GPU Memory State Preservation:**
시스템은 **항상** 모든 Forensic 분석 단계에서 GPU 메모리 상태를 보존해야 한다.

**세부사항:**
- SER 모델 로드 후 메모리에 유지
- 스테이지 간 불필요한 모델 언로드 방지
- 메모리 상태 로깅

**WHY:** 모델 재로딩은 45초/파일 오버헤드 발생.
**IMPACT:** 모델 유지 실패 시 처리 시간 2.3시간 증가.

---

**U2 - Thermal Monitoring:**
시스템은 **항상** GPU 온도를 모니터링하고 5초 간격으로 기록해야 한다.

**세부사항:**
- nvidia-smi 또는 pynvml 통한 온도 조회
- 온도 로그 파일 생성
- 80°C 초과 시 경고 로그

**WHY:** 열 쓰로틀링 예방 및 성능 일관성 보장.
**IMPACT:** 모니터링 실패 시 예측 불가능한 성능 저하.

---

**U3 - Memory Usage Tracking:**
시스템은 **항상** GPU 및 시스템 메모리 사용량을 추적해야 한다.

**세부사항:**
- 배치 처리 시작/종료 시 메모리 기록
- 메모리 사용량 임계값 알림
- 메모리 누수 감지

**WHY:** OOM 예방 및 최적 배치 사이즈 결정.
**IMPACT:** 추적 실패 시 메모리 관련 장애 발생.

---

**U4 - Processing Continuity:**
시스템은 **항상** 개별 파일 처리 실패가 전체 배치에 영향을 주지 않도록 해야 한다.

**세부사항:**
- 파일별 예외 처리
- 실패 파일 로깅 및 스킵
- 처리 완료 후 실패 파일 리포트

**WHY:** 단일 파일 오류로 전체 배치 중단 방지.
**IMPACT:** 격리 실패 시 전체 처리 중단.

---

### 5.2 Event-Driven Requirements (이벤트 기반)

**E1 - SER Model Loading Optimization:**
**WHEN** 첫 번째 오디오 파일 처리가 시작되면 **THEN** 시스템은 SER 모델을 GPU에 로드하고 세션 동안 유지해야 한다.

**세부사항:**
- Primary model (wav2vec2-large-robust-12-ft-emotion-msp-dim) 로드
- Secondary model (speechbrain/emotion-recognition-wav2vec2-IEMOCAP) 로드
- 모델 warmup 추론 수행

**WHY:** 모델 사전 로딩으로 45초/파일 오버헤드 제거.
**IMPACT:** 지연 로딩 시 183파일 기준 2.3시간 낭비.

---

**E2 - Thermal Throttling Activation:**
**WHEN** GPU 온도가 80°C를 초과하면 **THEN** 시스템은 배치 사이즈를 50% 감소시키고 처리 간격을 2초 추가해야 한다.

**세부사항:**
- 온도 임계값: 80°C
- 배치 사이즈 감소: 현재 값의 50%
- 쿨다운 간격: 2초
- 온도 70°C 미만 시 정상 모드 복귀

**WHY:** 열 쓰로틀링 예방으로 일관된 성능 유지.
**IMPACT:** 미대응 시 85°C 도달 후 하드웨어 강제 쓰로틀링.

---

**E3 - Memory Pressure Response:**
**WHEN** GPU 메모리 사용률이 90%를 초과하면 **THEN** 시스템은 torch.cuda.empty_cache()를 호출하고 배치 사이즈를 감소시켜야 한다.

**세부사항:**
- 메모리 임계값: 90%
- torch.cuda.empty_cache() 호출
- 배치 사이즈 25% 감소
- 메모리 해제 실패 시 CPU 폴백

**WHY:** OOM 예방 및 안정적 처리 보장.
**IMPACT:** 미대응 시 CUDA OOM으로 전체 처리 중단.

---

**E4 - Stage Pipeline Trigger:**
**WHEN** STT 단계의 첫 번째 파일 처리가 완료되면 **THEN** 시스템은 해당 파일의 Forensic 분석을 시작하면서 다음 파일의 STT를 병렬로 진행해야 한다.

**세부사항:**
- 프로듀서-컨슈머 패턴 적용
- STT 결과 큐 생성
- Forensic 컨슈머 비동기 시작
- 큐 사이즈 제한: 5파일

**WHY:** 스테이지 오버랩으로 처리 시간 50% 단축.
**IMPACT:** 순차 처리 시 리소스 유휴 시간 발생.

---

**E5 - Batch Completion Logging:**
**WHEN** 배치 처리가 완료되면 **THEN** 시스템은 성능 메트릭 리포트를 생성해야 한다.

**세부사항:**
- 총 처리 시간
- 파일별 평균 처리 시간
- GPU 활용률 통계
- 최대/평균 온도
- 실패 파일 목록

**WHY:** 최적화 효과 측정 및 병목 분석.
**IMPACT:** 리포트 부재 시 성능 개선 검증 불가.

---

### 5.3 State-Driven Requirements (상태 기반)

**S1 - GPU Memory Threshold Adjustment:**
**IF** 통합 메모리가 120GB 미만으로 사용 가능하면 **THEN** 시스템은 메모리 임계값을 30GB로 설정해야 한다.

**세부사항:**
- 기본 임계값: 30GB
- 저메모리 환경: 10GB
- 동적 조정 로직 적용

**WHY:** EdgeXpert 128GB 환경에 최적화된 임계값 설정.
**IMPACT:** 100MB 임계값 유지 시 캐시 효율 저하.

---

**S2 - Forensic GPU Mode:**
**IF** GPU가 사용 가능하고 메모리가 충분하면 **THEN** 시스템은 SER 추론을 GPU에서 수행해야 한다.

**세부사항:**
- torch.cuda.is_available() 확인
- GPU 메모리 최소 8GB 확보
- 조건 미충족 시 CPU 폴백

**WHY:** GPU 추론이 CPU 대비 10배 이상 빠름.
**IMPACT:** CPU 폴백 시 처리 시간 10배 증가.

---

**S3 - Model Cache State:**
**IF** SER 모델이 이미 로드되어 있으면 **THEN** 시스템은 재로딩을 스킵해야 한다.

**세부사항:**
- 모델 로드 상태 플래그 확인
- 로드된 모델 재사용
- 모델 버전 불일치 시 재로드

**WHY:** 불필요한 모델 재로딩 방지.
**IMPACT:** 재로딩 시 45초/파일 오버헤드.

---

**S4 - Thermal Cooldown State:**
**IF** 시스템이 열 쿨다운 모드이면 **THEN** 시스템은 GPU 온도가 70°C 미만이 될 때까지 감소된 배치 사이즈를 유지해야 한다.

**세부사항:**
- 쿨다운 진입 온도: 80°C
- 쿨다운 해제 온도: 70°C
- 상태 전이 로깅

**WHY:** 히스테리시스로 잦은 모드 전환 방지.
**IMPACT:** 히스테리시스 미적용 시 성능 불안정.

---

**S5 - Pipeline Backpressure:**
**IF** Forensic 처리 큐가 5파일 이상이면 **THEN** 시스템은 STT 처리를 일시 중지해야 한다.

**세부사항:**
- 큐 사이즈 임계값: 5파일
- STT 일시 중지
- 큐 사이즈 3 미만 시 재개

**WHY:** 메모리 과다 사용 및 처리 지연 방지.
**IMPACT:** 백프레셔 미적용 시 메모리 폭주.

---

### 5.4 Unwanted Requirements (금지 동작)

**N1 - Model Reload Per File:**
시스템은 파일마다 SER 모델을 재로드**하면 안 된다**.

**세부사항:**
- 세션 시작 시 1회 로드
- 파일 간 모델 유지
- 명시적 unload 호출만 허용

**WHY:** 파일별 재로딩은 45초/파일 오버헤드.
**IMPACT:** 재로딩 시 183파일 기준 2.3시간 낭비.

---

**N2 - GPU Thermal Damage:**
시스템은 GPU 온도가 85°C를 초과하도록 **허용하면 안 된다**.

**세부사항:**
- 85°C 도달 시 강제 처리 중단
- 70°C까지 대기 후 재개
- 하드웨어 보호 최우선

**WHY:** 85°C 초과 시 하드웨어 손상 위험.
**IMPACT:** 열 손상 시 영구적 성능 저하 또는 고장.

---

**N3 - Memory Threshold Underutilization:**
시스템은 메모리 임계값을 100MB 이하로 설정**하면 안 된다** (EdgeXpert 환경).

**세부사항:**
- 최소 임계값: 1GB
- 권장 임계값: 30GB
- 환경 자동 감지

**WHY:** 100MB는 128GB 환경에서 0.08% 활용에 불과.
**IMPACT:** 과소 설정 시 불필요한 캐시 무효화.

---

**N4 - Blocking I/O in GPU Thread:**
시스템은 GPU 추론 스레드에서 블로킹 I/O 작업을 수행**하면 안 된다**.

**세부사항:**
- 파일 I/O는 별도 스레드
- 네트워크 I/O 비동기 처리
- GPU 스레드는 연산만 수행

**WHY:** 블로킹 I/O는 GPU 유휴 시간 유발.
**IMPACT:** I/O 블로킹 시 GPU 활용률 저하.

---

### 5.5 Optional Requirements (선택적 기능)

**O1 - CUDA Graph Caching for SER:**
**가능하면** CUDA Graph를 사용하여 SER 추론 커널을 캐싱해야 한다.

**세부사항:**
- CUDA Graph 캡처
- 고정 입력 크기 추론
- 10-20% 추가 성능 향상

**WHY:** CUDA Graph는 커널 런칭 오버헤드 제거.
**IMPACT:** 미구현 시 추가 성능 향상 기회 상실.

---

**O2 - ARM64 CPU Optimization:**
**가능하면** ARM64 네이티브 최적화를 I/O 및 전처리에 적용해야 한다.

**세부사항:**
- ARM NEON SIMD 활용
- 20코어 병렬 I/O
- CPU-bound 작업 최적화

**WHY:** ARM CPU 리소스 최대 활용.
**IMPACT:** 미구현 시 CPU 활용률 저조.

---

**O3 - Real-Time Dashboard:**
**가능하면** 처리 진행 상황을 실시간 대시보드로 제공해야 한다.

**세부사항:**
- 처리 진행률 표시
- GPU/메모리/온도 모니터링
- 예상 완료 시간

**WHY:** 사용자 경험 향상 및 상태 가시성.
**IMPACT:** 미구현 시 CLI 로그만 제공.

---

## 6. Performance Requirements

### 6.1 Phase 1: Quick Wins (2-3x Faster)

**목표:** 기존 대비 2-3배 처리 속도 향상

**PR-001: 메모리 임계값 최적화**
- 현재: 100MB
- 목표: 30GB
- 효과: 캐시 효율 300배 향상

**PR-002: SER GPU 추론 활성화**
- 현재: CPU 추론
- 목표: GPU 추론
- 효과: 추론 속도 10배 향상

**PR-003: torch.cuda.empty_cache() 전략적 호출**
- 현재: 미호출 또는 과다 호출
- 목표: 배치 완료 후 1회 호출
- 효과: 메모리 단편화 감소

**측정 지표:**
- 처리 시간: 15시간 → 5-7시간
- GPU 활용률 (Forensic): 5% → 60%
- 메모리 효율: 0.08% → 25%

### 6.2 Phase 2: Integration (3-4x Faster)

**목표:** 기존 대비 3-4배 처리 속도 향상 (누적)

**PR-004: 통합 메모리 관리자**
- 현재: 스테이지별 독립 관리
- 목표: 통합 메모리 매니저
- 효과: 스테이지 간 메모리 충돌 방지

**PR-005: 스테이지별 배치 설정**
- 현재: 고정 배치 사이즈
- 목표: 스테이지별 동적 배치 사이즈
- 효과: 리소스 최적 활용

**PR-006: 열 모니터링 통합**
- 현재: 모니터링 없음
- 목표: 5초 간격 모니터링 + 동적 쓰로틀링
- 효과: 열 쓰로틀링 0%

**측정 지표:**
- 처리 시간: 5-7시간 → 4-5시간
- GPU 활용률 (Forensic): 60% → 80%
- 열 쓰로틀링 발생: 0회

### 6.3 Phase 3: Advanced (4x+ Faster)

**목표:** 기존 대비 4배 이상 처리 속도 향상 (누적)

**PR-007: 스테이지 파이프라이닝**
- 현재: 순차 처리
- 목표: STT + Forensic 오버랩
- 효과: 파이프라인 효율 50% 향상

**PR-008: CUDA Graph 캐싱 (선택)**
- 현재: 매 추론 커널 런칭
- 목표: CUDA Graph 캐싱
- 효과: 추가 10-20% 향상

**PR-009: ARM64 CPU 최적화 (선택)**
- 현재: 기본 Python I/O
- 목표: ARM NEON + 병렬 I/O
- 효과: I/O 병목 해소

**측정 지표:**
- 처리 시간: 4-5시간 → 3-4시간
- GPU 활용률 (Forensic): 80% → 90%+
- 파이프라인 효율: 50% 향상

---

## 7. Technical Constraints

### 7.1 Hardware Constraints

- **단일 GPU:** 다중 GPU 병렬 처리 불가
- **통합 메모리:** 128GB 제한, Zero-copy만 지원
- **열 관리:** 85°C 초과 시 쓰로틀링
- **폼팩터:** 미니PC 능동 냉각 한계

### 7.2 Software Constraints

- **Python 3.12+:** ARM64 호환 필수
- **CUDA 12.6:** Grace Blackwell 지원
- **WhisperX 호환:** 기존 파이프라인 인터페이스 유지
- **SpeechBrain 호환:** SER 모델 인터페이스 유지

### 7.3 Compatibility Constraints

- **기존 API 유지:** WhisperXService, SERService 인터페이스 호환
- **정확도 유지:** WER/감정 인식 정확도 변화 < 1%
- **Docker 지원:** 컨테이너 환경 정상 동작

---

## 8. Implementation Phases

### Phase 1: Quick Wins (Primary Goal)

**1.1 메모리 임계값 수정** (4시간)
- 파일: `src/voice_man/config/whisperx_config.py`
- 변경: `MEMORY_THRESHOLD_MB = 100` → `30000`
- 테스트: 메모리 프로파일링

**1.2 SER GPU 추론 활성화** (8시간)
- 파일: `src/voice_man/services/forensic/ser_service.py`
- 변경: `device="auto"` 기본값, GPU 우선 로직
- 추가: 모델 사전 로딩, 세션 캐싱
- 테스트: GPU 추론 벤치마크

**1.3 torch.cuda.empty_cache() 최적화** (4시간)
- 파일: `src/voice_man/services/e2e_test_service.py`
- 변경: 배치 완료 후 1회 호출
- 제거: 파일별 empty_cache() 호출
- 테스트: 메모리 사용 패턴

**산출물:**
- 수정된 config 파일
- GPU 가속 SER 서비스
- 벤치마크 결과 리포트

### Phase 2: Integration (Secondary Goal)

**2.1 UnifiedMemoryManager 통합** (12시간)
- 파일: `src/voice_man/services/forensic/memory_manager.py` (신규)
- 기능: 스테이지별 메모리 할당, 통합 모니터링
- 통합: SPEC-EDGEXPERT-001 UnifiedMemoryManager 재사용

**2.2 스테이지별 배치 설정** (8시간)
- 파일: `src/voice_man/config/batch_config.py` (신규)
- 기능: STT 배치, Forensic 배치 독립 설정
- 설정: GPU 메모리 기반 동적 조정

**2.3 ThermalManager 통합** (8시간)
- 파일: `src/voice_man/services/forensic/thermal_manager.py` (신규)
- 기능: GPU 온도 모니터링, 동적 쓰로틀링
- 통합: SPEC-EDGEXPERT-001 ThermalManager 재사용

**산출물:**
- 통합 메모리 관리자
- 스테이지별 배치 설정
- 열 관리 시스템

### Phase 3: Advanced (Optional Goal)

**3.1 스테이지 파이프라이닝** (16시간)
- 파일: `src/voice_man/services/forensic/pipeline_orchestrator.py` (신규)
- 기능: 프로듀서-컨슈머 패턴, 비동기 큐
- 구현: asyncio 기반 파이프라인

**3.2 CUDA Graph 캐싱** (12시간)
- 파일: `src/voice_man/services/forensic/cuda_graph_cache.py` (신규)
- 기능: SER 추론 CUDA Graph 캡처
- 조건: 고정 입력 크기

**3.3 ARM64 CPU 최적화** (8시간)
- 파일: `src/voice_man/services/forensic/arm_optimizer.py` (신규)
- 기능: ARM NEON 활용, 병렬 I/O
- 범위: 전처리, 파일 로딩

**산출물:**
- 파이프라인 오케스트레이터
- CUDA Graph 캐시 (선택)
- ARM64 최적화 모듈 (선택)

---

## 9. Risk Mitigation

### High Risk

**R1 - SER Model GPU Compatibility:**
- **위험:** SER 모델 GPU 이식 시 정확도 저하
- **완화:** Phase 1에서 정확도 벤치마크, CPU 폴백 유지
- **대안:** 정확도 저하 시 CPU 모드 유지

**R2 - Memory OOM:**
- **위험:** 30GB 임계값에서도 OOM 발생
- **완화:** 점진적 임계값 증가, 동적 조정 로직
- **대안:** 10GB부터 시작하여 안정화

### Medium Risk

**R3 - Thermal Throttling:**
- **위험:** 열 관리에도 불구하고 쓰로틀링 발생
- **완화:** 보수적 임계값 (75°C), 적극적 쿨다운
- **대안:** 배치 사이즈 대폭 감소

**R4 - Pipeline Deadlock:**
- **위험:** 프로듀서-컨슈머 패턴에서 데드락
- **완화:** 타임아웃 설정, 백프레셔 로직
- **대안:** 순차 처리 폴백

### Low Risk

**R5 - ARM64 Optimization Complexity:**
- **위험:** ARM64 최적화 구현 복잡도
- **완화:** 선택적 구현, 기본 Python 유지
- **대안:** Phase 3 스킵

---

## 10. Traceability Matrix

| Requirement ID | Component | Phase | Test Case |
|----------------|-----------|-------|-----------|
| U1 | SERService | 1 | TC-GPU-001 |
| U2 | ThermalManager | 2 | TC-THM-001 |
| U3 | MemoryManager | 2 | TC-MEM-001 |
| U4 | E2ETestService | 1 | TC-ERR-001 |
| E1 | SERService | 1 | TC-GPU-002 |
| E2 | ThermalManager | 2 | TC-THM-002 |
| E3 | MemoryManager | 2 | TC-MEM-002 |
| E4 | PipelineOrchestrator | 3 | TC-PIP-001 |
| E5 | E2ETestService | 1 | TC-RPT-001 |
| S1 | MemoryManager | 2 | TC-MEM-003 |
| S2 | SERService | 1 | TC-GPU-003 |
| S3 | SERService | 1 | TC-GPU-004 |
| S4 | ThermalManager | 2 | TC-THM-003 |
| S5 | PipelineOrchestrator | 3 | TC-PIP-002 |
| N1 | SERService | 1 | TC-GPU-005 |
| N2 | ThermalManager | 2 | TC-THM-004 |
| N3 | MemoryManager | 2 | TC-MEM-004 |
| N4 | PipelineOrchestrator | 3 | TC-PIP-003 |
| O1 | CUDAGraphCache | 3 | TC-OPT-001 |
| O2 | ARMOptimizer | 3 | TC-OPT-002 |
| O3 | Dashboard | 3 | TC-OPT-003 |
| PR-001 | Config | 1 | TC-PERF-001 |
| PR-002 | SERService | 1 | TC-PERF-002 |
| PR-003 | E2ETestService | 1 | TC-PERF-003 |
| PR-004 | MemoryManager | 2 | TC-PERF-004 |
| PR-005 | BatchConfig | 2 | TC-PERF-005 |
| PR-006 | ThermalManager | 2 | TC-PERF-006 |
| PR-007 | PipelineOrchestrator | 3 | TC-PERF-007 |

---

## 11. References

### Related SPECs

- **SPEC-FORENSIC-001**: 범죄 프로파일링 기반 음성 포렌식 분석 시스템 (완료)
- **SPEC-GPUOPT-001**: WhisperX Pipeline GPU Optimization (계획)
- **SPEC-EDGEXPERT-001**: MSI EdgeXpert WhisperX GPU 최적화 (계획)
- **SPEC-PARALLEL-001**: GPU 기반 병렬처리 최적화 시스템 (완료)

### Technical Documentation

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA pynvml Documentation](https://pypi.org/project/nvidia-ml-py/)
- [SpeechBrain Documentation](https://speechbrain.github.io/)
- [HuggingFace Transformers GPU Usage](https://huggingface.co/docs/transformers/perf_train_gpu_one)

### Academic Papers

- Schuller, B. W. (2018). "Speech Emotion Recognition: Two Decades in a Nutshell"
- PyTorch Team. (2024). "CUDA Graph Trees for Dynamic Workloads"

---

## 12. Appendix

### A. Terminology

- **SER (Speech Emotion Recognition):** 음성 기반 감정 인식
- **Forensic Pipeline:** 포렌식 분석 파이프라인 (SER + 스트레스 분석 + 스코어링)
- **Stage Pipelining:** 파이프라인 스테이지 간 병렬 처리
- **Thermal Throttling:** 열로 인한 성능 제한
- **Backpressure:** 다운스트림 과부하 시 업스트림 제어
- **CUDA Graph:** GPU 커널 시퀀스 사전 캡처 및 재사용

### B. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-10 | 지니 | Initial SPEC creation |

---

**문서 끝**
