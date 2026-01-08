# SPEC-PARALLEL-001 인수 기준

## 개요

본 문서는 GPU 기반 병렬처리 최적화 시스템의 인수 조건 및 테스트 시나리오를 정의합니다.

---

## Phase별 성능 기준

### Phase 1: 즉시 최적화 (4배 향상)

#### 성능 목표
- **처리 시간**: 15분 이내 (183개 파일 기준)
- **CPU 활용률**: 80% 이상
- **메모리 사용량**: 95GB 미만
- **처리 성공률**: 100%

#### 측정 방법
```bash
# 처리 시간 측정
time python scripts/process_audio_files.py --input data/audio/

# CPU 활용률 모니터링
htop

# 메모리 사용량 확인
free -h
```

#### 합격 조건
- ✅ 183개 파일을 15분 이내에 처리 완료
- ✅ 처리 중 CPU 활용률 평균 80% 이상
- ✅ 메모리 사용량 피크 95GB 미만
- ✅ 모든 파일 처리 성공 (실패율 0%)

---

### Phase 2: GPU 활성화 (20배 향상)

#### 성능 목표
- **처리 시간**: 3분 이내 (183개 파일 기준)
- **GPU 활용률**: 70% 이상
- **GPU 메모리 오류**: 0건
- **WER 변화**: < 1% (정확도 유지)

#### 측정 방법
```bash
# GPU 활용률 모니터링
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# WER 비교
python tests/compare_wer.py --baseline results/phase1/ --current results/phase2/
```

#### 합격 조건
- ✅ 183개 파일을 3분 이내에 처리 완료
- ✅ 처리 중 GPU 활용률 평균 70% 이상
- ✅ CUDA Out of Memory 오류 0건 발생
- ✅ WER 증가율 1% 미만 (기준: Phase 1 결과)

---

### Phase 3: 완전 파이프라인 (50배 향상)

#### 성능 목표
- **처리 시간**: 1.2분 이내 (183개 파일 기준)
- **GPU 활용률**: 90% 이상
- **화자 분리 정확도**: 90% 이상 (DER 기준)
- **메모리 누수**: 0건

#### 측정 방법
```bash
# 화자 분리 정확도 (DER: Diarization Error Rate)
python tests/evaluate_diarization.py --ground-truth data/labels/ --predictions results/phase3/

# 메모리 누수 확인 (10회 반복 테스트)
for i in {1..10}; do
    python scripts/process_audio_files.py --input data/audio/sample_10/
    free -m
done
```

#### 합격 조건
- ✅ 183개 파일을 1.5분 이내에 처리 완료 (목표 1.2분)
- ✅ 처리 중 GPU 활용률 평균 90% 이상
- ✅ DER (Diarization Error Rate) 10% 이하 (화자 분리 정확도 90%)
- ✅ 10회 반복 테스트 후 메모리 사용량 증가 < 5%

---

## 테스트 시나리오

### Scenario 1: 정상 처리 (Phase 1)

#### Given (전제 조건)
```
- 183개 m4a 파일이 data/audio/ 디렉토리에 존재
- 시스템 메모리 114GB 사용 가능
- CPU 20코어 사용 가능
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/ --workers 18 --batch-size 15
```

#### Then (예상 결과)
```
✅ 15분 이내에 모든 파일 처리 완료
✅ analysis-results/ 디렉토리에 183개 결과 파일 생성
✅ 처리 로그에 오류 0건
✅ 원본 파일 무결성 유지 (MD5 체크섬 일치)
✅ performance_report.json 생성 및 지표 기록
```

---

### Scenario 2: GPU 활용 처리 (Phase 2)

#### Given (전제 조건)
```
- CUDA 12.3 이상 설치
- NVIDIA GB10 GPU 사용 가능
- faster-whisper 1.0.3 이상 설치
- 10개 샘플 오디오 파일 준비 (data/audio/sample_10/)
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/sample_10/ --device cuda --batch-size 20
```

#### Then (예상 결과)
```
✅ 10초 이내에 모든 파일 처리 완료
✅ GPU 활용률 로그에서 평균 70% 이상 확인
✅ GPU 메모리 사용량 16GB 미만
✅ CUDA 오류 0건
✅ 전사 정확도 Phase 1 대비 WER 변화 < 1%
```

---

### Scenario 3: GPU 메모리 부족 복구 (Phase 2)

#### Given (전제 조건)
```
- GPU 메모리 8GB (제한된 환경 시뮬레이션)
- 초기 배치 크기 32 (의도적으로 큰 값)
- 20개 오디오 파일 준비
```

#### When (실행 조건)
```bash
# GPU 메모리 제한 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python scripts/process_audio_files.py --input data/audio/sample_20/ --batch-size 32
```

#### Then (예상 결과)
```
✅ CUDA Out of Memory 오류 발생 시 자동 배치 크기 감소 (32 → 16 → 8 → 4)
✅ 배치 크기 조정 후 처리 성공
✅ 로그에 "Reducing batch size to X" 메시지 기록
✅ 모든 파일 최종 처리 완료
```

---

### Scenario 4: CPU 폴백 모드 (Phase 2)

#### Given (전제 조건)
```
- GPU 사용 불가 환경 (CUDA 미설치 또는 GPU 없음)
- 10개 샘플 오디오 파일
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/sample_10/ --device auto
```

#### Then (예상 결과)
```
✅ GPU 감지 실패 로그 출력
✅ CPU 모드로 자동 전환 (device="cpu", compute_type="int8")
✅ 처리 완료 (속도는 느리지만 정상 동작)
✅ 로그에 "GPU not available, using CPU fallback" 메시지 기록
```

---

### Scenario 5: WhisperX 전체 파이프라인 (Phase 3)

#### Given (전제 조건)
```
- whisperx 3.1.5 이상 설치
- pyannote-audio 3.1.1 이상 설치
- HF_TOKEN 환경 변수 설정
- 5개 샘플 오디오 파일 (다중 화자 포함)
```

#### When (실행 조건)
```bash
export HF_TOKEN="your_token"
python scripts/process_audio_files.py --input data/audio/sample_5/ --use-whisperx --diarization
```

#### Then (예상 결과)
```
✅ 30초 이내에 모든 파일 처리 완료
✅ 결과 JSON에 "speaker" 필드 포함
✅ DER (Diarization Error Rate) 10% 이하
✅ GPU 활용률 90% 이상
✅ 타임스탬프 정확도 ±0.05초 이내 (WAV2VEC2 정렬)
```

---

### Scenario 6: 대규모 배치 처리 (Phase 3)

#### Given (전제 조건)
```
- 183개 전체 오디오 파일
- WhisperX 파이프라인 활성화
- GPU 메모리 16GB 사용 가능
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/ --use-whisperx --batch-size 16
```

#### Then (예상 결과)
```
✅ 1.5분 이내에 183개 파일 모두 처리 완료
✅ 처리 성공률 100% (실패 0건)
✅ performance_report.json에 다음 지표 기록:
   - total_time_minutes: < 1.5
   - avg_time_per_file_seconds: < 0.5
   - gpu_utilization: > 90%
   - failed_files: []
✅ 메모리 사용량 95GB 미만 유지
```

---

### Scenario 7: 메모리 누수 방지 (Phase 3)

#### Given (전제 조건)
```
- 10개 샘플 파일
- 10회 연속 처리 테스트
```

#### When (실행 조건)
```bash
for i in {1..10}; do
    echo "Run $i"
    python scripts/process_audio_files.py --input data/audio/sample_10/
    free -m | grep Mem | awk '{print $3}'
done
```

#### Then (예상 결과)
```
✅ 1회차와 10회차 메모리 사용량 차이 < 5%
✅ GPU 메모리 사용량 누적 증가 없음
✅ 프로세스 종료 후 메모리 해제 확인
✅ 로그에 GC (Garbage Collection) 트리거 기록
```

---

### Scenario 8: 원본 파일 무결성 검증 (모든 Phase)

#### Given (전제 조건)
```
- 10개 샘플 파일의 MD5 체크섬 사전 계산
- checksums.txt 파일에 저장
```

#### When (실행 조건)
```bash
# 처리 전 체크섬
md5sum data/audio/sample_10/* > checksums_before.txt

# 처리 실행
python scripts/process_audio_files.py --input data/audio/sample_10/

# 처리 후 체크섬
md5sum data/audio/sample_10/* > checksums_after.txt
```

#### Then (예상 결과)
```
✅ checksums_before.txt와 checksums_after.txt 동일
✅ 원본 파일 크기 변경 없음
✅ 원본 파일 수정 시간 (mtime) 변경 없음
✅ 분석 결과는 analysis-results/ 디렉토리에만 저장
```

---

### Scenario 9: 실패 파일 재시도 (모든 Phase)

#### Given (전제 조건)
```
- 10개 파일 중 2개는 의도적으로 손상된 파일
- 재시도 로직 활성화 (최대 3회)
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/corrupted_10/ --max-retries 3
```

#### Then (예상 결과)
```
✅ 정상 파일 8개는 1회 시도로 성공
✅ 손상된 파일 2개는 3회 재시도 후 실패 기록
✅ failed_files.json에 실패 파일 목록 및 오류 원인 저장
✅ 처리 중단 없이 전체 작업 완료
✅ 로그에 "Retrying file X (attempt Y/3)" 메시지 기록
```

---

### Scenario 10: 성능 리포트 생성 (Phase 3)

#### Given (전제 조건)
```
- 183개 파일 처리 완료
```

#### When (실행 조건)
```bash
python scripts/process_audio_files.py --input data/audio/ --generate-report
```

#### Then (예상 결과)
```
✅ performance_report.json 파일 생성
✅ 리포트에 다음 필드 포함:
   {
     "total_files": 183,
     "successful": 183,
     "failed": 0,
     "total_time_minutes": 1.2,
     "avg_time_per_file_seconds": 0.39,
     "gpu_utilization": 92,
     "cpu_utilization": 85,
     "memory_peak_mb": 92000,
     "failed_files": []
   }
✅ 터미널에 요약 통계 출력
```

---

## 품질 기준

### 기능 품질
- **처리 완료율**: 100% (실패율 0%)
- **WER 유지**: 기준 대비 변화 < 1%
- **화자 분리 정확도**: DER 10% 이하
- **타임스탬프 정확도**: ±0.05초 이내

### 성능 품질
- **처리 속도**: 1.5분 이내 (183개 파일 기준)
- **GPU 활용률**: 85% 이상
- **메모리 효율**: 시스템 RAM의 80% 미만 사용
- **메모리 누수**: 10회 반복 테스트 후 증가율 < 5%

### 안정성 품질
- **오류 복구**: GPU OOM 발생 시 자동 배치 크기 조정
- **CPU 폴백**: GPU 미사용 시 CPU 모드 정상 동작
- **재시도 로직**: 일시적 오류 최대 3회 재시도
- **원본 보존**: 처리 전후 파일 무결성 100% 유지

---

## TRUST 5 품질 게이트

### Test-first (테스트 우선)
- ✅ 테스트 커버리지 85% 이상
- ✅ 모든 시나리오에 대한 자동화 테스트 구현
- ✅ CI/CD 파이프라인에 통합

### Readable (가독성)
- ✅ 코드 린터 (ruff, black) 통과
- ✅ 함수/변수명 명확성 확인
- ✅ 복잡도 메트릭 (Cyclomatic Complexity) < 10

### Unified (일관성)
- ✅ 코드 스타일 가이드 준수 (PEP 8)
- ✅ Import 정렬 (isort) 적용
- ✅ 타입 힌트 일관성 (mypy) 검증

### Secured (보안)
- ✅ 원본 파일 읽기 전용 접근
- ✅ 환경 변수로 민감 정보 관리 (HF_TOKEN)
- ✅ 로그에 개인정보 미포함

### Trackable (추적 가능)
- ✅ Git 커밋 메시지 규칙 준수
- ✅ SPEC-PARALLEL-001 태그 포함
- ✅ 변경 이력 문서화

---

## 회귀 테스트

### Phase 1 회귀 테스트
- **목적**: Phase 2/3 구현 후에도 CPU 모드 정상 동작 확인
- **방법**: GPU 비활성화 후 Phase 1 시나리오 재실행
- **합격 조건**: 15분 이내 처리 완료

### Phase 2 회귀 테스트
- **목적**: Phase 3 구현 후에도 faster-whisper 단독 동작 확인
- **방법**: WhisperX 비활성화 후 Phase 2 시나리오 재실행
- **합격 조건**: 3분 이내 처리 완료

---

## 최종 인수 조건

### 필수 조건 (Must Have)
- ✅ Phase 3 성능 목표 달성 (1.5분 이내)
- ✅ 모든 테스트 시나리오 통과
- ✅ TRUST 5 품질 게이트 통과
- ✅ 원본 파일 무결성 100% 보장

### 권장 조건 (Should Have)
- ✅ 성능 리포트 자동 생성
- ✅ 로그 구조화 및 분석 도구 제공
- ✅ 사용자 가이드 문서 작성

### 선택 조건 (Nice to Have)
- ⏸ Multi-GPU 지원 (O1 요구사항)
- ⏸ Celery 분산 처리 (O2 요구사항)
- ⏸ Web UI 대시보드

---

## 문서화 요구사항

### 필수 문서
- ✅ README.md 업데이트 (설치 및 사용법)
- ✅ CHANGELOG.md 업데이트 (SPEC-PARALLEL-001 변경 사항)
- ✅ API 문서 (함수/클래스 docstring)

### 권장 문서
- ✅ 성능 벤치마크 리포트
- ✅ 트러블슈팅 가이드
- ✅ GPU 환경 설정 가이드

---

## 배포 전 체크리스트

- [ ] 모든 테스트 시나리오 통과 확인
- [ ] TRUST 5 품질 게이트 통과 확인
- [ ] 코드 리뷰 완료
- [ ] 성능 벤치마크 측정 및 문서화
- [ ] 사용자 가이드 작성 완료
- [ ] Git 태그 생성 (v2.0.0-parallel)
- [ ] CHANGELOG.md 업데이트
- [ ] 프로덕션 환경 배포 테스트

---

**문서 끝**
