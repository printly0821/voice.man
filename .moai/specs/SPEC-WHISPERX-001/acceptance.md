# SPEC-WHISPERX-001 인수 조건

## 개요

**SPEC ID**: SPEC-WHISPERX-001
**제목**: WhisperX 통합 파이프라인 시스템
**우선순위**: HIGH
**상태**: Planned

---

## 기능적 인수 조건

### AC-F1: WhisperX 파이프라인 초기화

**Given** 유효한 Hugging Face 토큰이 환경 변수로 설정되어 있을 때
**When** WhisperXPipeline 클래스가 초기화되면
**Then** Whisper, WAV2VEC2, Pyannote 모델이 모두 GPU에 로드되어야 한다

**검증 방법**:
```python
def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    import os
    os.environ["HF_TOKEN"] = "valid_token"

    pipeline = WhisperXPipeline()

    assert pipeline._whisper_model is not None
    assert pipeline._align_model is not None
    assert pipeline._diarize_model is not None
    assert pipeline.device == "cuda"
```

---

### AC-F2: 오디오 전사 및 word-level 타임스탬프

**Given** 유효한 오디오 파일이 입력되었을 때
**When** 파이프라인이 처리를 완료하면
**Then** 각 단어에 100ms 이내 정확도의 타임스탬프가 할당되어야 한다

**검증 방법**:
```python
def test_word_level_timestamps():
    """Word-level 타임스탬프 테스트"""
    pipeline = WhisperXPipeline()
    result = pipeline.process("test_audio.wav")

    assert len(result.word_segments) > 0

    for word_seg in result.word_segments:
        assert "start" in word_seg
        assert "end" in word_seg
        assert "word" in word_seg
        assert word_seg["end"] > word_seg["start"]
        # 단어 길이 합리성 검증 (10초 미만)
        assert word_seg["end"] - word_seg["start"] < 10.0
```

---

### AC-F3: 화자 분리 및 ID 할당

**Given** 2인 이상 대화가 포함된 오디오 파일이 입력되었을 때
**When** 파이프라인이 화자 분리를 완료하면
**Then** 각 세그먼트에 화자 ID가 할당되고 동일 화자에게 일관된 ID가 유지되어야 한다

**검증 방법**:
```python
def test_speaker_diarization():
    """화자 분리 테스트"""
    pipeline = WhisperXPipeline()
    result = pipeline.process("two_speakers_audio.wav", num_speakers=2)

    # 화자가 2명인지 확인
    assert len(result.speakers) == 2

    # 모든 세그먼트에 화자 ID 할당 확인
    for segment in result.segments:
        assert "speaker" in segment
        assert segment["speaker"] in ["SPEAKER_00", "SPEAKER_01"]
```

---

### AC-F4: 화자별 발화 통계

**Given** 화자 분리가 완료된 결과가 있을 때
**When** 화자 통계가 요청되면
**Then** 화자별 발화 시간, 비율, 턴 수가 정확하게 계산되어야 한다

**검증 방법**:
```python
def test_speaker_statistics():
    """화자 통계 테스트"""
    pipeline = WhisperXPipeline()
    result = pipeline.process("conversation_audio.wav")

    stats = result.speaker_stats

    assert "total_speakers" in stats
    assert "total_duration" in stats
    assert "speaker_details" in stats

    total_percentage = sum(
        d["percentage"] for d in stats["speaker_details"]
    )
    # 총 비율이 100%에 근접해야 함
    assert abs(total_percentage - 100.0) < 1.0
```

---

### AC-F5: 기존 인터페이스 호환성

**Given** 기존 DiarizationService 인터페이스가 있을 때
**When** WhisperX로 내부 구현이 교체되면
**Then** 기존 테스트 케이스가 모두 통과해야 한다

**검증 방법**:
```python
def test_backward_compatibility():
    """기존 인터페이스 호환성 테스트"""
    service = DiarizationService()

    # 기존 메서드 시그니처 확인
    result = await service.diarize_speakers("test_audio.wav", num_speakers=2)

    # 기존 반환 타입 확인
    assert isinstance(result, DiarizationResult)
    assert hasattr(result, "speakers")
    assert hasattr(result, "total_duration")
    assert hasattr(result, "num_speakers")
```

---

## 비기능적 인수 조건

### AC-NF1: 처리 성능

**Given** 183개의 m4a 오디오 파일이 있을 때
**When** 배치 처리가 실행되면
**Then** 총 처리 시간이 1.5분(90초) 이내여야 한다

**검증 방법**:
```python
def test_processing_performance():
    """처리 성능 테스트"""
    import time

    audio_files = glob.glob("test_data/*.m4a")
    assert len(audio_files) == 183

    service = WhisperXService()

    start_time = time.time()
    results = service.process_batch(audio_files)
    end_time = time.time()

    total_time = end_time - start_time
    assert total_time < 90  # 1.5분 이내

    # 성공률 100%
    success_count = sum(1 for r in results if r["status"] == "success")
    assert success_count == 183
```

---

### AC-NF2: GPU 활용률

**Given** GPU가 사용 가능한 환경에서
**When** 파이프라인이 실행되면
**Then** GPU 활용률이 평균 85% 이상이어야 한다

**검증 방법**:
```python
def test_gpu_utilization():
    """GPU 활용률 테스트"""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    utilizations = []

    # 처리 중 GPU 사용률 모니터링
    for _ in range(10):
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilizations.append(util.gpu)
        time.sleep(1)

    avg_utilization = sum(utilizations) / len(utilizations)
    assert avg_utilization >= 85
```

---

### AC-NF3: 메모리 안정성

**Given** GPU 메모리가 제한된 환경에서
**When** 대량 파일이 처리되면
**Then** GPU 메모리 사용률이 95%를 초과하지 않고 OOM이 발생하지 않아야 한다

**검증 방법**:
```python
def test_memory_stability():
    """메모리 안정성 테스트"""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    service = WhisperXService()

    # 다수 파일 처리
    for audio_file in audio_files[:50]:
        result = service.process_audio(audio_file)

        # 메모리 사용률 확인
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_ratio = info.used / info.total

        assert usage_ratio < 0.95, f"메모리 사용률 초과: {usage_ratio:.1%}"
```

---

### AC-NF4: 테스트 커버리지

**Given** 전체 WhisperX 관련 코드가 있을 때
**When** 테스트 스위트가 실행되면
**Then** 코드 커버리지가 85% 이상이어야 한다

**검증 방법**:
```bash
pytest tests/ --cov=src/voice_man --cov-report=term-missing

# 예상 출력:
# TOTAL                                        1234    123    90%
# 커버리지가 85% 이상이어야 함
```

---

## 에지 케이스 테스트

### AC-EC1: Hugging Face 토큰 없음

**Given** HF_TOKEN 환경 변수가 설정되지 않았을 때
**When** 파이프라인 초기화가 시도되면
**Then** 명확한 오류 메시지와 함께 ValueError가 발생해야 한다

**검증 방법**:
```python
def test_missing_hf_token():
    """HF 토큰 누락 테스트"""
    import os
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    with pytest.raises(ValueError) as exc_info:
        pipeline = WhisperXPipeline()

    assert "HF_TOKEN" in str(exc_info.value)
```

---

### AC-EC2: 지원하지 않는 오디오 포맷

**Given** 지원하지 않는 포맷(예: .avi)의 파일이 입력되었을 때
**When** 오디오 변환이 시도되면
**Then** 명확한 오류 메시지와 함께 ValueError가 발생해야 한다

**검증 방법**:
```python
def test_unsupported_format():
    """지원하지 않는 포맷 테스트"""
    converter = AudioConverterService()

    with pytest.raises(ValueError) as exc_info:
        converter.convert_to_wav("video.avi")

    assert "지원하지 않는 포맷" in str(exc_info.value)
```

---

### AC-EC3: 빈 오디오 파일

**Given** 무음만 포함된 오디오 파일이 입력되었을 때
**When** 파이프라인이 처리를 시도하면
**Then** 빈 세그먼트 목록과 함께 성공적으로 완료되어야 한다

**검증 방법**:
```python
def test_silent_audio():
    """무음 오디오 테스트"""
    pipeline = WhisperXPipeline()
    result = pipeline.process("silent_audio.wav")

    # 오류 없이 완료
    assert result is not None
    # 세그먼트가 비어있거나 매우 적음
    assert len(result.segments) == 0 or len(result.text.strip()) < 10
```

---

### AC-EC4: 긴 오디오 파일 (30분 초과)

**Given** 30분을 초과하는 오디오 파일이 입력되었을 때
**When** 파이프라인이 처리를 시도하면
**Then** 10분 청크로 분할되어 처리되고 결과가 병합되어야 한다

**검증 방법**:
```python
def test_long_audio_chunking():
    """긴 오디오 청크 분할 테스트"""
    chunker = AudioChunkerService()

    # 45분 오디오
    assert chunker.should_chunk("45min_audio.wav") == True

    chunks = chunker.split_audio("45min_audio.wav")

    # 5개 청크로 분할됨 (45분 / 10분 + 오버랩)
    assert len(chunks) >= 4

    # 각 청크가 10분 이하
    for chunk_path, start, end in chunks:
        assert end - start <= 600 + 30  # 청크 길이 + 오버랩
```

---

### AC-EC5: GPU 메모리 부족

**Given** GPU 메모리가 부족한 상황에서
**When** 모델 로딩이 시도되면
**Then** 순차 로딩 모드로 전환되어 처리가 완료되어야 한다

**검증 방법**:
```python
def test_low_gpu_memory():
    """GPU 메모리 부족 테스트"""
    # 메모리 사용률 70% 이상 상황 시뮬레이션
    pipeline = WhisperXPipeline()
    pipeline._check_memory_and_adjust()

    if pipeline._sequential_loading:
        # 순차 로딩 모드에서도 정상 처리
        result = pipeline.process("test_audio.wav")
        assert result is not None
```

---

## 보안 인수 조건

### AC-SEC1: 토큰 하드코딩 금지

**Given** 전체 소스 코드가 있을 때
**When** 코드 검사가 수행되면
**Then** Hugging Face 토큰이 하드코딩되어 있지 않아야 한다

**검증 방법**:
```python
def test_no_hardcoded_tokens():
    """토큰 하드코딩 검사"""
    import re

    source_files = glob.glob("src/**/*.py", recursive=True)

    hf_token_pattern = r"hf_[A-Za-z0-9]{20,}"

    for file_path in source_files:
        with open(file_path) as f:
            content = f.read()
            matches = re.findall(hf_token_pattern, content)
            assert len(matches) == 0, f"토큰 하드코딩 발견: {file_path}"
```

---

### AC-SEC2: 임시 파일 정리

**Given** 오디오 변환이 수행된 후
**When** 처리가 완료되면
**Then** 모든 임시 파일이 삭제되어야 한다

**검증 방법**:
```python
def test_temp_file_cleanup():
    """임시 파일 정리 테스트"""
    import tempfile

    temp_dir = tempfile.gettempdir()
    initial_count = len(os.listdir(temp_dir))

    with AudioConverterService() as converter:
        converter.convert_to_wav("test.m4a")

    final_count = len(os.listdir(temp_dir))

    # 임시 파일 증가 없음
    assert final_count <= initial_count
```

---

## 통합 테스트 시나리오

### 시나리오 1: 전체 파이프라인 실행

```gherkin
Feature: WhisperX 전체 파이프라인

  Scenario: 단일 파일 전체 처리
    Given 5분 길이의 2인 대화 m4a 파일이 있다
    And HF_TOKEN 환경 변수가 설정되어 있다
    When 파이프라인으로 처리한다
    Then 전사 텍스트가 생성된다
    And 모든 단어에 타임스탬프가 할당된다
    And 2명의 화자가 식별된다
    And 각 세그먼트에 화자 ID가 할당된다
    And 화자별 발화 통계가 생성된다
```

---

### 시나리오 2: 배치 처리

```gherkin
Feature: 배치 오디오 처리

  Scenario: 다중 파일 배치 처리
    Given 10개의 m4a 파일이 있다
    When 배치 처리를 실행한다
    Then 모든 파일이 성공적으로 처리된다
    And 총 처리 시간이 개별 처리 시간 합보다 짧다
    And 진행률이 실시간으로 업데이트된다
```

---

### 시나리오 3: 오류 복구

```gherkin
Feature: 오류 복구

  Scenario: 일부 파일 실패 시 계속 처리
    Given 10개의 파일 중 1개가 손상되어 있다
    When 배치 처리를 실행한다
    Then 손상된 파일은 실패로 기록된다
    And 나머지 9개 파일은 성공적으로 처리된다
    And 실패한 파일 목록이 리포트에 포함된다
```

---

## 품질 게이트

### TRUST 5 체크리스트

- [ ] **Test-first**: 테스트 커버리지 85% 이상
- [ ] **Readable**: 코드 스타일 가이드 준수 (ruff 검사 통과)
- [ ] **Unified**: 일관된 네이밍 컨벤션 및 포맷
- [ ] **Secured**: 토큰 하드코딩 없음, 임시 파일 정리
- [ ] **Trackable**: 성능 메트릭 로깅, 처리 리포트 생성

---

## 완료 정의 (Definition of Done)

1. **기능 완료**
   - [ ] 모든 기능적 인수 조건 통과
   - [ ] 모든 비기능적 인수 조건 통과
   - [ ] 모든 에지 케이스 테스트 통과
   - [ ] 모든 보안 인수 조건 통과

2. **품질 완료**
   - [ ] 테스트 커버리지 85% 이상
   - [ ] ruff 린터 경고 0개
   - [ ] TRUST 5 품질 게이트 통과

3. **문서 완료**
   - [ ] API 문서 업데이트
   - [ ] README 업데이트 (설치 및 사용법)
   - [ ] 성능 리포트 작성

4. **배포 완료**
   - [ ] 의존성 업데이트 (pyproject.toml)
   - [ ] 환경 변수 문서화
   - [ ] 기존 테스트 회귀 없음

---

**문서 끝**
