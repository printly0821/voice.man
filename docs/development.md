# 개발 가이드

Voice Man 프로젝트에 기여하고 개발 환경을 설정하는 방법을 안내합니다.

## 개발 환경 설정

### 1. 포크 및 클론

```bash
# 저장소 포크 (GitHub 웹 인터페이스)

# 로컬에 클론
git clone https://github.com/yourusername/voice.man.git
cd voice.man
```

### 2. 가상 환경 생성

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate  # Windows
```

### 3. 개발 의존성 설치

```bash
pip install -e ".[dev]"
```

### 4. pre-commit 후크 설정

```bash
# pre-commit 설치
pip install pre-commit

# 후크 활성화
pre-commit install
```

### 5. 개발 서버 시작

```bash
uvicorn voice_man.main:app --reload --port 8000
```

## 프로젝트 구조

```
voice.man/
├── src/
│   └── voice_man/
│       ├── __init__.py
│       ├── main.py                 # FastAPI 애플리케이션
│       ├── schemas.py              # Pydantic 스키마
│       ├── models/
│       │   ├── __init__.py
│       │   ├── database.py         # SQLAlchemy 모델
│       │   └── diarization.py      # 화자 분리 모델
│       ├── services/
│       │   ├── __init__.py
│       │   ├── audio_service.py    # 오디오 처리 서비스
│       │   ├── stt_service.py      # STT 서비스
│       │   └── diarization_service.py  # 화자 분리 서비스
│       └── api/
│           └── v1/
│               ├── __init__.py
│               ├── audio.py        # 오디오 관련 엔드포인트
│               └── analysis.py     # 분석 관련 엔드포인트
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixture
│   ├── unit/                       # 단위 테스트
│   ├── integration/                # 통합 테스트
│   └── acceptance/                 # 인수 테스트
├── alembic/                        # 데이터베이스 마이그레이션
├── docs/                           # 문서
├── .moai/                          # MoAI 설정
├── pyproject.toml                  # 프로젝트 설정
├── alembic.ini                     # Alembic 설정
└── .env.example                    # 환경 변수 예시
```

## 코딩 표준

### Python 스타일 가이드

**줄 길이**: 최대 100자

```python
# 좋은 예
def process_audio_file(
    file_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioMetadata:
    """오디오 파일을 처리합니다."""
    ...

# 나쁜 예
def process_audio_file(file_path: str, sample_rate: int = 16000, channels: int = 1) -> AudioMetadata:
    ...
```

**타입 힌트**: 모든 함수에 타입 힌트 포함

```python
# 좋은 예
async def upload_audio(file: UploadFile) -> AudioUploadResponse:
    ...

# 나쁜 예
async def upload_audio(file):
    ...
```

**Docstring**: Google 스타일

```python
def compute_sha256_hash(content: bytes) -> str:
    """
    콘텐츠의 SHA-256 해시를 계산합니다.

    Args:
        content: 해시를 계산할 바이너리 데이터

    Returns:
        SHA-256 해시 문자열 (16진수)

    Example:
        >>> compute_sha256_hash(b"hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    ...
```

**변수 이름**: 영어, 명확하고 의미있는 이름

```python
# 좋은 예
speaker_count = len(speakers)
transcription_confidence = 0.95

# 나쁜 예
n = len(s)  # 의미 불명
sc = 0.95  # 약어 사용
```

### 임포트 순서

1. 표준 라이브러리
2. 서드파티 라이브러리
3. 로컬 임포트

```python
# 1. 표준 라이브러리
from pathlib import Path
from typing import List, Optional

# 2. 서드파티 라이브러리
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 3. 로컬 임포트
from voice_man.models.database import AudioFile
from voice_man.services import AudioService
```

## 테스트 작성

### 단위 테스트

```python
# tests/unit/test_services.py
import pytest
from voice_man.services.diarization_service import DiarizationService

@pytest.fixture
def diarization_service():
    """화자 분리 서비스 픽스처"""
    return DiarizationService()

def test_diarize_speakers_success(diarization_service):
    """화자 분리 성공 테스트"""
    # Given
    audio_path = "tests/fixtures/sample.wav"

    # When
    result = await diarization_service.diarize_speakers(audio_path)

    # Then
    assert result.num_speakers > 0
    assert result.total_duration > 0

def test_diarize_speakers_invalid_file(diarization_service):
    """잘못된 파일 화자 분리 테스트"""
    # Given
    audio_path = "invalid/path.wav"

    # When & Then
    with pytest.raises(ValueError, match="오디오 파일을 처리할 수 없습니다"):
        await diarization_service.diarize_speakers(audio_path)
```

### 통합 테스트

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient
from voice_man.main import app

@pytest.mark.asyncio
async def test_upload_audio_flow():
    """오디오 업로드 통합 테스트"""
    # Given
    client = AsyncClient(app=app, base_url="http://test")
    audio_file = open("tests/fixtures/sample.mp3", "rb")

    # When
    response = await client.post(
        "/api/v1/audio/upload",
        files={"file": audio_file}
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert "sha256_hash" in data
```

### 인수 테스트 (Gherkin)

```python
# tests/acceptance/test_audio_upload.feature
Feature: 오디오 파일 업로드
  Scenario: 정상적인 mp3 파일 업로드
    Given 유효한 mp3 파일이 존재한다
    When 파일을 업로드한다
    Then 파일 ID가 반환된다
    And SHA-256 해시가 생성된다

  Scenario: 지원하지 않는 파일 형식
    Given txt 파일이 존재한다
    When 파일을 업로드한다
    Then 400 에러가 반환된다
    And "지원하지 않는 파일 형식입니다" 메시지가 표시된다
```

```python
# tests/acceptance/test_audio_upload.py
from pytest_bdd import given, when, then, scenario

@pytest.fixture
def audio_file():
    return open("tests/fixtures/sample.mp3", "rb")

@scenario("audio_upload.feature", "정상적인 mp3 파일 업로드")
def test_upload_valid_mp3():
    pass

@given("유효한 mp3 파일이 존재한다")
def valid_mp3_file(audio_file):
    pass

@when("파일을 업로드한다")
async def upload_file(client, audio_file):
    client.response = await client.post(
        "/api/v1/audio/upload",
        files={"file": audio_file}
    )

@then("파일 ID가 반환된다")
def check_file_id():
    assert "file_id" in client.response.json()
```

### 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 특정 파일만 테스트
pytest tests/unit/test_services.py

# 커버리지 리포트
pytest --cov=voice_man --cov-report=html

# 특정 마커만 테스트
pytest -m unit
pytest -m integration
pytest -m acceptance
```

## 품질 도구

### Ruff (린팅 및 포맷팅)

```bash
# 린트 체크
ruff check .

# 자동 수정
ruff check --fix .

# 포맷팅
ruff format .
```

### MyPy (타입 체크)

```bash
# 타입 체크
mypy src/voice_man

# 엄격 모드
mypy --strict src/voice_man
```

### pytest-cov (커버리지)

```bash
# 커버리지 리포트
pytest --cov=voice_man --cov-report=term-missing

# HTML 리포트
pytest --cov=voice_man --cov-report=html
open htmlcov/index.html  # macOS
```

## 워크플로우

### 1. 새 기능 개발

```bash
# 1. feature 브랜치 생성
git checkout -b feature/add-crime-detection

# 2. 코드 작성
# ...

# 3. 테스트 작성
pytest tests/

# 4. 린트 및 타입 체크
ruff check .
mypy src/

# 5. 커밋
git add .
git commit -m "feat: 범죄 발언 태깅 기능 추가"

# 6. 푸시 및 PR
git push origin feature/add-crime-detection
```

### 2. 버그 수정

```bash
# 1. bugfix 브랜치 생성
git checkout -b fix/upload-error-handling

# 2. 버그 수정
# ...

# 3. 회귀 테스트
pytest tests/

# 4. 커밋
git add .
git commit -m "fix: 파일 업로드 에러 핸들링 개선"

# 5. 푸시 및 PR
git push origin fix/upload-error-handling
```

### 3. 커밋 메시지 규칙 (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (세미콜론 누락 등)
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경

**예시**:
```
feat(stt): Whisper 모델 로드 최적화

GPU 메모리 사용량을 줄이기 위해 모델 양자화를 적용합니다.

- torch.quantization 사용
- 모델 캐싱 추가
- 메모리 사용량 40% 감소

Closes #123
```

## 디버깅

### VS Code 설정

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "voice_man.main:app",
        "--reload",
        "--port",
        "8000"
      ],
      "justMyCode": false
    }
  ]
}
```

### 로깅

```python
import logging

logger = logging.getLogger(__name__)

# 로그 사용
logger.info("파일 업로드 시작: %s", filename)
logger.error("변환 실패: %s", str(error), exc_info=True)
logger.debug("화자 분리 결과: %s", diarization_result)
```

## 문서화

### 코드 문서

```python
def merge_stt_and_diarization(
    stt_segments: List[TranscriptSegment],
    diarization_result: DiarizationResult,
) -> List[TranscriptSegment]:
    """
    STT 세그먼트와 화자 분리 결과를 병합합니다.

    이 함수는 STT 변환 결과와 화자 분리 결과를 시간 기반으로 병합하여
    각 텍스트 세그먼트에 화자 ID를 할당합니다. 겹치는 구간이 가장 큰
    화자를 해당 세그먼트의 화자로 선택합니다.

    Args:
        stt_segments: STT 변환된 텍스트 세그먼트 목록
        diarization_result: 화자 분리 결과

    Returns:
        화자 ID가 할당된 세그먼트 목록

    Raises:
        ValueError: 세그먼트나 화자 정보가 비어있는 경우

    Example:
        >>> segments = [TranscriptSegment(start_time=0.0, end_time=2.5, ...)]
        >>> diarization = DiarizationResult(speakers=[...])
        >>> merge_stt_and_diarization(segments, diarization)
        [TranscriptSegment(speaker_id="SPEAKER_00", ...)]

    Note:
        이 함수는 세그먼트와 화자 구간의 겹침 정도를 계산하여
        가장 관련성이 높은 화자를 할당합니다.
    """
    ...
```

### API 문서

```python
@app.post(
    "/api/v1/audio/upload",
    response_model=AudioUploadResponse,
    summary="오디오 파일 업로드",
    description="""
    오디오 파일을 업로드하고 SHA-256 해시를 생성합니다.

    ## 지원 파일 형식
    - MP3
    - WAV
    - M4A
    - FLAC
    - OGG

    ## 제한 사항
    - 최대 파일 크기: 500MB
    - 최소 파일 크기: 1KB
    """,
    responses={
        200: {"description": "업로드 성공"},
        400: {"description": "잘못된 파일 형식"},
    },
    tags=["Audio"],
)
async def upload_audio_file(file: UploadFile) -> AudioUploadResponse:
    ...
```

## 성능 프로파일링

```python
import cProfile
import pstats

# 프로파일링
profiler = cProfile.Profile()
profiler.enable()

# 실행할 코드
result = process_audio(file_path)

profiler.disable()

# 결과 분석
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 상위 10개 함수
```

## CI/CD 파이프라인

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check .

      - name: Type check with mypy
        run: mypy src/

      - name: Run tests
        run: pytest --cov=voice_man

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

**관련 문서**:
- [아키텍처](architecture.md)
- [API 레퍼런스](api-reference.md)
- [배포 가이드](deployment.md)
