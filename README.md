# Voice Man

음성 녹취 증거 분석 시스템

## Features

- 오디오 파일 업로드 및 관리
- FFmpeg 기반 오디오 전처리
- Whisper STT 엔진 통합
- pyannote-audio 화자 분리
- 데이터베이스 저장 및 조회

## Installation

```bash
pip install -e .
```

## Running

```bash
uvicorn voice_man.main:app --reload
```

## API Documentation

`http://localhost:8000/docs`
