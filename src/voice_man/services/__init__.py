"""
서비스 패키지

비즈니스 로직 서비스 모듈과 diarization_service를 export합니다.
"""

# diarization_service export
from voice_man.services.diarization_service import DiarizationService

# services.py의 유틸리티 함수들 export (절대 경로 사용)
import sys
from pathlib import Path

# services.py 파일의 경로
services_py_path = Path(__file__).parent.parent / "services.py"

# services.py 모듈을 동적으로 로드
import importlib.util

spec = importlib.util.spec_from_file_location("voice_man.services_module", services_py_path)
services_module = importlib.util.module_from_spec(spec)
sys.modules["voice_man.services_module"] = services_module
spec.loader.exec_module(services_module)

# services.py의 함수들을 export
from voice_man.services_module import (
    compute_sha256_hash,
    generate_file_id,
    is_supported_audio_format,
    extract_audio_metadata,
    detect_corrupted_file,
    transcribe_audio,
)

# Memory management services
from voice_man.services.memory import (
    MemoryManager,
    MemoryPredictor,
    ServiceCleanupProtocol,
    FileMemoryStats,
    MemoryPressureStatus,
    PredictionResult,
    MemoryPressureLevel,
)

__all__ = [
    "DiarizationService",
    "compute_sha256_hash",
    "generate_file_id",
    "is_supported_audio_format",
    "extract_audio_metadata",
    "detect_corrupted_file",
    "transcribe_audio",
    # Memory management
    "MemoryManager",
    "MemoryPredictor",
    "ServiceCleanupProtocol",
    "FileMemoryStats",
    "MemoryPressureStatus",
    "PredictionResult",
    "MemoryPressureLevel",
]
