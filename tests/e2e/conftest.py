"""
E2E Test Fixtures and Configuration.

Provides fixtures for E2E batch processing tests with GPU parallel processing.
"""

import asyncio
import pytest
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def ref_call_directory() -> Path:
    """Return path to ref/call directory with test audio files."""
    return Path(__file__).parent.parent.parent / "ref" / "call"


@pytest.fixture
def results_directory() -> Path:
    """Return path to results output directory."""
    results_dir = Path(__file__).parent.parent.parent / "ref" / "call" / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture
def sample_audio_files(ref_call_directory: Path) -> List[Path]:
    """Return list of m4a files from ref/call directory."""
    if not ref_call_directory.exists():
        return []
    return sorted(ref_call_directory.glob("*.m4a"))


@pytest.fixture
def mock_whisperx_service() -> MagicMock:
    """Create mock WhisperXService for testing."""
    mock = MagicMock()
    mock.process_audio = AsyncMock(
        return_value=MagicMock(
            text="Test transcript",
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Test segment",
                    "speaker": "SPEAKER_00",
                }
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
    )
    return mock


@pytest.fixture
def mock_gpu_monitor() -> MagicMock:
    """Create mock GPUMonitorService for testing."""
    mock = MagicMock()
    mock.is_gpu_available.return_value = True
    mock.get_gpu_memory_stats.return_value = {
        "total_mb": 24576,
        "used_mb": 4096,
        "free_mb": 20480,
        "usage_percentage": 16.7,
        "available": True,
    }
    mock.check_memory_status.return_value = {
        "warning": False,
        "critical": False,
        "auto_adjust_recommended": False,
        "usage_percentage": 16.7,
        "message": "GPU memory usage normal: 16.7%",
    }
    mock.get_recommended_batch_size.return_value = 15
    mock.clear_gpu_cache.return_value = None
    return mock


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create mock MemoryManager for testing."""
    mock = MagicMock()
    mock.force_garbage_collection.return_value = 100
    mock.should_collect.return_value = False
    mock.get_memory_summary.return_value = {
        "current_mb": 512,
        "peak_mb": 1024,
        "system_memory_percent": 45.0,
    }
    return mock


@pytest.fixture
def e2e_test_config() -> dict:
    """Default E2E test configuration."""
    return {
        "batch_size": 15,
        "max_batch_size": 32,
        "min_batch_size": 2,
        "num_speakers": 2,
        "language": "ko",
        "device": "cuda",
        "compute_type": "float16",
        "max_retries": 3,
        "retry_delays": [5, 15, 30],
        "dynamic_batch_adjustment": True,
        "gpu_memory_warning_threshold": 80.0,
        "gpu_memory_critical_threshold": 95.0,
    }
