"""
Memory Management Services

Provides comprehensive memory management for voice analysis pipeline.

Memory Optimization (2026-01):
- Smart device selection (GPU/CPU) based on file size and memory
- Dynamic batch size optimization based on available RAM
- Comprehensive memory tracking and cleanup
"""

# Core memory management
from voice_man.services.memory.memory_manager import (
    MemoryManager,
    MemoryPredictor,
    ServiceCleanupProtocol,
    FileMemoryStats,
    MemoryPressureStatus,
    PredictionResult,
    MemoryPressureLevel,
)

# Memory optimization services
from voice_man.services.memory.smart_device_selector import SmartDeviceSelector
from voice_man.services.memory.batch_optimizer import BatchSizeOptimizer

__all__ = [
    # Core memory management
    "MemoryManager",
    "MemoryPredictor",
    "ServiceCleanupProtocol",
    "FileMemoryStats",
    "MemoryPressureStatus",
    "PredictionResult",
    "MemoryPressureLevel",
    # Memory optimization
    "SmartDeviceSelector",
    "BatchSizeOptimizer",
]
