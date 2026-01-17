"""
EdgeXpert GPU Optimization Services

MSI EdgeXpert (NVIDIA Grace Blackwell) 환경에서 WhisperX 파이프라인 최적화.
"""

from voice_man.services.edgexpert.unified_memory_manager import UnifiedMemoryManager
from voice_man.services.edgexpert.cuda_stream_processor import CUDAStreamProcessor
from voice_man.services.edgexpert.hardware_accelerated_codec import HardwareAcceleratedCodec
from voice_man.services.edgexpert.blackwell_optimizer import BlackWellOptimizer
from voice_man.services.edgexpert.arm_cpu_pipeline import ARMCPUPipeline
from voice_man.services.edgexpert.thermal_manager import ThermalManager
from voice_man.services.edgexpert.edgexpert_orchestrator import (
    EdgeXpertOrchestrator,
    OrchestratorConfig,
    OperationPhase,
    PerformanceMetrics,
)
from voice_man.services.edgexpert.edgexpert_whisperx_pipeline import (
    EdgeXpertWhisperXPipeline,
    create_edgexpert_pipeline,
)

__all__ = [
    "UnifiedMemoryManager",
    "CUDAStreamProcessor",
    "HardwareAcceleratedCodec",
    "BlackWellOptimizer",
    "ARMCPUPipeline",
    "ThermalManager",
    "EdgeXpertOrchestrator",
    "OrchestratorConfig",
    "OperationPhase",
    "PerformanceMetrics",
    "EdgeXpertWhisperXPipeline",
    "create_edgexpert_pipeline",
]
