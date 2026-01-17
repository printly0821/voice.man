"""
GPU Optimization Services for WhisperX Pipeline

Phase 1: Quick Wins (6-7x target)
- OptimizedWhisperXPipeline: torch.compile() + Mixed Precision
- TranscriptionCache: L1 memory cache + L2 disk cache
- ProgressiveBatchProcessor: Progressive batch size adjustment

Phase 2: Intermediate Optimization (20-30x target)
- FasterWhisperWrapper: CTranslate2-based implementation
- MultiGPUOrchestrator: Multi-GPU detection and load balancing
- DynamicBatchProcessor: Dynamic batch size (4-32) with GPU memory awareness

Phase 3: Advanced Optimization (50-100x target)
- TensorRTCompiler: ONNX to TensorRT conversion
- QuantizationEngine: PTQ and INT8/FP16 hybrid quantization
- CUDAGraphOptimized: CUDA Graph Trees and kernel fusion
- RobustPipeline: Error handling with retry and fallback

Reference: SPEC-GPUOPT-001
"""

from voice_man.services.gpu_optimization.cuda_graph_optimized import (
    CUDAGraphConfig,
    CUDAGraphOptimized,
    CUDAGraphStats,
)
from voice_man.services.gpu_optimization.dynamic_batch_processor import (
    AudioMetadata,
    BatchPlan,
    DynamicBatchConfig,
    DynamicBatchProcessor,
)
from voice_man.services.gpu_optimization.faster_whisper_wrapper import (
    FasterWhisperWrapper,
)
from voice_man.services.gpu_optimization.multi_gpu_orchestrator import (
    GPUInfo,
    MultiGPUOrchestrator,
    OrchestratorConfig,
)
from voice_man.services.gpu_optimization.optimized_whisperx_pipeline import (
    OptimizedWhisperXPipeline,
)
from voice_man.services.gpu_optimization.progressive_batch_processor import (
    BatchConfig,
    BatchResult,
    ProgressiveBatchProcessor,
)
from voice_man.services.gpu_optimization.quantization_engine import (
    QuantizationConfig,
    QuantizationEngine,
    QuantizationResult,
)
from voice_man.services.gpu_optimization.robust_pipeline import (
    CircuitBreaker,
    CircuitBreakerError,
    ErrorCategory,
    ErrorInfo,
    ErrorSeverity,
    PipelineResult,
    RobustConfig,
    RobustPipeline,
)
from voice_man.services.gpu_optimization.tensorrt_compiler import (
    TensorRTBuildResult,
    TensorRTCompiler,
    TensorRTConfig,
)
from voice_man.services.gpu_optimization.transcription_cache import (
    CacheConfig,
    CacheEntry,
    TranscriptionCache,
)

__all__ = [
    # Phase 1
    "OptimizedWhisperXPipeline",
    "TranscriptionCache",
    "ProgressiveBatchProcessor",
    # Phase 2
    "FasterWhisperWrapper",
    "MultiGPUOrchestrator",
    "DynamicBatchProcessor",
    # Phase 3
    "TensorRTCompiler",
    "QuantizationEngine",
    "CUDAGraphOptimized",
    "RobustPipeline",
    # Config classes
    "CacheConfig",
    "CacheEntry",
    "BatchConfig",
    "BatchResult",
    "DynamicBatchConfig",
    "DynamicBatchProcessor",
    "AudioMetadata",
    "BatchPlan",
    "GPUInfo",
    "OrchestratorConfig",
    "TensorRTConfig",
    "TensorRTBuildResult",
    "QuantizationConfig",
    "QuantizationResult",
    "CUDAGraphConfig",
    "CUDAGraphStats",
    "RobustConfig",
    "PipelineResult",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorInfo",
    "CircuitBreaker",
    "CircuitBreakerError",
]
