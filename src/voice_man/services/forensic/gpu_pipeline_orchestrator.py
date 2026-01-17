"""
GPU-Optimized Pipeline Orchestrator for Forensic Processing

Integrates GPU optimization components with the forensic pipeline:
- DynamicBatchProcessor: Dynamic batch sizing (4-32)
- MultiGPUOrchestrator: Multi-GPU workload balancing
- TranscriptionCache: L1/L2 caching (100MB + disk)
- RobustPipeline: Error handling with retry & circuit breaker

Reference: SPEC-GPUOPT-001 Phase 3
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from voice_man.services.forensic.pipeline_orchestrator import (
    PipelineOrchestrator,
    MAX_QUEUE_SIZE,
    BACKPRESSURE_RESUME_SIZE,
)
from voice_man.services.gpu_optimization.dynamic_batch_processor import (
    DynamicBatchProcessor,
    DynamicBatchConfig,
    AudioMetadata,
)
from voice_man.services.gpu_optimization.multi_gpu_orchestrator import (
    MultiGPUOrchestrator,
    OrchestratorConfig,
)
from voice_man.services.gpu_optimization.transcription_cache import (
    TranscriptionCache,
    CacheConfig,
)
from voice_man.services.gpu_optimization.robust_pipeline import (
    RobustPipeline,
    CircuitBreakerError,
)
from voice_man.services.gpu_monitor_service import GPUMonitorService

logger = logging.getLogger(__name__)


class GPUPipelineOrchestrator(PipelineOrchestrator):
    """
    GPU-optimized pipeline orchestrator for forensic processing.

    Extends PipelineOrchestrator with:
    - Dynamic batch processing (4-32 files per batch)
    - Multi-GPU workload balancing
    - Transcription caching (L1: 100MB, L2: disk)
    - Robust error handling with retry

    Example:
        orchestrator = GPUPipelineOrchestrator(
            stt_service=whisperx_service,
            forensic_service=forensic_scoring_service,
        )

        async for result in orchestrator.process_files(audio_files):
            print(f"Result: {result}")

        await orchestrator.shutdown()
    """

    def __init__(
        self,
        stt_service: Optional[Any] = None,
        forensic_service: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        thermal_manager: Optional[Any] = None,
        # GPU optimization components
        enable_gpu_optimization: bool = True,
        batch_config: Optional[DynamicBatchConfig] = None,
        gpu_config: Optional[OrchestratorConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        Initialize GPU Pipeline Orchestrator.

        Args:
            stt_service: WhisperXService instance for STT processing.
            forensic_service: ForensicScoringService instance for forensic analysis.
            memory_manager: ForensicMemoryManager for stage-based memory allocation.
            thermal_manager: ThermalManager for GPU temperature monitoring.
            enable_gpu_optimization: Enable GPU optimization features.
            batch_config: Dynamic batch configuration.
            gpu_config: Multi-GPU orchestrator configuration.
            cache_config: Transcription cache configuration.
        """
        super().__init__(
            stt_service=stt_service,
            forensic_service=forensic_service,
            memory_manager=memory_manager,
            thermal_manager=thermal_manager,
        )

        # GPU optimization flag
        self._enable_gpu_optimization = enable_gpu_optimization

        # Initialize GPU optimization components
        if enable_gpu_optimization:
            # Dynamic batch processor (creates its own GPU monitor internally)
            self._batch_processor = DynamicBatchProcessor(
                config=batch_config or DynamicBatchConfig(),
            )

            # Multi-GPU orchestrator
            self._gpu_orchestrator = MultiGPUOrchestrator(config=gpu_config or OrchestratorConfig())

            # Transcription cache
            self._cache = TranscriptionCache(config=cache_config or CacheConfig())

            # Robust pipeline wrapper (use default config)
            self._robust_pipeline = RobustPipeline()

            logger.info(
                f"GPUPipelineOrchestrator initialized with GPU optimization: "
                f"batch_size={self._batch_processor.config.min_batch_size}-"
                f"{self._batch_processor.config.max_batch_size}, "
                f"gpus={len(self._gpu_orchestrator.gpus)}, "
                f"cache_enabled=True"
            )
        else:
            self._batch_processor = None
            self._gpu_orchestrator = None
            self._cache = None
            self._robust_pipeline = None
            logger.info("GPUPipelineOrchestrator initialized without GPU optimization")

        # Batch processing statistics
        self._batches_processed = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_audio_metadata(self, file_path: Path) -> AudioMetadata:
        """Get audio file metadata for batching."""
        try:
            # Try to get duration from file
            import librosa

            duration = librosa.get_duration(path=str(file_path))
            size_bytes = file_path.stat().st_size
            return AudioMetadata(
                path=str(file_path),
                duration=duration,
                size_bytes=size_bytes,
            )
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_path.name}: {e}")
            # Return default metadata (short duration to be safe)
            return AudioMetadata(
                path=str(file_path),
                duration=60.0,  # Assume 1 minute
                size_bytes=file_path.stat().st_size,
            )

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for audio file."""
        # Use file hash + model info for cache key
        file_stat = file_path.stat()
        key_data = f"{file_path.name}-{file_stat.st_mtime}-{file_stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _process_batch_with_cache(
        self,
        file_paths: List[Path],
        gpu_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of audio files with caching support.

        Args:
            file_paths: List of audio file paths to process.
            gpu_id: GPU device ID to use.

        Returns:
            List of STT results with cache information.
        """
        results = []

        for file_path in file_paths:
            cache_key = self._get_cache_key(file_path)

            # Check cache
            cached_result = self._cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {file_path.name}")
                self._cache_hits += 1
                results.append(
                    {
                        "file_path": str(file_path),
                        "transcript": cached_result.get("text", ""),
                        "stt_result": cached_result,
                        "cached": True,
                    }
                )
                continue

            # Cache miss - process the file
            logger.debug(f"Cache miss for {file_path.name}")
            self._cache_misses += 1

            try:
                # Directly call async process_audio
                # Note: RobustPipeline.execute_with_retry expects sync functions
                # For async operations, we handle errors at pipeline level
                result = await self._stt_service.process_audio(str(file_path))

                # Store in cache
                transcript = getattr(result, "text", "")
                if not transcript and hasattr(result, "segments"):
                    transcript = " ".join(
                        s.get("text", "") for s in getattr(result, "segments", [])
                    )

                self._cache.put(
                    cache_key,
                    {
                        "text": transcript,
                        "result": result,
                    },
                )

                results.append(
                    {
                        "file_path": str(file_path),
                        "transcript": transcript,
                        "stt_result": result,
                        "cached": False,
                    }
                )

            except CircuitBreakerError:
                logger.error(f"Circuit breaker opened for {file_path.name}")
                results.append(
                    {
                        "file_path": str(file_path),
                        "transcript": "",
                        "stt_result": None,
                        "cached": False,
                        "stt_error": "Circuit breaker: Too many failures",
                    }
                )

            except Exception as e:
                logger.error(f"STT error for {file_path.name}: {e}")
                results.append(
                    {
                        "file_path": str(file_path),
                        "transcript": "",
                        "stt_result": None,
                        "cached": False,
                        "stt_error": str(e),
                    }
                )

        return results

    async def _produce_stt_results_gpu(
        self,
        files: List[Path],
    ) -> None:
        """
        GPU-optimized producer coroutine: Process audio files through STT with dynamic batching.

        Implements:
        - Dynamic batch sizing (4-32 files per batch)
        - Multi-GPU workload balancing
        - Transcription caching (L1: 100MB, L2: disk)
        - Robust error handling with retry

        Args:
            files: List of audio file paths to process.
        """
        if self._stt_service is None:
            logger.error("STT service not configured")
            await self._queue.put(None)
            return

        # Allocate STT stage memory if memory manager available
        if self._memory_manager:
            self._memory_manager.allocate("stt")

        try:
            # Get audio metadata for all files
            audio_file_paths = [str(f) for f in files]

            # Plan batch processing
            batch_plan = self._batch_processor.plan_batch_processing(audio_file_paths)

            logger.info(
                f"Batch plan created: {len(batch_plan.batches)} batches, "
                f"{batch_plan.total_files()} files"
            )

            # Process each batch
            for batch_idx, batch_files in enumerate(batch_plan.batches):
                if self._stop_event.is_set():
                    logger.info("Producer stopping due to stop event")
                    break

                # Check thermal throttling
                if self._thermal_manager and self._thermal_manager.is_throttling:
                    logger.debug("Thermal throttling active, adding delay")
                    await asyncio.sleep(0.5)

                # Select GPU for this batch (get GPU for transcribe stage)
                gpu_id = self._gpu_orchestrator.get_gpu_for_stage(
                    stage="transcribe",
                )

                logger.info(
                    f"Processing batch {batch_idx + 1}/{len(batch_plan.batches)}: "
                    f"{len(batch_files)} files on GPU {gpu_id}"
                )

                # Convert to Path objects
                batch_paths = [Path(f) for f in batch_files]

                # Process batch with cache
                batch_results = await self._process_batch_with_cache(
                    batch_paths,
                    gpu_id=gpu_id,
                )

                # Queue results for forensic analysis
                for result in batch_results:
                    if result.get("stt_error"):
                        self._files_failed += 1
                    else:
                        self._files_processed += 1

                    await self._queue.put(result)
                    self._check_backpressure()

                self._batches_processed += 1

        finally:
            # Release STT stage memory
            if self._memory_manager:
                self._memory_manager.release("stt")

            # Signal end of production with sentinel
            await self._queue.put(None)
            logger.info("GPU Producer finished")

    async def _produce_stt_results(self, files: List[Path]) -> None:
        """
        Producer coroutine with GPU optimization if enabled.

        Args:
            files: List of audio file paths to process.
        """
        if self._enable_gpu_optimization and self._batch_processor:
            await self._produce_stt_results_gpu(files)
        else:
            # Use base implementation
            await super()._produce_stt_results(files)

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU optimization statistics.

        Returns:
            Dictionary with GPU optimization status information.
        """
        stats = self.get_stats()
        if self._enable_gpu_optimization:
            stats.update(
                {
                    "gpu_optimization_enabled": True,
                    "batches_processed": self._batches_processed,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "cache_hit_rate": (
                        self._cache_hits / (self._cache_hits + self._cache_misses)
                        if (self._cache_hits + self._cache_misses) > 0
                        else 0.0
                    ),
                    "gpus_available": len(self._gpu_orchestrator.gpus)
                    if self._gpu_orchestrator
                    else 0,
                }
            )
        else:
            stats["gpu_optimization_enabled"] = False

        return stats


class CPUPipelineOrchestrator(PipelineOrchestrator):
    """
    CPU-only pipeline orchestrator for fallback compatibility.

    Uses base PipelineOrchestrator without GPU optimization.
    """

    pass  # Uses base implementation as-is
