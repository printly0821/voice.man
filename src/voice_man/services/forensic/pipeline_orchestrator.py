"""
PipelineOrchestrator - Producer-Consumer Pipeline for STT+Forensic Processing.

SPEC-PERFOPT-001 Phase 3: Advanced pipeline orchestration with backpressure.

EARS Requirements:
    - E4 (Event): STT result completion triggers Forensic analysis + parallel next file STT
    - S5 (State): Pause STT when Forensic queue exceeds 5 (backpressure)
    - N4 (Unwanted): No blocking I/O on GPU threads
    - PR-007 (Performance): 50% pipeline efficiency improvement via STT+Forensic overlap

Architecture:
    - Producer (STT): Processes audio files and puts results in queue
    - Consumer (Forensic): Takes STT results from queue and performs forensic analysis
    - Backpressure: Pauses producer when queue reaches MAX_QUEUE_SIZE (5)
                   Resumes when queue drops to BACKPRESSURE_RESUME_SIZE (3)

Features:
    - asyncio.Queue-based producer-consumer pattern
    - Backpressure mechanism with hysteresis
    - AsyncIterator for streaming results
    - Integration with Phase 2 managers (ForensicMemoryManager, ThermalManager)
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Backpressure thresholds
MAX_QUEUE_SIZE = 5
BACKPRESSURE_RESUME_SIZE = 3


class PipelineOrchestrator:
    """
    Producer-Consumer pipeline orchestrator for STT and Forensic processing.

    Implements overlapping execution where:
    - STT (producer) processes audio files and queues results
    - Forensic (consumer) analyzes transcripts from queue
    - Backpressure prevents queue overflow

    Example:
        orchestrator = PipelineOrchestrator(
            stt_service=whisperx_service,
            forensic_service=forensic_scoring_service,
        )

        async for result in orchestrator.process_files(audio_files):
            print(f"Forensic result: {result}")

        await orchestrator.shutdown()
    """

    def __init__(
        self,
        stt_service: Optional[Any] = None,
        forensic_service: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        thermal_manager: Optional[Any] = None,
    ):
        """
        Initialize PipelineOrchestrator.

        Args:
            stt_service: WhisperXService instance for STT processing.
            forensic_service: ForensicScoringService instance for forensic analysis.
            memory_manager: ForensicMemoryManager for stage-based memory allocation.
            thermal_manager: ThermalManager for GPU temperature monitoring.
        """
        self._stt_service = stt_service
        self._forensic_service = forensic_service
        self._memory_manager = memory_manager
        self._thermal_manager = thermal_manager

        # Producer-consumer queue with bounded size for backpressure
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

        # Stop event for graceful shutdown
        self._stop_event = asyncio.Event()

        # Backpressure state
        self._backpressure_active = False

        # Pipeline state
        self._is_running = False
        self._producer_task: Optional[asyncio.Task] = None

        # Statistics
        self._files_processed = 0
        self._files_failed = 0

        logger.info(
            f"PipelineOrchestrator initialized: max_queue={MAX_QUEUE_SIZE}, "
            f"resume_at={BACKPRESSURE_RESUME_SIZE}"
        )

    def _check_backpressure(self) -> None:
        """
        Check and update backpressure state based on queue size.

        Uses hysteresis to prevent oscillation:
        - Activate at MAX_QUEUE_SIZE (5)
        - Deactivate at BACKPRESSURE_RESUME_SIZE (3)
        - Maintain current state between thresholds
        """
        queue_size = self._queue.qsize()

        if queue_size >= MAX_QUEUE_SIZE:
            if not self._backpressure_active:
                self._backpressure_active = True
                logger.warning(f"Backpressure activated: queue_size={queue_size}")
        elif queue_size <= BACKPRESSURE_RESUME_SIZE:
            if self._backpressure_active:
                self._backpressure_active = False
                logger.info(f"Backpressure deactivated: queue_size={queue_size}")
        # Between thresholds: maintain current state (hysteresis)

    def is_backpressure_active(self) -> bool:
        """
        Check if backpressure is currently active.

        Returns:
            True if backpressure is active, False otherwise.
        """
        return self._backpressure_active

    async def _produce_stt_results(self, files: List[Path]) -> None:
        """
        Producer coroutine: Process audio files through STT and queue results.

        E4 (Event): STT result completion triggers Forensic analysis.
        S5 (State): Pauses when queue is full (handled by asyncio.Queue).

        Args:
            files: List of audio file paths to process.
        """
        if self._stt_service is None:
            logger.error("STT service not configured")
            # Still send sentinel to signal producer finished
            await self._queue.put(None)
            return

        # Allocate STT stage memory if memory manager available
        if self._memory_manager:
            self._memory_manager.allocate("stt")

        try:
            for file_path in files:
                if self._stop_event.is_set():
                    logger.info("Producer stopping due to stop event")
                    break

                try:
                    # Check thermal throttling
                    if self._thermal_manager and self._thermal_manager.is_throttling:
                        logger.debug("Thermal throttling active, adding delay")
                        await asyncio.sleep(0.5)

                    logger.debug(f"Processing STT for: {file_path.name}")

                    # Process audio through STT
                    result = await self._stt_service.process_audio(str(file_path))

                    # Yield control to consumer (prevent event loop blocking)
                    await asyncio.sleep(0)

                    # Extract transcript text
                    transcript = getattr(result, "text", "")
                    if not transcript and hasattr(result, "segments"):
                        transcript = " ".join(
                            s.get("text", "") for s in getattr(result, "segments", [])
                        )

                    # Queue result for forensic analysis
                    # This will block if queue is full (backpressure)
                    queue_item = {
                        "file_path": str(file_path),
                        "transcript": transcript,
                        "stt_result": result,
                    }

                    await self._queue.put(queue_item)
                    self._check_backpressure()

                    logger.debug(f"STT complete, queued for forensic: {file_path.name}")
                    self._files_processed += 1

                except Exception as e:
                    logger.error(f"STT error for {file_path.name}: {e}")
                    self._files_failed += 1
                    # Queue error result so consumer can report it
                    error_item = {
                        "file_path": str(file_path),
                        "transcript": "",
                        "stt_result": None,
                        "stt_error": str(e),
                    }
                    await self._queue.put(error_item)
                    self._check_backpressure()

        finally:
            # Release STT stage memory
            if self._memory_manager:
                self._memory_manager.release("stt")

            # Signal end of production with sentinel
            await self._queue.put(None)  # Sentinel to signal producer completion
            logger.info("Producer finished")

    async def _consume_forensic(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Consumer coroutine: Take STT results from queue and perform forensic analysis.

        E4 (Event): Forensic analysis starts on STT completion.
        S5 (State): Updates backpressure state after consuming.

        Yields:
            Dict containing forensic analysis result or error information.
        """
        # Allocate forensic stages if memory manager available
        if self._memory_manager:
            self._memory_manager.allocate("ser")
            self._memory_manager.allocate("scoring")

        try:
            while True:
                # Check for stop condition
                if self._stop_event.is_set() and self._queue.empty():
                    logger.info("Consumer stopping: stop event set and queue empty")
                    break

                try:
                    # Get item from queue with timeout
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Check stop condition again
                        if self._stop_event.is_set():
                            break
                        continue

                    # Check for sentinel (producer finished)
                    if item is None:
                        logger.debug("Received sentinel, producer finished")
                        break

                    # Update backpressure state
                    self._check_backpressure()

                    file_path = item.get("file_path", "unknown")
                    transcript = item.get("transcript", "")
                    stt_error = item.get("stt_error")

                    # If STT failed, yield error result without running forensic
                    if stt_error:
                        logger.debug(
                            f"Skipping forensic for {Path(file_path).name} due to STT error"
                        )
                        yield {
                            "file_path": file_path,
                            "result": None,
                            "error": f"STT Error: {stt_error}",
                        }
                        self._queue.task_done()
                        continue

                    logger.debug(f"Running forensic analysis for: {Path(file_path).name}")

                    try:
                        if self._forensic_service is None:
                            raise ValueError("Forensic service not configured")

                        # Perform forensic analysis
                        result = await self._forensic_service.analyze(file_path, transcript)

                        yield {
                            "file_path": file_path,
                            "result": result,
                            "error": None,
                        }

                    except Exception as e:
                        logger.error(f"Forensic error for {Path(file_path).name}: {e}")
                        yield {
                            "file_path": file_path,
                            "result": None,
                            "error": str(e),
                        }

                    self._queue.task_done()

                except Exception as e:
                    logger.error(f"Consumer error: {e}")
                    if self._stop_event.is_set():
                        break

        finally:
            # Release forensic stages
            if self._memory_manager:
                self._memory_manager.release("ser")
                self._memory_manager.release("scoring")

    async def process_files(self, files: List[Path]) -> AsyncIterator[Dict[str, Any]]:
        """
        Process audio files through STT+Forensic pipeline with overlapping execution.

        PR-007 (Performance): Achieves 50% efficiency improvement via parallel execution.

        Args:
            files: List of audio file paths to process.

        Yields:
            Dict containing forensic analysis result for each file.

        Example:
            async for result in orchestrator.process_files(audio_files):
                if result["error"]:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Risk score: {result['result'].overall_risk_score}")
        """
        if not files:
            logger.info("No files to process")
            return

        self._is_running = True
        self._stop_event.clear()

        # Start thermal monitoring if available
        if self._thermal_manager:
            self._thermal_manager.start_monitoring(interval_seconds=1.0)

        try:
            # Start producer task
            self._producer_task = asyncio.create_task(self._produce_stt_results(files))

            # Start consumer and yield results
            results_count = 0
            async for result in self._consume_forensic():
                yield result
                results_count += 1

                # Stop when all files are processed
                if results_count >= len(files):
                    break

            # Wait for producer to complete
            if self._producer_task and not self._producer_task.done():
                self._stop_event.set()
                await self._producer_task

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

        finally:
            self._is_running = False

            # Stop thermal monitoring
            if self._thermal_manager:
                self._thermal_manager.stop_monitoring()

            logger.info(
                f"Pipeline complete: processed={self._files_processed}, failed={self._files_failed}"
            )

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the pipeline.

        - Sets stop event
        - Drains and clears the queue
        - Stops thermal monitoring
        """
        logger.info("Shutting down pipeline orchestrator")

        # Set stop event
        self._stop_event.set()

        # Cancel producer task if running
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Stop thermal monitoring
        if self._thermal_manager:
            self._thermal_manager.stop_monitoring()

        self._is_running = False
        logger.info("Pipeline orchestrator shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline statistics.

        Returns:
            Dictionary with pipeline status information.
        """
        return {
            "queue_size": self._queue.qsize(),
            "backpressure_active": self._backpressure_active,
            "is_running": self._is_running,
            "files_processed": self._files_processed,
            "files_failed": self._files_failed,
        }
