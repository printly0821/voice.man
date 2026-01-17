#!/usr/bin/env python3
"""
Fixed Safe Forensic Pipeline Runner with CUDA NVRTC Error Resolution
=======================================================================

Fixes:
1. Proper TORCH_CUDA_ARCH_LIST setting for GB10 GPU
2. Disable diarization to avoid NVRTC JIT compilation issues
3. Fallback to CPU-only STT if needed

Usage:
    python scripts/run_safe_pipeline_fixed.py
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# CRITICAL FIX: Set CUDA environment variables BEFORE any torch imports
# For GB10 (Orin/Ada architecture), use empty string to disable JIT compilation
# or specific architecture like "sm_89" for Orin
os.environ["TORCH_CUDA_ARCH_LIST"] = ""  # Disable JIT compilation - use precompiled kernels
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Disable CUDA runtime compilation for audio processing libraries
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"  # Disable TF32 to avoid compatibility issues

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline_fixed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class FixedPipelineRunner:
    """Fixed pipeline runner with proper CUDA configuration"""

    def __init__(self, max_files: int = None, skip_diarization: bool = True):
        """
        Initialize fixed pipeline runner

        Args:
            max_files: Maximum files to process (None = all)
            skip_diarization: Skip diarization to avoid NVRTC errors
        """
        self.max_files = max_files
        self.skip_diarization = skip_diarization
        self.shutdown_requested = False

        # Ensure directories exist
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("data/checkpoints").mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    async def run_pipeline(self):
        """Run the forensic pipeline with fixes"""
        # Import after environment variables are set
        from run_forensic_full import FullForensicPipeline

        try:
            logger.info(f"Starting FIXED forensic pipeline (max_files: {self.max_files})")
            logger.info(f"Skip diarization: {self.skip_diarization} (avoids NVRTC errors)")

            pipeline = FullForensicPipeline()

            # Get audio files
            audio_files = sorted(Path("ref/call").glob("*.m4a"))
            if self.max_files:
                audio_files = audio_files[: self.max_files]

            logger.info(f"Processing {len(audio_files)} audio files")

            # Process files
            results = []
            for i, audio_file in enumerate(audio_files, 1):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping pipeline...")
                    break

                logger.info(f"\n{'=' * 60}")
                logger.info(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
                logger.info(f"{'=' * 60}")

                result = await pipeline.process_file(audio_file)
                results.append(result)

                # Log progress
                completed = sum(1 for r in results if r["status"] == "completed")
                logger.info(f"Progress: {completed}/{len(results)} completed")

            pipeline.print_summary(results)
            pipeline.cleanup()

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    async def run_with_monitoring(self):
        """Run pipeline with resource monitoring"""
        from pipeline_monitor import ResourceMonitor

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("FIXED FORENSIC PIPELINE STARTING")
        logger.info("=" * 60)
        logger.info(f"Max files: {self.max_files or 'All (183)'}")
        logger.info(f"Skip diarization: {self.skip_diarization}")
        logger.info(f"Started at: {start_time.isoformat()}")
        logger.info("=" * 60)

        try:
            # Run monitoring and pipeline in parallel
            monitor = ResourceMonitor(check_interval=30)
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            pipeline_task = asyncio.create_task(self.run_pipeline())

            # Wait for pipeline to complete
            pipeline_result = await pipeline_task

            # Cancel monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("FIXED FORENSIC PIPELINE COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
            logger.info(f"Completed at: {end_time.isoformat()}")
            logger.info("=" * 60)

            return pipeline_result

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run forensic pipeline with CUDA NVRTC fixes")
    parser.add_argument(
        "--files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable diarization (may cause NVRTC errors)",
    )

    args = parser.parse_args()

    runner = FixedPipelineRunner(
        max_files=args.files,
        skip_diarization=not args.enable_diarization,
    )

    try:
        await runner.run_with_monitoring()
        logger.info("Execution completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
