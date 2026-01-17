#!/usr/bin/env python3
"""
Safe Forensic Pipeline Runner with Parallel Monitoring
======================================================

Runs the forensic pipeline with simultaneous resource monitoring.
Provides automatic pause/resume and error recovery.

Usage:
    python scripts/run_safe_pipeline.py
    python scripts/run_safe_pipeline.py --stage full
    python scripts/run_safe_pipeline.py --files 50
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/safe_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SafePipelineRunner:
    """Run forensic pipeline with parallel monitoring"""

    def __init__(self, max_files: int = None, stage: str = "full"):
        """
        Initialize safe pipeline runner

        Args:
            max_files: Maximum files to process (None = all)
            stage: Pipeline stage to run (pilot, medium, full, all)
        """
        self.max_files = max_files
        self.stage = stage
        self.monitoring_task = None
        self.pipeline_task = None
        self.shutdown_requested = False

        # Ensure directories exist
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("data/checkpoints").mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    async def run_monitoring(self):
        """Run resource monitoring in parallel"""
        from pipeline_monitor import ResourceMonitor

        monitor = ResourceMonitor(check_interval=30)

        try:
            logger.info("Starting resource monitor...")
            await monitor.start_monitoring()
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        finally:
            monitor.stop_monitoring()
            monitor.print_summary()

    async def run_pipeline(self):
        """Run the actual forensic pipeline"""
        from run_forensic_pipeline import ForensicPipelineRunner

        try:
            logger.info(
                f"Starting forensic pipeline (stage: {self.stage}, max_files: {self.max_files})"
            )

            runner = ForensicPipelineRunner(
                audio_dir=Path("ref/call"),
                enable_gpu=True,
            )

            results = await runner.run_staged_pipeline(max_files=self.max_files)

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    async def run_with_monitoring(self):
        """Run pipeline with parallel monitoring"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("SAFE FORENSIC PIPELINE STARTING")
        logger.info("=" * 60)
        logger.info(f"Stage: {self.stage}")
        logger.info(f"Max files: {self.max_files or 'All (195)'}")
        logger.info(f"Started at: {start_time.isoformat()}")
        logger.info("=" * 60)

        try:
            # Run monitoring and pipeline in parallel
            monitoring_task = asyncio.create_task(self.run_monitoring())
            pipeline_task = asyncio.create_task(self.run_pipeline())

            # Wait for pipeline to complete or error
            pipeline_result = await pipeline_task

            # Cancel monitoring when pipeline completes
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("SAFE FORENSIC PIPELINE COMPLETED")
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
        finally:
            # Ensure monitoring is stopped
            if monitoring_task and not monitoring_task.done():
                monitoring_task.cancel()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run forensic pipeline with safety monitoring")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pilot", "medium", "full", "all"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )

    args = parser.parse_args()

    runner = SafePipelineRunner(max_files=args.files, stage=args.stage)

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
