#!/usr/bin/env python3
"""
Batch processing script for 183 audio files.

This script processes the 183 audio files from /ref/call directory using
the enhanced batch processing service with memory optimization and progress tracking.

Usage:
    python scripts/process_audio_files.py
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_man.services.batch_service import BatchProcessor, BatchConfig
from voice_man.services.progress_service import ProgressTracker, ProgressConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("batch_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def mock_analyze_audio(file_path: Path) -> dict:
    """
    Mock audio analysis function.

    In production, this would call the actual STT and analysis pipeline.
    For now, it simulates the analysis process.
    """
    logger.info(f"Analyzing {file_path.name}")

    # Simulate processing time (0.1-0.5 seconds)
    import random

    await asyncio.sleep(random.uniform(0.1, 0.5))

    # Simulate occasional failures (5% failure rate)
    if random.random() < 0.05:
        raise ValueError(f"Simulated processing error for {file_path.name}")

    # Return mock analysis result
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "status": "success",
        "duration_seconds": random.uniform(30, 300),
        "transcript_length": random.randint(100, 5000),
        "speaker_count": random.randint(1, 3),
    }


def print_progress_summary(progress_tracker: ProgressTracker):
    """Print a formatted progress summary."""
    summary = progress_tracker.get_progress_summary()

    print("\n" + "=" * 60)
    print(f"PROGRESS UPDATE - Batch {summary['current_batch']}/{summary['total_batches']}")
    print("=" * 60)
    print(f"  Files Processed: {summary['completed_files']}/{summary['total_files']}")
    print(f"  Progress: {summary['overall_progress_percentage']:.1f}%")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Failed Files: {summary['failed_files']}")
    print(f"  ETA: {summary['eta_formatted']}")
    print("=" * 60 + "\n")


async def main():
    """Main processing function."""
    # Configuration
    AUDIO_DIR = Path("/Users/innojini/Dev/voice.man/ref/call")
    RESULTS_DIR = Path("/Users/innojini/Dev/voice.man/reports")
    BATCH_SIZE = 10
    MAX_WORKERS = 2

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get all audio files
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
    logger.info(f"Found {len(audio_files)} audio files to process")

    if not audio_files:
        logger.error("No audio files found!")
        return

    # Initialize batch processor with memory optimization
    config = BatchConfig(
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        retry_count=3,
        continue_on_error=True,
        enable_memory_cleanup=True,
    )
    processor = BatchProcessor(config)

    # Initialize progress tracker
    progress_tracker = ProgressTracker(ProgressConfig(eta_window_size=10))
    progress_tracker.start_overall(
        total_files=len(audio_files),
        total_batches=(len(audio_files) + BATCH_SIZE - 1) // BATCH_SIZE,
    )

    # Define progress callback
    def progress_callback(progress):
        progress_tracker.update_batch_progress(
            progress.current_batch, progress.processed, progress.failed
        )
        print_progress_summary(progress_tracker)

    # Process all files
    logger.info("Starting batch processing...")
    start_time = datetime.now()

    results = await processor.process_all(
        audio_files, mock_analyze_audio, progress_callback=progress_callback
    )

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Get final statistics
    statistics = processor.get_statistics()

    # Print final summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total Files: {statistics.total_files}")
    print(f"  Successful: {statistics.successful_files}")
    print(f"  Failed: {statistics.failed_files}")
    print(f"  Success Rate: {statistics.successful_files / statistics.total_files:.1%}")
    print(f"  Total Time: {processing_time:.2f} seconds ({processing_time / 60:.1f} minutes)")
    print(f"  Avg Time Per File: {processing_time / statistics.total_files:.2f} seconds")
    print(f"  Total Attempts: {statistics.total_attempts}")
    print(f"  Avg Attempts Per File: {statistics.average_attempts_per_file:.2f}")
    print("=" * 60 + "\n")

    # Save failed files list
    failed_files = processor.get_failed_files()
    if failed_files:
        failed_file_path = RESULTS_DIR / "failed_files.txt"
        with open(failed_file_path, "w") as f:
            f.write("Failed Files:\n")
            f.write("=" * 60 + "\n")
            for file_path in failed_files:
                f.write(f"  - {file_path}\n")
        logger.info(f"Failed files list saved to {failed_file_path}")

    # Save processing results
    results_file = RESULTS_DIR / "processing_results.txt"
    with open(results_file, "w") as f:
        f.write("Batch Processing Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Total Files: {statistics.total_files}\n")
        f.write(f"Successful: {statistics.successful_files}\n")
        f.write(f"Failed: {statistics.failed_files}\n")
        f.write(f"Processing Time: {processing_time:.2f}s\n")
        f.write("\nDetailed Results:\n")
        f.write("-" * 60 + "\n")

        for result in results:
            f.write(f"\nFile: {result.file_path}\n")
            f.write(f"  Status: {result.status}\n")
            f.write(f"  Attempts: {result.attempts}\n")
            if result.status == "success" and result.data:
                f.write(f"  Details: {result.data}\n")
            elif result.status == "failed":
                f.write(f"  Error: {result.error}\n")

    logger.info(f"Processing results saved to {results_file}")
    logger.info("Batch processing completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
