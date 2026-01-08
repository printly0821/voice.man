#!/usr/bin/env python3
"""
E2E Batch Test Script for WhisperX Processing.

CLI script for running end-to-end batch tests on audio files with GPU parallel processing.

Usage:
    python scripts/e2e_batch_test.py --input-dir ref/call --output-dir ref/call/reports/results
    python scripts/e2e_batch_test.py --input-dir ref/call --batch-size 20 --num-speakers 2

SPEC-E2ETEST-001 Requirements:
- U1: BatchProcessor based GPU parallel batch processing
- U4: Original file integrity verification (checksum)
- E4: Dynamic batch adjustment on GPU memory shortage
- S2: Failed file retry queue (exponential backoff)
- N1: GPU memory must not exceed 95%
- N3: Original file modification prohibited
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from voice_man.services.e2e_test_service import (
    E2ETestConfig,
    E2ETestRunner,
    E2ETestResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E2E Batch Test for WhisperX Audio Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python scripts/e2e_batch_test.py --input-dir ref/call

  # Specify output directory and batch size
  python scripts/e2e_batch_test.py --input-dir ref/call --output-dir results --batch-size 20

  # Process with specific speaker count
  python scripts/e2e_batch_test.py --input-dir ref/call --num-speakers 2

  # Use CPU instead of GPU
  python scripts/e2e_batch_test.py --input-dir ref/call --device cpu
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing audio files to process",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output reports (default: input-dir/results)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Initial batch size for processing (default: 15)",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size ceiling (default: 32)",
    )

    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=2,
        help="Minimum batch size floor (default: 2)",
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (default: auto-detect)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="Language code for transcription (default: ko)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Device to use for processing (default: cuda)",
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "float32", "int8"],
        default="float16",
        help="Compute type for inference (default: float16)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed files (default: 3)",
    )

    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Disable checksum verification",
    )

    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch size adjustment",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect files and show what would be processed without processing",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    return parser.parse_args()


def create_progress_callback(total_files: int):
    """Create a progress callback with tqdm if available.

    Args:
        total_files: Total number of files to process

    Returns:
        Tuple of (callback function, progress bar instance or None)
    """
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_files,
            desc="Processing",
            unit="file",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        def callback(current: int, total: int, elapsed: float, filename: str, status: str):
            pbar.update(1)
            pbar.set_postfix_str(f"{filename[:30]}... [{status}]")

        return callback, pbar
    else:

        def callback(current: int, total: int, elapsed: float, filename: str, status: str):
            progress = current / total * 100
            logger.info(f"[{progress:5.1f}%] {current}/{total} - {filename} [{status}]")

        return callback, None


def print_result_summary(result: E2ETestResult) -> None:
    """Print a summary of the E2E test result."""
    success_rate = result.success_rate * 100

    print("\n" + "=" * 60)
    print("E2E BATCH TEST RESULTS")
    print("=" * 60)

    print(f"\nFiles Processed: {result.processed_files}/{result.total_files}")
    print(f"Success Rate:    {success_rate:.1f}%")
    print(f"Failed Files:    {result.failed_files}")
    print(f"Total Time:      {result.total_time_seconds:.2f}s")
    print(f"Avg Time/File:   {result.avg_time_per_file:.2f}s")
    print(f"Checksum:        {'Verified' if result.checksum_verified else 'FAILED'}")

    if result.gpu_stats:
        print("\nGPU Statistics:")
        for key, value in result.gpu_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    if result.failed_files > 0:
        print("\nFailed Files:")
        for fr in result.get_failed_files():
            print(f"  - {Path(fr.file_path).name}: {fr.error}")

    print("=" * 60)


async def main() -> int:
    """Main entry point for E2E batch test script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    input_dir = Path(args.input_dir).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir / "results"

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Create configuration
    config = E2ETestConfig(
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        min_batch_size=args.min_batch_size,
        num_speakers=args.num_speakers,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        max_retries=args.max_retries,
        retry_delays=[5, 15, 30],
        dynamic_batch_adjustment=not args.no_dynamic_batch,
        enable_checksum_verification=not args.no_checksum,
    )

    # Initialize services (lazy loading)
    whisperx_service = None
    gpu_monitor = None
    memory_manager = None

    if not args.dry_run:
        try:
            # Import and initialize services
            from voice_man.services.whisperx_service import WhisperXService
            from voice_man.services.gpu_monitor_service import GPUMonitorService
            from voice_man.services.memory_service import MemoryManager

            logger.info("Initializing WhisperX service...")
            whisperx_service = WhisperXService(
                model_size="large-v3",
                device=config.device,
                language=config.language,
                compute_type=config.compute_type,
            )

            if config.device == "cuda":
                gpu_monitor = GPUMonitorService()
                memory_manager = MemoryManager(enable_gpu_monitoring=True)
                logger.info("GPU monitoring enabled")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            if not args.dry_run:
                return 1

    # Create runner
    runner = E2ETestRunner(
        config=config,
        whisperx_service=whisperx_service,
        gpu_monitor=gpu_monitor,
        memory_manager=memory_manager,
    )

    # Collect files
    logger.info("Collecting audio files...")
    try:
        files = await runner.collect_files(input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if args.limit:
        files = files[: args.limit]
        logger.info(f"Limited to {args.limit} files")

    logger.info(f"Found {len(files)} audio files")

    if not files:
        logger.warning("No audio files found in input directory")
        return 0

    # Dry run - just show what would be processed
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would process {len(files)} files:")
        for i, f in enumerate(files[:20], 1):
            print(f"  {i}. {f.name}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
        print("\nConfiguration:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Max retries: {config.max_retries}")
        print(f"  Device: {config.device}")
        print(f"  Language: {config.language}")
        return 0

    # Create progress callback
    progress_callback, pbar = create_progress_callback(len(files))

    # Run E2E test
    logger.info("Starting E2E batch test...")
    start_time = datetime.now()

    try:
        result = await runner.run(files, progress_callback=progress_callback)
    finally:
        if pbar:
            pbar.close()

    end_time = datetime.now()
    logger.info(f"E2E test completed in {(end_time - start_time).total_seconds():.2f}s")

    # Print summary
    print_result_summary(result)

    # Generate reports
    logger.info("Generating reports...")
    generated = runner.generate_report(result, output_dir)

    print("\nGenerated Reports:")
    for report_type, path in generated.items():
        print(f"  {report_type}: {path}")

    # Cleanup
    if whisperx_service:
        whisperx_service.unload()

    # Return exit code based on success rate
    if result.success_rate < 0.95:
        logger.warning(f"Success rate below 95%: {result.success_rate * 100:.1f}%")
        return 1

    if not result.checksum_verified:
        logger.error("Checksum verification failed - files may have been modified")
        return 2

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
