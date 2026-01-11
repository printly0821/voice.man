#!/usr/bin/env python3
"""
Optimized Parallel Batch Forensic Analysis Script

Process multiple audio files in parallel using asyncio + ThreadPoolExecutor
with shared model pool for memory efficiency.

Key Optimizations:
- Shared model pool (no duplicate model loading)
- Stage-based memory management
- Asyncio for I/O operations
- Distil-Whisper support for 4-6x faster transcription

Usage:
    python scripts/batch_forensic_optimized.py --input-dir /tmp/failed_calls --workers 2
    python scripts/batch_forensic_optimized.py --input-dir ref/call --workers 3 --remaining-file /tmp/remaining_files_after_first.txt
"""

# ============================================================================
# PyTorch 2.6+ Compatibility Patch
# ============================================================================
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, weights_only=None, **kwargs):
    """Patched torch.load to default weights_only=False for backward compatibility."""
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)


torch.load = _patched_torch_load
# ============================================================================

import argparse
import asyncio
import gc
import json
import logging
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from voice_man.services.forensic.audio_feature_service import AudioFeatureService
from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
from voice_man.services.forensic.ser_service import SERService
from voice_man.services.forensic.cross_validation_service import CrossValidationService
from voice_man.services.forensic.crime_language_service import (
    CrimeLanguageAnalysisService,
)
from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService
from voice_man.services.whisperx_service import WhisperXService
from voice_man.reports.html_generator import ForensicHTMLGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_e2e_transcription_cache(
    e2e_results_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Load E2E test results to reuse existing transcriptions.

    Args:
        e2e_results_path: Path to e2e_test_report.json

    Returns:
        Dictionary mapping audio file paths to transcription data
    """
    cache = {}
    try:
        with open(e2e_results_path) as f:
            data = json.load(f)

        for file_result in data.get("result", {}).get("file_results", []):
            file_path = file_result.get("file_path")
            if file_path and file_result.get("status") == "success":
                cache[file_path] = {
                    "segments": file_result.get("segments", []),
                    "text": file_result.get("transcript_text", ""),
                }

        logger.info(f"Loaded {len(cache)} transcriptions from E2E results")
    except Exception as e:
        logger.warning(f"Could not load E2E results: {e}")

    return cache


class SharedModelPool:
    """
    Thread-safe shared model pool for efficient GPU memory usage.

    All workers share the same model instances instead of each loading their own.
    This reduces memory usage from ~24GB per worker to ~24GB total.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._whisper_service = None
            self._forensic_services = None
            self._html_generator = None
            self._service_lock = threading.Lock()
            self._whisper_lock = threading.Lock()
            self._forensic_lock = threading.Lock()
            self._initialized = True
            logger.info("SharedModelPool initialized")

    def get_whisper_service(
        self, model_size: str = "large-v3", device: str = "cuda", language: str = "ko"
    ) -> WhisperXService:
        """Get or create shared WhisperX service."""
        with self._whisper_lock:
            if self._whisper_service is None:
                logger.info(f"Initializing shared WhisperX service: {model_size}")
                self._whisper_service = WhisperXService(
                    model_size=model_size,
                    device=device,
                    language=language,
                    compute_type="float16",
                )
                logger.info("WhisperX service initialized and shared")
            return self._whisper_service

    def get_forensic_services(self) -> Dict[str, Any]:
        """Get or create shared forensic services."""
        with self._forensic_lock:
            if self._forensic_services is None:
                logger.info("Initializing shared forensic services")

                audio_feature_service = AudioFeatureService()
                stress_analysis_service = StressAnalysisService()
                ser_service = SERService()
                crime_language_service = CrimeLanguageAnalysisService()
                cross_validation_service = CrossValidationService(
                    crime_language_service=crime_language_service,
                    ser_service=ser_service,
                )
                forensic_scoring_service = ForensicScoringService(
                    audio_feature_service=audio_feature_service,
                    stress_analysis_service=stress_analysis_service,
                    crime_language_service=crime_language_service,
                    ser_service=ser_service,
                    cross_validation_service=cross_validation_service,
                )

                self._forensic_services = {
                    "audio_feature": audio_feature_service,
                    "stress": stress_analysis_service,
                    "ser": ser_service,
                    "crime": crime_language_service,
                    "cross_validation": cross_validation_service,
                    "scoring": forensic_scoring_service,
                }
                logger.info("Forensic services initialized and shared")
            return self._forensic_services

    def get_html_generator(self) -> ForensicHTMLGenerator:
        """Get or create shared HTML generator."""
        with self._service_lock:
            if self._html_generator is None:
                self._html_generator = ForensicHTMLGenerator()
            return self._html_generator

    def cleanup_whisper(self):
        """Clean up Whisper service to free GPU memory."""
        with self._whisper_lock:
            if self._whisper_service is not None:
                try:
                    # Unload models if service has unload method
                    if hasattr(self._whisper_service, "unload"):
                        self._whisper_service.unload()
                    del self._whisper_service
                    self._whisper_service = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Whisper service cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up Whisper: {e}")

    def cleanup_all(self):
        """Clean up all services."""
        logger.info("Cleaning up all services...")
        self.cleanup_whisper()

        if self._forensic_services is not None:
            for name, service in self._forensic_services.items():
                try:
                    if hasattr(service, "unload"):
                        service.unload()
                    del service
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")
            self._forensic_services = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All services cleaned up")


async def process_single_file_async(
    audio_path: str,
    output_dir: str,
    model_pool: SharedModelPool,
    file_idx: int,
    model_size: str,
    e2e_transcription: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single audio file through forensic analysis using shared models.

    Args:
        audio_path: Path to audio file
        output_dir: Output directory for results
        model_pool: Shared model pool
        file_idx: File index for logging
        model_size: Whisper model size
        e2e_transcription: Optional existing transcription to reuse

    Returns:
        Dictionary containing analysis results
    """
    worker_logger = logging.getLogger(f"Worker-{file_idx}")
    audio_name = Path(audio_path).name

    try:
        worker_logger.info(f"Processing: {audio_name}")

        # Check if we have existing transcription
        use_cached_transcription = False
        if e2e_transcription:
            worker_logger.info("Reusing existing transcription [CACHED]")
            use_cached_transcription = True

        # Get shared services
        whisper_service = model_pool.get_whisper_service(model_size=model_size)

        # Run transcription (this is async, runs in thread pool)
        worker_logger.info(f"[{audio_name}] Computing transcription...")
        transcript_result = await whisper_service.process_audio(
            str(audio_path),
            existing_transcription=e2e_transcription if use_cached_transcription else None,
        )

        # Important: Clean up GPU memory after transcription
        # This is the heaviest stage and uses the most memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        worker_logger.info(f"[{audio_name}] Transcription complete")

        # Get forensic services
        forensic_services = model_pool.get_forensic_services()

        # Run forensic analysis
        worker_logger.info(f"[{audio_name}] Running forensic analysis...")
        forensic_score = await forensic_services["scoring"].analyze(
            audio_path=str(audio_path),
            transcript=transcript_result.text,
        )

        # Prepare result
        result = {
            "analysis_id": forensic_score.analysis_id,
            "audio_file": str(audio_path),
            "analyzed_at": forensic_score.analyzed_at.isoformat(),
            "audio_duration_seconds": forensic_score.audio_duration_seconds,
            "overall_risk_score": forensic_score.overall_risk_score,
            "overall_risk_level": forensic_score.overall_risk_level,
            "summary": forensic_score.summary,
            "recommendations": forensic_score.recommendations,
            "flags": forensic_score.flags,
            "category_scores": [cs.model_dump() for cs in forensic_score.category_scores],
            "transcript_summary": {
                "text": (
                    transcript_result.text[:500] + "..."
                    if len(transcript_result.text) > 500
                    else transcript_result.text
                ),
                "num_segments": len(transcript_result.segments),
                "speakers": len(transcript_result.speakers),
            },
            "model_used": model_size,
            "cached_transcription": use_cached_transcription,
        }

        # Generate HTML report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        html_file = (
            output_path
            / f"forensic_result_{result['analyzed_at'][:10].replace(':', '').replace('-', '')}_{result['analysis_id']}.html"
        )

        html_generator = model_pool.get_html_generator()
        html_generator.generate(result, str(html_file))
        worker_logger.info(f"[{audio_name}] HTML saved: {html_file}")

        # Clean up GPU memory after forensic analysis
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        worker_logger.error(f"[{audio_name}] Analysis failed: {e}")
        import traceback

        worker_logger.error(traceback.format_exc())
        return {"error": str(e), "audio_file": str(audio_path)}


async def process_batch_async(
    audio_files: List[Path],
    output_dir: str,
    num_workers: int,
    model_size: str,
    e2e_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Process audio files in parallel with controlled concurrency.

    Args:
        audio_files: List of audio file paths
        output_dir: Output directory for results
        num_workers: Maximum number of concurrent workers
        model_size: Whisper model size
        e2e_cache: E2E transcription cache

    Returns:
        List of results
    """
    # Initialize shared model pool
    model_pool = SharedModelPool()

    # Semaphore to control concurrency
    semaphore = asyncio.Semaphore(num_workers)

    async def process_with_semaphore(file_path: Path, idx: int):
        """Process file with semaphore-controlled concurrency."""
        async with semaphore:
            cached_transcript = e2e_cache.get(str(file_path))
            return await process_single_file_async(
                str(file_path),
                output_dir,
                model_pool,
                idx,
                model_size,
                cached_transcript,
            )

    # Create tasks for all files
    tasks = [
        process_with_semaphore(file_path, i)
        for i, file_path in enumerate(audio_files)
    ]

    # Process with progress tracking
    results = []
    failed = []
    start_time = datetime.now()
    completed_count = 0
    total_count = len(tasks)

    logger.info(f"Starting parallel processing with {num_workers} workers")
    logger.info(f"Model: {model_size}, Files: {total_count}")

    # Process tasks as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed_count += 1

        if "error" in result:
            failed.append(result)
            logger.error(
                f"[{completed_count}/{total_count}] Failed: {Path(result['audio_file']).name} - {result['error']}"
            )
        else:
            results.append(result)
            cached_indicator = " [CACHED]" if result.get("cached_transcription") else ""
            logger.info(
                f"[{completed_count}/{total_count}] Completed: {Path(result['audio_file']).name}{cached_indicator} (Risk: {result['overall_risk_score']:.1f}/100)"
            )

        # Progress every 10 files
        if completed_count % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / completed_count
            remaining = (total_count - completed_count) * avg_time
            logger.info(
                f"Progress: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%) | "
                f"Avg: {avg_time:.1f}s/file | ETA: {remaining/60:.1f}min"
            )

    # Cleanup
    model_pool.cleanup_all()

    return results, failed


def main():
    parser = argparse.ArgumentParser(
        description="Optimized parallel batch forensic analysis with shared models"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing audio files")
    parser.add_argument(
        "--output-dir",
        default="ref/call/results/forensic",
        help="Output directory for results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--remaining-file", help="File containing list of remaining files")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "distil-large-v3",
        ],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--e2e-results",
        help="Path to E2E test report JSON to reuse existing transcriptions",
    )
    args = parser.parse_args()

    # Load E2E transcription cache if provided
    e2e_cache: Dict[str, Dict[str, Any]] = {}
    if args.e2e_results:
        e2e_cache = load_e2e_transcription_cache(args.e2e_results)

    # Get list of files to process
    if args.remaining_file:
        with open(args.remaining_file) as f:
            file_names = [line.strip() for line in f if line.strip()]
        audio_files = [
            Path(args.input_dir) / name
            for name in file_names
            if (Path(args.input_dir) / name).exists()
        ]
    else:
        audio_files = list(Path(args.input_dir).glob("*.m4a"))

    if args.limit:
        audio_files = audio_files[: args.limit]

    model_display = (
        f"Distil-Whisper ({args.model})" if args.model.startswith("distil-") else args.model
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZED BATCH FORENSIC ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Files to process: {len(audio_files)}")
    logger.info(f"Parallel workers:  {args.workers}")
    logger.info(f"Model:            {model_display}")
    if e2e_cache:
        logger.info(f"E2E cache:        {len(e2e_cache)} transcriptions available")
    logger.info("=" * 60)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process files in parallel
    start_time = datetime.now()

    # Run async processing
    results, failed = asyncio.run(
        process_batch_async(
            audio_files=audio_files,
            output_dir=args.output_dir,
            num_workers=args.workers,
            model_size=args.model,
            e2e_cache=e2e_cache,
        )
    )

    # Generate summary
    duration = (datetime.now() - start_time).total_seconds()
    cached_count = sum(1 for r in results if r.get("cached_transcription"))

    summary = {
        "total_files": len(audio_files),
        "successful": len(results),
        "failed": len(failed),
        "success_rate": len(results) / len(audio_files) if audio_files else 0,
        "duration_seconds": duration,
        "avg_time_per_file": duration / len(audio_files) if audio_files else 0,
        "model_used": args.model,
        "cached_transcriptions": cached_count,
        "results": results,
        "failed_files": failed,
    }

    # Save summary
    summary_file = (
        Path(args.output_dir)
        / f"batch_summary_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("OPTIMIZED BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total Files:        {summary['total_files']}")
    logger.info(f"Successful:         {summary['successful']}")
    logger.info(f"Failed:             {summary['failed']}")
    logger.info(f"Success Rate:       {summary['success_rate']:.1%}")
    logger.info(f"Model Used:         {args.model}")
    logger.info(f"Cached Transcripts: {cached_count}")
    logger.info(f"Duration:           {duration / 60:.1f} minutes")
    logger.info(f"Avg Time/File:      {summary['avg_time_per_file']:.1f} seconds")
    logger.info(f"Summary File:       {summary_file}")
    logger.info("=" * 60)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
