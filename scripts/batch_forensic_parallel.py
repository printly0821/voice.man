#!/usr/bin/env python3
"""
Parallel Batch Forensic Analysis Script

Process multiple audio files in parallel using multiprocessing.
Each process handles its own GPU memory allocation.

Features:
- Distil-Whisper model support for 4-6x faster transcription
- E2E transcription result reuse to skip redundant processing
- Progress tracking and summary reports

Usage:
    python scripts/batch_forensic_parallel.py --input-dir /tmp/failed_calls --workers 2
    python scripts/batch_forensic_parallel.py --input-dir ref/call --workers 3 --limit 10
    python scripts/batch_forensic_parallel.py --input-dir ref/call --model distil-large-v3 --e2e-results ref/call/results/e2e_test_report.json
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
import multiprocessing as mp
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def process_single_file(
    audio_path: str,
    output_dir: str,
    worker_id: int,
    gpu_id: int = 0,
    model_size: str = "large-v3",
    e2e_transcription: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single audio file through forensic analysis.
    Runs in a separate process with its own GPU memory.

    Args:
        audio_path: Path to audio file
        output_dir: Output directory for results
        worker_id: Worker identifier
        gpu_id: GPU device ID
        model_size: Whisper model size (e.g., large-v3, distil-large-v3)
        e2e_transcription: Optional existing transcription to reuse
    """
    import logging
    import torch

    # Set up logging for this worker
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    worker_logger.setLevel(logging.INFO)

    try:
        # Set GPU device for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        worker_logger.info(f"[Worker-{worker_id}] Processing: {Path(audio_path).name}")
        worker_logger.info(f"[Worker-{worker_id}] CUDA available: {torch.cuda.is_available()}")
        worker_logger.info(f"[Worker-{worker_id}] Model: {model_size}")

        # Check if we have existing transcription
        use_cached_transcription = False
        if e2e_transcription:
            worker_logger.info(
                f"[Worker-{worker_id}] Reusing existing transcription (skipping STT/alignment)"
            )
            use_cached_transcription = True

        # Initialize services (each worker gets its own instances)
        # ForensicScoringService needs these services internally
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

        # Initialize WhisperX and HTML generator
        whisperx_service = WhisperXService(
            model_size=model_size, device="cuda", language="ko", compute_type="float16"
        )
        html_generator = ForensicHTMLGenerator()

        # Run transcription
        worker_logger.info(f"[Worker-{worker_id}] Computing transcription...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Process with optional existing transcription
            transcript_result = loop.run_until_complete(
                whisperx_service.process_audio(
                    str(audio_path),
                    existing_transcription=e2e_transcription if use_cached_transcription else None,
                )
            )

            # Run forensic analysis - analyze() handles all sub-analyses internally
            worker_logger.info(f"[Worker-{worker_id}] Running forensic analysis...")
            forensic_score = loop.run_until_complete(
                forensic_scoring_service.analyze(
                    audio_path=str(audio_path),
                    transcript=transcript_result.text,
                )
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
                    "text": transcript_result.text[:500] + "..."
                    if len(transcript_result.text) > 500
                    else transcript_result.text,
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
            html_generator.generate(result, str(html_file))
            worker_logger.info(f"[Worker-{worker_id}] HTML saved: {html_file}")

            # Cleanup
            del whisperx_service
            del ser_service
            del forensic_scoring_service
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loop.close()
            return result

        except Exception as e:
            worker_logger.error(f"[Worker-{worker_id}] Analysis failed: {e}")
            return {"error": str(e), "audio_file": str(audio_path)}

    except Exception as e:
        worker_logger.error(f"[Worker-{worker_id}] Fatal error: {e}")
        return {"error": str(e), "audio_file": str(audio_path)}

    finally:
        # Always cleanup GPU memory, even on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch forensic analysis with Distil-Whisper support"
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
        help="Whisper model size (default: large-v3). Use distil-large-v3 for 4-6x faster transcription.",
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
    logger.info(f"Processing {len(audio_files)} files with {args.workers} parallel workers")
    logger.info(f"Model: {model_display}")
    if e2e_cache:
        logger.info(f"E2E cache: {len(e2e_cache)} transcriptions available for reuse")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process files in parallel
    results = []
    failed = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for i, audio_path in enumerate(audio_files):
            # Round-robin GPU assignment (all use GPU 0 for now)
            gpu_id = 0

            # Get cached transcription if available
            cached_transcript = e2e_cache.get(str(audio_path))

            future = executor.submit(
                process_single_file,
                str(audio_path),
                args.output_dir,
                i,
                gpu_id,
                args.model,
                cached_transcript,
            )
            future_to_file[future] = audio_path

        # Process completed tasks
        for future in as_completed(future_to_file):
            audio_path = future_to_file[future]
            try:
                result = future.result()
                if "error" in result:
                    failed.append(result)
                    logger.error(f"Failed: {audio_path.name} - {result['error']}")
                else:
                    results.append(result)
                    cached_indicator = " [CACHED]" if result.get("cached_transcription") else ""
                    logger.info(
                        f"Completed: {audio_path.name}{cached_indicator} (Risk: {result['overall_risk_score']:.1f}/100)"
                    )
            except Exception as e:
                logger.error(f"Exception processing {audio_path.name}: {e}")
                failed.append({"error": str(e), "audio_file": str(audio_path)})

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
        / f"batch_summary_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info("PARALLEL BATCH PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total Files:        {summary['total_files']}")
    logger.info(f"Successful:         {summary['successful']}")
    logger.info(f"Failed:             {summary['failed']}")
    logger.info(f"Success Rate:       {summary['success_rate']:.1%}")
    logger.info(f"Model Used:         {args.model}")
    logger.info(f"Cached Transcripts: {cached_count}")
    logger.info(f"Duration:           {duration / 60:.1f} minutes")
    logger.info(f"Avg Time/File:      {summary['avg_time_per_file']:.1f} seconds")
    logger.info("=" * 50)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
