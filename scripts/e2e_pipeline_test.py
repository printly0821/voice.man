#!/usr/bin/env python3
"""
E2E Pipeline Test Script - STT + Forensic Integration.

Uses PipelineOrchestrator for parallel Producer-Consumer processing:
- Producer: WhisperX STT transcription
- Consumer: ForensicScoringService analysis

SPEC-PERFOPT-001 Phase 3 Integration Test.

Usage:
    python scripts/e2e_pipeline_test.py --input-dir ref/call/e2e_test_top10
    python scripts/e2e_pipeline_test.py --input-dir ref/call/e2e_test_top10 --output-dir results
"""

# ============================================================================
# PyTorch 2.6+ Compatibility Patch for weights_only
# ============================================================================
import lightning_fabric.utilities.cloud_io as _cloud_io

_original_pl_load = _cloud_io._load


def _patched_pl_load(path_or_url, map_location=None, weights_only=None):
    """Patched _load to default weights_only=False for backward compatibility."""
    if weights_only is None:
        weights_only = False
    return _original_pl_load(path_or_url, map_location, weights_only)


_cloud_io._load = _patched_pl_load

try:
    import pyannote.audio.core.model as _pyannote_model

    _pyannote_model.pl_load = _patched_pl_load
except ImportError:
    pass
# ============================================================================

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_man.services.whisperx_service import WhisperXService
from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService
from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator
from voice_man.services.forensic.memory_manager import ForensicMemoryManager
from voice_man.services.forensic.thermal_manager import ThermalManager
from voice_man.services.forensic.audio_feature_service import AudioFeatureService
from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
from voice_man.services.forensic.ser_service import SERService
from voice_man.services.forensic.cross_validation_service import CrossValidationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E2E Pipeline Test - STT + Forensic with PipelineOrchestrator",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: input-dir/pipeline_results)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Expected number of speakers (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    return parser.parse_args()


def collect_audio_files(directory: Path) -> List[Path]:
    """Collect audio files from directory."""
    extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


async def run_pipeline_test(
    input_dir: Path,
    output_dir: Path,
    num_speakers: int,
    device: str,
) -> dict:
    """Run the E2E pipeline test."""

    # Collect files
    files = collect_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        return {"error": "No files found"}

    logger.info(f"Found {len(files)} audio files")

    # Initialize services
    logger.info("Initializing WhisperX service...")
    stt_service = WhisperXService(
        model_size="large-v3",
        device=device,
        language="ko",
    )

    logger.info("Initializing Forensic services...")
    audio_feature_service = AudioFeatureService()
    stress_analysis_service = StressAnalysisService()
    crime_language_service = CrimeLanguageAnalysisService()
    # SPEC-PERFOPT-001: Disable ensemble for faster processing (single model mode)
    # Ensemble: ~10min/file, Single: ~5min/file expected (50% improvement)
    ser_service = SERService(use_ensemble=False)
    cross_validation_service = CrossValidationService(
        ser_service=ser_service,
        crime_language_service=crime_language_service,
    )

    logger.info("Initializing ForensicScoringService...")
    # SPEC-PERFOPT-001: Enable Text-Only Mode for maximum speed
    # Skip both SER and Audio Feature analysis (500s+ each in parallel)
    # Text-based analysis only (gaslighting, threat, coercion detection preserved)
    # Expected improvement: ~10min → ~2min per file
    forensic_service = ForensicScoringService(
        audio_feature_service=audio_feature_service,
        stress_analysis_service=stress_analysis_service,
        crime_language_service=crime_language_service,
        ser_service=ser_service,
        cross_validation_service=cross_validation_service,
        skip_ser=True,  # Skip SER for faster processing
        skip_audio_features=True,  # Skip Audio Features for Text-Only Mode
    )

    logger.info("Initializing memory and thermal managers...")
    memory_manager = ForensicMemoryManager()
    thermal_manager = ThermalManager()

    # Start thermal monitoring
    thermal_manager.start_monitoring(interval_seconds=2.0)

    # Create PipelineOrchestrator
    logger.info("Creating PipelineOrchestrator...")
    orchestrator = PipelineOrchestrator(
        stt_service=stt_service,
        forensic_service=forensic_service,
        memory_manager=memory_manager,
        thermal_manager=thermal_manager,
    )

    # Track results
    results = []
    start_time = time.time()
    processed_count = 0
    success_count = 0
    error_count = 0

    logger.info(f"Starting pipeline processing of {len(files)} files...")
    logger.info("=" * 60)

    try:
        # Process files using PipelineOrchestrator
        async for result in orchestrator.process_files(files):
            processed_count += 1
            file_path = result.get("file_path", "unknown")
            file_name = Path(file_path).name
            error = result.get("error")

            if error:
                error_count += 1
                logger.warning(f"[{processed_count}/{len(files)}] FAILED: {file_name} - {error}")
                results.append(
                    {
                        "file": file_name,
                        "status": "failed",
                        "error": error,
                    }
                )
            else:
                success_count += 1
                forensic_result = result.get("result")
                logger.info(f"[{processed_count}/{len(files)}] SUCCESS: {file_name}")

                # Extract key metrics from forensic result
                result_summary = {
                    "file": file_name,
                    "status": "success",
                }

                if forensic_result:
                    # Add forensic summary if available
                    if hasattr(forensic_result, "overall_score"):
                        result_summary["overall_score"] = forensic_result.overall_score
                    if hasattr(forensic_result, "risk_level"):
                        result_summary["risk_level"] = forensic_result.risk_level

                results.append(result_summary)

            # Show progress
            elapsed = time.time() - start_time
            avg_per_file = elapsed / processed_count
            remaining = (len(files) - processed_count) * avg_per_file

            logger.info(
                f"Progress: {processed_count}/{len(files)} | "
                f"Success: {success_count} | Failed: {error_count} | "
                f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s"
            )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    finally:
        # Shutdown orchestrator and managers
        await orchestrator.shutdown()
        thermal_manager.stop_monitoring()

    total_time = time.time() - start_time

    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "total_files": len(files),
        "processed": processed_count,
        "success": success_count,
        "failed": error_count,
        "success_rate": f"{(success_count / len(files) * 100):.1f}%",
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_file": round(total_time / len(files), 2) if files else 0,
        "results": results,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE TEST COMPLETE")
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Success: {success_count} ({success_count / len(files) * 100:.1f}%)")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average per file: {total_time / len(files):.1f}s")
    logger.info(f"Results saved to: {result_file}")

    return summary


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir or args.input_dir / "pipeline_results"

    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Num speakers: {args.num_speakers}")

    # Run test
    try:
        result = asyncio.run(
            run_pipeline_test(
                input_dir=args.input_dir,
                output_dir=output_dir,
                num_speakers=args.num_speakers,
                device=args.device,
            )
        )

        if result.get("error"):
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
