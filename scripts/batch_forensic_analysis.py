#!/usr/bin/env python3
"""
Batch Forensic Analysis Script

Run full forensic analysis on all audio files and generate reports.

This script performs:
1. Audio feature extraction (stress, pitch, etc.)
2. Speech emotion recognition (SER)
3. Crime language pattern detection
4. Cross-validation (text vs voice)
5. Forensic scoring
6. HTML/PDF report generation

Usage:
    python scripts/batch_forensic_analysis.py --input-dir ref/call --output-dir ref/call/results/forensic
    python scripts/batch_forensic_analysis.py --input-dir ref/call --limit 5  # Test with 5 files
"""

# ============================================================================
# PyTorch 2.6+ Compatibility Patch for weights_only
# Required for pyannote-audio model loading with newer PyTorch versions
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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm
import torch

# Add src to path
import sys

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
from voice_man.reports.pdf_generator import ForensicPDFGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitor GPU memory and trigger cleanup before OOM."""

    def __init__(
        self,
        warning_threshold: float = 70.0,  # Warning at 70%
        critical_threshold: float = 85.0,  # Critical at 85%
        oom_threshold: float = 95.0,  # Near OOM at 95%
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.oom_threshold = oom_threshold

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}

        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "usage_percent": (allocated / total) * 100,
        }

    def check_pressure(self) -> str:
        """Check memory pressure level: 'normal', 'warning', 'critical', 'oom'"""
        stats = self.get_memory_usage()
        if not stats["available"]:
            return "normal"

        usage = stats["usage_percent"]
        if usage >= self.oom_threshold:
            return "oom"
        elif usage >= self.critical_threshold:
            return "critical"
        elif usage >= self.warning_threshold:
            return "warning"
        return "normal"

    def should_trigger_cleanup(self) -> bool:
        """Check if cleanup should be triggered."""
        pressure = self.check_pressure()
        return pressure in ("warning", "critical", "oom")

    def log_status(self, context: str = ""):
        """Log current memory status."""
        stats = self.get_memory_usage()
        if stats["available"]:
            logger.info(
                f"{context}GPU: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB "
                f"({stats['usage_percent']:.1f}%)"
            )
        else:
            logger.info(f"{context}GPU: Not available")


class ForensicBatchProcessor:
    """Batch processor for forensic analysis with GPU memory management."""

    def __init__(
        self,
        whisperx_service: WhisperXService,
        audio_feature_service: AudioFeatureService,
        stress_analysis_service: StressAnalysisService,
        ser_service: SERService,
        cross_validation_service: CrossValidationService,
        crime_language_service: CrimeLanguageAnalysisService,
        forensic_scoring_service: ForensicScoringService,
        html_generator: ForensicHTMLGenerator,
        pdf_generator: Optional[ForensicPDFGenerator] = None,
        enable_memory_monitoring: bool = True,
        reload_interval: int = 20,
    ):
        self.whisperx_service = whisperx_service
        self.audio_feature_service = audio_feature_service
        self.stress_analysis_service = stress_analysis_service
        self.ser_service = ser_service
        self.cross_validation_service = cross_validation_service
        self.crime_language_service = crime_language_service
        self.forensic_scoring_service = forensic_scoring_service
        self.html_generator = html_generator
        self.pdf_generator = pdf_generator

        # Memory management
        self.enable_memory_monitoring = enable_memory_monitoring
        self.reload_interval = reload_interval
        self._processed_count = 0

        if enable_memory_monitoring:
            self._memory_monitor = GPUMemoryMonitor()
        else:
            self._memory_monitor = None

    def _cleanup_gpu_memory(self) -> None:
        """Aggressive GPU memory cleanup after each file."""
        try:
            # 1. Clear Python garbage
            gc.collect()

            # 2. Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # PyTorch 2.0+ IPC memory collection
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()

            # 3. Reset peak memory stats for accurate monitoring
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # 4. Log memory status
            if self.enable_memory_monitoring:
                self._memory_monitor.log_status("After cleanup: ")

        except Exception as e:
            logger.warning(f"GPU cleanup encountered error: {e}")

    def _check_memory_before_processing(self, audio_path: Path) -> bool:
        """Check memory pressure before processing. Return False if should skip/retry."""
        if not self.enable_memory_monitoring or not self._memory_monitor:
            return True

        pressure = self._memory_monitor.check_pressure()

        if pressure == "oom":
            logger.error(
                f"GPU memory at OOM threshold. Cannot process {audio_path.name}. "
                "Forcing cleanup and continuing..."
            )
            self._cleanup_gpu_memory()
            # Re-check after cleanup
            pressure = self._memory_monitor.check_pressure()
            if pressure == "oom":
                return False
        elif pressure == "critical":
            logger.warning(
                f"GPU memory at critical level before processing {audio_path.name}. "
                "Forcing cleanup..."
            )
            self._cleanup_gpu_memory()

        return True

    def _maybe_reload_models(self) -> None:
        """Periodically reload models to clear accumulated state."""
        self._processed_count += 1

        if self._processed_count >= self.reload_interval:
            logger.info(
                f"Processed {self._processed_count} files. "
                "Reloading models to clear accumulated state..."
            )

            # Unload WhisperX (largest memory consumer)
            try:
                if hasattr(self.whisperx_service, "unload"):
                    self.whisperx_service.unload()
                # Create new WhisperXService instance to reload models
                from voice_man.services.whisperx_service import WhisperXService

                self.whisperx_service = WhisperXService(
                    model_size=self.whisperx_service.model_size,
                    device=self.whisperx_service.device,
                    language=self.whisperx_service.language,
                    compute_type=self.whisperx_service.compute_type,
                )
                logger.info("WhisperX service reloaded successfully")
            except Exception as e:
                logger.warning(f"WhisperX reload error: {e}")

            # Unload SER models if available
            try:
                if hasattr(self.ser_service, "unload_models"):
                    self.ser_service.unload_models()
            except Exception as e:
                logger.warning(f"SER unload error: {e}")

            self._processed_count = 0
            self._cleanup_gpu_memory()

    async def analyze_single_file(
        self,
        audio_path: Path,
        transcript_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform full forensic analysis on a single audio file.
        With GPU memory management.

        Args:
            audio_path: Path to audio file
            transcript_data: Optional pre-computed transcript data from E2E results

        Returns:
            Forensic analysis result dictionary
        """
        analysis_id = str(uuid.uuid4())[:8]
        logger.info(f"[{analysis_id}] Starting forensic analysis: {audio_path.name}")

        # Check memory before processing
        if not self._check_memory_before_processing(audio_path):
            return {
                "analysis_id": analysis_id,
                "audio_file": str(audio_path),
                "error": "GPU memory insufficient",
                "status": "skipped",
            }

        try:
            # Log memory before processing
            if self.enable_memory_monitoring and self._memory_monitor:
                self._memory_monitor.log_status(f"[{analysis_id}] Before: ")

            # Step 1: Get or compute transcription
            if transcript_data is None:
                logger.info(f"[{analysis_id}] Computing transcription...")
                transcript_result = await self.whisperx_service.process_audio(str(audio_path))
                transcript_text = transcript_result.text

                # Cleanup after WhisperX (largest memory consumer)
                del transcript_result
                self._cleanup_gpu_memory()
            else:
                transcript_text = transcript_data.get("transcript_text", "")

            # Step 2: Use ForensicScoringService.analyze for complete analysis
            logger.info(f"[{analysis_id}] Running forensic analysis...")
            forensic_score = await self.forensic_scoring_service.analyze(
                audio_path=str(audio_path),
                transcript=transcript_text,
            )

            # Build result from ForensicScoreResult
            result = {
                "analysis_id": analysis_id,
                "analyzed_at": datetime.now().isoformat(),
                "audio_file": str(audio_path),
                "audio_duration_seconds": forensic_score.audio_duration_seconds,
                "overall_risk_score": forensic_score.overall_risk_score,
                "overall_risk_level": forensic_score.overall_risk_level,
                "confidence_level": forensic_score.confidence_level,
                "summary": forensic_score.summary,
                "recommendations": forensic_score.recommendations,
                "flags": forensic_score.flags,
                "category_scores": [
                    {
                        "category": cat.category,
                        "score": cat.score,
                        "confidence": cat.confidence,
                        "evidence_count": cat.evidence_count,
                        "key_indicators": cat.key_indicators,
                    }
                    for cat in forensic_score.category_scores
                ],
                "gaslighting_analysis": {
                    "intensity_score": forensic_score.gaslighting_analysis.intensity_score,
                    "patterns_detected": forensic_score.gaslighting_analysis.patterns_detected,
                    "manipulation_techniques": forensic_score.gaslighting_analysis.manipulation_techniques,
                    "victim_impact_level": forensic_score.gaslighting_analysis.victim_impact_level,
                },
                "threat_assessment": {
                    "threat_level": forensic_score.threat_assessment.threat_level,
                    "threat_types": forensic_score.threat_assessment.threat_types,
                    "immediacy": forensic_score.threat_assessment.immediacy,
                    "specificity": forensic_score.threat_assessment.specificity,
                    "credibility_score": forensic_score.threat_assessment.credibility_score,
                },
                "deception_analysis": {
                    "deception_probability": forensic_score.deception_analysis.deception_probability,
                    "voice_text_consistency": forensic_score.deception_analysis.voice_text_consistency,
                    "emotional_authenticity": forensic_score.deception_analysis.emotional_authenticity,
                    "linguistic_markers_count": forensic_score.deception_analysis.linguistic_markers_count,
                    "behavioral_indicators": forensic_score.deception_analysis.behavioral_indicators,
                },
                "transcript": {
                    "text": transcript_text,
                },
            }

            # Delete forensic_score to free memory
            del forensic_score

            logger.info(
                f"[{analysis_id}] Analysis complete. Risk score: {result['overall_risk_score']:.1f}/100"
            )

            return result

        except Exception as e:
            logger.error(f"[{analysis_id}] Analysis failed: {e}")
            return {
                "analysis_id": analysis_id,
                "audio_file": str(audio_path),
                "error": str(e),
                "status": "failed",
            }
        finally:
            # Always cleanup after each file, regardless of success/failure
            self._cleanup_gpu_memory()

            # Check if models need periodic reload
            self._maybe_reload_models()

            # Log memory after processing
            if self.enable_memory_monitoring and self._memory_monitor:
                self._memory_monitor.log_status(f"[{analysis_id}] After: ")

    def _get_processed_files(self, output_dir: Path) -> set:
        """Get set of audio file names that have already been processed.

        Args:
            output_dir: Directory containing forensic HTML reports

        Returns:
            Set of audio file names that have reports
        """
        processed = set()
        if output_dir and output_dir.exists():
            for html_file in output_dir.glob("forensic_result_*.html"):
                try:
                    with open(html_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Extract audio file name from HTML content
                        import re

                        # Match Korean audio file names: 통화 녹음 XXX_YYMMDD_HHMMSS.m4a
                        match = re.search(r"통화 녹음[^<>\"]+\.m4a", content)
                        if match:
                            file_name = match.group(0).strip()
                            # Remove any HTML tags or artifacts
                            file_name = re.sub(r"<[^>]+>", "", file_name).strip()
                            if file_name.endswith(".m4a"):
                                processed.add(file_name)
                except Exception:
                    pass
        return processed

    async def process_batch(
        self,
        audio_files: List[Path],
        transcript_map: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: Optional[Path] = None,
        generate_pdf: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a batch of audio files through forensic analysis.

        Args:
            audio_files: List of audio file paths
            transcript_map: Optional map of file paths to transcript data
            output_dir: Output directory for reports
            generate_pdf: Whether to generate PDF reports

        Returns:
            Batch processing summary
        """
        # Get already processed files to skip
        processed_files = self._get_processed_files(output_dir) if output_dir else set()
        skipped = len([f for f in audio_files if f.name in processed_files])

        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files")

        total = len(audio_files)
        results = []
        failed = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        with tqdm.tqdm(total=total, desc="Forensic Analysis", unit="file") as pbar:
            for audio_path in audio_files:
                pbar.set_description(f"Processing: {audio_path.name[:40]}")

                # Skip if already processed
                if audio_path.name in processed_files:
                    logger.info(f"Skipping already processed: {audio_path.name}")
                    pbar.update(1)
                    continue

                # Get transcript data if available
                transcript_data = None
                if transcript_map:
                    transcript_data = transcript_map.get(str(audio_path))

                # Run analysis
                result = await self.analyze_single_file(audio_path, transcript_data)

                if "error" in result:
                    failed.append(result)
                else:
                    results.append(result)

                    # Generate HTML report
                    if output_dir:
                        html_path = (
                            output_dir
                            / f"forensic_result_{result['analyzed_at'][:10].replace(':', '').replace('-', '')}_{result['analysis_id']}.html"
                        )
                        self.html_generator.generate(result, str(html_path))

                        # Generate PDF report if requested
                        if generate_pdf and self.pdf_generator:
                            pdf_path = (
                                output_dir
                                / f"forensic_result_{result['analyzed_at'][:10].replace(':', '').replace('-', '')}_{result['analysis_id']}.pdf"
                            )
                            try:
                                await self.pdf_generator.generate(result, str(pdf_path))
                            except Exception as e:
                                logger.warning(f"PDF generation failed for {audio_path.name}: {e}")

                pbar.update(1)

        # Generate summary
        summary = {
            "total_files": total,
            "successful": len(results),
            "failed": len(failed),
            "success_rate": len(results) / total if total > 0 else 0,
            "results": results,
            "failed_files": failed,
            "timestamp": datetime.now().isoformat(),
        }

        # Save summary JSON
        if output_dir:
            summary_path = (
                output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


async def load_e2e_transcripts(e2e_report_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load transcript data from E2E test report.

    Args:
        e2e_report_path: Path to E2E test report JSON

    Returns:
        Map of file paths to transcript data
    """
    with open(e2e_report_path, "r", encoding="utf-8") as f:
        e2e_data = json.load(f)

    file_results = e2e_data.get("result", {}).get("file_results", [])

    transcript_map = {}
    for result in file_results:
        if result.get("status") == "success":
            file_path = result.get("file_path")
            if file_path:
                transcript_map[file_path] = {
                    "transcript_text": result.get("transcript_text", ""),
                    "segments": result.get("segments", []),
                    "speakers": result.get("speakers", []),
                }

    logger.info(f"Loaded {len(transcript_map)} transcripts from E2E report")
    return transcript_map


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Batch forensic analysis for audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing audio files",
    )

    parser.add_argument(
        "--e2e-report",
        help="Path to E2E test report JSON (for pre-computed transcripts)",
    )

    parser.add_argument(
        "--output-dir",
        default="ref/call/results/forensic",
        help="Output directory for forensic reports",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate PDF reports",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for WhisperX (default: cuda)",
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size",
    )

    parser.add_argument(
        "--reload-interval",
        type=int,
        default=20,
        help="Reload models every N files to clear memory (default: 20)",
    )

    parser.add_argument(
        "--no-memory-monitor",
        action="store_true",
        help="Disable GPU memory monitoring",
    )

    args = parser.parse_args()

    # Collect audio files
    input_dir = Path(args.input_dir)
    audio_files = sorted(input_dir.glob("*.m4a"))

    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return 1

    # Apply limit
    if args.limit:
        audio_files = audio_files[: args.limit]

    logger.info(f"Processing {len(audio_files)} audio files")

    # Initialize services
    logger.info("Initializing services...")

    whisperx_service = WhisperXService(
        model_size=args.model,
        device=args.device,
        language="ko",
        compute_type="float16",
    )

    # Initialize independent services first
    audio_feature_service = AudioFeatureService()
    stress_analysis_service = StressAnalysisService()
    ser_service = SERService()
    crime_language_service = CrimeLanguageAnalysisService()

    # Initialize dependent services
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
    html_generator = ForensicHTMLGenerator()

    pdf_generator = None
    if args.pdf:
        pdf_generator = ForensicPDFGenerator()

    # Create processor with memory management
    processor = ForensicBatchProcessor(
        whisperx_service=whisperx_service,
        audio_feature_service=audio_feature_service,
        stress_analysis_service=stress_analysis_service,
        ser_service=ser_service,
        cross_validation_service=cross_validation_service,
        crime_language_service=crime_language_service,
        forensic_scoring_service=forensic_scoring_service,
        html_generator=html_generator,
        pdf_generator=pdf_generator,
        enable_memory_monitoring=not args.no_memory_monitor,
        reload_interval=args.reload_interval,
    )

    # Log GPU memory monitoring status
    if not args.no_memory_monitor:
        logger.info("GPU memory monitoring: ENABLED")
        logger.info(f"Model reload interval: every {args.reload_interval} files")
    else:
        logger.info("GPU memory monitoring: DISABLED")

    # Load E2E transcripts if available
    transcript_map = None
    if args.e2e_report:
        e2e_path = Path(args.e2e_report)
        if e2e_path.exists():
            logger.info(f"Loading transcripts from {e2e_path}")
            transcript_map = await load_e2e_transcripts(e2e_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    start_time = datetime.now()
    logger.info("Starting batch forensic analysis...")

    summary = await processor.process_batch(
        audio_files=audio_files,
        transcript_map=transcript_map,
        output_dir=output_dir,
        generate_pdf=args.pdf,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print summary
    print("\n" + "=" * 60)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total Files:    {summary['total_files']}")
    print(f"  Successful:     {summary['successful']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Success Rate:   {summary['success_rate']:.1%}")
    print(f"  Duration:       {duration:.1f}s ({duration / 60:.1f} minutes)")
    print(f"  Avg Time/File:  {duration / summary['total_files']:.1f}s")
    print("=" * 60)

    # Cleanup
    whisperx_service.unload()

    if summary["failed"] > 0:
        logger.warning(f"{summary['failed']} files failed analysis")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
