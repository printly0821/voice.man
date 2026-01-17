#!/usr/bin/env python3
"""
Full Forensic Pipeline with Asset Tracking
==========================================

Runs STT + Forensic analysis (without diarization due to CUDA issues)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline_execution.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set PyTorch environment
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class FullForensicPipeline:
    """Complete forensic pipeline with asset tracking"""

    def __init__(self):
        """Initialize all services"""
        logger.info("Initializing forensic pipeline...")
        from voice_man.services.asset_tracking import (
            AssetRegistry,
            CoreAssetManager,
            AssetStatus,
        )
        from voice_man.services.whisperx_service import WhisperXService
        from voice_man.services.forensic.audio_feature_service import (
            AudioFeatureService,
        )
        from voice_man.services.forensic.stress_analysis_service import (
            StressAnalysisService,
        )
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.services.forensic.forensic_scoring_service import (
            ForensicScoringService,
        )

        # Asset tracking
        self.asset_registry = AssetRegistry()
        self.core_asset_manager = CoreAssetManager()

        # STT service
        self.stt_service = WhisperXService(device="cuda", language="ko")

        # Forensic services
        self.audio_feature_service = AudioFeatureService()
        self.stress_analysis_service = StressAnalysisService()
        self.ser_service = SERService()
        self.crime_language_service = CrimeLanguageAnalysisService()
        self.cross_validation_service = CrossValidationService(
            crime_language_service=self.crime_language_service,
            ser_service=self.ser_service,
        )
        self.forensic_service = ForensicScoringService(
            audio_feature_service=self.audio_feature_service,
            stress_analysis_service=self.stress_analysis_service,
            crime_language_service=self.crime_language_service,
            ser_service=self.ser_service,
            cross_validation_service=self.cross_validation_service,
            skip_ser=False,  # Enable SER
            skip_audio_features=False,  # Enable audio features
        )

        logger.info("All services initialized")

    async def process_file(self, audio_file: Path) -> Dict:
        """
        Process a single file through STT + Forensic pipeline

        Args:
            audio_file: Path to audio file

        Returns:
            Processing result
        """
        from voice_man.services.asset_tracking import AssetStatus

        file_path_str = str(audio_file)

        # Register asset
        asset = self.core_asset_manager.register_audio_file(
            file_path_str,
            copy_to_storage=False,
        )
        logger.info(f"Registered asset: {asset.asset_id}")

        try:
            # Step 1: STT (transcription only, no diarization)
            logger.info(f"STT processing: {audio_file.name}")
            stt_result = await self.stt_service.transcribe_only(file_path_str)

            if "error" in stt_result:
                raise Exception(f"STT failed: {stt_result['error']}")

            # Extract text from segments (WhisperX raw result format)
            segments = stt_result.get("segments", [])
            text = " ".join(seg.get("text", "") for seg in segments)

            if not text:
                raise Exception("Transcription produced empty text")

            logger.info(f"STT complete: {len(segments)} segments, {len(text)} chars")

            # Step 2: SER (Speech Emotion Recognition) - analyzes audio directly
            logger.info(f"SER processing: {audio_file.name}")
            try:
                ser_result = self.ser_service.analyze_ensemble_from_file(file_path_str)
                logger.info(f"SER complete: {ser_result.ensemble_emotion.emotion_type}")
            except Exception as e:
                logger.warning(f"SER failed (continuing): {e}")
                ser_result = None

            # Step 3: Crime Language Analysis - analyzes transcript text
            logger.info(f"Crime language analysis: {audio_file.name}")
            try:
                crime_result = self.crime_language_service.analyze_comprehensive(text)
                logger.info(f"Crime analysis complete: risk={crime_result.overall_risk_score:.2f}")
            except Exception as e:
                logger.warning(f"Crime analysis failed (continuing): {e}")
                crime_result = None

            # Step 4: Forensic Scoring - comprehensive analysis
            logger.info(f"Forensic scoring: {audio_file.name}")
            try:
                forensic_result = await self.forensic_service.analyze(
                    audio_path=file_path_str,
                    transcript=text,
                )
                logger.info(f"Forensic complete: risk={forensic_result.overall_risk_score:.1f}")
            except Exception as e:
                logger.warning(f"Forensic scoring failed (continuing): {e}")
                forensic_result = None

            # Update asset status
            self.asset_registry.update_asset(asset.asset_id, status=AssetStatus.COMPLETED)

            return {
                "file": file_path_str,
                "asset_id": asset.asset_id,
                "status": "completed",
                "stt": {"segments": len(segments), "chars": len(text)},
                "ser": ser_result.ensemble_emotion.emotion_type if ser_result else None,
                "crime": crime_result.overall_risk_score if crime_result else None,
                "forensic": forensic_result.overall_risk_score if forensic_result else None,
            }

        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            self.asset_registry.update_asset(
                asset.asset_id, status=AssetStatus.FAILED, error_message=str(e)
            )
            return {
                "file": file_path_str,
                "asset_id": asset.asset_id,
                "status": "failed",
                "error": str(e),
            }

    async def run_batch(self, files: List[Path]) -> List[Dict]:
        """
        Process a batch of files

        Args:
            files: List of audio file paths

        Returns:
            List of processing results
        """
        results = []

        for i, audio_file in enumerate(files, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{i}/{len(files)}] Processing: {audio_file.name}")
            logger.info(f"{'=' * 60}")

            result = await self.process_file(audio_file)
            results.append(result)

        return results

    def print_summary(self, results: List[Dict]):
        """Print processing summary"""
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")

        logger.info(f"\n{'=' * 60}")
        logger.info("EXECUTION SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total: {len(results)}")
        logger.info(f"Completed: {completed} ({completed / len(results) * 100:.1f}%)")
        logger.info(f"Failed: {failed}")

        if completed > 0:
            forensic_scores = [r["forensic"] for r in results if r.get("forensic") is not None]
            if forensic_scores:
                avg_score = sum(forensic_scores) / len(forensic_scores)
                logger.info(f"Avg forensic score: {avg_score:.1f}")

        logger.info(f"{'=' * 60}\n")

    def cleanup(self):
        """Cleanup resources"""
        self.stt_service.unload()


async def main():
    """Main entry point"""
    import sys

    # Get audio files
    audio_files = sorted(Path("ref/call").glob("*.m4a"))

    # Determine batch size from command line argument (default: 10 for Stage 1)
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Process specified batch size
    if batch_size == 10:
        stage_name = "Stage 1: Pilot"
    elif batch_size == 50:
        stage_name = "Stage 2: Medium"
    else:
        stage_name = f"Stage: {batch_size} files"

    batch_files = audio_files[:batch_size]

    logger.info(f"\nStarting {stage_name} batch ({len(batch_files)} files)")

    pipeline = FullForensicPipeline()
    results = await pipeline.run_batch(batch_files)
    pipeline.print_summary(results)
    pipeline.cleanup()

    return results


if __name__ == "__main__":
    asyncio.run(main())
