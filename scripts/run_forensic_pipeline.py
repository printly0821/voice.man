#!/usr/bin/env python3
"""
Forensic Pipeline Runner with Asset Tracking
=============================================

Integrates GPU pipeline orchestrator with staged batch processing:
- Stage 1: 10 files (pilot)
- Stage 2: 50 files (medium)
- Stage 3: All files (full)

Usage:
    python scripts/run_forensic_pipeline.py
    python scripts/run_forensic_pipeline.py --stage pilot --files 10
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline_execution.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ForensicPipelineRunner:
    """
    Integrated forensic pipeline runner with asset tracking
    """

    def __init__(
        self,
        audio_dir: Path = Path("ref/call"),
        checkpoint_dir: str = "data/checkpoints",
        enable_gpu: bool = True,
    ):
        """Initialize pipeline runner"""
        self.audio_dir = audio_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_gpu = enable_gpu

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Initialize services
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all required services"""
        logger.info("Initializing services...")

        # Asset tracking
        from voice_man.services.asset_tracking import (
            AssetRegistry,
            CoreAssetManager,
            StagePipeline,
            StagedBatchConfig,
            ProcessingStage,
            AssetStatus,
        )

        self.asset_registry = AssetRegistry()
        self.core_asset_manager = CoreAssetManager()
        self.stage_pipeline = StagePipeline(
            config=StagedBatchConfig(
                pilot_batch_size=10,
                medium_batch_size=50,
                full_batch_size=184,  # Will be updated based on actual file count
                checkpoint_dir=str(self.checkpoint_dir),
            ),
            asset_registry=self.asset_registry,
        )

        # GPU Pipeline
        from voice_man.services.forensic.gpu_pipeline_orchestrator import (
            GPUPipelineOrchestrator,
        )
        from voice_man.services.whisperx_service import WhisperXService
        from voice_man.services.forensic.forensic_scoring_service import (
            ForensicScoringService,
        )

        # Initialize services (with placeholder dependencies)
        self.stt_service = WhisperXService(device="cuda" if self.enable_gpu else "cpu")
        self.forensic_service = self._create_forensic_service()

        self.gpu_orchestrator = GPUPipelineOrchestrator(
            stt_service=self.stt_service,
            forensic_service=self.forensic_service,
            enable_gpu_optimization=self.enable_gpu,
        )

        logger.info("Services initialized successfully")

    def _create_forensic_service(self):
        """Create forensic scoring service with all dependencies"""
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

        # Initialize independent services first (no dependencies)
        audio_feature_service = AudioFeatureService()
        stress_analysis_service = StressAnalysisService()
        ser_service = SERService()
        crime_language_service = CrimeLanguageAnalysisService()

        # Initialize dependent services
        cross_validation_service = CrossValidationService(
            crime_language_service=crime_language_service,
            ser_service=ser_service,
        )

        # Finally initialize ForensicScoringService with all dependencies
        return ForensicScoringService(
            audio_feature_service=audio_feature_service,
            stress_analysis_service=stress_analysis_service,
            crime_language_service=crime_language_service,
            ser_service=ser_service,
            cross_validation_service=cross_validation_service,
        )

    def get_audio_files(self, limit: int = None) -> List[Path]:
        """
        Get audio files for processing

        Args:
            limit: Maximum number of files to return

        Returns:
            List of audio file paths
        """
        audio_files = sorted(self.audio_dir.glob("*.m4a"))

        if limit:
            audio_files = audio_files[:limit]

        logger.info(f"Found {len(audio_files)} audio files")
        return audio_files

    async def process_stage(
        self,
        file_paths: List[Path],
        stage_name: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """
        Process a single stage

        Args:
            file_paths: List of file paths to process
            stage_name: Stage name (pilot, medium, full)
            batch_id: Batch ID for tracking

        Returns:
            Processing result summary
        """
        from voice_man.services.asset_tracking.staged_batch_processor import (
            StageResult,
            ProcessingStage,
        )
        from voice_man.services.asset_tracking import AssetStatus

        logger.info(f"Starting {stage_name} stage with {len(file_paths)} files")
        start_time = datetime.now(timezone.utc)

        # Register all files as assets
        asset_ids = []
        for file_path in file_paths:
            asset = self.core_asset_manager.register_audio_file(
                str(file_path),
                copy_to_storage=False,  # Don't copy, just track
            )
            asset_ids.append(asset.asset_id)

        # Process files through GPU pipeline
        processed = []
        failed = []
        results = []

        try:
            async for result in self.gpu_orchestrator.process_files(file_paths):
                file_path = str(result["file_path"])  # Convert Path to str
                asset = self.core_asset_manager.get_asset_by_original_path(file_path)

                if result.get("error"):
                    failed.append(asset.asset_id)
                    self.asset_registry.update_asset(
                        asset.asset_id,
                        status=AssetStatus.FAILED,
                        error_message=result["error"],
                    )
                    logger.error(f"Failed to process {file_path}: {result['error']}")
                else:
                    processed.append(asset.asset_id)
                    self.asset_registry.update_asset(
                        asset.asset_id,
                        status=AssetStatus.COMPLETED,
                    )
                    results.append(result)
                    logger.info(f"Processed: {file_path}")

        except Exception as e:
            logger.error(f"Stage processing error: {e}")
            raise

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Create stage result
        stage_result = StageResult(
            stage=ProcessingStage.FORENSIC_SCORING,
            batch_id=batch_id,
            total_files=len(file_paths),
            processed_files=len(processed),
            failed_files=len(failed),
            skipped_files=0,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

        # Save summary
        summary = {
            "stage": stage_name,
            "batch_id": batch_id,
            "total_files": len(file_paths),
            "processed": len(processed),
            "failed": len(failed),
            "success_rate": stage_result.success_rate,
            "duration_seconds": duration,
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
        }

        logger.info(
            f"{stage_name} stage completed: "
            f"{summary['processed']}/{summary['total_files']} files "
            f"({summary['success_rate']:.1%}) in {duration:.1f}s"
        )

        return summary

    async def run_staged_pipeline(self, max_files: int = None):
        """
        Run complete staged pipeline

        Args:
            max_files: Maximum files to process (for testing)
        """
        from voice_man.services.asset_tracking.staged_batch_processor import (
            BatchCheckpoint,
            ProcessingStage,
        )

        # Get all audio files
        all_files = self.get_audio_files(limit=max_files)
        total_count = len(all_files)

        # Update config based on actual file count
        if total_count < 184:
            self.stage_pipeline.config.full_batch_size = total_count
            self.stage_pipeline.config.medium_batch_size = min(50, total_count)
            self.stage_pipeline.config.pilot_batch_size = min(10, total_count)

        results = {}

        try:
            # Stage 1: Pilot (10 files)
            if total_count >= 10:
                pilot_files = all_files[:10]
                batch_id = self.stage_pipeline.processor.create_batch_id("pilot")
                results["pilot"] = await self.process_stage(pilot_files, "pilot", batch_id)

                # Check if pilot succeeded
                if results["pilot"]["failed"] > 0:
                    logger.warning(
                        f"Pilot stage had {results['pilot']['failed']} failures, "
                        "proceeding with caution"
                    )

            # Stage 2: Medium (50 files or all if less)
            if total_count >= 50:
                medium_files = all_files[:50]
                batch_id = self.stage_pipeline.processor.create_batch_id("medium")
                results["medium"] = await self.process_stage(medium_files, "medium", batch_id)
            elif total_count > 10:
                # Skip medium if less than 50 but more than 10
                logger.info(f"Skipping medium stage (only {total_count} files)")

            # Stage 3: Full (all files)
            batch_id = self.stage_pipeline.processor.create_batch_id("full")
            results["full"] = await self.process_stage(all_files, "full", batch_id)

        finally:
            # Cleanup
            await self.gpu_orchestrator.shutdown()

        # Print final summary
        self._print_final_summary(results)

        return results

    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final execution summary"""
        print("\n" + "=" * 60)
        print("FORENSIC PIPELINE EXECUTION SUMMARY")
        print("=" * 60)

        for stage_name, result in results.items():
            print(f"\n{stage_name.upper()} STAGE:")
            print(f"  Files: {result['processed']}/{result['total_files']}")
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Duration: {result['duration_seconds']:.1f}s")

        print("\n" + "=" * 60)

        # Calculate overall stats
        total_processed = sum(r["processed"] for r in results.values())
        total_files = results["full"]["total_files"] if "full" in results else 0
        overall_rate = total_processed / total_files if total_files > 0 else 0

        print(f"\nOVERALL: {total_processed}/{total_files} files ({overall_rate:.1%})")
        print("=" * 60 + "\n")


@click.command()
@click.option(
    "--stage",
    type=click.Choice(["pilot", "medium", "full", "all"]),
    default="all",
    help="Pipeline stage to run",
)
@click.option(
    "--files",
    type=int,
    default=None,
    help="Maximum number of files to process (for testing)",
)
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("ref/call"),
    help="Directory containing audio files",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Enable GPU acceleration",
)
def main(stage: str, files: int, audio_dir: Path, gpu: bool):
    """
    Run forensic pipeline with asset tracking
    """
    runner = ForensicPipelineRunner(
        audio_dir=audio_dir,
        enable_gpu=gpu,
    )

    # Run the pipeline
    asyncio.run(runner.run_staged_pipeline(max_files=files))


if __name__ == "__main__":
    main()
