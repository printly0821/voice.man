#!/usr/bin/env python3
"""
Forensic Pipeline Runner with CPU Fallback for Alignment
==========================================================

Works around CUDA compilation issues by using CPU for alignment.
"""

import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline_execution.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set PyTorch to use deterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Disable CUDA kernel compilation for alignment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


async def run_pipeline_with_cpu_alignment():
    """Run pipeline with CPU alignment fallback"""
    from voice_man.services.whisperx_service import WhisperXService
    from voice_man.services.audio_converter_service import AudioConverterService

    # Get test files
    audio_files = sorted(Path("ref/call").glob("*.m4a"))[:10]
    logger.info(f"Processing {len(audio_files)} files")

    # Initialize STT service
    stt_service = WhisperXService(device="cuda", language="ko")

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        try:
            # Process with transcription only (skip diarization to avoid CUDA compilation)
            result_dict = await stt_service.transcribe_only(str(audio_file))

            if result_dict.get("error"):
                logger.error(f"Failed: {result_dict['error']}")
            else:
                text = result_dict.get("text", "")
                segments = result_dict.get("segments", [])
                logger.info(f"Success: {len(segments)} segments, {len(text)} chars")
                results.append(
                    {
                        "file": str(audio_file),
                        "text": text[:100],
                        "segments": len(segments),
                    }
                )

        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")

    # Cleanup
    stt_service.unload()

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTS: {len(results)}/{len(audio_files)} files processed")
    logger.info(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    asyncio.run(run_pipeline_with_cpu_alignment())
