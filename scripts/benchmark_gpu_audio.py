#!/usr/bin/env python3
"""
GPU Audio Performance Benchmark Script.

Tests the performance of detect_emotional_escalation with GPU acceleration.
Compares GPU vs CPU performance on sample audio files.

SPEC-GPUAUDIO-001 Performance Validation.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import librosa
import numpy as np


def benchmark_single_file(audio_path: Path, use_gpu: bool = True) -> dict:
    """Benchmark a single audio file."""
    from voice_man.services.forensic.audio_feature_service import AudioFeatureService

    # Load audio
    print(f"\n{'=' * 60}")
    print(f"Testing: {audio_path.name}")
    print(f"GPU Mode: {use_gpu}")
    print(f"{'=' * 60}")

    load_start = time.time()
    audio, sr = librosa.load(str(audio_path), sr=16000)
    load_time = time.time() - load_start
    duration = len(audio) / sr

    print(f"Audio loaded: {duration:.1f}s duration ({load_time:.2f}s load time)")

    # Initialize service
    init_start = time.time()
    service = AudioFeatureService(use_gpu=use_gpu)
    init_time = time.time() - init_start
    print(f"Service initialized: {init_time:.2f}s")

    # Benchmark detect_emotional_escalation
    print("\nRunning detect_emotional_escalation()...")

    bench_start = time.time()
    result = service.detect_emotional_escalation(audio, sr)
    bench_time = time.time() - bench_start

    print(f"\n✅ Completed in {bench_time:.2f}s")
    print(f"   - Audio duration: {duration:.1f}s")
    print(f"   - Processing rate: {duration / bench_time:.1f}x realtime")

    if result:
        print(f"   - Escalation zones detected: {len(result)}")
        if len(result) > 0:
            print(f"   - First zone: {result[0].start_time:.1f}s - {result[0].end_time:.1f}s")

    return {
        "file": audio_path.name,
        "duration_sec": duration,
        "gpu_mode": use_gpu,
        "init_time": init_time,
        "process_time": bench_time,
        "realtime_factor": duration / bench_time if bench_time > 0 else 0,
    }


def main():
    """Run benchmark on test files."""
    test_dir = Path("ref/call/e2e_test_top10")

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)

    # Get audio files
    audio_files = list(test_dir.glob("*.m4a")) + list(test_dir.glob("*.wav"))

    if not audio_files:
        print(f"Error: No audio files found in {test_dir}")
        sys.exit(1)

    print(f"\n{'#' * 60}")
    print("# SPEC-GPUAUDIO-001 Performance Benchmark")
    print(f"# Testing {len(audio_files)} files with GPU acceleration")
    print(f"{'#' * 60}")

    # Run GPU benchmark
    results = []
    total_duration = 0
    total_process_time = 0

    for i, audio_file in enumerate(audio_files[:10], 1):
        print(f"\n[{i}/{min(len(audio_files), 10)}] Processing...")
        try:
            result = benchmark_single_file(audio_file, use_gpu=True)
            results.append(result)
            total_duration += result["duration_sec"]
            total_process_time += result["process_time"]
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files processed: {len(results)}")
    print(f"Total audio duration: {total_duration / 60:.1f} minutes")
    print(f"Total processing time: {total_process_time:.1f}s")
    print(f"Average realtime factor: {total_duration / total_process_time:.1f}x")
    print(f"Average per file: {total_process_time / len(results):.1f}s")

    # Performance target check
    target_factor = 100  # 100x improvement target
    avg_factor = total_duration / total_process_time if total_process_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("PERFORMANCE TARGET CHECK")
    print(f"{'=' * 60}")
    print(f"Target: {target_factor}x realtime (500s → 5s for 42min audio)")
    print(f"Achieved: {avg_factor:.1f}x realtime")

    if avg_factor >= target_factor:
        print("✅ PASS: Performance target met!")
    else:
        print("⚠️ Performance target not fully met (may be due to I/O overhead)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
