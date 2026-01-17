#!/usr/bin/env python3
"""
Comprehensive benchmark for audio duration extraction methods.

This script analyzes different approaches to getting audio file duration:
1. ffprobe subprocess (current approach)
2. ffmpeg-python wrapper
3. Python libraries (mutagen, tinytag)
4. Direct header parsing
5. File system stat (baseline comparison)

Measures:
- Execution time
- Memory usage (RSS)
- CPU usage
- Accuracy

Usage:
    python scripts/benchmark_duration_extraction.py
"""

import gc
import json
import os
import resource
import subprocess
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    method: str
    file_path: str
    file_size_mb: float
    duration_sec: float
    execution_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_time_user_ms: float
    cpu_time_system_ms: float
    success: bool
    error: Optional[str] = None


class DurationBenchmark:
    """Benchmark different duration extraction methods."""

    def __init__(self, test_audio_dir: Optional[str] = None):
        """
        Initialize benchmark.

        Args:
            test_audio_dir: Directory containing test audio files
        """
        self.test_audio_dir = Path(test_audio_dir or "ref/call")
        self.results: List[BenchmarkResult] = []

        # Check available methods
        self.available_methods = self._check_available_methods()

    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which duration extraction methods are available."""
        methods = {
            "ffprobe_subprocess": self._check_ffprobe(),
            "ffmpeg_python": self._check_ffmpeg_python(),
            "mutagen": self._check_mutagen(),
            "tinytag": self._check_tinytag(),
            "wavelib": True,  # Built-in
        }

        print("Available Methods:")
        for method, available in methods.items():
            status = "✓" if available else "✗"
            print(f"  {status} {method}")

        return methods

    def _check_ffprobe(self) -> bool:
        """Check if ffprobe is available."""
        try:
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return True
        except Exception:
            return False

    def _check_ffmpeg_python(self) -> bool:
        """Check if ffmpeg-python is available."""
        try:
            import ffmpeg

            return True
        except ImportError:
            return False

    def _check_mutagen(self) -> bool:
        """Check if mutagen is available."""
        try:
            import mutagen

            return True
        except ImportError:
            return False

    def _check_tinytag(self) -> bool:
        """Check if tinytag is available."""
        try:
            import tinytag

            return True
        except ImportError:
            return False

    # ========================================================================
    # Duration Extraction Methods
    # ========================================================================

    def _method_ffprobe_subprocess(self, file_path: str) -> float:
        """
        Method 1: ffprobe subprocess (CURRENT APPROACH).

        Uses subprocess.run() to call ffprobe CLI.
        This is the current implementation in audio_chunker.py.

        Pros: Accurate, widely compatible
        Cons: Subprocess overhead, potential encoding issues
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else 0.0

    def _method_ffmpeg_python(self, file_path: str) -> float:
        """
        Method 2: ffmpeg-python wrapper.

        Uses ffmpeg-python library which wraps ffprobe.

        Pros: Pythonic API, handles encoding
        Cons: Still uses subprocess internally, additional dependency
        """
        import ffmpeg

        probe = ffmpeg.probe(file_path)
        return float(probe["format"]["duration"])

    def _method_mutagen(self, file_path: str) -> float:
        """
        Method 3: mutagen library.

        Pure Python library for audio metadata.
        Reads file headers without decoding.

        Pros: No subprocess, low memory, fast
        Cons: May be less accurate for some formats
        """
        from mutagen import File

        audio_file = File(file_path)
        if audio_file is None:
            return 0.0

        # mutagen returns duration in seconds
        return audio_file.info.length

    def _method_tinytag(self, file_path: str) -> float:
        """
        Method 4: tinytag library.

        Lightweight library for reading audio metadata.

        Pros: Very lightweight, designed for this use case
        Cons: May not support all formats
        """
        from tinytag import TinyTag

        tag = TinyTag.get(file_path)
        return tag.duration

    def _method_wavelib(self, file_path: str) -> float:
        """
        Method 5: wavelib (built-in).

        Only works for WAV files.
        For other formats, returns 0.

        Pros: Built-in, no dependencies
        Cons: WAV only, would need conversion first
        """
        try:
            import wave

            with wave.open(file_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except Exception:
            # Not a WAV file or unsupported
            return 0.0

    def _method_file_stat(self, file_path: str) -> float:
        """
        Method 6: file system stat (BASELINE).

        Just gets file size - cannot determine duration.
        Used for baseline comparison of overhead.

        Returns: 0.0 (duration cannot be determined)
        """
        Path(file_path).stat()
        return 0.0

    # ========================================================================
    # Benchmark Execution
    # ========================================================================

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _run_benchmark(
        self,
        method: str,
        method_func: Callable[[str], float],
        file_path: str,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            method: Method name
            method_func: Function to extract duration
            file_path: Path to audio file

        Returns:
            BenchmarkResult with metrics
        """
        # Get file info
        file_stat = Path(file_path).stat()
        file_size_mb = file_stat.st_size / (1024 * 1024)

        # Force garbage collection before measurement
        gc.collect()

        # Start memory tracking
        tracemalloc.start()
        memory_before = self._get_memory_usage_mb()

        # Start CPU time measurement
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        start_time = time.perf_counter()

        # Run method
        success = True
        error = None
        duration = 0.0

        try:
            duration = method_func(file_path)
        except Exception as e:
            success = False
            error = str(e)

        # End measurements
        end_time = time.perf_counter()
        ru_after = resource.getrusage(resource.RUSAGE_SELF)

        memory_after = self._get_memory_usage_mb()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate metrics
        execution_time_ms = (end_time - start_time) * 1000
        memory_peak_mb = peak / (1024 * 1024)
        memory_delta_mb = memory_after - memory_before

        cpu_time_user_ms = (ru_after.ru_utime - ru_before.ru_utime) * 1000
        cpu_time_system_ms = (ru_after.ru_stime - ru_before.ru_stime) * 1000

        return BenchmarkResult(
            method=method,
            file_path=file_path,
            file_size_mb=file_size_mb,
            duration_sec=duration,
            execution_time_ms=execution_time_ms,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak_mb,
            memory_delta_mb=memory_delta_mb,
            cpu_time_user_ms=cpu_time_user_ms,
            cpu_time_system_ms=cpu_time_system_ms,
            success=success,
            error=error,
        )

    def _get_test_files(self) -> List[Path]:
        """Get list of test audio files."""
        if not self.test_audio_dir.exists():
            print(f"Warning: Test directory not found: {self.test_audio_dir}")
            return []

        # Get audio files
        audio_files = []
        for ext in ["*.m4a", "*.mp3", "*.wav", "*.flac", "*.ogg"]:
            audio_files.extend(self.test_audio_dir.glob(ext))

        return sorted(audio_files)

    def run(self, max_files: int = 10) -> List[BenchmarkResult]:
        """
        Run all benchmarks.

        Args:
            max_files: Maximum number of files to test

        Returns:
            List of BenchmarkResults
        """
        test_files = self._get_test_files()

        if not test_files:
            print("No test files found")
            return []

        # Limit files
        test_files = test_files[:max_files]

        print(f"\nBenchmarking {len(test_files)} files...")
        print("=" * 80)

        methods_to_test = [
            ("file_stat", self._method_file_stat),
            ("ffprobe_subprocess", self._method_ffprobe_subprocess),
        ]

        # Add optional methods if available
        if self.available_methods["ffmpeg_python"]:
            methods_to_test.append(("ffmpeg_python", self._method_ffmpeg_python))
        if self.available_methods["mutagen"]:
            methods_to_test.append(("mutagen", self._method_mutagen))
        if self.available_methods["tinytag"]:
            methods_to_test.append(("tinytag", self._method_tinytag))

        methods_to_test.append(("wavelib", self._method_wavelib))

        # Run benchmarks
        for i, file_path in enumerate(test_files, 1):
            print(f"\n[{i}/{len(test_files)}] Testing: {file_path.name}")
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {file_size_mb:.2f} MB")

            for method_name, method_func in methods_to_test:
                result = self._run_benchmark(method_name, method_func, str(file_path))
                self.results.append(result)

                status = "✓" if result.success else "✗"
                if result.success:
                    print(
                        f"  {status} {method_name:20s}: "
                        f"{result.execution_time_ms:6.1f}ms, "
                        f"Δ{result.memory_delta_mb:+6.2f}MB, "
                        f"duration={result.duration_sec:.1f}s"
                    )
                else:
                    print(
                        f"  {status} {method_name:20s}: FAILED - {result.error or 'Unknown error'}"
                    )

        return self.results

    # ========================================================================
    # Analysis and Reporting
    # ========================================================================

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze benchmark results.

        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            return {}

        # Group by method
        by_method: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.method not in by_method:
                by_method[result.method] = []
            by_method[result.method].append(result)

        # Calculate statistics per method
        stats = {}
        for method, results in by_method.items():
            successful = [r for r in results if r.success]

            if not successful:
                stats[method] = {
                    "success_rate": 0.0,
                    "avg_time_ms": 0.0,
                    "avg_memory_delta_mb": 0.0,
                    "avg_memory_peak_mb": 0.0,
                    "sample_count": 0,
                }
                continue

            stats[method] = {
                "success_rate": len(successful) / len(results),
                "avg_time_ms": sum(r.execution_time_ms for r in successful) / len(successful),
                "avg_memory_delta_mb": (
                    sum(r.memory_delta_mb for r in successful) / len(successful)
                ),
                "avg_memory_peak_mb": (sum(r.memory_peak_mb for r in successful) / len(successful)),
                "sample_count": len(successful),
            }

        return {
            "by_method": stats,
            "all_results": self.results,
        }

    def print_report(self):
        """Print benchmark analysis report."""
        analysis = self.analyze()

        if not analysis:
            print("No results to analyze")
            return

        stats = analysis["by_method"]

        print("\n" + "=" * 80)
        print("BENCHMARK ANALYSIS REPORT")
        print("=" * 80)

        # Summary table
        print("\nPerformance Summary:")
        print("-" * 80)
        print(
            f"{'Method':<25} {'Success':>8} {'Time(ms)':>12} {'ΔMem(MB)':>12} "
            f"{'PeakMem(MB)':>14} {'Samples':>8}"
        )
        print("-" * 80)

        # Sort by average execution time
        sorted_methods = sorted(stats.items(), key=lambda x: x[1]["avg_time_ms"])

        for method, stat in sorted_methods:
            print(
                f"{method:<25} {stat['success_rate']:>7.1%} "
                f"{stat['avg_time_ms']:>11.1f} "
                f"{stat['avg_memory_delta_mb']:>+11.2f} "
                f"{stat['avg_memory_peak_mb']:>14.2f} "
                f"{stat['sample_count']:>8d}"
            )

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Find best performing methods
        successful_methods = {m: s for m, s in stats.items() if s["success_rate"] > 0.5}

        if successful_methods:
            # Fastest
            fastest = min(successful_methods.items(), key=lambda x: x[1]["avg_time_ms"])
            print(f"\n✓ Fastest Method: {fastest[0]}")
            print(f"  - Average time: {fastest[1]['avg_time_ms']:.1f}ms")
            print(f"  - Memory delta: {fastest[1]['avg_memory_delta_mb']:+.2f}MB")

            # Lowest memory
            lowest_mem = min(successful_methods.items(), key=lambda x: x[1]["avg_memory_delta_mb"])
            print(f"\n✓ Lowest Memory: {lowest_mem[0]}")
            print(f"  - Memory delta: {lowest_mem[1]['avg_memory_delta_mb']:+.2f}MB")
            print(f"  - Average time: {lowest_mem[1]['avg_time_ms']:.1f}ms")

            # Compare with current (ffprobe_subprocess)
            if "ffprobe_subprocess" in successful_methods:
                current = successful_methods["ffprobe_subprocess"]
                print(f"\nCurrent Method (ffprobe_subprocess):")
                print(f"  - Average time: {current['avg_time_ms']:.1f}ms")
                print(f"  - Memory delta: {current['avg_memory_delta_mb']:+.2f}MB")

                # Find alternatives that are better
                better_time = [
                    (m, s)
                    for m, s in successful_methods.items()
                    if s["avg_time_ms"] < current["avg_time_ms"] * 0.9
                ]
                better_memory = [
                    (m, s)
                    for m, s in successful_methods.items()
                    if s["avg_memory_delta_mb"] < current["avg_memory_delta_mb"] - 1.0
                ]

                if better_time:
                    print(f"\n✓ Faster Alternatives (>10% faster):")
                    for m, s in better_time:
                        improvement = (
                            (current["avg_time_ms"] - s["avg_time_ms"])
                            / current["avg_time_ms"]
                            * 100
                        )
                        print(f"  - {m}: {improvement:.1f}% faster")

                if better_memory:
                    print(f"\n✓ Lower Memory Alternatives (>1MB less):")
                    for m, s in better_memory:
                        savings = current["avg_memory_delta_mb"] - s["avg_memory_delta_mb"]
                        print(f"  - {m}: {savings:.2f}MB less memory")

    def save_results(self, output_path: str):
        """
        Save benchmark results to JSON.

        Args:
            output_path: Path to output JSON file
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "test_directory": str(self.test_audio_dir),
            "available_methods": self.available_methods,
            "results": [
                {
                    "method": r.method,
                    "file_path": r.file_path,
                    "file_size_mb": r.file_size_mb,
                    "duration_sec": r.duration_sec,
                    "execution_time_ms": r.execution_time_ms,
                    "memory_before_mb": r.memory_before_mb,
                    "memory_after_mb": r.memory_after_mb,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_delta_mb": r.memory_delta_mb,
                    "cpu_time_user_ms": r.cpu_time_user_ms,
                    "cpu_time_system_ms": r.cpu_time_system_ms,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Run benchmark and print report."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark audio duration extraction methods")
    parser.add_argument(
        "--dir",
        default="ref/call",
        help="Directory containing test audio files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to test",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = DurationBenchmark(test_audio_dir=args.dir)
    benchmark.run(max_files=args.max_files)

    # Print report
    benchmark.print_report()

    # Save results
    if args.output:
        benchmark.save_results(args.output)
    else:
        # Auto-generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"ref/call/benchmark_results/duration_benchmark_{timestamp}.json"
        benchmark.save_results(output_path)


if __name__ == "__main__":
    main()
