#!/usr/bin/env python3
"""
CPU 모드 Whisper 벤치마크
openai-whisper를 사용하여 CPU에서 실행 가능한 벤치마크

Reference: SPEC-GPUOPT-001
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import whisper
import librosa

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    """단일 파일 처리 결과"""

    file_name: str
    duration_seconds: float
    processing_time_seconds: float
    success: bool
    error_message: str = ""
    transcription_text: str = ""


@dataclass
class BenchmarkMetrics:
    """벤치마크 메트릭스"""

    total_processing_time: float = 0.0
    avg_time_per_file: float = 0.0
    files_per_minute: float = 0.0
    realtime_factor: float = 0.0  # 처리시간/오디오길이 (낮을수록 좋음)
    total_audio_duration: float = 0.0
    success_count: int = 0
    failure_count: int = 0


class SimpleWhisperBenchmark:
    """CPU 모드 Whisper 벤치마크"""

    def __init__(
        self,
        audio_dir: Path = Path("/home/innojini/dev/voice.man/ref/call"),
        output_dir: Path = Path("/home/innojini/dev/voice.man/ref/call/benchmark_results"),
    ):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SimpleWhisperBenchmark initialized")

    def get_test_files(self, limit: int = 10) -> List[Path]:
        """벤치마크 대상 오디오 파일 가져오기"""
        audio_files = []
        extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

        for ext in extensions:
            files = list(self.audio_dir.glob(f"*{ext}"))
            audio_files.extend(files)

        # 파일 크기 및 이름으로 정렬 (작은 파일부터)
        audio_files.sort(key=lambda x: (x.stat().st_size, x.name))
        return audio_files[:limit]

    def get_audio_duration(self, audio_path: Path) -> float:
        """오디오 파일 길이 가져오기"""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as e:
            logger.warning(f"Failed to get duration for {audio_path.name}: {e}")
            return 0.0

    def process_file(self, audio_path: Path) -> FileResult:
        """단일 파일 처리"""
        start_time = time.time()

        try:
            duration = self.get_audio_duration(audio_path)

            # openai-whisper로 처리 (CPU 모드)
            logger.info(f"Processing: {audio_path.name}")

            # medium 모델 사용 (large-v3는 CPU에서 너무 느림)
            model = whisper.load_model("medium")
            result = model.transcribe(str(audio_path), language="ko", fp16=False)

            processing_time = time.time() - start_time

            return FileResult(
                file_name=audio_path.name,
                duration_seconds=duration,
                processing_time_seconds=processing_time,
                success=True,
                transcription_text=result["text"],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process {audio_path.name}: {e}")

            return FileResult(
                file_name=audio_path.name,
                duration_seconds=0.0,
                processing_time_seconds=processing_time,
                success=False,
                error_message=str(e),
            )

    def run_benchmark(self, num_files: int = 10) -> dict:
        """벤치마크 실행"""
        logger.info("\n" + "=" * 60)
        logger.info("WHISPER CPU BENCHMARK")
        logger.info("=" * 60 + "\n")

        # 테스트 파일 가져오기
        test_files = self.get_test_files(limit=num_files)
        logger.info(f"Found {len(test_files)} audio files for benchmarking")

        # 파일 정보 출력
        total_duration = 0.0
        for f in test_files:
            duration = self.get_audio_duration(f)
            total_duration += duration
            logger.info(f"  - {f.name}: {duration:.1f}s")

        logger.info(
            f"\nTotal audio duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)\n"
        )

        # 벤치마크 실행
        results = []
        start_time = time.time()

        for i, audio_file in enumerate(test_files, 1):
            logger.info(f"\n[{i}/{len(test_files)}] Processing: {audio_file.name}")
            result = self.process_file(audio_file)
            results.append(result)

        total_benchmark_time = time.time() - start_time

        # 메트릭스 계산
        metrics = BenchmarkMetrics()
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        metrics.success_count = len(successful_results)
        metrics.failure_count = len(failed_results)

        if successful_results:
            processing_times = [r.processing_time_seconds for r in successful_results]
            metrics.total_processing_time = sum(processing_times)
            metrics.avg_time_per_file = metrics.total_processing_time / len(successful_results)
            metrics.total_audio_duration = total_duration
            metrics.files_per_minute = (
                len(successful_results) / (metrics.total_processing_time / 60)
                if metrics.total_processing_time > 0
                else 0
            )
            metrics.realtime_factor = (
                metrics.total_processing_time / total_duration if total_duration > 0 else 0
            )

        # 결과 정리
        benchmark_result = {
            "mode": "cpu",
            "model": "whisper-medium",
            "num_files": num_files,
            "total_benchmark_time_seconds": total_benchmark_time,
            "metrics": {
                "processing_time": {
                    "total_seconds": metrics.total_processing_time,
                    "avg_per_file_seconds": metrics.avg_time_per_file,
                },
                "throughput": {
                    "files_per_minute": metrics.files_per_minute,
                    "total_audio_duration_seconds": metrics.total_audio_duration,
                    "realtime_factor": metrics.realtime_factor,
                },
                "reliability": {
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "success_rate": metrics.success_count
                    / (metrics.success_count + metrics.failure_count)
                    if (metrics.success_count + metrics.failure_count) > 0
                    else 0.0,
                },
            },
            "files": [
                {
                    "file_name": r.file_name,
                    "duration_seconds": r.duration_seconds,
                    "processing_time_seconds": r.processing_time_seconds,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }

        # 결과 저장
        output_file = self.output_dir / f"benchmark_cpu_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_result, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {output_file}")

        # 결과 출력
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60 + "\n")

        logger.info("Processing Time:")
        logger.info(f"  Total:       {metrics.total_processing_time:.2f}s")
        logger.info(f"  Avg/File:    {metrics.avg_time_per_file:.2f}s")

        logger.info("\nThroughput:")
        logger.info(f"  Files/min:   {metrics.files_per_minute:.2f}")
        logger.info(f"  Audio Time:  {metrics.total_audio_duration:.2f}s")

        logger.info("\nRealtime Factor (lower is better):")
        logger.info(f"  RT Factor:  {metrics.realtime_factor:.2f}x")

        logger.info("\nReliability:")
        logger.info(f"  Success:     {metrics.success_count}/{len(results)}")
        logger.info(f"  Success Rate: {metrics.success_count / len(results) * 100:.1f}%")

        logger.info("\n" + "=" * 60 + "\n")

        return benchmark_result


def main():
    """메인 실행 함수"""
    benchmark = SimpleWhisperBenchmark()
    result = benchmark.run_benchmark(num_files=10)
    return result


if __name__ == "__main__":
    main()
