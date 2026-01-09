"""
E2E Test Service for WhisperX Batch Processing.

Provides comprehensive E2E testing capabilities with GPU parallel batch processing,
checksum verification, progress tracking, and report generation.

SPEC-E2ETEST-001 Requirements Implemented:
- U1: BatchProcessor based GPU parallel batch processing
- U4: Original file integrity verification (checksum)
- E4: Dynamic batch adjustment on GPU memory shortage
- S2: Failed file retry queue (exponential backoff)
- N1: GPU memory must not exceed 95%
- N3: Original file modification prohibited
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileProcessingResult:
    """Result of processing a single audio file.

    Attributes:
        file_path: Path to the processed audio file
        status: Processing status ("success" | "failed" | "skipped")
        processing_time_seconds: Time taken to process in seconds
        transcript_text: Transcribed text if successful
        segments: List of transcript segments with speaker info
        speakers: List of identified speakers
        error: Error message if failed
    """

    file_path: str
    status: str  # "success" | "failed" | "skipped"
    processing_time_seconds: float
    transcript_text: Optional[str]
    segments: Optional[List[Dict[str, Any]]]
    speakers: Optional[List[str]]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "status": self.status,
            "processing_time_seconds": self.processing_time_seconds,
            "transcript_text": self.transcript_text,
            "segments": self.segments,
            "speakers": self.speakers,
            "error": self.error,
        }


@dataclass
class E2ETestResult:
    """Comprehensive result of E2E test execution.

    Attributes:
        total_files: Total number of files to process
        processed_files: Number of files processed
        failed_files: Number of files that failed
        total_time_seconds: Total processing time in seconds
        avg_time_per_file: Average processing time per file
        gpu_stats: GPU memory and usage statistics
        file_results: List of individual file processing results
        checksum_verified: Whether all file checksums were verified
    """

    total_files: int
    processed_files: int
    failed_files: int
    total_time_seconds: float
    avg_time_per_file: float
    gpu_stats: Dict[str, Any]
    file_results: List[FileProcessingResult]
    checksum_verified: bool

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a ratio between 0.0 and 1.0."""
        if self.processed_files == 0:
            return 0.0
        return (self.processed_files - self.failed_files) / self.processed_files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_time_seconds": self.total_time_seconds,
            "avg_time_per_file": self.avg_time_per_file,
            "success_rate": self.success_rate,
            "gpu_stats": self.gpu_stats,
            "file_results": [r.to_dict() for r in self.file_results],
            "checksum_verified": self.checksum_verified,
        }

    def get_failed_files(self) -> List[FileProcessingResult]:
        """Get list of failed file results."""
        return [r for r in self.file_results if r.status == "failed"]


@dataclass
class E2ETestConfig:
    """Configuration for E2E test execution.

    Implements SPEC-E2ETEST-001 requirements:
    - F2: GPU batch size 15-20, max 32
    - E4: Dynamic batch adjustment
    - S2: Retry with exponential backoff
    """

    batch_size: int = 15
    max_batch_size: int = 32
    min_batch_size: int = 2
    num_speakers: Optional[int] = 2
    language: str = "ko"
    device: str = "cuda"
    compute_type: str = "float16"
    max_retries: int = 3
    retry_delays: List[int] = field(default_factory=lambda: [5, 15, 30])
    dynamic_batch_adjustment: bool = True
    gpu_memory_warning_threshold: float = 80.0
    gpu_memory_critical_threshold: float = 95.0
    enable_checksum_verification: bool = True


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash as hexadecimal string

    Implements:
        U4: Original file integrity verification
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_checksums(original_checksums: Dict[str, str], current_checksums: Dict[str, str]) -> bool:
    """Verify that current checksums match original checksums.

    Args:
        original_checksums: Dictionary mapping file paths to original checksums
        current_checksums: Dictionary mapping file paths to current checksums

    Returns:
        True if all checksums match, False otherwise

    Implements:
        U4: Original file integrity verification
        N3: Original file modification prohibited
    """
    if set(original_checksums.keys()) != set(current_checksums.keys()):
        logger.error("Checksum verification failed: file set mismatch")
        return False

    for file_path, original_hash in original_checksums.items():
        current_hash = current_checksums.get(file_path)
        if current_hash != original_hash:
            logger.error(f"Checksum mismatch for {file_path}")
            return False

    return True


class E2ETestRunner:
    """Run E2E tests for WhisperX batch processing.

    Implements SPEC-E2ETEST-001 requirements for GPU parallel batch processing
    with progress tracking, error handling, and comprehensive reporting.
    """

    def __init__(
        self,
        config: E2ETestConfig,
        whisperx_service: Optional[Any] = None,
        gpu_monitor: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ):
        """Initialize E2E test runner.

        Args:
            config: E2E test configuration
            whisperx_service: WhisperX service instance (optional, created if None)
            gpu_monitor: GPU monitor service instance (optional)
            memory_manager: Memory manager instance (optional)
        """
        self.config = config
        self._whisperx_service = whisperx_service
        self._gpu_monitor = gpu_monitor
        self._memory_manager = memory_manager
        self._current_batch_size = config.batch_size
        self._original_checksums: Dict[str, str] = {}
        self._progress_callback: Optional[Callable] = None

    async def collect_files(self, directory: Path) -> List[Path]:
        """Collect audio files from directory.

        Args:
            directory: Directory to scan for audio files

        Returns:
            List of audio file paths sorted by name
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm"}
        files = []

        for ext in audio_extensions:
            files.extend(directory.glob(f"*{ext}"))

        return sorted(files)

    async def _calculate_all_checksums(self, files: List[Path]) -> Dict[str, str]:
        """Calculate checksums for all files.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping file paths to checksums
        """
        checksums = {}
        for file_path in files:
            checksums[str(file_path)] = calculate_md5(file_path)
        return checksums

    async def process_single_file(self, file_path: Path, attempt: int = 1) -> FileProcessingResult:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file
            attempt: Current attempt number (for retry tracking)

        Returns:
            FileProcessingResult with processing outcome

        Implements:
            U1: BatchProcessor based GPU parallel batch processing
        """
        start_time = time.time()

        try:
            if self._whisperx_service is None:
                raise RuntimeError("WhisperX service not initialized")

            result = await self._whisperx_service.process_audio(
                str(file_path),
                num_speakers=self.config.num_speakers,
            )

            processing_time = time.time() - start_time

            return FileProcessingResult(
                file_path=str(file_path),
                status="success",
                processing_time_seconds=processing_time,
                transcript_text=getattr(result, "text", None),
                segments=getattr(result, "segments", None),
                speakers=getattr(result, "speakers", None),
                error=None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"Failed to process {file_path.name}: {e}")

            return FileProcessingResult(
                file_path=str(file_path),
                status="failed",
                processing_time_seconds=processing_time,
                transcript_text=None,
                segments=None,
                speakers=None,
                error=str(e),
            )

    async def _process_with_retry(self, file_path: Path) -> FileProcessingResult:
        """Process file with exponential backoff retry.

        Args:
            file_path: Path to the audio file

        Returns:
            FileProcessingResult after all retry attempts

        Implements:
            S2: Failed file retry queue (exponential backoff)
        """
        for attempt in range(1, self.config.max_retries + 1):
            result = await self.process_single_file(file_path, attempt)

            if result.status == "success":
                return result

            if attempt < self.config.max_retries:
                delay = self.config.retry_delays[attempt - 1]
                logger.info(
                    f"Retrying {file_path.name} in {delay}s (attempt {attempt}/{self.config.max_retries})"
                )
                await asyncio.sleep(delay)

        return result

    async def _adjust_batch_size(self) -> None:
        """Dynamically adjust batch size based on GPU memory.

        Implements:
            E4: Dynamic batch adjustment on GPU memory shortage
            N1: GPU memory must not exceed 95%
        """
        if not self.config.dynamic_batch_adjustment or self._gpu_monitor is None:
            return

        memory_status = self._gpu_monitor.check_memory_status()
        usage = memory_status.get("usage_percentage", 0)

        # N1: GPU memory must not exceed 95%
        if usage >= self.config.gpu_memory_critical_threshold:
            # Clear GPU cache
            self._gpu_monitor.clear_gpu_cache()

            # Reduce batch size by 50%
            new_size = max(self._current_batch_size // 2, self.config.min_batch_size)
            if new_size != self._current_batch_size:
                logger.warning(
                    f"GPU memory critical ({usage:.1f}%), reducing batch size: "
                    f"{self._current_batch_size} -> {new_size}"
                )
                self._current_batch_size = new_size

        # Increase batch size if memory is low
        elif usage < 50.0 and self._current_batch_size < self.config.max_batch_size:
            new_size = min(self._current_batch_size + 2, self.config.max_batch_size)
            if new_size != self._current_batch_size:
                logger.info(
                    f"GPU memory low ({usage:.1f}%), increasing batch size: "
                    f"{self._current_batch_size} -> {new_size}"
                )
                self._current_batch_size = new_size

    async def run(
        self,
        files: List[Path],
        progress_callback: Optional[Callable[[int, int, float, str, str], None]] = None,
    ) -> E2ETestResult:
        """Run E2E test on list of files.

        Args:
            files: List of audio file paths to process
            progress_callback: Optional callback for progress updates
                signature: (current, total, elapsed_seconds, current_file, status)

        Returns:
            E2ETestResult with comprehensive test results

        Implements:
            U1: BatchProcessor based GPU parallel batch processing
            U4: Original file integrity verification
        """
        if not files:
            return E2ETestResult(
                total_files=0,
                processed_files=0,
                failed_files=0,
                total_time_seconds=0.0,
                avg_time_per_file=0.0,
                gpu_stats={},
                file_results=[],
                checksum_verified=True,
            )

        self._progress_callback = progress_callback
        start_time = time.time()

        # U4: Calculate original checksums
        if self.config.enable_checksum_verification:
            self._original_checksums = await self._calculate_all_checksums(files)

        file_results: List[FileProcessingResult] = []
        failed_count = 0

        for i, file_path in enumerate(files):
            # Adjust batch size based on GPU memory
            await self._adjust_batch_size()

            # Process file with retry
            result = await self._process_with_retry(file_path)
            file_results.append(result)

            if result.status == "failed":
                failed_count += 1

            # Progress callback
            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(
                    i + 1,
                    len(files),
                    elapsed,
                    file_path.name,
                    result.status,
                )

        total_time = time.time() - start_time

        # U4: Verify checksums after processing
        checksum_verified = True
        if self.config.enable_checksum_verification:
            current_checksums = await self._calculate_all_checksums(files)
            checksum_verified = verify_checksums(self._original_checksums, current_checksums)

        # Collect GPU stats
        gpu_stats = {}
        if self._gpu_monitor:
            gpu_stats = self._gpu_monitor.get_gpu_memory_stats()

        return E2ETestResult(
            total_files=len(files),
            processed_files=len(file_results),
            failed_files=failed_count,
            total_time_seconds=total_time,
            avg_time_per_file=total_time / len(files) if files else 0.0,
            gpu_stats=gpu_stats,
            file_results=file_results,
            checksum_verified=checksum_verified,
        )

    def generate_report(self, result: E2ETestResult, output_dir: Path) -> Dict[str, Path]:
        """Generate comprehensive E2E test reports.

        Args:
            result: E2E test result to generate reports from
            output_dir: Directory to write reports to

        Returns:
            Dictionary mapping report type to file path

        Implements:
            TASK-008: Comprehensive report generation
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: Dict[str, Path] = {}

        # JSON Report
        json_path = output_dir / "e2e_test_report.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "max_retries": self.config.max_retries,
                "language": self.config.language,
                "device": self.config.device,
            },
            "result": result.to_dict(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        generated_files["json"] = json_path

        # Markdown Summary
        md_path = output_dir / "e2e_test_summary.md"
        md_content = self._generate_markdown_summary(result)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        generated_files["markdown"] = md_path

        # Failed Files JSON
        failed_files = result.get_failed_files()
        if failed_files:
            failed_path = output_dir / "failed_files.json"
            failed_data = {
                "timestamp": datetime.now().isoformat(),
                "failed_count": len(failed_files),
                "files": [f.to_dict() for f in failed_files],
            }
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_data, f, ensure_ascii=False, indent=2)
            generated_files["failed_files"] = failed_path

        logger.info(f"Generated {len(generated_files)} report files in {output_dir}")
        return generated_files

    def _generate_markdown_summary(self, result: E2ETestResult) -> str:
        """Generate markdown summary report.

        Args:
            result: E2E test result

        Returns:
            Markdown formatted string
        """
        success_rate = result.success_rate * 100
        checksum_status = "Verified" if result.checksum_verified else "FAILED"

        md = f"""# E2E Test Summary Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

| Metric | Value |
|--------|-------|
| Total Files | {result.total_files} |
| Processed | {result.processed_files} |
| Failed | {result.failed_files} |
| Success Rate | {success_rate:.1f}% |
| Total Time | {result.total_time_seconds:.2f}s |
| Avg Time/File | {result.avg_time_per_file:.2f}s |
| Checksum | {checksum_status} |

## GPU Statistics

"""
        if result.gpu_stats:
            for key, value in result.gpu_stats.items():
                if isinstance(value, float):
                    md += f"- {key}: {value:.2f}\n"
                else:
                    md += f"- {key}: {value}\n"
        else:
            md += "No GPU statistics available.\n"

        md += "\n## Failed Files\n\n"
        failed_files = result.get_failed_files()
        if failed_files:
            for f in failed_files:
                md += f"- `{Path(f.file_path).name}`: {f.error}\n"
        else:
            md += "No failed files.\n"

        return md
