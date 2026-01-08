"""
Performance report service for batch processing metrics.

Provides comprehensive performance tracking, metrics collection, and report
generation based on EARS requirements (SPEC-PARALLEL-001).

EARS Requirements Implemented:
- U1: Performance metrics logging (batch time, GPU memory, CPU utilization)
- E3: Performance report generation on completion
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a single batch processing run."""

    batch_id: int
    file_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    processing_time_seconds: float = 0.0
    gpu_memory_samples: List[float] = field(default_factory=list)
    gpu_memory_peak_mb: float = 0.0
    cpu_utilization_samples: List[float] = field(default_factory=list)
    cpu_utilization_avg: float = 0.0
    device_used: str = "cpu"
    successful_files: int = 0
    failed_files: int = 0


@dataclass
class FailedFileRecord:
    """Record of a failed file."""

    file_path: str
    error_message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PerformanceReportService:
    """
    Service for tracking and reporting performance metrics.

    Implements U1 (performance metrics logging) and E3 (report generation).
    """

    def __init__(self):
        """Initialize performance report service."""
        self._batches: Dict[int, BatchMetrics] = {}
        self._failed_files: List[FailedFileRecord] = []
        self._overall_start_time: Optional[float] = None
        self._overall_end_time: Optional[float] = None
        self._total_files_processed: int = 0
        self._gpu_processing_count: int = 0
        self._cpu_processing_count: int = 0

    def start_batch(
        self, batch_id: int, file_count: int, device: str = "cpu"
    ) -> None:
        """
        Start tracking a new batch.

        Args:
            batch_id: Unique identifier for the batch
            file_count: Number of files in the batch
            device: Device used for processing ("cuda" or "cpu")

        Implements:
            U1: Performance metrics logging - batch time tracking
        """
        if self._overall_start_time is None:
            self._overall_start_time = time.time()

        self._batches[batch_id] = BatchMetrics(
            batch_id=batch_id,
            file_count=file_count,
            start_time=time.time(),
            device_used=device,
        )

        logger.info(f"Started batch {batch_id} with {file_count} files on {device}")

    def end_batch(
        self,
        batch_id: int,
        successful_files: int = 0,
        failed_files: int = 0,
    ) -> None:
        """
        End tracking for a batch.

        Args:
            batch_id: Batch identifier
            successful_files: Number of successfully processed files
            failed_files: Number of failed files

        Implements:
            U1: Performance metrics logging - batch completion
        """
        if batch_id not in self._batches:
            logger.warning(f"Batch {batch_id} not found, creating new record")
            self._batches[batch_id] = BatchMetrics(batch_id=batch_id)

        batch = self._batches[batch_id]
        batch.end_time = time.time()
        batch.processing_time_seconds = batch.end_time - batch.start_time
        batch.successful_files = successful_files
        batch.failed_files = failed_files

        # Calculate averages
        if batch.gpu_memory_samples:
            batch.gpu_memory_peak_mb = max(batch.gpu_memory_samples)
        if batch.cpu_utilization_samples:
            batch.cpu_utilization_avg = sum(batch.cpu_utilization_samples) / len(
                batch.cpu_utilization_samples
            )

        # Update totals
        self._total_files_processed += successful_files + failed_files
        if batch.device_used == "cuda":
            self._gpu_processing_count += successful_files
        else:
            self._cpu_processing_count += successful_files

        self._overall_end_time = time.time()

        logger.info(
            f"Completed batch {batch_id}: {successful_files} success, "
            f"{failed_files} failed, {batch.processing_time_seconds:.2f}s"
        )

    def record_gpu_memory(self, batch_id: Optional[int] = None) -> None:
        """
        Record current GPU memory usage.

        Args:
            batch_id: Optional batch to associate with (uses latest if None)

        Implements:
            U1: Performance metrics logging - GPU memory tracking
        """
        # Get current batch
        if batch_id is None and self._batches:
            batch_id = max(self._batches.keys())

        if batch_id not in self._batches:
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem_info.used / (1024 * 1024)

            self._batches[batch_id].gpu_memory_samples.append(used_mb)
            logger.debug(f"Recorded GPU memory: {used_mb:.0f} MB")

        except ImportError:
            logger.debug("pynvml not available, skipping GPU memory recording")
        except Exception as e:
            logger.debug(f"Could not record GPU memory: {e}")

    def record_cpu_utilization(self, batch_id: Optional[int] = None) -> None:
        """
        Record current CPU utilization.

        Args:
            batch_id: Optional batch to associate with (uses latest if None)

        Implements:
            U1: Performance metrics logging - CPU utilization tracking
        """
        # Get current batch
        if batch_id is None and self._batches:
            batch_id = max(self._batches.keys())

        if batch_id not in self._batches:
            return

        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            self._batches[batch_id].cpu_utilization_samples.append(cpu_percent)
            logger.debug(f"Recorded CPU utilization: {cpu_percent:.1f}%")

        except ImportError:
            logger.debug("psutil not available, skipping CPU recording")
        except Exception as e:
            logger.debug(f"Could not record CPU utilization: {e}")

    def record_failed_file(self, file_path: str, error_message: str) -> None:
        """
        Record a failed file.

        Args:
            file_path: Path to the failed file
            error_message: Error description
        """
        record = FailedFileRecord(file_path=file_path, error_message=error_message)
        self._failed_files.append(record)
        logger.warning(f"Recorded failed file: {file_path} - {error_message}")

    def get_batch_report(self, batch_id: int) -> Dict:
        """
        Get report for a specific batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Dictionary with batch metrics

        Implements:
            U1: Performance metrics reporting
        """
        if batch_id not in self._batches:
            return {"error": f"Batch {batch_id} not found"}

        batch = self._batches[batch_id]
        return {
            "batch_id": batch.batch_id,
            "file_count": batch.file_count,
            "processing_time_seconds": batch.processing_time_seconds,
            "gpu_memory_peak_mb": batch.gpu_memory_peak_mb,
            "cpu_utilization_avg": batch.cpu_utilization_avg,
            "device_used": batch.device_used,
            "successful_files": batch.successful_files,
            "failed_files": batch.failed_files,
        }

    def generate_final_report(self) -> Dict:
        """
        Generate comprehensive final performance report.

        Returns:
            Dictionary with complete performance statistics

        Implements:
            E3: Performance report generation on completion
        """
        # Calculate totals
        total_processing_time = 0.0
        total_successful = 0
        total_failed = 0
        all_gpu_memory_peaks = []
        all_cpu_utils = []

        for batch in self._batches.values():
            total_processing_time += batch.processing_time_seconds
            total_successful += batch.successful_files
            total_failed += batch.failed_files
            if batch.gpu_memory_peak_mb > 0:
                all_gpu_memory_peaks.append(batch.gpu_memory_peak_mb)
            if batch.cpu_utilization_avg > 0:
                all_cpu_utils.append(batch.cpu_utilization_avg)

        total_files = total_successful + total_failed

        # Calculate GPU vs CPU ratio
        total_device_files = self._gpu_processing_count + self._cpu_processing_count
        gpu_ratio = (
            self._gpu_processing_count / total_device_files
            if total_device_files > 0
            else 0.0
        )

        report = {
            # Summary statistics
            "total_files_processed": total_files,
            "successful_files": total_successful,
            "failed_files_count": total_failed,
            "success_rate": total_successful / total_files if total_files > 0 else 0.0,
            # Timing
            "total_processing_time": total_processing_time,
            "average_file_processing_time": (
                total_processing_time / total_files if total_files > 0 else 0.0
            ),
            "overall_wall_time": (
                (self._overall_end_time - self._overall_start_time)
                if self._overall_start_time and self._overall_end_time
                else 0.0
            ),
            # Device statistics
            "gpu_vs_cpu_ratio": gpu_ratio,
            "gpu_processed_files": self._gpu_processing_count,
            "cpu_processed_files": self._cpu_processing_count,
            # Memory statistics
            "gpu_memory_peak_mb": max(all_gpu_memory_peaks) if all_gpu_memory_peaks else 0.0,
            "gpu_memory_avg_mb": (
                sum(all_gpu_memory_peaks) / len(all_gpu_memory_peaks)
                if all_gpu_memory_peaks
                else 0.0
            ),
            # CPU statistics
            "cpu_utilization_avg": (
                sum(all_cpu_utils) / len(all_cpu_utils) if all_cpu_utils else 0.0
            ),
            # Batch statistics
            "total_batches": len(self._batches),
            "batch_reports": [
                self.get_batch_report(bid) for bid in sorted(self._batches.keys())
            ],
            # Failed files
            "failed_files": [
                {"file_path": f.file_path, "error": f.error_message, "timestamp": f.timestamp}
                for f in self._failed_files
            ],
            # Metadata
            "report_generated_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Generated final report: {total_files} files, "
            f"{total_successful} successful, {total_failed} failed, "
            f"{total_processing_time:.2f}s total time"
        )

        return report

    def reset(self) -> None:
        """Reset all metrics for a new processing session."""
        self._batches.clear()
        self._failed_files.clear()
        self._overall_start_time = None
        self._overall_end_time = None
        self._total_files_processed = 0
        self._gpu_processing_count = 0
        self._cpu_processing_count = 0
        logger.info("Performance metrics reset")

    def get_progress_summary(self) -> Dict:
        """
        Get current progress summary.

        Returns:
            Dictionary with current progress information
        """
        completed_batches = sum(
            1 for b in self._batches.values() if b.end_time > 0
        )

        return {
            "total_batches_started": len(self._batches),
            "completed_batches": completed_batches,
            "total_files_processed": self._total_files_processed,
            "failed_files_count": len(self._failed_files),
            "elapsed_time": (
                time.time() - self._overall_start_time
                if self._overall_start_time
                else 0.0
            ),
        }
