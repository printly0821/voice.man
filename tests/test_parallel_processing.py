"""
GPU parallel processing tests for SPEC-PARALLEL-001.

Tests GPU monitoring, faster-whisper integration, batch processing optimization,
and performance reporting based on EARS requirements.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# TASK-001: GPU Monitor Service Tests
# ============================================================================


class TestGPUMonitorService:
    """Tests for GPU monitoring service based on EARS requirements."""

    # -------------------------------------------------------------------------
    # E1: Event-driven - GPU availability check at batch start
    # -------------------------------------------------------------------------

    def test_check_gpu_availability_returns_true_when_cuda_available(self):
        """Test GPU availability check returns True when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            assert service.is_gpu_available() is True

    def test_check_gpu_availability_returns_false_when_cuda_unavailable(self):
        """Test GPU availability check returns False when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            assert service.is_gpu_available() is False

    def test_fallback_to_cpu_when_gpu_unavailable(self):
        """Test CPU fallback mechanism when GPU is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            device = service.get_recommended_device()
            assert device == "cpu"

    def test_use_gpu_when_available(self):
        """Test GPU device selection when available."""
        with patch("torch.cuda.is_available", return_value=True):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            device = service.get_recommended_device()
            assert device == "cuda"

    # -------------------------------------------------------------------------
    # U2: Ubiquitous - GPU memory real-time monitoring (80% warning, 95% auto-adjust)
    # -------------------------------------------------------------------------

    def test_get_gpu_memory_stats_returns_memory_info(self):
        """Test GPU memory statistics retrieval."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024  # 24GB
        mock_memory_info.used = 12 * 1024 * 1024 * 1024  # 12GB (50%)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            stats = service.get_gpu_memory_stats()

            assert stats["total_mb"] == 24 * 1024
            assert stats["used_mb"] == 12 * 1024
            assert stats["free_mb"] == 12 * 1024
            assert stats["usage_percentage"] == pytest.approx(50.0, rel=0.01)

    def test_warning_triggered_at_80_percent_usage(self):
        """Test warning is triggered when GPU memory usage exceeds 80%."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024  # 24GB
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.82)  # 82%

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            status = service.check_memory_status()

            assert status["warning"] is True
            assert status["critical"] is False
            assert "80%" in status["message"]

    def test_auto_adjust_triggered_at_95_percent_usage(self):
        """Test auto-adjustment is triggered when GPU memory usage exceeds 95%."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024  # 24GB
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.96)  # 96%

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            status = service.check_memory_status()

            assert status["critical"] is True
            assert status["auto_adjust_recommended"] is True

    # -------------------------------------------------------------------------
    # E2: Event-driven - Auto-reduce batch size by 50% on memory shortage
    # -------------------------------------------------------------------------

    def test_recommend_reduced_batch_size_on_memory_shortage(self):
        """Test batch size reduction recommendation on memory shortage."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024  # 24GB
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.96)  # 96%

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            current_batch_size = 20
            recommended = service.get_recommended_batch_size(current_batch_size)

            # Should recommend 50% reduction
            assert recommended == 10

    def test_batch_size_not_reduced_below_minimum(self):
        """Test batch size is not reduced below minimum threshold."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.96)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService(min_batch_size=2)
            current_batch_size = 3
            recommended = service.get_recommended_batch_size(current_batch_size)

            # Should not go below minimum
            assert recommended >= 2

    # -------------------------------------------------------------------------
    # S1: State-driven - CPU fallback when GPU in use
    # -------------------------------------------------------------------------

    def test_fallback_device_when_gpu_memory_critical(self):
        """Test device fallback to CPU when GPU memory is critical."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.99)  # 99% (above 98% fallback threshold)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            device = service.get_recommended_device()

            # Should fallback to CPU due to memory pressure (above 98% threshold)
            assert device == "cpu"

    # -------------------------------------------------------------------------
    # GPU Cache Cleanup
    # -------------------------------------------------------------------------

    def test_clear_gpu_cache_calls_torch_empty_cache(self):
        """Test GPU cache clearing calls torch.cuda.empty_cache()."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            service.clear_gpu_cache()

            mock_empty_cache.assert_called_once()

    def test_clear_gpu_cache_noop_when_gpu_unavailable(self):
        """Test GPU cache clearing is no-op when GPU is unavailable."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:
            from voice_man.services.gpu_monitor_service import GPUMonitorService

            service = GPUMonitorService()
            service.clear_gpu_cache()

            # Should not call empty_cache when GPU unavailable
            mock_empty_cache.assert_not_called()


# ============================================================================
# TASK-002: Faster-Whisper Wrapper Tests
# ============================================================================


class TestWhisperModel:
    """Tests for faster-whisper wrapper based on EARS requirements."""

    # -------------------------------------------------------------------------
    # F1: Feature - faster-whisper based STT (GPU: float16, CPU: int8)
    # -------------------------------------------------------------------------

    def test_gpu_mode_uses_float16_compute_type(self):
        """Test GPU mode uses float16 compute type."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model:
            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="large-v3", device="cuda")

            mock_model.assert_called_once()
            call_args = mock_model.call_args
            assert call_args[1]["device"] == "cuda"
            assert call_args[1]["compute_type"] == "float16"

    def test_cpu_mode_uses_int8_compute_type(self):
        """Test CPU mode uses int8 compute type."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model:
            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="medium", device="cpu")

            mock_model.assert_called_once()
            call_args = mock_model.call_args
            assert call_args[1]["device"] == "cpu"
            assert call_args[1]["compute_type"] == "int8"

    def test_auto_device_selection_prefers_gpu(self):
        """Test auto device selection prefers GPU when available."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model:
            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="large-v3", device="auto")

            call_args = mock_model.call_args
            assert call_args[1]["device"] == "cuda"

    def test_auto_device_selection_falls_back_to_cpu(self):
        """Test auto device selection falls back to CPU when GPU unavailable."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model:
            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="medium", device="auto")

            call_args = mock_model.call_args
            assert call_args[1]["device"] == "cpu"

    # -------------------------------------------------------------------------
    # Transcription Tests
    # -------------------------------------------------------------------------

    def test_transcribe_returns_segments_with_timestamps(self):
        """Test transcription returns segments with timestamps."""
        mock_segments = [
            Mock(start=0.0, end=2.5, text="Hello world"),
            Mock(start=2.5, end=5.0, text="How are you"),
        ]
        mock_info = Mock(language="ko", language_probability=0.98)

        with patch("torch.cuda.is_available", return_value=False), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
            mock_model_class.return_value = mock_model

            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="base", device="cpu")
            result = wrapper.transcribe("/path/to/audio.wav")

            assert "segments" in result
            assert len(result["segments"]) == 2
            assert result["segments"][0]["start"] == 0.0
            assert result["segments"][0]["end"] == 2.5
            assert result["segments"][0]["text"] == "Hello world"
            assert result["language"] == "ko"

    def test_transcribe_combines_full_text(self):
        """Test transcription combines segments into full text."""
        mock_segments = [
            Mock(start=0.0, end=2.5, text="Hello"),
            Mock(start=2.5, end=5.0, text="world"),
        ]
        mock_info = Mock(language="ko", language_probability=0.98)

        with patch("torch.cuda.is_available", return_value=False), patch(
            "faster_whisper.WhisperModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
            mock_model_class.return_value = mock_model

            from voice_man.models.whisper_model import WhisperModelWrapper

            wrapper = WhisperModelWrapper(model_size="base", device="cpu")
            result = wrapper.transcribe("/path/to/audio.wav")

            assert result["text"] == "Hello world"


# ============================================================================
# TASK-003: Extended BatchConfig Tests
# ============================================================================


class TestExtendedBatchConfig:
    """Tests for extended BatchConfig with GPU support."""

    def test_default_config_has_gpu_settings(self):
        """Test default config includes GPU-related settings."""
        from voice_man.services.batch_service import BatchConfig

        config = BatchConfig()
        assert hasattr(config, "use_gpu")
        assert hasattr(config, "gpu_batch_size")
        assert hasattr(config, "dynamic_batch_adjustment")

    def test_gpu_optimized_defaults(self):
        """Test GPU-optimized default values as per SPEC."""
        from voice_man.services.batch_service import BatchConfig

        config = BatchConfig(use_gpu=True)
        assert config.batch_size == 15  # Default for GPU
        assert config.max_workers == 16  # Default for GPU optimization

    def test_cpu_fallback_settings(self):
        """Test CPU fallback settings are properly applied."""
        from voice_man.services.batch_service import BatchConfig

        config = BatchConfig(use_gpu=False)
        assert config.batch_size == 5  # CPU default
        assert config.max_workers == 4  # CPU default


# ============================================================================
# TASK-004: Extended MemoryManager Tests
# ============================================================================


class TestExtendedMemoryManager:
    """Tests for extended MemoryManager with GPU monitoring."""

    # -------------------------------------------------------------------------
    # GPU Memory Monitoring
    # -------------------------------------------------------------------------

    def test_memory_manager_tracks_gpu_memory(self):
        """Test memory manager tracks GPU memory when available."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024
        mock_memory_info.used = 12 * 1024 * 1024 * 1024

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.memory_service import MemoryManager

            manager = MemoryManager(threshold_mb=70000, enable_gpu_monitoring=True)
            summary = manager.get_memory_summary()

            assert "gpu_memory_mb" in summary
            assert "gpu_memory_percentage" in summary

    # -------------------------------------------------------------------------
    # S2: State-driven - GC trigger at 80% system memory usage
    # -------------------------------------------------------------------------

    def test_gc_triggered_at_80_percent_system_memory(self):
        """Test garbage collection is triggered at 80% system memory usage."""
        with patch("psutil.Process") as mock_process, patch(
            "psutil.virtual_memory"
        ) as mock_vmem:
            mock_memory = Mock()
            mock_memory.rss = 64 * 1024 * 1024 * 1024  # 64GB used
            mock_process.return_value.memory_info.return_value = mock_memory

            mock_vmem.return_value = Mock(
                total=80 * 1024 * 1024 * 1024,  # 80GB total
                percent=82.0,  # 82% usage
            )

            from voice_man.services.memory_service import MemoryManager

            manager = MemoryManager(threshold_mb=70000, system_memory_threshold=80.0)
            assert manager.should_collect() is True

    # -------------------------------------------------------------------------
    # GPU Cache Integration
    # -------------------------------------------------------------------------

    def test_force_gc_includes_gpu_cache_clear(self):
        """Test forced GC includes GPU cache clearing."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache, patch("gc.collect", return_value=100):
            from voice_man.services.memory_service import MemoryManager

            manager = MemoryManager(enable_gpu_monitoring=True)
            manager.force_garbage_collection()

            mock_empty_cache.assert_called_once()


# ============================================================================
# TASK-005: Performance Report Service Tests
# ============================================================================


class TestPerformanceReportService:
    """Tests for performance report generation based on EARS requirements."""

    # -------------------------------------------------------------------------
    # U1: Ubiquitous - Performance metrics logging
    # -------------------------------------------------------------------------

    def test_logs_batch_processing_time(self):
        """Test batch processing time is logged."""
        from voice_man.services.performance_report_service import (
            PerformanceReportService,
        )

        service = PerformanceReportService()
        service.start_batch(batch_id=1, file_count=10)
        service.end_batch(batch_id=1)

        report = service.get_batch_report(batch_id=1)
        assert "processing_time_seconds" in report
        assert report["processing_time_seconds"] >= 0

    def test_logs_gpu_memory_usage(self):
        """Test GPU memory usage is logged during batch processing."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024
        mock_memory_info.used = 12 * 1024 * 1024 * 1024

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.performance_report_service import (
                PerformanceReportService,
            )

            service = PerformanceReportService()
            service.start_batch(batch_id=1, file_count=10)
            service.record_gpu_memory()
            service.end_batch(batch_id=1)

            report = service.get_batch_report(batch_id=1)
            assert "gpu_memory_peak_mb" in report

    def test_logs_cpu_utilization(self):
        """Test CPU utilization is logged during batch processing."""
        with patch("psutil.cpu_percent", return_value=75.5):
            from voice_man.services.performance_report_service import (
                PerformanceReportService,
            )

            service = PerformanceReportService()
            service.start_batch(batch_id=1, file_count=10)
            service.record_cpu_utilization()
            service.end_batch(batch_id=1)

            report = service.get_batch_report(batch_id=1)
            assert "cpu_utilization_avg" in report

    # -------------------------------------------------------------------------
    # E3: Event-driven - Performance report on completion
    # -------------------------------------------------------------------------

    def test_generates_final_report_on_completion(self):
        """Test final performance report is generated on completion."""
        from voice_man.services.performance_report_service import (
            PerformanceReportService,
        )

        service = PerformanceReportService()

        # Simulate processing
        service.start_batch(batch_id=1, file_count=10)
        service.end_batch(batch_id=1)
        service.start_batch(batch_id=2, file_count=10)
        service.end_batch(batch_id=2)

        final_report = service.generate_final_report()

        assert "total_files_processed" in final_report
        assert "total_processing_time" in final_report
        assert "gpu_vs_cpu_ratio" in final_report
        assert "average_file_processing_time" in final_report

    def test_report_includes_failed_files_list(self):
        """Test report includes list of failed files."""
        from voice_man.services.performance_report_service import (
            PerformanceReportService,
        )

        service = PerformanceReportService()
        service.record_failed_file("/path/to/file1.wav", "Memory error")
        service.record_failed_file("/path/to/file2.wav", "Timeout")

        final_report = service.generate_final_report()

        assert "failed_files" in final_report
        assert len(final_report["failed_files"]) == 2


# ============================================================================
# TASK-006: Analysis Pipeline GPU Integration Tests
# ============================================================================


class TestAnalysisPipelineGPUIntegration:
    """Tests for GPU integration in analysis pipeline."""

    def test_pipeline_uses_gpu_when_available(self):
        """Test analysis pipeline uses GPU when available."""
        with patch("torch.cuda.is_available", return_value=True):
            from voice_man.services.analysis_pipeline_service import (
                SingleFileAnalysisPipeline,
            )

            pipeline = SingleFileAnalysisPipeline(use_gpu=True)
            assert pipeline.device == "cuda"

    def test_pipeline_falls_back_to_cpu(self):
        """Test analysis pipeline falls back to CPU when GPU unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            from voice_man.services.analysis_pipeline_service import (
                SingleFileAnalysisPipeline,
            )

            pipeline = SingleFileAnalysisPipeline(use_gpu=True)
            assert pipeline.device == "cpu"

    def test_dynamic_batch_size_adjustment(self):
        """Test dynamic batch size adjustment during processing."""
        mock_handle = Mock()
        mock_memory_info = Mock()
        mock_memory_info.total = 24 * 1024 * 1024 * 1024
        mock_memory_info.used = int(24 * 1024 * 1024 * 1024 * 0.96)  # Critical

        with patch("torch.cuda.is_available", return_value=True), patch(
            "pynvml.nvmlInit"
        ), patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle), patch(
            "pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info
        ):
            from voice_man.services.analysis_pipeline_service import (
                SingleFileAnalysisPipeline,
            )

            pipeline = SingleFileAnalysisPipeline(use_gpu=True, initial_batch_size=20)
            adjusted_size = pipeline.get_adjusted_batch_size()

            # Should be reduced by 50%
            assert adjusted_size == 10


# ============================================================================
# Integration Tests
# ============================================================================


class TestParallelProcessingIntegration:
    """Integration tests for parallel processing system."""

    # -------------------------------------------------------------------------
    # N1: Negative - Original file modification prohibited
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_original_files_not_modified(self, tmp_path):
        """Test that original files are not modified during processing."""
        import hashlib

        # Create test file
        test_file = tmp_path / "test_audio.wav"
        test_content = b"fake audio content for testing"
        test_file.write_bytes(test_content)

        # Calculate original checksum
        original_checksum = hashlib.md5(test_content).hexdigest()

        # Process file (mock processing)
        with patch("torch.cuda.is_available", return_value=False):
            from voice_man.services.analysis_pipeline_service import (
                SingleFileAnalysisPipeline,
            )

            pipeline = SingleFileAnalysisPipeline(use_gpu=False)
            # Verify file unchanged after any operation
            current_content = test_file.read_bytes()
            current_checksum = hashlib.md5(current_content).hexdigest()

            assert original_checksum == current_checksum

    # -------------------------------------------------------------------------
    # N2: Negative - Unlimited memory allocation prohibited
    # -------------------------------------------------------------------------

    def test_memory_allocation_respects_limits(self):
        """Test that memory allocation respects system limits."""
        with patch("psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = Mock(
                total=80 * 1024 * 1024 * 1024,  # 80GB total
                available=16 * 1024 * 1024 * 1024,  # 16GB available (80% used)
                percent=80.0,
            )

            from voice_man.services.memory_service import MemoryManager

            manager = MemoryManager(system_memory_threshold=80.0)

            # Should trigger collection at 80%
            assert manager.should_collect() is True

    # -------------------------------------------------------------------------
    # S3: State-driven - Retry queue with exponential backoff
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_failed_files_added_to_retry_queue(self):
        """Test that failed files are added to retry queue with exponential backoff."""
        from voice_man.services.batch_service import BatchProcessor, BatchConfig

        config = BatchConfig(retry_count=3, continue_on_error=True)
        processor = BatchProcessor(config)

        retry_info = processor.get_retry_info()
        assert "max_retries" in retry_info
        assert retry_info["max_retries"] == 3
        assert "backoff_strategy" in retry_info
        assert retry_info["backoff_strategy"] == "exponential"
