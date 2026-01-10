"""
PipelineOrchestrator Unit Tests

SPEC-PERFOPT-001 Phase 3: Advanced producer-consumer pipeline orchestration.
TDD RED Phase - Tests written FIRST before implementation.

EARS Requirements:
    - E4 (Event): STT result completion triggers Forensic analysis + parallel next file STT
    - S5 (State): Pause STT when Forensic queue exceeds 5 (backpressure)
    - N4 (Unwanted): No blocking I/O on GPU threads
    - PR-007 (Performance): 50% pipeline efficiency improvement via STT+Forensic overlap

Key Features:
    - asyncio.Queue-based producer-consumer pattern
    - Backpressure mechanism (MAX_QUEUE_SIZE=5, resume at 3)
    - AsyncIterator for streaming results
    - Integration with Phase 2 managers (ForensicMemoryManager, ThermalManager)
"""

import asyncio
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockSTTResult:
    """Mock STT result for testing."""

    file_path: str
    text: str
    segments: List[Dict[str, Any]]
    speakers: List[str]


@dataclass
class MockForensicResult:
    """Mock Forensic result for testing."""

    file_path: str
    overall_risk_score: float
    risk_level: str
    analysis_id: str


@pytest.fixture
def mock_stt_service():
    """Create a mock WhisperX service."""
    service = MagicMock()
    service.process_audio = AsyncMock()
    return service


@pytest.fixture
def mock_forensic_service():
    """Create a mock ForensicScoringService."""
    service = MagicMock()
    service.analyze = AsyncMock()
    return service


@pytest.fixture
def mock_memory_manager():
    """Create a mock ForensicMemoryManager."""
    manager = MagicMock()
    manager.allocate = MagicMock(return_value=True)
    manager.release = MagicMock(return_value=True)
    manager.is_allocated = MagicMock(return_value=False)
    manager.get_memory_stats = MagicMock(return_value={"total_allocated_mb": 0})
    return manager


@pytest.fixture
def mock_thermal_manager():
    """Create a mock ThermalManager."""
    manager = MagicMock()
    manager.is_throttling = False
    manager.is_critical = False
    manager.get_current_temperature = MagicMock(return_value=60)
    manager.register_throttle_callback = MagicMock()
    manager.start_monitoring = MagicMock()
    manager.stop_monitoring = MagicMock()
    return manager


@pytest.fixture
def sample_audio_files(tmp_path: Path) -> List[Path]:
    """Create sample audio file paths for testing."""
    files = []
    for i in range(5):
        file_path = tmp_path / f"audio_{i:03d}.wav"
        file_path.touch()
        files.append(file_path)
    return files


# ============================================================================
# Test: Import and Initialization
# ============================================================================


class TestPipelineOrchestratorImport:
    """Test that PipelineOrchestrator can be imported."""

    def test_import_pipeline_orchestrator(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 3 implementation
        WHEN: Importing PipelineOrchestrator
        THEN: The import should succeed without errors
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        assert PipelineOrchestrator is not None

    def test_import_constants(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 3 implementation
        WHEN: Importing module constants
        THEN: MAX_QUEUE_SIZE and BACKPRESSURE_RESUME_SIZE should be available
        """
        from voice_man.services.forensic.pipeline_orchestrator import (
            MAX_QUEUE_SIZE,
            BACKPRESSURE_RESUME_SIZE,
        )

        assert MAX_QUEUE_SIZE == 5
        assert BACKPRESSURE_RESUME_SIZE == 3


class TestPipelineOrchestratorInitialization:
    """Test PipelineOrchestrator initialization."""

    def test_initialization_default(self):
        """
        GIVEN: PipelineOrchestrator class
        WHEN: Initialized with no parameters
        THEN: Should create instance with default settings
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator is not None
        assert orchestrator._stt_service is None
        assert orchestrator._forensic_service is None

    def test_initialization_with_services(
        self, mock_stt_service, mock_forensic_service, mock_memory_manager, mock_thermal_manager
    ):
        """
        GIVEN: PipelineOrchestrator class
        WHEN: Initialized with all services
        THEN: Should store service references
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
            memory_manager=mock_memory_manager,
            thermal_manager=mock_thermal_manager,
        )

        assert orchestrator._stt_service is mock_stt_service
        assert orchestrator._forensic_service is mock_forensic_service
        assert orchestrator._memory_manager is mock_memory_manager
        assert orchestrator._thermal_manager is mock_thermal_manager

    def test_has_asyncio_queue(self):
        """
        GIVEN: PipelineOrchestrator class
        WHEN: Initialized
        THEN: Should have an asyncio.Queue with maxsize=5
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert hasattr(orchestrator, "_queue")
        assert isinstance(orchestrator._queue, asyncio.Queue)
        assert orchestrator._queue.maxsize == 5

    def test_has_stop_event(self):
        """
        GIVEN: PipelineOrchestrator class
        WHEN: Initialized
        THEN: Should have an asyncio.Event for stopping
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert hasattr(orchestrator, "_stop_event")
        assert isinstance(orchestrator._stop_event, asyncio.Event)

    def test_backpressure_initial_state(self):
        """
        GIVEN: PipelineOrchestrator class
        WHEN: Initialized
        THEN: Backpressure should be inactive
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator._backpressure_active is False


# ============================================================================
# Test: Backpressure Logic (TASK-002)
# ============================================================================


class TestBackpressureLogic:
    """Test backpressure mechanism per S5 requirement."""

    @pytest.mark.asyncio
    async def test_backpressure_activates_at_queue_size_5(self):
        """
        GIVEN: PipelineOrchestrator with queue nearing capacity
        WHEN: Queue size reaches MAX_QUEUE_SIZE (5)
        THEN: Backpressure should activate

        S5 (State): Pause STT when Forensic queue exceeds 5
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Simulate filling the queue
        for i in range(5):
            await orchestrator._queue.put({"file": f"test_{i}.wav"})

        # Check backpressure state
        orchestrator._check_backpressure()

        assert orchestrator._backpressure_active is True
        assert orchestrator.is_backpressure_active() is True

    @pytest.mark.asyncio
    async def test_backpressure_resumes_at_queue_size_3(self):
        """
        GIVEN: PipelineOrchestrator with backpressure active
        WHEN: Queue size drops to BACKPRESSURE_RESUME_SIZE (3)
        THEN: Backpressure should deactivate

        S5 (State): Resume when queue drops to 3
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Fill queue to activate backpressure
        for i in range(5):
            await orchestrator._queue.put({"file": f"test_{i}.wav"})
        orchestrator._check_backpressure()
        assert orchestrator._backpressure_active is True

        # Drain queue to resume size
        for _ in range(2):
            await orchestrator._queue.get()
        orchestrator._check_backpressure()

        assert orchestrator._backpressure_active is False

    @pytest.mark.asyncio
    async def test_backpressure_hysteresis(self):
        """
        GIVEN: PipelineOrchestrator
        WHEN: Queue size is between 3 and 5
        THEN: Backpressure should maintain current state (hysteresis)
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Fill queue to 4 (between thresholds)
        for i in range(4):
            await orchestrator._queue.put({"file": f"test_{i}.wav"})

        # Initially inactive
        orchestrator._check_backpressure()
        assert orchestrator._backpressure_active is False

        # Fill to 5, activate backpressure
        await orchestrator._queue.put({"file": "test_4.wav"})
        orchestrator._check_backpressure()
        assert orchestrator._backpressure_active is True

        # Drain to 4 - should stay active (hysteresis)
        await orchestrator._queue.get()
        orchestrator._check_backpressure()
        assert orchestrator._backpressure_active is True  # Still active due to hysteresis

    def test_is_backpressure_active_method(self):
        """
        GIVEN: PipelineOrchestrator instance
        WHEN: is_backpressure_active() is called
        THEN: Should return current backpressure state
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator.is_backpressure_active() is False

        orchestrator._backpressure_active = True
        assert orchestrator.is_backpressure_active() is True


# ============================================================================
# Test: STT Producer (TASK-003)
# ============================================================================


class TestSTTProducer:
    """Test STT producer functionality."""

    @pytest.mark.asyncio
    async def test_produce_stt_results_processes_files(self, mock_stt_service, sample_audio_files):
        """
        GIVEN: PipelineOrchestrator with STT service
        WHEN: _produce_stt_results() is called with file list
        THEN: Should process each file and put results in queue

        E4 (Event): STT result completion triggers Forensic analysis
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav",
            text="Test transcript",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test"}],
            speakers=["SPEAKER_00"],
        )

        orchestrator = PipelineOrchestrator(stt_service=mock_stt_service)

        # Use only 3 files to avoid queue filling up
        test_files = sample_audio_files[:3]

        # Start producer
        producer_task = asyncio.create_task(orchestrator._produce_stt_results(test_files))

        # Wait for producer to complete
        await asyncio.wait_for(producer_task, timeout=5.0)

        # Verify STT service was called for each file
        assert mock_stt_service.process_audio.call_count == len(test_files)

        # Queue should have items + sentinel
        assert orchestrator._queue.qsize() == len(test_files) + 1  # +1 for sentinel

    @pytest.mark.asyncio
    async def test_producer_respects_backpressure(self, mock_stt_service, sample_audio_files):
        """
        GIVEN: PipelineOrchestrator with backpressure active
        WHEN: Producer tries to add to full queue
        THEN: Should wait for queue to have space

        S5 (State): Pause STT when queue full
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav",
            text="Test",
            segments=[],
            speakers=[],
        )

        orchestrator = PipelineOrchestrator(stt_service=mock_stt_service)

        # Pre-fill queue to capacity (leave room for 1 item)
        for i in range(4):
            await orchestrator._queue.put({"file": f"prefill_{i}.wav"})

        # Start producer - will add 1 item then block trying to add sentinel
        producer_task = asyncio.create_task(
            orchestrator._produce_stt_results([sample_audio_files[0]])
        )

        # Give producer time to process
        await asyncio.sleep(0.05)

        # Producer might not be done yet (blocked trying to add sentinel)
        # Drain items to allow completion
        while not orchestrator._queue.empty():
            await orchestrator._queue.get()

        # Wait for producer to finish
        await asyncio.wait_for(producer_task, timeout=2.0)

        # Verify file was processed
        assert mock_stt_service.process_audio.call_count == 1

    @pytest.mark.asyncio
    async def test_producer_handles_stt_error_gracefully(
        self, mock_stt_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with failing STT service
        WHEN: STT processing raises exception
        THEN: Should log error and continue with next file

        N4 (Unwanted): No blocking on errors
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        # First call fails, rest succeed
        mock_stt_service.process_audio.side_effect = [
            Exception("STT Error"),
            MockSTTResult(file_path="test.wav", text="Success", segments=[], speakers=[]),
            MockSTTResult(file_path="test.wav", text="Success", segments=[], speakers=[]),
        ]

        orchestrator = PipelineOrchestrator(stt_service=mock_stt_service)

        producer_task = asyncio.create_task(
            orchestrator._produce_stt_results(sample_audio_files[:3])
        )

        # Wait for producer to complete
        await asyncio.wait_for(producer_task, timeout=5.0)

        # Should still process all files
        assert mock_stt_service.process_audio.call_count == 3

        # Queue should have 3 items (1 error + 2 success) + sentinel
        assert orchestrator._queue.qsize() == 4


# ============================================================================
# Test: Forensic Consumer (TASK-004)
# ============================================================================


class TestForensicConsumer:
    """Test Forensic consumer functionality."""

    @pytest.mark.asyncio
    async def test_consume_forensic_yields_results(self, mock_forensic_service):
        """
        GIVEN: PipelineOrchestrator with Forensic service and queue items
        WHEN: _consume_forensic() is iterated
        THEN: Should yield ForensicResult for each queue item

        E4 (Event): Forensic analysis starts on STT completion
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=75.5,
            risk_level="HIGH",
            analysis_id="test-123",
        )

        orchestrator = PipelineOrchestrator(forensic_service=mock_forensic_service)

        # Put items in queue
        await orchestrator._queue.put(
            {
                "file_path": "test.wav",
                "transcript": "Test transcript",
            }
        )

        # Consume one result
        results = []
        orchestrator._stop_event.set()  # Will stop after processing queue

        async for result in orchestrator._consume_forensic():
            results.append(result)
            break  # Stop after first result

        assert len(results) == 1
        assert mock_forensic_service.analyze.call_count == 1

    @pytest.mark.asyncio
    async def test_consumer_updates_backpressure_state(self, mock_forensic_service):
        """
        GIVEN: PipelineOrchestrator with backpressure active
        WHEN: Consumer processes items from queue
        THEN: Should update backpressure state after each item

        S5 (State): Resume when queue size drops
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=50.0,
            risk_level="MODERATE",
            analysis_id="test-456",
        )

        orchestrator = PipelineOrchestrator(forensic_service=mock_forensic_service)

        # Fill queue and activate backpressure
        for i in range(5):
            await orchestrator._queue.put(
                {
                    "file_path": f"test_{i}.wav",
                    "transcript": f"Transcript {i}",
                }
            )
        orchestrator._check_backpressure()
        assert orchestrator._backpressure_active is True

        # Consume items
        async for result in orchestrator._consume_forensic():
            if orchestrator._queue.qsize() <= 3:
                orchestrator._stop_event.set()
                break

        # Backpressure should be deactivated
        assert orchestrator._backpressure_active is False

    @pytest.mark.asyncio
    async def test_consumer_handles_forensic_error(self, mock_forensic_service):
        """
        GIVEN: PipelineOrchestrator with failing Forensic service
        WHEN: Forensic analysis raises exception
        THEN: Should yield error result and continue

        N4 (Unwanted): No blocking on errors
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_forensic_service.analyze.side_effect = Exception("Forensic Error")

        orchestrator = PipelineOrchestrator(forensic_service=mock_forensic_service)

        await orchestrator._queue.put(
            {
                "file_path": "test.wav",
                "transcript": "Test",
            }
        )

        results = []
        orchestrator._stop_event.set()

        async for result in orchestrator._consume_forensic():
            results.append(result)
            break

        assert len(results) == 1
        assert results[0].get("error") is not None


# ============================================================================
# Test: Main Pipeline (process_files)
# ============================================================================


class TestProcessFiles:
    """Test the main process_files method."""

    @pytest.mark.asyncio
    async def test_process_files_returns_async_iterator(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with services
        WHEN: process_files() is called
        THEN: Should return AsyncIterator
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav",
            text="Test",
            segments=[],
            speakers=[],
        )
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=50.0,
            risk_level="MODERATE",
            analysis_id="test-123",
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        result_iter = orchestrator.process_files(sample_audio_files[:2])

        # Should be async iterable
        assert hasattr(result_iter, "__aiter__")
        assert hasattr(result_iter, "__anext__")

    @pytest.mark.asyncio
    async def test_process_files_overlaps_stt_and_forensic(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with services
        WHEN: process_files() processes multiple files
        THEN: STT and Forensic should run in parallel (overlap)

        PR-007 (Performance): 50% efficiency improvement via overlap
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        stt_call_times = []
        forensic_call_times = []

        async def mock_stt_process(audio_path, **kwargs):
            stt_call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)  # Simulate processing time
            return MockSTTResult(
                file_path=str(audio_path),
                text="Test",
                segments=[],
                speakers=[],
            )

        async def mock_forensic_analyze(audio_path, transcript):
            forensic_call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.03)  # Simulate processing time
            return MockForensicResult(
                file_path=audio_path,
                overall_risk_score=50.0,
                risk_level="MODERATE",
                analysis_id="test",
            )

        mock_stt_service.process_audio = mock_stt_process
        mock_forensic_service.analyze = mock_forensic_analyze

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        results = []
        async for result in orchestrator.process_files(sample_audio_files[:3]):
            results.append(result)

        assert len(results) == 3

        # Verify parallel execution: forensic should start before all STT completes
        if len(stt_call_times) >= 2 and len(forensic_call_times) >= 1:
            # Second STT should start before or during first forensic
            assert forensic_call_times[0] < stt_call_times[-1] + 0.1

    @pytest.mark.asyncio
    async def test_process_files_empty_list(self, mock_stt_service, mock_forensic_service):
        """
        GIVEN: PipelineOrchestrator with services
        WHEN: process_files() is called with empty list
        THEN: Should return immediately with no results
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        results = []
        async for result in orchestrator.process_files([]):
            results.append(result)

        assert len(results) == 0


# ============================================================================
# Test: Shutdown
# ============================================================================


class TestShutdown:
    """Test graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_stop_event(self):
        """
        GIVEN: PipelineOrchestrator instance
        WHEN: shutdown() is called
        THEN: Should set stop event
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        await orchestrator.shutdown()

        assert orchestrator._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_clears_queue(self):
        """
        GIVEN: PipelineOrchestrator with items in queue
        WHEN: shutdown() is called
        THEN: Should drain and clear the queue
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Add items to queue
        for i in range(3):
            await orchestrator._queue.put({"test": i})

        await orchestrator.shutdown()

        assert orchestrator._queue.empty()

    @pytest.mark.asyncio
    async def test_shutdown_stops_thermal_monitoring(self, mock_thermal_manager):
        """
        GIVEN: PipelineOrchestrator with ThermalManager
        WHEN: shutdown() is called
        THEN: Should stop thermal monitoring
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(thermal_manager=mock_thermal_manager)

        await orchestrator.shutdown()

        mock_thermal_manager.stop_monitoring.assert_called_once()


# ============================================================================
# Test: Memory and Thermal Integration
# ============================================================================


class TestMemoryAndThermalIntegration:
    """Test integration with Phase 2 managers."""

    @pytest.mark.asyncio
    async def test_uses_memory_manager_for_stage_allocation(
        self, mock_stt_service, mock_forensic_service, mock_memory_manager, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with ForensicMemoryManager
        WHEN: Pipeline processes files
        THEN: Should allocate and release stage memory
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav", text="Test", segments=[], speakers=[]
        )
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav", overall_risk_score=50.0, risk_level="MODERATE", analysis_id="test"
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
            memory_manager=mock_memory_manager,
        )

        async for _ in orchestrator.process_files(sample_audio_files[:1]):
            pass

        # Verify memory manager was used
        mock_memory_manager.allocate.assert_called()
        mock_memory_manager.release.assert_called()

    @pytest.mark.asyncio
    async def test_respects_thermal_throttling(
        self, mock_stt_service, mock_forensic_service, mock_thermal_manager, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with ThermalManager in throttling state
        WHEN: Pipeline processes files
        THEN: Should slow down processing when throttling
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_thermal_manager.is_throttling = True

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav", text="Test", segments=[], speakers=[]
        )
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav", overall_risk_score=50.0, risk_level="MODERATE", analysis_id="test"
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
            thermal_manager=mock_thermal_manager,
        )

        # Processing should still work even when throttling
        results = []
        async for result in orchestrator.process_files(sample_audio_files[:1]):
            results.append(result)

        assert len(results) == 1


# ============================================================================
# Test: Statistics and Monitoring
# ============================================================================


class TestStatisticsAndMonitoring:
    """Test statistics and monitoring functionality."""

    def test_get_stats_returns_pipeline_status(self):
        """
        GIVEN: PipelineOrchestrator instance
        WHEN: get_stats() is called
        THEN: Should return pipeline statistics
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        stats = orchestrator.get_stats()

        assert "queue_size" in stats
        assert "backpressure_active" in stats
        assert "is_running" in stats

    @pytest.mark.asyncio
    async def test_stats_reflect_queue_state(self):
        """
        GIVEN: PipelineOrchestrator with items in queue
        WHEN: get_stats() is called
        THEN: Should reflect current queue size
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        for i in range(3):
            await orchestrator._queue.put({"test": i})

        stats = orchestrator.get_stats()

        assert stats["queue_size"] == 3


# ============================================================================
# Test: E2ETestService Integration (TASK-006)
# ============================================================================


class TestE2ETestServiceIntegration:
    """Test integration with E2ETestService."""

    def test_orchestrator_can_be_used_with_e2e_runner(
        self, mock_stt_service, mock_forensic_service, mock_memory_manager, mock_thermal_manager
    ):
        """
        GIVEN: E2ETestRunner configuration
        WHEN: PipelineOrchestrator is passed to E2ETestRunner
        THEN: Should integrate without errors
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
            memory_manager=mock_memory_manager,
            thermal_manager=mock_thermal_manager,
        )

        # Verify orchestrator has all required interfaces for E2E integration
        assert hasattr(orchestrator, "process_files")
        assert hasattr(orchestrator, "shutdown")
        assert hasattr(orchestrator, "get_stats")
        assert hasattr(orchestrator, "is_backpressure_active")

    @pytest.mark.asyncio
    async def test_orchestrator_provides_progress_tracking(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator processing files
        WHEN: Processing multiple files
        THEN: get_stats() should reflect progress
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav",
            text="Test transcript",
            segments=[],
            speakers=[],
        )
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=50.0,
            risk_level="MODERATE",
            analysis_id="test-123",
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        results = []
        async for result in orchestrator.process_files(sample_audio_files[:2]):
            results.append(result)
            stats = orchestrator.get_stats()
            assert "files_processed" in stats

        final_stats = orchestrator.get_stats()
        assert final_stats["files_processed"] >= 2

    @pytest.mark.asyncio
    async def test_orchestrator_handles_mixed_success_failure(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with some files failing
        WHEN: Processing files with mixed results
        THEN: Should continue processing and track failures
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        # First file succeeds, second fails, third succeeds
        mock_stt_service.process_audio.side_effect = [
            MockSTTResult(file_path="1.wav", text="Success 1", segments=[], speakers=[]),
            Exception("STT Failed"),
            MockSTTResult(file_path="3.wav", text="Success 3", segments=[], speakers=[]),
        ]
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=50.0,
            risk_level="MODERATE",
            analysis_id="test-123",
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        results = []
        async for result in orchestrator.process_files(sample_audio_files[:3]):
            results.append(result)

        # Should have 3 results (2 success + 1 error)
        assert len(results) == 3

        # One result should be an error
        error_results = [r for r in results if r.get("error") is not None]
        assert len(error_results) >= 1

        # Stats should reflect failure
        stats = orchestrator.get_stats()
        assert stats["files_failed"] >= 1


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_process_single_file(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with single file
        WHEN: Processing one file
        THEN: Should complete successfully
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        mock_stt_service.process_audio.return_value = MockSTTResult(
            file_path="test.wav",
            text="Single file test",
            segments=[],
            speakers=[],
        )
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=75.0,
            risk_level="HIGH",
            analysis_id="single-test",
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        results = []
        async for result in orchestrator.process_files(sample_audio_files[:1]):
            results.append(result)

        assert len(results) == 1
        assert results[0]["error"] is None

    @pytest.mark.asyncio
    async def test_shutdown_during_processing(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator actively processing
        WHEN: shutdown() is called during processing
        THEN: Should stop gracefully
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        async def slow_stt(audio_path, **kwargs):
            await asyncio.sleep(1.0)  # Simulate slow processing
            return MockSTTResult(file_path=str(audio_path), text="Slow", segments=[], speakers=[])

        mock_stt_service.process_audio = slow_stt
        mock_forensic_service.analyze.return_value = MockForensicResult(
            file_path="test.wav",
            overall_risk_score=50.0,
            risk_level="MODERATE",
            analysis_id="test",
        )

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        # Start processing in background
        async def collect_results(async_iter):
            collected = []
            try:
                async for r in async_iter:
                    collected.append(r)
            except Exception:
                pass
            return collected

        async def process_and_shutdown():
            _ = asyncio.create_task(
                collect_results(orchestrator.process_files(sample_audio_files[:3]))
            )
            await asyncio.sleep(0.1)  # Let processing start
            await orchestrator.shutdown()

        # Should not hang
        await asyncio.wait_for(process_and_shutdown(), timeout=2.0)

        # Verify shutdown state
        assert orchestrator._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_no_services_configured(self, sample_audio_files):
        """
        GIVEN: PipelineOrchestrator without services
        WHEN: Processing files
        THEN: Should handle gracefully (no crash, producer exits quickly)
        """
        from voice_man.services.forensic.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Without STT service, producer exits immediately
        # Consumer will wait briefly then also exit
        results = []

        async def collect_with_timeout():
            try:
                async for result in orchestrator.process_files(sample_audio_files[:1]):
                    results.append(result)
            except Exception:
                pass

        # Should complete quickly (producer has no STT service, returns immediately)
        await asyncio.wait_for(collect_with_timeout(), timeout=2.0)

        # Should not crash, no results produced (no STT service)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_queue_size_never_exceeds_max(
        self, mock_stt_service, mock_forensic_service, sample_audio_files
    ):
        """
        GIVEN: PipelineOrchestrator with bounded queue
        WHEN: Processing many files
        THEN: Queue size should never exceed MAX_QUEUE_SIZE
        """
        from voice_man.services.forensic.pipeline_orchestrator import (
            PipelineOrchestrator,
            MAX_QUEUE_SIZE,
        )

        async def fast_stt(audio_path, **kwargs):
            await asyncio.sleep(0.01)
            return MockSTTResult(file_path=str(audio_path), text="Fast", segments=[], speakers=[])

        async def slow_forensic(audio_path, transcript):
            await asyncio.sleep(0.1)  # Forensic slower than STT
            return MockForensicResult(
                file_path=audio_path,
                overall_risk_score=50.0,
                risk_level="MODERATE",
                analysis_id="test",
            )

        mock_stt_service.process_audio = fast_stt
        mock_forensic_service.analyze = slow_forensic

        orchestrator = PipelineOrchestrator(
            stt_service=mock_stt_service,
            forensic_service=mock_forensic_service,
        )

        max_observed_queue_size = 0

        async for result in orchestrator.process_files(sample_audio_files):
            current_size = orchestrator._queue.qsize()
            max_observed_queue_size = max(max_observed_queue_size, current_size)

        # Queue should never exceed MAX_QUEUE_SIZE due to backpressure
        assert max_observed_queue_size <= MAX_QUEUE_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
