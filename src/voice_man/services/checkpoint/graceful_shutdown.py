"""
Graceful Shutdown Handler for Batch Processing

Handles system signals (SIGTERM, SIGINT) to ensure clean shutdown:
- Save current state before exit
- Cleanup resources
- Close database connections
- Flush pending checkpoints
- Prevent data loss on crash

Usage:
    shutdown = GracefulShutdown()

    # Register cleanup callback
    shutdown.register_callback(lambda: save_checkpoint())

    # Enable signal handling
    shutdown.setup()

    # Check if shutdown requested
    if shutdown.is_shutdown_requested():
        print("Shutting down gracefully...")
        cleanup()
        sys.exit(0)
"""

import logging
import signal
import sys
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    Context manager for handling shutdown during critical sections.

    Usage:
        with ShutdownHandler(checkpoint_manager):
            # Critical operation that shouldn't be interrupted
            process_batch()
    """

    def __init__(self, checkpoint_manager=None, timeout: float = 30.0):
        """
        Initialize shutdown handler.

        Args:
            checkpoint_manager: Optional checkpoint manager to save state
            timeout: Maximum time to wait for cleanup (seconds)
        """
        self.checkpoint_manager = checkpoint_manager
        self.timeout = timeout
        self._original_handlers = {}

    def __enter__(self):
        """Enter critical section - temporarily ignore signals."""
        # Ignore SIGINT and SIGTERM during critical section
        self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit critical section - restore signal handlers."""
        # Restore original handlers
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        return False


class GracefulShutdown:
    """
    Graceful shutdown handler for batch processing.

    Features:
    - Signal handling (SIGTERM, SIGINT)
    - Multiple cleanup callbacks
    - Configurable shutdown timeout
    - Thread-safe shutdown flag
    - Force shutdown option

    Usage:
        shutdown = GracefulShutdown()

        @shutdown.on_shutdown
        def save_state():
            checkpoint_manager.save_batch_checkpoint(...)

        shutdown.setup()

        # In your processing loop
        while processing and not shutdown.is_shutdown_requested():
            process_item()
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize graceful shutdown handler.

        Args:
            timeout: Maximum time to wait for cleanup (seconds)
        """
        self.timeout = timeout
        self._shutdown_requested = False
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._original_handlers: dict = {}

        logger.info(f"GracefulShutdown initialized (timeout={timeout}s)")

    def register_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a cleanup callback to be called on shutdown.

        Args:
            callback: Function to call during shutdown
        """
        with self._lock:
            self._callbacks.append(callback)
        logger.debug(f"Registered shutdown callback: {callback.__name__}")

    def on_shutdown(self, func: Callable[[], None]) -> Callable[[], None]:
        """
        Decorator for registering shutdown callbacks.

        Args:
            func: Function to register as callback

        Returns:
            Original function unchanged

        Usage:
            @shutdown.on_shutdown
            def cleanup():
                save_checkpoint()
        """
        self.register_callback(func)
        return func

    def setup(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Signal handlers registered for SIGTERM and SIGINT")

    def teardown(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

        logger.info("Signal handlers restored")

    def _handle_signal(self, signum, frame) -> None:
        """
        Handle shutdown signal.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum})")

        # Request shutdown
        self.request_shutdown(signal_name)

    def request_shutdown(self, reason: str = "manual") -> None:
        """
        Request graceful shutdown.

        Args:
            reason: Reason for shutdown (for logging)
        """
        with self._lock:
            if self._shutdown_requested:
                logger.warning("Shutdown already requested, ignoring")
                return

            self._shutdown_requested = True
            logger.info(f"Shutdown requested: {reason}")

        # Execute callbacks
        self._execute_callbacks()

    def _execute_callbacks(self) -> None:
        """Execute all registered cleanup callbacks."""
        logger.info(f"Executing {len(self._callbacks)} cleanup callbacks...")

        for i, callback in enumerate(self._callbacks, 1):
            callback_name = getattr(callback, "__name__", f"callback_{i}")
            logger.info(f"Running callback {i}/{len(self._callbacks)}: {callback_name}")

            try:
                # Run callback with timeout
                import threading

                result = [None]
                exception = [None]

                def run_callback():
                    try:
                        result[0] = callback()
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=run_callback)
                thread.start()
                thread.join(timeout=self.timeout)

                if thread.is_alive():
                    logger.error(f"Callback {callback_name} timed out after {self.timeout}s")
                elif exception[0]:
                    logger.error(f"Callback {callback_name} failed: {exception[0]}")
                else:
                    logger.info(f"Callback {callback_name} completed")

            except Exception as e:
                logger.error(f"Failed to execute callback {callback_name}: {e}")

        logger.info("All cleanup callbacks completed")

    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.

        Returns:
            True if shutdown requested, False otherwise
        """
        with self._lock:
            return self._shutdown_requested

    def wait_for_shutdown(self, check_interval: float = 0.5) -> None:
        """
        Block until shutdown is requested.

        Args:
            check_interval: Time to wait between checks (seconds)

        Usage:
            # Start background processing
            thread = threading.Thread(target=process_files)
            thread.start()

            # Wait for shutdown signal
            shutdown.wait_for_shutdown()

            # Cleanup
            thread.join()
        """
        logger.info("Waiting for shutdown signal...")
        while not self.is_shutdown_requested():
            time.sleep(check_interval)
        logger.info("Shutdown signal received")

    def force_shutdown(self, exit_code: int = 0) -> None:
        """
        Force immediate shutdown without cleanup.

        Args:
            exit_code: Exit code for sys.exit
        """
        logger.warning(f"Force shutdown requested, exiting with code {exit_code}")
        sys.exit(exit_code)

    def reset(self) -> None:
        """Reset shutdown state (for testing)."""
        with self._lock:
            self._shutdown_requested = False
        logger.info("Shutdown state reset")

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()
        return False


class CheckpointShutdownHandler(GracefulShutdown):
    """
    Specialized shutdown handler for checkpoint-based workflows.

    Automatically saves checkpoint before shutdown.

    Usage:
        handler = CheckpointShutdownHandler(checkpoint_manager)

        handler.setup()

        # Process files...
        # On SIGTERM/SIGINT, checkpoint is automatically saved
    """

    def __init__(self, checkpoint_manager, timeout: float = 30.0):
        """
        Initialize checkpoint shutdown handler.

        Args:
            checkpoint_manager: CheckpointManager instance
            timeout: Maximum time to wait for cleanup
        """
        super().__init__(timeout=timeout)
        self.checkpoint_manager = checkpoint_manager

        # Register automatic checkpoint save
        self.register_callback(self._save_checkpoint_on_shutdown)

    def _save_checkpoint_on_shutdown(self) -> None:
        """Save current checkpoint before shutdown."""
        if self.checkpoint_manager is None:
            logger.warning("No checkpoint manager available")
            return

        try:
            workflow_id = self.checkpoint_manager.get_current_workflow_id()
            if workflow_id:
                logger.info(f"Saving checkpoint for workflow {workflow_id} before shutdown")

                # Mark workflow as crashed (will be resumed)
                from .state_store import WorkflowStatus

                self.checkpoint_manager.state_store.update_workflow_state(
                    workflow_id,
                    status=WorkflowStatus.CRASHED,
                )

                logger.info("Workflow marked as crashed for resume")
            else:
                logger.warning("No current workflow to checkpoint")

        except Exception as e:
            logger.error(f"Failed to save checkpoint on shutdown: {e}")
