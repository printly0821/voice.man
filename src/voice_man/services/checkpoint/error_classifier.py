"""
Error Classification for Intelligent Retry Strategies

Categorizes errors into types with appropriate retry strategies:
- TRANSIENT: Temporary errors (network, GPU busy) - retry with backoff
- RECOVERABLE: Can be fixed (memory, cleanup) - retry after mitigation
- PERMANENT: Skip file (corrupt data) - no retry

Usage:
    classifier = ErrorClassifier()

    try:
        result = process_file()
    except Exception as e:
        category, severity = classifier.classify_error(e)
        strategy = classifier.determine_retry_strategy(category)

        if strategy.should_retry:
            # Retry with appropriate strategy
            await asyncio.sleep(strategy.backoff_seconds)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Category of error determining retry strategy."""

    TRANSIENT = "transient"  # Temporary errors, retry with backoff
    RECOVERABLE = "recoverable"  # Recoverable with mitigation
    PERMANENT = "permanent"  # Permanent failure, skip
    UNKNOWN = "unknown"  # Uncategorized, treat as transient


class ErrorSeverity(str, Enum):
    """Severity level for error handling and logging."""

    LOW = "low"  # Warning, can continue
    MEDIUM = "medium"  # Needs attention, may retry
    HIGH = "high"  # Critical, needs intervention
    CRITICAL = "critical"  # System failure


@dataclass
class RetryStrategy:
    """
    Retry strategy for an error category.

    Attributes:
        should_retry: Whether to retry the operation
        max_retries: Maximum number of retry attempts
        backoff_seconds: Initial backoff delay in seconds
        exponential_backoff: Whether to use exponential backoff
        cleanup_before_retry: Whether to perform cleanup before retry
        fallback_to_cpu: Whether to fallback to CPU on GPU errors
    """

    should_retry: bool
    max_retries: int
    backoff_seconds: float
    exponential_backoff: bool = True
    cleanup_before_retry: bool = False
    fallback_to_cpu: bool = False


class ErrorClassifier:
    """
    Intelligent error classifier for retry strategies.

    Features:
    - Pattern-based error classification
    - Configurable retry strategies per category
    - GPU/CUDA error detection
    - Memory error detection
    - Network/IO error detection
    - File corruption detection

    Error Patterns:
        CUDA/GPU Errors:
        - "out of memory" -> RECOVERABLE (cleanup + retry)
        - "CUDA error" -> TRANSIENT (retry with backoff)
        - "NVRTC error" -> TRANSIENT (retry with backoff)

        Memory Errors:
        - "MemoryError" -> RECOVERABLE (cleanup + retry)
        - "Allocation failed" -> RECOVERABLE (cleanup + retry)

        Network/IO Errors:
        - "Connection refused" -> TRANSIENT (retry with backoff)
        - "Timeout" -> TRANSIENT (retry with backoff)
        - "Network unreachable" -> TRANSIENT (retry with backoff)

        File Errors:
        - "File not found" -> PERMANENT (skip file)
        - "Permission denied" -> PERMANENT (skip file)
        - "Corrupt data" -> PERMANENT (skip file)

        ValueError/RuntimeError:
        - Context-dependent classification
    """

    # Default error patterns with categories
    ERROR_PATTERNS: Dict[str, ErrorCategory] = {
        # CUDA/GPU errors
        r"cuda.*out of memory": ErrorCategory.RECOVERABLE,
        r"cuda.*error": ErrorCategory.TRANSIENT,
        r"gpu.*out of memory": ErrorCategory.RECOVERABLE,
        r"nvrtc.*error": ErrorCategory.TRANSIENT,
        r"device.*side assert": ErrorCategory.PERMANENT,
        # Memory errors
        r"memoryerror": ErrorCategory.RECOVERABLE,
        r"allocation.*failed": ErrorCategory.RECOVERABLE,
        r"heap.*overflow": ErrorCategory.RECOVERABLE,
        r"stack.*overflow": ErrorCategory.PERMANENT,
        # Network/IO errors
        r"connection.*refused": ErrorCategory.TRANSIENT,
        r"connection.*reset": ErrorCategory.TRANSIENT,
        r"timeout": ErrorCategory.TRANSIENT,
        r"network.*unreachable": ErrorCategory.TRANSIENT,
        r"temporary.*failure": ErrorCategory.TRANSIENT,
        # File errors
        r"file not found": ErrorCategory.PERMANENT,
        r"no such file": ErrorCategory.PERMANENT,
        r"permission denied": ErrorCategory.PERMANENT,
        r"access denied": ErrorCategory.PERMANENT,
        r"corrupt.*data": ErrorCategory.PERMANENT,
        r"invalid.*header": ErrorCategory.PERMANENT,
        r"truncated.*file": ErrorCategory.PERMANENT,
        # Audio-specific errors
        r"unsupported.*codec": ErrorCategory.PERMANENT,
        r"invalid.*sample.*rate": ErrorCategory.PERMANENT,
        r"audio.*duration.*zero": ErrorCategory.PERMANENT,
        # Model errors
        r"model.*not found": ErrorCategory.PERMANENT,
        r"weight.*load.*failed": ErrorCategory.PERMANENT,
    }

    # Retry strategies per category
    RETRY_STRATEGIES: Dict[ErrorCategory, RetryStrategy] = {
        ErrorCategory.TRANSIENT: RetryStrategy(
            should_retry=True,
            max_retries=3,
            backoff_seconds=2.0,
            exponential_backoff=True,
            cleanup_before_retry=False,
        ),
        ErrorCategory.RECOVERABLE: RetryStrategy(
            should_retry=True,
            max_retries=2,
            backoff_seconds=5.0,
            exponential_backoff=False,
            cleanup_before_retry=True,
            fallback_to_cpu=True,  # Fallback for GPU memory issues
        ),
        ErrorCategory.PERMANENT: RetryStrategy(
            should_retry=False,
            max_retries=0,
            backoff_seconds=0.0,
        ),
        ErrorCategory.UNKNOWN: RetryStrategy(
            should_retry=True,
            max_retries=1,
            backoff_seconds=1.0,
            exponential_backoff=False,
        ),
    }

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, ErrorCategory]] = None,
        custom_strategies: Optional[Dict[ErrorCategory, RetryStrategy]] = None,
    ):
        """
        Initialize error classifier.

        Args:
            custom_patterns: Custom error patterns to add/override defaults
            custom_strategies: Custom retry strategies to add/override defaults
        """
        self.patterns = self.ERROR_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.strategies = self.RETRY_STRATEGIES.copy()
        if custom_strategies:
            self.strategies.update(custom_strategies)

        # Compile patterns for performance
        self._compiled_patterns: List[Tuple[re.Pattern, ErrorCategory]] = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in self.patterns.items()
        ]

        logger.info(
            f"ErrorClassifier initialized with {len(self.patterns)} patterns "
            f"and {len(self.strategies)} strategies"
        )

    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an exception into category and severity.

        Args:
            error: Exception to classify

        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        error_full = f"{error_type}: {error_message}"

        logger.debug(f"Classifying error: {error_full}")

        # Check against patterns
        for pattern, category in self._compiled_patterns:
            if pattern.search(error_full):
                severity = self._determine_severity(category, error_message)
                logger.info(
                    f"Error classified as {category.value} ({severity.value}): "
                    f"{error_type} - {error_message[:100]}"
                )
                return category, severity

        # Type-based fallback classification
        category, severity = self._classify_by_type(error_type, error_message)
        logger.info(
            f"Error classified by type as {category.value} ({severity.value}): {error_type}"
        )
        return category, severity

    def _classify_by_type(
        self, error_type: str, error_message: str
    ) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error based on exception type when pattern matching fails."""
        # CUDA/GPU errors
        if "cuda" in error_type.lower() or "gpu" in error_message:
            if "out of memory" in error_message:
                return ErrorCategory.RECOVERABLE, ErrorSeverity.MEDIUM
            return ErrorCategory.TRANSIENT, ErrorSeverity.HIGH

        # Timeout errors
        if "timeout" in error_type.lower() or "timeout" in error_message:
            return ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM

        # Network/IO errors
        if (
            "connection" in error_type.lower()
            or "network" in error_type.lower()
            or "http" in error_type.lower()
        ):
            return ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM

        # File not found
        if "filenotfound" in error_type.lower() or "no such file" in error_message:
            return ErrorCategory.PERMANENT, ErrorSeverity.HIGH

        # ValueError (often input-related)
        if error_type == "ValueError":
            return ErrorCategory.PERMANENT, ErrorSeverity.MEDIUM

        # RuntimeError (various causes)
        if error_type == "RuntimeError":
            if "internal" in error_message or "bug" in error_message:
                return ErrorCategory.PERMANENT, ErrorSeverity.HIGH
            return ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM

        # KeyError (missing data)
        if error_type == "KeyError":
            return ErrorCategory.PERMANENT, ErrorSeverity.MEDIUM

        # AttributeError (missing attribute/method)
        if error_type == "AttributeError":
            return ErrorCategory.PERMANENT, ErrorSeverity.HIGH

        # Default to unknown (transient with limited retry)
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    def _determine_severity(self, category: ErrorCategory, error_message: str) -> ErrorSeverity:
        """Determine severity level based on category and message."""
        if category == ErrorCategory.PERMANENT:
            # Check for critical permanent errors
            if any(
                critical in error_message
                for critical in ["corrupt", "assert", "internal", "critical"]
            ):
                return ErrorSeverity.CRITICAL
            return ErrorSeverity.HIGH

        if category == ErrorCategory.RECOVERABLE:
            if "memory" in error_message or "allocation" in error_message:
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.LOW

        if category == ErrorCategory.TRANSIENT:
            if "timeout" in error_message:
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def determine_retry_strategy(
        self, category: ErrorCategory, custom_strategy: Optional[RetryStrategy] = None
    ) -> RetryStrategy:
        """
        Get retry strategy for an error category.

        Args:
            category: Error category
            custom_strategy: Optional custom strategy for this specific case

        Returns:
            RetryStrategy for the category
        """
        if custom_strategy:
            return custom_strategy

        return self.strategies.get(category, self.strategies[ErrorCategory.UNKNOWN])

    def get_backoff_seconds(
        self, attempt: int, category: ErrorCategory, strategy: Optional[RetryStrategy] = None
    ) -> float:
        """
        Calculate backoff delay for a retry attempt.

        Args:
            attempt: Attempt number (0-indexed)
            category: Error category
            strategy: Optional custom strategy

        Returns:
            Backoff delay in seconds
        """
        if strategy is None:
            strategy = self.determine_retry_strategy(category)

        if not strategy.exponential_backoff:
            return strategy.backoff_seconds

        # Exponential backoff: base * 2^attempt
        backoff = strategy.backoff_seconds * (2**attempt)
        return min(backoff, 60.0)  # Cap at 60 seconds

    def should_fallback_to_cpu(self, error: Exception) -> bool:
        """
        Check if error warrants CPU fallback.

        Args:
            error: Exception to check

        Returns:
            True if should fallback to CPU
        """
        category, _ = self.classify_error(error)
        strategy = self.determine_retry_strategy(category)
        return strategy.fallback_to_cpu

    def should_cleanup_before_retry(self, error: Exception) -> bool:
        """
        Check if cleanup is needed before retry.

        Args:
            error: Exception to check

        Returns:
            True if cleanup should be performed
        """
        category, _ = self.classify_error(error)
        strategy = self.determine_retry_strategy(category)
        return strategy.cleanup_before_retry

    def add_custom_pattern(
        self, pattern: str, category: ErrorCategory, override: bool = False
    ) -> None:
        """
        Add a custom error pattern.

        Args:
            pattern: Regex pattern for error matching
            category: Category to assign to matched errors
            override: Whether to override existing patterns
        """
        if not override and pattern in self.patterns:
            logger.warning(f"Pattern already exists, not overriding: {pattern}")
            return

        self.patterns[pattern] = category
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), cat) for p, cat in self.patterns.items()
        ]
        logger.info(f"Added custom pattern: {pattern} -> {category.value}")

    def set_custom_strategy(self, category: ErrorCategory, strategy: RetryStrategy) -> None:
        """
        Set custom retry strategy for a category.

        Args:
            category: Error category
            strategy: Custom retry strategy
        """
        self.strategies[category] = strategy
        logger.info(f"Set custom strategy for {category.value}: {strategy}")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get classifier statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_patterns": len(self.patterns),
            "total_strategies": len(self.strategies),
            "patterns_by_category": {
                category.value: sum(1 for _, cat in self._compiled_patterns if cat == category)
                for category in ErrorCategory
            },
            "strategies": {
                category.value: {
                    "should_retry": strategy.should_retry,
                    "max_retries": strategy.max_retries,
                    "backoff_seconds": strategy.backoff_seconds,
                    "exponential_backoff": strategy.exponential_backoff,
                    "cleanup_before_retry": strategy.cleanup_before_retry,
                    "fallback_to_cpu": strategy.fallback_to_cpu,
                }
                for category, strategy in self.strategies.items()
            },
        }
