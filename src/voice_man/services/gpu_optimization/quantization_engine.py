"""
Quantization Engine for WhisperX Pipeline

Phase 3 Advanced Optimization (50-100x speedup target):
- Post-Training Quantization (PTQ)
- INT8/FP16 hybrid quantization
- Dynamic vs Static quantization
- Quantization-aware evaluation
- EARS Requirements: E5 (fallback to non-quantized)

Reference: SPEC-GPUOPT-001 Phase 3
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Quantization configuration."""

    # Quantization mode
    mode: str = "dynamic_int8"  # dynamic_int8, static_int8, fp16, hybrid

    # Precision settings
    activation_dtype: str = "int8"  # int8, fp16
    weight_dtype: str = "int8"  # int8, fp16, fp32

    # Calibration settings (for static quantization)
    calibration_batch_size: int = 32
    calibration_samples: int = 100

    # Accuracy preservation
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.95  # WER threshold

    # Fallback settings
    fallback_to_fp16: bool = True
    fallback_to_fp32: bool = True

    # Layer-wise settings
    skip_layers: List[str] = field(default_factory=list)  # Layers to skip quantization
    quantize_per_channel: bool = True


@dataclass
class QuantizationResult:
    """Result of quantization process."""

    success: bool
    quantized_model: Optional[Any] = None
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    accuracy_drop: float = 0.0
    inference_speedup: float = 0.0
    error_message: Optional[str] = None
    quantization_mode_used: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "original_size_mb": self.original_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "compression_ratio": self.compression_ratio,
            "accuracy_drop": self.accuracy_drop,
            "inference_speedup": self.inference_speedup,
            "error_message": self.error_message,
            "quantization_mode_used": self.quantization_mode_used,
        }


class QuantizationEngine:
    """
    Quantization engine for Whisper models.

    Features:
    - Dynamic INT8 quantization (fastest, minimal accuracy loss)
    - Static INT8 quantization (requires calibration)
    - FP16 quantization (memory efficient, good accuracy)
    - Hybrid INT8/FP16 quantization (best balance)
    - Automatic accuracy evaluation
    - Fallback to higher precision if accuracy drops

    EARS Requirements:
    - E5: Fallback to non-quantized if quantization fails
    - U2: Accuracy preservation monitoring

    Performance Target: 1.5-2x speedup (cumulative 50-100x)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize Quantization Engine.

        Args:
            config: Quantization configuration (default: default config)
        """
        self.config = config or QuantizationConfig()

        # Check quantization availability
        self._torch_available = self._check_torch()

        logger.info(
            f"QuantizationEngine initialized: "
            f"mode={self.config.mode}, "
            f"torch={'available' if self._torch_available else 'unavailable'}"
        )

    def _check_torch(self) -> bool:
        """Check if PyTorch with quantization support is available."""
        try:
            import torch  # noqa: F401

            # Check for quantization support
            if hasattr(torch, "quantization"):
                return True
            return False
        except ImportError:
            logger.info("PyTorch not available for quantization")
            return False

    def quantize_model(
        self,
        model: Any,
        calibration_data: Optional[List[Any]] = None,
        evaluate_fn: Optional[Callable[[Any], float]] = None,
    ) -> QuantizationResult:
        """
        Quantize a PyTorch model.

        Args:
            model: PyTorch model to quantize
            calibration_data: Calibration data for static quantization
            evaluate_fn: Optional function to evaluate model accuracy

        Returns:
            QuantizationResult with quantization status
        """
        if not self._torch_available:
            # E5: Fallback if quantization unavailable
            logger.warning("PyTorch unavailable, returning original model")
            return QuantizationResult(
                success=False,
                quantized_model=model,
                error_message="PyTorch unavailable for quantization",
                quantization_mode_used="none",
            )

        # Get original model size
        original_size = self._get_model_size(model)

        try:
            import torch
            from torch.ao import quantization as tq

            model.eval()

            # Apply quantization based on mode
            if self.config.mode == "dynamic_int8":
                quantized_model = self._apply_dynamic_int8(model)
            elif self.config.mode == "static_int8":
                if calibration_data is None:
                    logger.warning(
                        "Static quantization requires calibration data, "
                        "falling back to dynamic quantization"
                    )
                    quantized_model = self._apply_dynamic_int8(model)
                else:
                    quantized_model = self._apply_static_int8(model, calibration_data)
            elif self.config.mode == "fp16":
                quantized_model = self._apply_fp16(model)
            elif self.config.mode == "hybrid":
                quantized_model = self._apply_hybrid(model)
            else:
                logger.warning(f"Unknown mode: {self.config.mode}, using FP16")
                quantized_model = self._apply_fp16(model)

            # Get quantized size
            quantized_size = self._get_model_size(quantized_model)

            # Evaluate accuracy if function provided
            accuracy_drop = 0.0
            speedup = 1.0

            if evaluate_fn and self.config.preserve_accuracy:
                original_accuracy = evaluate_fn(model)
                quantized_accuracy = evaluate_fn(quantized_model)
                accuracy_drop = original_accuracy - quantized_accuracy

                # Check if accuracy drop is acceptable
                if accuracy_drop > (1.0 - self.config.accuracy_threshold):
                    logger.warning(
                        f"Accuracy drop {accuracy_drop:.2%} exceeds threshold, falling back to FP16"
                    )
                    if self.config.fallback_to_fp16:
                        quantized_model = self._apply_fp16(model)
                        quantized_size = self._get_model_size(quantized_model)
                        accuracy_drop = 0.0

            # Estimate speedup (rough estimate)
            if self.config.mode in ("dynamic_int8", "static_int8"):
                speedup = 2.0  # INT8: ~2x speedup
            elif self.config.mode == "fp16":
                speedup = 1.5  # FP16: ~1.5x speedup

            result = QuantizationResult(
                success=True,
                quantized_model=quantized_model,
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=original_size / quantized_size if quantized_size > 0 else 1.0,
                accuracy_drop=accuracy_drop,
                inference_speedup=speedup,
                quantization_mode_used=self.config.mode,
            )

            logger.info(
                f"Quantization complete: "
                f"mode={self.config.mode}, "
                f"size={original_size:.1f}MB -> {quantized_size:.1f}MB, "
                f"compression={result.compression_ratio:.2f}x, "
                f"speedup={speedup:.2f}x"
            )

            return result

        except Exception as e:
            logger.error(f"Quantization failed: {e}")

            # Fallback to FP16 if enabled
            if self.config.fallback_to_fp16:
                logger.info("Falling back to FP16 quantization")
                try:
                    quantized_model = self._apply_fp16(model)
                    quantized_size = self._get_model_size(quantized_model)

                    return QuantizationResult(
                        success=True,
                        quantized_model=quantized_model,
                        original_size_mb=original_size,
                        quantized_size_mb=quantized_size,
                        compression_ratio=original_size / quantized_size
                        if quantized_size > 0
                        else 1.0,
                        quantization_mode_used="fp16_fallback",
                    )
                except Exception as fp16_error:
                    logger.error(f"FP16 fallback failed: {fp16_error}")

            # Final fallback to original model
            if self.config.fallback_to_fp32:
                logger.info("Using original model (FP32)")
                return QuantizationResult(
                    success=False,
                    quantized_model=model,
                    original_size_mb=original_size,
                    quantized_size_mb=original_size,
                    compression_ratio=1.0,
                    error_message=str(e),
                    quantization_mode_used="fp32_fallback",
                )

            return QuantizationResult(
                success=False,
                error_message=str(e),
                quantization_mode_used="none",
            )

    def _apply_dynamic_int8(self, model: Any) -> Any:
        """Apply dynamic INT8 quantization."""
        import torch
        from torch.ao.quantization import quantize_dynamic

        # Configure dynamic quantization
        qconfig_spec = {
            torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
            torch.nn.Conv2d: torch.ao.quantization.default_dynamic_qconfig,
        }

        quantized_model = quantize_dynamic(
            model,
            qconfig_spec,
            dtype=torch.qint8,
        )

        logger.debug("Applied dynamic INT8 quantization")
        return quantized_model

    def _apply_static_int8(self, model: Any, calibration_data: List[Any]) -> Any:
        """Apply static INT8 quantization with calibration."""
        import torch
        from torch.ao.quantization import (
            get_default_qconfig,
            prepare,
            convert,
        )

        # Configure quantization
        model.qconfig = get_default_qconfig("fbgemm")

        # Prepare for calibration
        prepared_model = prepare(model)

        # Calibrate
        logger.info(f"Calibrating with {len(calibration_data)} samples...")
        prepared_model.eval()

        with torch.no_grad():
            for data in calibration_data[: self.config.calibration_samples]:
                # Forward pass for calibration
                try:
                    _ = prepared_model(data)
                except Exception:
                    # Skip calibration data that doesn't work
                    continue

        # Convert to quantized model
        quantized_model = convert(prepared_model)

        logger.debug("Applied static INT8 quantization")
        return quantized_model

    def _apply_fp16(self, model: Any) -> Any:
        """Apply FP16 (half precision) quantization."""
        import torch

        quantized_model = model.half()

        logger.debug("Applied FP16 quantization")
        return quantized_model

    def _apply_hybrid(self, model: Any) -> Any:
        """Apply hybrid INT8/FP16 quantization."""
        import torch
        from torch.ao.quantization import quantize_dynamic

        # Apply dynamic INT8 to linear layers
        qconfig_spec = {
            torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
        }

        quantized_model = quantize_dynamic(
            model,
            qconfig_spec,
            dtype=torch.qint8,
        )

        # Convert remaining to FP16
        for module in quantized_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.half()

        logger.debug("Applied hybrid INT8/FP16 quantization")
        return quantized_model

    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        import torch

        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    def get_quantization_stats(self, model: Any) -> Dict[str, Any]:
        """
        Get quantization statistics for a model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with statistics
        """
        import torch

        stats = {
            "model_size_mb": self._get_model_size(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "dtype_precision": str(next(model.parameters()).dtype),
        }

        # Check if model is quantized
        is_quantized = False
        for module in model.modules():
            if hasattr(module, "weight_quantizer") or hasattr(module, "activation_quantizer"):
                is_quantized = True
                break

        stats["is_quantized"] = is_quantized
        return stats

    def dequantize_model(self, model: Any) -> Any:
        """
        Dequantize a quantized model.

        Args:
            model: Quantized model

        Returns:
            Dequantized model
        """
        import torch

        # Convert back to FP32
        model = model.float()

        logger.debug("Model dequantized to FP32")
        return model

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "torch_available": self._torch_available,
            "quantization_mode": self.config.mode,
            "preserve_accuracy": self.config.preserve_accuracy,
            "accuracy_threshold": self.config.accuracy_threshold,
            "fallback_enabled": {
                "fp16": self.config.fallback_to_fp16,
                "fp32": self.config.fallback_to_fp32,
            },
        }

    def is_available(self) -> bool:
        """Check if quantization is available."""
        return self._torch_available
