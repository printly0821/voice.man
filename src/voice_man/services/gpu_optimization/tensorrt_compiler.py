"""
TensorRT Compiler for WhisperX Pipeline

Phase 3 Advanced Optimization (50-100x speedup target):
- ONNX to TensorRT conversion
- FP16/INT8 precision support
- Dynamic shape optimization
- EARS Requirements: E5 (TensorRT fallback)

Reference: SPEC-GPUOPT-001 Phase 3
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TensorRTConfig:
    """TensorRT compilation configuration."""

    # Precision settings
    precision: str = "fp16"  # fp32, fp16, int8

    # Optimization settings
    max_workspace_size: int = 1 << 30  # 1GB default
    optimization_level: int = 3  # 0-5, higher = more optimization

    # Dynamic shapes (for variable audio lengths)
    enable_dynamic_shapes: bool = True
    min_audio_length: int = 1  # 1 second
    opt_audio_length: int = 300  # 5 minutes
    max_audio_length: int = 3600  # 1 hour

    # Build settings
    build_flags: List[str] = field(
        default_factory=lambda: [
            "enable_fp16",
            "enable_timing_cache",
        ]
    )

    # Fallback settings
    fallback_to_onnx: bool = True
    fallback_to_torch: bool = True


@dataclass
class TensorRTBuildResult:
    """Result of TensorRT engine build."""

    success: bool
    engine_path: Optional[str] = None
    build_time_seconds: float = 0.0
    engine_size_mb: float = 0.0
    error_message: Optional[str] = None
    precision_used: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "engine_path": self.engine_path,
            "build_time_seconds": self.build_time_seconds,
            "engine_size_mb": self.engine_size_mb,
            "error_message": self.error_message,
            "precision_used": self.precision_used,
        }


class TensorRTCompiler:
    """
    TensorRT compiler for Whisper models.

    Features:
    - ONNX to TensorRT conversion
    - FP16/INT8 precision support
    - Dynamic shape optimization for variable audio lengths
    - Automatic fallback to ONNX/Torch if compilation fails

    EARS Requirements:
    - E5: Fallback to ONNX/Torch if TensorRT unavailable
    - S7: TensorRT optimization for inference speed

    Performance Target: 2-3x over Faster-Whisper (cumulative 50-100x)
    """

    def __init__(self, config: Optional[TensorRTConfig] = None):
        """
        Initialize TensorRT Compiler.

        Args:
            config: Compiler configuration (default: default config)
        """
        self.config = config or TensorRTConfig()

        # Check TensorRT availability
        self._tensorrt_available = self._check_tensorrt()
        self._onnx_available = self._check_onnx()

        logger.info(
            f"TensorRTCompiler initialized: "
            f"TensorRT={'available' if self._tensorrt_available else 'unavailable'}, "
            f"ONNX={'available' if self._onnx_available else 'unavailable'}"
        )

    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt  # noqa: F401

            return True
        except ImportError:
            logger.info("TensorRT not available. Install with: pip install tensorrt")
            return False

    def _check_onnx(self) -> bool:
        """Check if ONNX is available."""
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            logger.info("ONNX not available. Install with: pip install onnx onnxruntime")
            return False

    def convert_onnx_to_tensorrt(
        self,
        onnx_model_path: str,
        output_engine_path: Optional[str] = None,
        calibration_data: Optional[List[Any]] = None,
    ) -> TensorRTBuildResult:
        """
        Convert ONNX model to TensorRT engine.

        Args:
            onnx_model_path: Path to ONNX model
            output_engine_path: Output path for TensorRT engine
            calibration_data: Calibration data for INT8 precision

        Returns:
            TensorRTBuildResult with build status
        """
        import time

        start_time = time.time()

        if not self._tensorrt_available:
            # E5: Fallback if TensorRT unavailable
            if self.config.fallback_to_onnx and self._onnx_available:
                logger.info("TensorRT unavailable, using ONNX fallback")
                return TensorRTBuildResult(
                    success=False,
                    error_message="TensorRT unavailable, using ONNX fallback",
                    precision_used="onnx",
                )
            elif self.config.fallback_to_torch:
                logger.info("TensorRT unavailable, using Torch fallback")
                return TensorRTBuildResult(
                    success=False,
                    error_message="TensorRT unavailable, using Torch fallback",
                    precision_used="torch",
                )

            return TensorRTBuildResult(
                success=False,
                error_message="TensorRT unavailable and no fallback enabled",
            )

        onnx_path = Path(onnx_model_path)
        if not onnx_path.exists():
            return TensorRTBuildResult(
                success=False,
                error_message=f"ONNX model not found: {onnx_model_path}",
            )

        # Generate output path if not provided
        if output_engine_path is None:
            output_engine_path = str(onnx_path.parent / f"{onnx_path.stem}.engine")

        try:
            import tensorrt as trt

            # Create builder and network
            logger.info(f"Building TensorRT engine: {onnx_model_path}")
            logger.info(f"Output path: {output_engine_path}")
            logger.info(f"Precision: {self.config.precision}")

            # Build engine
            result = self._build_engine(trt, onnx_path, Path(output_engine_path), calibration_data)

            build_time = time.time() - start_time
            result.build_time_seconds = build_time

            if result.success:
                engine_size_mb = Path(output_engine_path).stat().st_size / (1024 * 1024)
                result.engine_size_mb = engine_size_mb
                result.precision_used = self.config.precision

                logger.info(
                    f"TensorRT engine built successfully: "
                    f"{output_engine_path} ({engine_size_mb:.1f}MB, {build_time:.1f}s)"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {e}")
            return TensorRTBuildResult(
                success=False,
                build_time_seconds=time.time() - start_time,
                error_message=str(e),
            )

    def _build_engine(
        self,
        trt: Any,
        onnx_path: Path,
        output_path: Path,
        calibration_data: Optional[List[Any]],
    ) -> TensorRTBuildResult:
        """
        Build TensorRT engine from ONNX model.

        Args:
            trt: TensorRT module
            onnx_path: Path to ONNX model
            output_path: Output path for engine
            calibration_data: Optional calibration data for INT8

        Returns:
            TensorRTBuildResult
        """
        try:
            # Create builder
            trt_logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(trt_logger)

            # Create network
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )

            # Create parser
            parser = trt.ONNXParser(network, trt_logger)

            # Parse ONNX model
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    errors = [parser.get_error(i) for i in range(parser.num_errors)]
                    return TensorRTBuildResult(
                        success=False,
                        error_message=f"ONNX parsing failed: {errors}",
                    )

            # Create builder config
            config = builder.create_builder_config()

            # Set workspace size
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.config.max_workspace_size
            )

            # Set precision flags
            if self.config.precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.config.precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # INT8 requires calibration
                if calibration_data:
                    # TODO: Implement INT8 calibration
                    logger.warning("INT8 calibration not yet implemented")

            # Set optimization level
            if hasattr(config, "set_optimization_level"):
                config.set_optimization_level(self.config.optimization_level)

            # Build serialized network
            logger.info("Building engine...")
            serialized_engine = builder.build_serialized_network(network)

            if serialized_engine is None:
                return TensorRTBuildResult(
                    success=False,
                    error_message="Engine build failed",
                )

            # Save engine
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(serialized_engine)

            return TensorRTBuildResult(
                success=True,
                engine_path=str(output_path),
            )

        except Exception as e:
            return TensorRTBuildResult(
                success=False,
                error_message=f"Engine build failed: {e}",
            )

    def load_engine(self, engine_path: str) -> Optional[Any]:
        """
        Load TensorRT engine from file.

        Args:
            engine_path: Path to TensorRT engine file

        Returns:
            Loaded engine or None if failed
        """
        if not self._tensorrt_available:
            logger.warning("Cannot load engine: TensorRT unavailable")
            return None

        engine_path_obj = Path(engine_path)
        if not engine_path_obj.exists():
            logger.error(f"Engine file not found: {engine_path}")
            return None

        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(trt_logger)

            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                logger.error(f"Failed to load engine: {engine_path}")
                return None

            logger.info(f"TensorRT engine loaded: {engine_path}")
            return engine

        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            return None

    def export_to_onnx(
        self,
        model: Any,
        output_path: str,
        opset_version: int = 17,
        sample_input: Optional[Any] = None,
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            opset_version: ONNX opset version
            sample_input: Sample input for tracing

        Returns:
            True if successful
        """
        try:
            import torch

            model.eval()
            model.cuda()  # Ensure model is on GPU

            # Create sample input if not provided
            if sample_input is None:
                # Default: 16kHz, 5 seconds audio
                sample_input = torch.randn(1, 1, 16000 * 5).cuda()

            dynamic_axes = {
                "audio": {0: "batch_size", 1: "audio_length"},
            }

            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["audio"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

            logger.info(f"Model exported to ONNX: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return False

    def get_engine_info(self, engine_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about TensorRT engine.

        Args:
            engine_path: Path to TensorRT engine file

        Returns:
            Dictionary with engine info or None
        """
        engine = self.load_engine(engine_path)
        if engine is None:
            return None

        try:
            return {
                "path": engine_path,
                "num_bindings": engine.num_bindings,
                "max_batch_size": engine.max_batch_size,
                "size_mb": Path(engine_path).stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Failed to get engine info: {e}")
            return None

    def cleanup_engine(self, engine_path: str) -> bool:
        """
        Remove TensorRT engine file.

        Args:
            engine_path: Path to engine file

        Returns:
            True if successful
        """
        try:
            Path(engine_path).unlink(missing_ok=True)
            logger.info(f"Engine removed: {engine_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove engine: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get compiler statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "tensorrt_available": self._tensorrt_available,
            "onnx_available": self._onnx_available,
            "precision": self.config.precision,
            "optimization_level": self.config.optimization_level,
            "max_workspace_size_mb": self.config.max_workspace_size / (1024 * 1024),
            "dynamic_shapes_enabled": self.config.enable_dynamic_shapes,
        }

    def is_available(self) -> bool:
        """Check if TensorRT is available."""
        return self._tensorrt_available
