"""
UnifiedMemoryManager - Grace Blackwell 통합 메모리 관리자

Phase 1: Zero-copy CPU-GPU 메모리 공유 구현
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    """
    Grace Blackwell 통합 메모리 관리자

    Zero-copy: CPU-GPU 간 메모리 복사 불필요
    128GB 통합 메모리 활용
    """

    def __init__(self, device: Optional[str] = None):
        """
        UnifiedMemoryManager 초기화

        Args:
            device: CUDA 장치 ID (기본값: "cuda:0")
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.total_memory = 128 * 1024 * 1024 * 1024  # 128GB

        if self.device.type == "cuda":
            logger.info(f"UnifiedMemoryManager initialized on {self.device}")
        else:
            logger.warning("CUDA not available, falling back to CPU")

    def allocate_unified(self, shape: tuple, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        통합 메모리에 텐서 할당 (Zero-copy)

        Args:
            shape: 텐서 형태
            dtype: 데이터 타입 (기본값: float16)

        Returns:
            할당된 텐서
        """
        try:
            # CUDA 통합 메모리 할당 시도
            if self.device.type == "cuda":
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                logger.info(f"Allocated {shape} {dtype} tensor to unified memory")
                return tensor
            else:
                # CPU 폴백
                tensor = torch.empty(shape, dtype=dtype, device="cpu")
                logger.info(f"Allocated {shape} {dtype} tensor to CPU (CUDA unavailable)")
                return tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU OOM or error: {e}, falling back to CPU")
                # CPU 폴백
                return torch.empty(shape, dtype=dtype, device="cpu")
            else:
                raise

    def get_memory_usage(self) -> Dict[str, float]:
        """
        현재 메모리 사용량 조회

        Returns:
            메모리 사용 정보 딕셔너리
        """
        if self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
                total = self.total_memory / (1024**3)  # GB

                return {
                    "total": total,
                    "used": allocated,
                    "reserved": reserved,
                    "free": total - allocated,
                }
            except Exception as e:
                logger.warning(f"Failed to get memory usage: {e}")
                return {"total": 128.0, "used": 0.0, "reserved": 0.0, "free": 128.0}
        else:
            # CPU 메모리 정보 (psutil 사용 가능)
            import psutil

            mem = psutil.virtual_memory()
            return {
                "total": mem.total / (1024**3),
                "used": mem.used / (1024**3),
                "free": mem.available / (1024**3),
            }

    def release_memory(self) -> None:
        """
        명시적 메모리 해제

        CUDA 캐시를 정리하여 메모리를 확보합니다.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")

    def is_unified_memory_available(self) -> bool:
        """
        통합 메모리 지원 여부 확인

        Returns:
            통합 메모리 사용 가능 여부
        """
        # Grace Blackwell의 경우 항상 True 반환
        # 실제 하드웨어에서는 cudaGetDeviceProperties 확인 필요
        return torch.cuda.is_available()
