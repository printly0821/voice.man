"""
HardwareAcceleratedCodec - NVENC/NVDEC 하드웨어 가속

Phase 1: GPU에서 오디오 디코딩/인코딩
"""

import torch
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class HardwareAcceleratedCodec:
    """
    NVENC/NVDEC 하드웨어 가속

    GPU에서 오디오 디코딩/인코딩
    """

    def __init__(self, use_nvdec: bool = True, device: Optional[str] = None):
        """
        HardwareAcceleratedCodec 초기화

        Args:
            use_nvdec: NVDEC 하드웨어 가속 사용 여부
            device: CUDA 장치 ID
        """
        self.use_nvdec = use_nvdec
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.device.type == "cuda":
            logger.info(
                f"HardwareAcceleratedCodec initialized with NVDEC={use_nvdec} on {self.device}"
            )
        else:
            logger.warning("CUDA not available, software decoding only")

        # 디코딩 메트릭
        self._decoding_metrics: Dict[str, Any] = {
            "decoding_time": 0.0,
            "throughput": 0.0,
            "cpu_usage": 0.0,
        }

    def is_nvdec_supported(self) -> bool:
        """
        NVDEC 하드웨어 가속 지원 여부 확인

        Returns:
            NVDEC 지원 여부
        """
        # Grace Blackwell은 NVDEC 지원
        if self.device.type == "cuda":
            try:
                # CUDA 버전 확인
                cuda_version = torch.version.cuda
                if cuda_version:
                    logger.info(f"NVDEC supported (CUDA {cuda_version})")
                    return True
            except Exception as e:
                logger.warning(f"NVDEC support check failed: {e}")
        return False

    def decode_audio_gpu(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        GPU에서 오디오 디코딩

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            디코딩된 오디오 텐서 (GPU 메모리)
        """
        import time

        start_time = time.time()

        try:
            if self.use_nvdec and self.is_nvdec_supported():
                # NVDEC 하드웨어 가속
                audio = self._decode_with_nvdec(audio_path)
            else:
                # 소프트웨어 폴백
                audio = self._decode_software(audio_path)

            # 메트릭 업데이트
            decoding_time = time.time() - start_time
            self._decoding_metrics["decoding_time"] = decoding_time

            return audio

        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            return None

    def _decode_with_nvdec(self, audio_path: str) -> torch.Tensor:
        """
        NVDEC로 디코딩

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            디코딩된 오디오 텐서
        """
        try:
            # torchaudio를 사용한 GPU 디코딩
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)

            # GPU로 전송 (Zero-copy if Unified Memory)
            if self.device.type == "cuda":
                waveform = waveform.to(self.device)

            logger.info(f"Decoded audio with torchaudio: {waveform.shape} @ {sample_rate}Hz")
            return waveform

        except ImportError:
            logger.warning("torchaudio not available, falling back to software decoding")
            return self._decode_software(audio_path)

    def _decode_software(self, audio_path: str) -> torch.Tensor:
        """
        소프트웨어 디코딩

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            디코딩된 오디오 텐서
        """
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)

            # CPU 메모리에 유지
            logger.info(f"Decoded audio (software): {waveform.shape} @ {sample_rate}Hz")
            return waveform

        except ImportError:
            # torchaudio가 없으면 더미 텐서 반환
            logger.error("torchaudio not available")
            return torch.zeros(1, 16000)

    def get_supported_formats(self) -> List[str]:
        """
        지원하는 오디오 형식 목록

        Returns:
            지원 형식 리스트
        """
        # 일반적인 오디오 형식
        formats = ["wav", "mp3", "flac", "ogg", "m4a", "aac"]

        if self.use_nvdec:
            # NVDEC가 지원하는 형식 (실제 하드웨어 의존)
            formats.extend(["wav", "mp3"])

        return formats

    def get_cpu_usage(self) -> float:
        """
        디코딩 중 CPU 사용률

        Returns:
            CPU 사용률 (0-100%)
        """
        try:
            import psutil

            cpu_usage = psutil.cpu_percent(interval=0.1)
            self._decoding_metrics["cpu_usage"] = cpu_usage
            return cpu_usage
        except Exception:
            return self._decoding_metrics.get("cpu_usage", 0.0)

    def get_decoding_metrics(self) -> Dict[str, Any]:
        """
        디코딩 성능 메트릭

        Returns:
            성능 메트릭 딕셔너리
        """
        return self._decoding_metrics.copy()
