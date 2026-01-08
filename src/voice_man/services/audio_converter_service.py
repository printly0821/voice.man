"""
Audio Converter Service

Converts audio files to WhisperX-compatible format (16kHz mono WAV).
Implements SPEC-WHISPERX-001 requirements.

EARS Requirements Implemented:
- E2: Auto audio format conversion (m4a -> 16kHz mono WAV)
- N3: No leftover temporary files (context manager cleanup)
"""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import for ffmpeg
ffmpeg = None


def _import_ffmpeg():
    """Lazy import ffmpeg."""
    global ffmpeg
    if ffmpeg is None:
        try:
            import ffmpeg as ff

            ffmpeg = ff
        except ImportError:
            raise ImportError(
                "ffmpeg-python not installed. Install with: pip install ffmpeg-python"
            )
    return ffmpeg


class UnsupportedFormatError(Exception):
    """Raised when audio format is not supported."""

    def __init__(self, format_name: str):
        self.format_name = format_name
        super().__init__(f"Unsupported audio format: {format_name}")


class AudioConverterService:
    """
    Audio format conversion service.

    E2: Converts various audio formats to 16kHz mono WAV for WhisperX.
    N3: Uses context manager for proper temporary file cleanup.
    """

    # E2: Supported input formats
    SUPPORTED_FORMATS = {"m4a", "mp3", "wav", "flac", "ogg", "webm", "aac", "wma"}

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
    ):
        """
        Initialize audio converter service.

        Args:
            target_sample_rate: Target sample rate (default: 16000 Hz)
            target_channels: Target number of channels (default: 1 for mono)
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_format = "wav"

        logger.info(
            f"AudioConverterService initialized: "
            f"target={target_sample_rate}Hz, {target_channels}ch, {self.target_format}"
        )

    def is_supported_format(self, format_name: str) -> bool:
        """
        Check if audio format is supported.

        Args:
            format_name: File extension (without dot)

        Returns:
            True if format is supported
        """
        return format_name.lower() in self.SUPPORTED_FORMATS

    def get_file_format(self, file_path: str) -> str:
        """
        Get file format from path.

        Args:
            file_path: Path to audio file

        Returns:
            File extension (without dot)
        """
        return Path(file_path).suffix.lstrip(".").lower()

    async def _get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio info (sample_rate, channels, format)
        """
        try:
            ff = _import_ffmpeg()
            probe = ff.probe(file_path)

            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None,
            )

            if audio_stream:
                return {
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "format": self.get_file_format(file_path),
                    "duration": float(audio_stream.get("duration", 0)),
                }
        except Exception as e:
            logger.warning(f"Could not probe audio file: {e}")

        return {
            "sample_rate": 0,
            "channels": 0,
            "format": self.get_file_format(file_path),
            "duration": 0,
        }

    def _needs_conversion(self, audio_info: Dict[str, Any]) -> bool:
        """
        Check if audio file needs conversion.

        Args:
            audio_info: Audio file information

        Returns:
            True if conversion is needed
        """
        if audio_info["format"] != self.target_format:
            return True
        if audio_info["sample_rate"] != self.target_sample_rate:
            return True
        if audio_info["channels"] != self.target_channels:
            return True
        return False

    async def convert_to_wav(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert audio file to 16kHz mono WAV.

        E2: Auto audio format conversion.

        Args:
            input_path: Path to input audio file
            output_path: Path for output file (optional, creates temp file if not provided)

        Returns:
            Path to converted WAV file

        Raises:
            FileNotFoundError: If input file doesn't exist
            UnsupportedFormatError: If input format is not supported
        """
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        file_format = self.get_file_format(input_path)
        if not self.is_supported_format(file_format):
            raise UnsupportedFormatError(file_format)

        # Check if conversion is needed
        audio_info = await self._get_audio_info(input_path)
        if not self._needs_conversion(audio_info):
            logger.info(f"No conversion needed for: {input_path}")
            return input_path

        # Determine output path
        if output_path is None:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            output_path = str(Path(temp_dir) / f"{input_file.stem}_converted.wav")

        logger.info(f"Converting {input_path} -> {output_path}")

        try:
            ff = _import_ffmpeg()

            # Build ffmpeg command
            stream = ff.input(input_path)
            stream = stream.output(
                output_path,
                ar=self.target_sample_rate,
                ac=self.target_channels,
                acodec="pcm_s16le",
                format="wav",
            )
            stream = stream.overwrite_output()

            # Run conversion
            stream.run(quiet=True, capture_stderr=True)

            logger.info(f"Conversion complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

    @asynccontextmanager
    async def convert_context(self, input_path: str):
        """
        Context manager for audio conversion with automatic cleanup.

        N3: Ensures temporary files are cleaned up even on error.

        Args:
            input_path: Path to input audio file

        Yields:
            Path to converted WAV file

        Example:
            async with converter.convert_context("audio.m4a") as wav_path:
                # Use wav_path
                pass
            # Temp file automatically cleaned up
        """
        temp_file = None
        converted_path = None

        try:
            # Check if conversion is needed
            audio_info = await self._get_audio_info(input_path)

            if self._needs_conversion(audio_info):
                # Create temp file for conversion
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False,
                )
                temp_file.close()
                converted_path = temp_file.name

                await self.convert_to_wav(input_path, converted_path)
                yield converted_path
            else:
                # No conversion needed
                yield input_path

        finally:
            # N3: Clean up temp file
            if converted_path and Path(converted_path).exists():
                if converted_path != input_path:
                    try:
                        Path(converted_path).unlink()
                        logger.debug(f"Cleaned up temp file: {converted_path}")
                    except Exception as e:
                        logger.warning(f"Could not clean up temp file: {e}")

    async def get_duration(self, file_path: str) -> float:
        """
        Get audio file duration in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        audio_info = await self._get_audio_info(file_path)
        return audio_info.get("duration", 0.0)
