#!/usr/bin/env python3
"""Find audio files longer than specified duration."""

import subprocess
from pathlib import Path


def get_duration(file_path: str) -> float:
    """Get audio file duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-i",
                file_path,
                "-show_entries",
                "format=duration",
                "-v",
                "quiet",
                "-of",
                "csv=p=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else 0.0
    except Exception:
        return 0.0


def main():
    """Find all audio files and sort by duration."""
    call_dir = Path("ref/call")

    if not call_dir.exists():
        print(f"Directory not found: {call_dir}")
        return

    audio_files = list(call_dir.glob("*.m4a"))

    if not audio_files:
        print("No .m4a files found")
        return

    # Get durations
    results = []
    for audio_file in audio_files:
        duration = get_duration(str(audio_file))
        if duration > 0:
            results.append((duration, audio_file))

    # Sort by duration (longest first)
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"총 {len(results)}개 오디오 파일 발견")
    print(f"{'=' * 80}\n")

    # Print all files with duration
    for duration, audio_file in results:
        minutes = duration / 60
        status = ""
        if duration >= 1800:  # 30 minutes
            status = " ⭐ 30분 이상!"
        elif duration >= 600:  # 10 minutes
            status = " ✓ 10분 이상 (청킹 대상)"
        print(f"{duration:7.1f}초 ({minutes:5.1f}분){status} - {audio_file.name}")

    # Summary
    print(f"\n{'=' * 80}")
    long_files = [d for d, _ in results if d >= 1800]
    chunking_files = [d for d, _ in results if d >= 600]
    print(f"30분 이상: {len(long_files)}개")
    print(f"10분 이상 (청킹 대상): {len(chunking_files)}개")
    print(f"{'=' * 80}")

    # Return longest file for further testing
    if results:
        longest_duration, longest_file = results[0]
        print(f"\n가장 긴 파일: {longest_file} ({longest_duration / 60:.1f}분)")
        return str(longest_file)


if __name__ == "__main__":
    main()
