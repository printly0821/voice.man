#!/usr/bin/env python3
"""
Real-time Pipeline Monitor
===========================

Displays real-time progress of the forensic pipeline execution.

Usage:
    python scripts/monitor_pipeline.py
"""

import asyncio
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


class PipelineMonitor:
    """Real-time pipeline progress monitor"""

    def __init__(self, log_file: str = "logs/pipeline_full_183.log"):
        self.log_file = Path(log_file)
        self.last_size = 0
        self.start_time = None
        self.processed_files = 0
        self.total_files = 183
        self.last_update = time.time()

    def get_gpu_status(self):
        """Get GPU status"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    return {
                        "name": parts[0],
                        "temp": f"{parts[1]}°C",
                        "util": f"{parts[2]}%",
                        "power": f"{parts[3]}W",
                    }
        except:
            pass
        return {"name": "N/A", "temp": "N/A", "util": "N/A", "power": "N/A"}

    def get_process_status(self):
        """Get pipeline process status"""
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            for line in result.stdout.split("\n"):
                if "run_safe_pipeline_fixed" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return {"pid": parts[1], "cpu": f"{parts[2]}%", "mem": f"{parts[3]}%"}
        except:
            pass
        return None

    def parse_progress(self, line):
        """Parse progress from log line"""
        # Match: "Progress: X/Y completed"
        match = re.search(r"Progress:\s*(\d+)/(\d+)", line)
        if match:
            return int(match.group(1)), int(match.group(2))

        # Match: "[X/183] Processing:"
        match = re.search(r"\[(\d+)/183\].*Processing:", line)
        if match:
            return int(match.group(1)), 183

        return None

    def parse_forensic_result(self, line):
        """Parse forensic result"""
        # Match: "Forensic complete: risk=X.X"
        match = re.search(r"Forensic complete:\s+risk=([\d.]+)", line)
        if match:
            return float(match.group(1))
        return None

    def get_latest_results(self, num_lines: int = 100):
        """Get latest results from log"""
        if not self.log_file.exists():
            return []

        try:
            result = subprocess.run(
                ["tail", f"-{num_lines}", str(self.log_file)],
                capture_output=True,
                text=True,
                timeout=2,
            )

            results = []
            for line in result.stdout.split("\n"):
                forensic_match = self.parse_forensic_result(line)
                if forensic_match is not None:
                    results.append(forensic_match)

            return results[-5:]  # Return last 5 results
        except:
            return []

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("clear")

    def display_header(self):
        """Display monitor header"""
        print("\n" + "=" * 70)
        print("🔍 FORENSIC PIPELINE REAL-TIME MONITOR")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def display_stats(self):
        """Display current statistics"""
        gpu = self.get_gpu_status()
        proc = self.get_process_status()
        results = self.get_latest_results()

        # Progress bar
        progress_pct = (self.processed_files / self.total_files) * 100
        bar_length = 40
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n📊 진행률: {self.processed_files}/{self.total_files} ({progress_pct:.1f}%)")
        print(f"   [{bar}]")

        # Estimated time remaining
        if self.start_time and self.processed_files > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta_seconds = avg_time * remaining_files
            eta = str(timedelta(seconds=int(eta_seconds)))
            print(f"   ⏱ 경과 시간: {str(timedelta(seconds=int(elapsed)))} | 예상 남은 시간: {eta}")

        # GPU Status
        print(f"\n🖥️  GPU: {gpu['name']}")
        print(f"   온도: {gpu['temp']} | 사용률: {gpu['util']} | 전력: {gpu['power']}")

        # Process Status
        if proc:
            print(f"\n💻 프로세스 (PID: {proc['pid']}):")
            print(f"   CPU: {proc['cpu']} | 메모리: {proc['mem']}")

        # Recent Results
        if results:
            print("\n📈 최근 포렌식 점수:")
            for i, score in enumerate(results[-5:], 1):
                risk_level = "낮음" if score < 10 else "중간" if score < 30 else "높음"
                print(f"   {i}. 위험도: {score:.1f} ({risk_level})")

        print("\n" + "=" * 70)
        print("Ctrl+C to exit | Auto-refresh every 5 seconds")
        print("=" * 70 + "\n")

    def tail_log(self):
        """Tail the log file for new entries"""
        try:
            current_size = self.log_file.stat().st_size
            if current_size > self.last_size:
                result = subprocess.run(
                    [
                        "tail",
                        f"-{max(1, (current_size - self.last_size) // 100)}",
                        str(self.log_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )

                # Update processed files count
                for line in result.stdout.split("\n"):
                    progress = self.parse_progress(line)
                    if progress:
                        self.processed_files = max(self.processed_files, progress[0])
                        if not self.start_time:
                            self.start_time = time.time()

                self.last_size = current_size

                # Show recent processing activity
                lines = result.stdout.split("\n")[-10:]
                for line in lines:
                    if "[Processing:" in line or "Processing:" in line:
                        # Clean up the line
                        clean_line = re.sub(r"\[.*?\]", "", line)
                        clean_line = re.sub(r"\s+", " ", clean_line).strip()
                        if clean_line:
                            print(f"  🔄 {clean_line}")
                    elif "Forensic complete" in line:
                        match = re.search(r"risk=([\d.]+)", line)
                        if match:
                            print(f"  ✅ 포렌식 점수: {match.group(1)}")
                    elif "EXECUTION SUMMARY" in line:
                        print("  🎉 파이프라인 완료!")
                        return True  # Signal completion

        except Exception:
            pass

        return False

    async def monitor(self):
        """Main monitoring loop"""
        self.clear_screen()
        self.display_header()
        self.display_stats()

        print("📝 최신 활동:")

        # Check for completion first
        try:
            with open(self.log_file) as f:
                content = f.read()
                if "EXECUTION SUMMARY" in content and "Execution completed successfully" in content:
                    print("\n  🎉 파이프라인이 이미 완료되었습니다!")
                    completed = content.count("Forensic complete")
                    print(f"  ✅ 완료된 파일: {completed}")
                    return
        except:
            pass

        # Tail new log entries
        completed = self.tail_log()
        if completed:
            return

        # Wait and refresh
        print("\n⏳ 5초 후 새로고침...")

        for i in range(5, 0, -1):
            print(f"   {i}...", end="", flush=True)
            await asyncio.sleep(1)

        print("\r      ")

    async def run(self):
        """Run the monitor"""
        try:
            while True:
                self.clear_screen()
                await self.monitor()

                # Check if process is still running
                proc = self.get_process_status()
                if not proc and self.processed_files > 0:
                    # Process stopped but we had some progress
                    print("\n⚠️  파이프라인 프로세스가 중단되었습니다.")
                    break

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n👋 모니터링을 종료합니다.")
            print(f"\n최종 진행률: {self.processed_files}/{self.total_files}")


async def main():
    """Main entry point"""
    monitor = PipelineMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
