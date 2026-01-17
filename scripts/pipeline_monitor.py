#!/usr/bin/env python3
"""
Forensic Pipeline Resource Monitor
===================================

Monitors GPU temperature, memory usage, and system resources during pipeline execution.
Provides auto-pause/resume capability to prevent system crashes.

Features:
- GPU thermal monitoring (nvidia-smi)
- GPU memory tracking
- System RAM monitoring
- Automatic pause on threshold exceedance
- Progress tracking with resource awareness
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline_monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during pipeline execution"""

    # Thresholds for auto-pause
    GPU_TEMP_WARNING = 75  # Celsius
    GPU_TEMP_CRITICAL = 85  # Celsius
    GPU_MEM_WARNING = 90  # Percent
    GPU_MEM_CRITICAL = 95  # Percent
    RAM_WARNING = 85  # Percent
    RAM_CRITICAL = 90  # Percent

    def __init__(self, check_interval: int = 30):
        """
        Initialize resource monitor

        Args:
            check_interval: Seconds between checks (default: 30)
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.metrics_history: List[Dict] = []
        self.pause_requested = False

        # Create logs directory
        Path("logs").mkdir(parents=True, exist_ok=True)

    def get_gpu_metrics(self) -> Optional[Dict]:
        """
        Get GPU metrics using nvidia-smi

        Returns:
            Dict with GPU metrics or None if unavailable
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(", ")
            if len(parts) < 5:
                return None

            name = parts[0]
            temp = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
            utilization = float(parts[4])

            return {
                "name": name,
                "temperature_c": temp,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": (mem_used / mem_total * 100) if mem_total > 0 else 0,
                "utilization_percent": utilization,
            }

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return None

    def get_ram_metrics(self) -> Dict:
        """
        Get system RAM metrics

        Returns:
            Dict with RAM metrics
        """
        try:
            result = subprocess.run(
                ["free", "-m"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return {"total_mb": 0, "used_mb": 0, "percent": 0}

            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return {"total_mb": 0, "used_mb": 0, "percent": 0}

            parts = lines[1].split()
            if len(parts) < 3:
                return {"total_mb": 0, "used_mb": 0, "percent": 0}

            total = int(parts[1])
            used = int(parts[2])

            return {
                "total_mb": total,
                "used_mb": used,
                "percent": (used / total * 100) if total > 0 else 0,
            }

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to get RAM metrics: {e}")
            return {"total_mb": 0, "used_mb": 0, "percent": 0}

    def get_pipeline_progress(self) -> Dict:
        """
        Get pipeline progress from checkpoint files

        Returns:
            Dict with progress info
        """
        checkpoint_dir = Path("data/checkpoints")
        if not checkpoint_dir.exists():
            return {"processed": 0, "total": 195, "percent": 0}

        # Count completed assets
        completed = 0
        for checkpoint_file in checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    if data.get("status") == "completed":
                        completed += 1
            except (json.JSONDecodeError, IOError, OSError):
                pass

        return {
            "processed": completed,
            "total": 195,  # Will be updated dynamically
            "percent": (completed / 195 * 100) if 195 > 0 else 0,
        }

    def check_thresholds(self, gpu_metrics: Optional[Dict], ram_metrics: Dict) -> Dict:
        """
        Check if any thresholds are exceeded

        Args:
            gpu_metrics: GPU metrics dict
            ram_metrics: RAM metrics dict

        Returns:
            Dict with threshold status
        """
        status = {
            "gpu_temp": "normal",
            "gpu_mem": "normal",
            "ram": "normal",
            "should_pause": False,
            "warnings": [],
        }

        if gpu_metrics:
            temp = gpu_metrics["temperature_c"]
            mem_percent = gpu_metrics["memory_percent"]

            if temp >= self.GPU_TEMP_CRITICAL:
                status["gpu_temp"] = "critical"
                status["should_pause"] = True
                status["warnings"].append(f"GPU temperature critical: {temp}°C")
            elif temp >= self.GPU_TEMP_WARNING:
                status["gpu_temp"] = "warning"
                status["warnings"].append(f"GPU temperature high: {temp}°C")

            if mem_percent >= self.GPU_MEM_CRITICAL:
                status["gpu_mem"] = "critical"
                status["should_pause"] = True
                status["warnings"].append(f"GPU memory critical: {mem_percent:.1f}%")
            elif mem_percent >= self.GPU_MEM_WARNING:
                status["gpu_mem"] = "warning"
                status["warnings"].append(f"GPU memory high: {mem_percent:.1f}%")

        ram_percent = ram_metrics["percent"]
        if ram_percent >= self.RAM_CRITICAL:
            status["ram"] = "critical"
            status["should_pause"] = True
            status["warnings"].append(f"RAM critical: {ram_percent:.1f}%")
        elif ram_percent >= self.RAM_WARNING:
            status["ram"] = "warning"
            status["warnings"].append(f"RAM high: {ram_percent:.1f}%")

        return status

    async def monitor_once(self) -> Dict:
        """
        Perform a single monitoring check

        Returns:
            Dict with all metrics and status
        """
        timestamp = datetime.now().isoformat()

        gpu_metrics = self.get_gpu_metrics()
        ram_metrics = self.get_ram_metrics()
        progress = self.get_pipeline_progress()
        threshold_status = self.check_thresholds(gpu_metrics, ram_metrics)

        metrics = {
            "timestamp": timestamp,
            "gpu": gpu_metrics,
            "ram": ram_metrics,
            "progress": progress,
            "thresholds": threshold_status,
        }

        self.metrics_history.append(metrics)

        # Log status
        gpu_info = ""
        if gpu_metrics:
            gpu_info = (
                f" GPU: {gpu_metrics['temperature_c']}°C, "
                f"MEM: {gpu_metrics['memory_percent']:.1f}%, "
                f"UTIL: {gpu_metrics['utilization_percent']:.0f}%"
            )

        logger.info(
            f"Progress: {progress['processed']}/{progress['total']} ({progress['percent']:.1f}%)"
            f" |{gpu_info}"
            f" | RAM: {ram_metrics['percent']:.1f}%"
        )

        # Log warnings
        for warning in threshold_status["warnings"]:
            logger.warning(f"⚠️  {warning}")

        # Critical threshold handling
        if threshold_status["should_pause"]:
            logger.error("🛑 CRITICAL: Thresholds exceeded - PAUSE REQUESTED")
            self.pause_requested = True

        return metrics

    async def start_monitoring(self, duration_seconds: Optional[int] = None):
        """
        Start continuous monitoring

        Args:
            duration_seconds: Total duration to monitor (None = infinite)
        """
        self.monitoring = True
        self.pause_requested = False
        start_time = time.time()

        logger.info(f"Starting resource monitor (interval: {self.check_interval}s)")
        logger.info(f"Thresholds: GPU>{self.GPU_TEMP_WARNING}°C, RAM>{self.RAM_WARNING}%")

        try:
            while self.monitoring:
                # Check if duration exceeded
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info("Monitoring duration reached")
                    break

                # Perform monitoring check
                await self.monitor_once()

                # Check if pause was requested
                if self.pause_requested:
                    logger.warning("Pause requested - stopping monitoring")
                    break

                # Wait for next interval
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        finally:
            self.monitoring = False
            self.save_metrics()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Monitoring stopped")

    def save_metrics(self):
        """Save metrics history to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path(f"logs/monitoring_metrics_{timestamp}.json")

        try:
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def print_summary(self):
        """Print monitoring summary"""
        if not self.metrics_history:
            logger.info("No metrics collected")
            return

        gpu_temps = [m["gpu"]["temperature_c"] for m in self.metrics_history if m.get("gpu")]
        ram_usage = [m["ram"]["percent"] for m in self.metrics_history]

        print("\n" + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        print(f"Total checks: {len(self.metrics_history)}")

        if gpu_temps:
            print(f"GPU Temp: {min(gpu_temps):.0f} - {max(gpu_temps):.0f}°C")
            print(f"GPU Temp Avg: {sum(gpu_temps) / len(gpu_temps):.1f}°C")

        if ram_usage:
            print(f"RAM Usage: {min(ram_usage):.1f} - {max(ram_usage):.1f}%")

        progress = self.metrics_history[-1]["progress"]
        print(f"Final Progress: {progress['processed']}/{progress['total']} files")

        if self.pause_requested:
            print("⚠️  PAUSE REQUESTED due to threshold exceedance")

        print("=" * 60 + "\n")


async def main():
    """Main entry point for standalone monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor pipeline resources")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Total duration in seconds")
    args = parser.parse_args()

    monitor = ResourceMonitor(check_interval=args.interval)

    try:
        await monitor.start_monitoring(duration_seconds=args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        monitor.stop_monitoring()
        monitor.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
