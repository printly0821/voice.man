#!/usr/bin/env python3
"""
포렌식 증거 분석 실시간 모니터링 대시보드

기능:
- 10개 파일 배치별 진행 상황 표시
- 성능 메트릭 실시간 추적
- CPU/GPU 온도 모니터링
- 성능 저하 자동 감지
- 예상 완료 시간 계산
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import psutil


class ForensicMonitoringDashboard:
    """포렌식 모니터링 대시보드"""

    def __init__(self, log_file: Path = Path("logs/forensic_evidence_analysis.log")):
        """초기화"""
        self.log_file = log_file
        self.batch_stats = {}
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = time.time()

    def get_system_metrics(self) -> Dict:
        """시스템 메트릭 수집"""
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }

        # GPU 메트릭
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout:
                parts = result.stdout.strip().split(",")
                metrics["gpu_memory_used"] = float(parts[0])
                metrics["gpu_memory_total"] = float(parts[1])
                metrics["gpu_temp"] = float(parts[2])
                metrics["gpu_memory_percent"] = (
                    metrics["gpu_memory_used"] / metrics["gpu_memory_total"] * 100
                )
            else:
                metrics["gpu_available"] = False
        except Exception:
            metrics["gpu_available"] = False

        # CPU 온도
        try:
            result = subprocess.run(
                "cat /sys/class/thermal/thermal_zone0/temp",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout:
                metrics["cpu_temp"] = int(result.stdout.strip()) / 1000
            else:
                metrics["cpu_temp"] = None
        except Exception:
            metrics["cpu_temp"] = None

        return metrics

    def print_header(self) -> None:
        """헤더 출력"""
        os.system("clear" if os.name == "posix" else "cls")
        print("\n" + "=" * 100)
        print("🔍 포렌식 증거 분석 실시간 모니터링 대시보드".center(100))
        print("=" * 100)

    def print_batch_progress(self) -> None:
        """배치 진행 상황 출력"""
        print("\n📊 배치 진행 상황")
        print("-" * 100)

        # 배치별 통계
        if self.batch_stats:
            for batch_num, stats in sorted(self.batch_stats.items()):
                status = "✅" if stats["completed"] else "🔄"
                progress = (
                    f"{stats['processed']}/{stats['total']}"
                    if not stats["completed"]
                    else f"{stats['total']}/{stats['total']}"
                )

                print(
                    f"{status} 배치 {batch_num:2d}: "
                    f"{progress:>5s} 파일 | "
                    f"시간: {stats['elapsed']:.1f}초 | "
                    f"평균: {stats['avg_per_file']:.2f}초/파일 | "
                    f"성공률: {stats['success_rate']:.0f}%"
                )

    def print_system_status(self, metrics: Dict) -> None:
        """시스템 상태 출력"""
        print("\n💻 시스템 상태")
        print("-" * 100)

        # CPU
        cpu_status = "✅" if metrics["cpu_percent"] < 80 else "⚠️"
        print(
            f"{cpu_status} CPU: {metrics['cpu_percent']:>5.1f}% | "
            f"메모리: {metrics['memory_percent']:>5.1f}% | "
            f"디스크: {metrics['disk_percent']:>5.1f}%"
        )

        # GPU
        if metrics.get("gpu_available", True):
            gpu_status = "✅" if metrics.get("gpu_memory_percent", 0) < 90 else "⚠️"
            gpu_temp_status = "✅" if metrics.get("gpu_temp", 0) < 80 else "⚠️"
            print(
                f"{gpu_status} GPU 메모리: {metrics.get('gpu_memory_percent', 0):>5.1f}% | "
                f"{gpu_temp_status} GPU 온도: {metrics.get('gpu_temp', 0):>5.1f}°C"
            )

        # 온도
        if metrics.get("cpu_temp"):
            cpu_temp_status = "✅" if metrics["cpu_temp"] < 80 else "⚠️"
            print(f"{cpu_temp_status} CPU 온도: {metrics['cpu_temp']:>5.1f}°C")

    def print_performance_alerts(self, metrics: Dict) -> None:
        """성능 경고 출력"""
        print("\n⚠️ 성능 경고")
        print("-" * 100)

        alerts = []

        if metrics["cpu_percent"] > 85:
            alerts.append(f"🔴 CPU 높은 사용률: {metrics['cpu_percent']:.1f}%")

        if metrics["memory_percent"] > 85:
            alerts.append(f"🔴 메모리 부족: {metrics['memory_percent']:.1f}%")

        if metrics.get("gpu_memory_percent", 0) > 90:
            alerts.append(f"🔴 GPU 메모리 부족: {metrics['gpu_memory_percent']:.1f}%")

        if metrics.get("cpu_temp") and metrics["cpu_temp"] > 80:
            alerts.append(f"🟠 CPU 온도 높음: {metrics['cpu_temp']:.1f}°C")

        if metrics.get("gpu_temp") and metrics["gpu_temp"] > 80:
            alerts.append(f"🟠 GPU 온도 높음: {metrics['gpu_temp']:.1f}°C")

        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("  ✅ 모든 시스템 정상")

    def print_timeline(self) -> None:
        """타임라인 출력"""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)

        print("\n⏱️ 처리 시간")
        print("-" * 100)
        print(f"경과 시간: {minutes:02d}:{seconds:02d}")

        if self.batch_stats:
            total_batches = len(self.batch_stats)
            completed = sum(1 for b in self.batch_stats.values() if b["completed"])

            if completed > 0 and completed < total_batches:
                avg_batch_time = sum(b["elapsed"] for b in self.batch_stats.values()) / completed
                remaining_batches = total_batches - completed
                estimated_remaining = avg_batch_time * remaining_batches
                estimated_total = elapsed + estimated_remaining

                est_hours = int(estimated_remaining / 3600)
                est_minutes = int((estimated_remaining % 3600) / 60)
                est_seconds = int(estimated_remaining % 60)

                print(f"예상 남은 시간: {est_hours}시간 {est_minutes}분 {est_seconds}초")
                print(
                    f"예상 총 시간: {int(estimated_total / 3600)}시간 "
                    f"{int((estimated_total % 3600) / 60)}분"
                )

    def print_recommendations(self, metrics: Dict) -> None:
        """권장 사항 출력"""
        print("\n💡 권장 사항")
        print("-" * 100)

        recommendations = []

        if metrics["cpu_percent"] > 75:
            recommendations.append("• CPU 사용률이 높습니다. 다른 작업 종료를 고려하세요.")

        if metrics["memory_percent"] > 75:
            recommendations.append("• 메모리 부족 상태입니다. 배치 크기 축소를 고려하세요.")

        if metrics.get("gpu_memory_percent", 0) > 75:
            recommendations.append("• GPU 메모리 부족 상태입니다. 배치 크기 축소를 고려하세요.")

        if metrics.get("cpu_temp") and metrics["cpu_temp"] > 70:
            recommendations.append("• CPU 온도가 높습니다. 냉각을 강화하세요.")

        if metrics.get("gpu_temp") and metrics["gpu_temp"] > 70:
            recommendations.append("• GPU 온도가 높습니다. 냉각을 강화하세요.")

        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("  ✅ 시스템이 최적 상태입니다.")

    def display_dashboard(self) -> None:
        """대시보드 표시"""
        while True:
            try:
                self.print_header()

                metrics = self.get_system_metrics()

                self.print_batch_progress()
                self.print_system_status(metrics)
                self.print_performance_alerts(metrics)
                self.print_timeline()
                self.print_recommendations(metrics)

                print("\n" + "=" * 100)
                print(
                    f"마지막 업데이트: {metrics['timestamp']} | "
                    "입력: Q(종료), P(일시정지), R(재개)".center(100)
                )
                print("=" * 100 + "\n")

                time.sleep(5)  # 5초마다 업데이트

            except KeyboardInterrupt:
                print("\n대시보드 종료")
                break
            except Exception as e:
                print(f"대시보드 오류: {e}")
                time.sleep(5)

    def update_batch_stats(self, batch_num: int, stats: Dict) -> None:
        """배치 통계 업데이트"""
        self.batch_stats[batch_num] = stats


def start_monitoring_dashboard() -> None:
    """모니터링 대시보드 시작"""
    dashboard = ForensicMonitoringDashboard()
    dashboard.display_dashboard()


if __name__ == "__main__":
    start_monitoring_dashboard()
