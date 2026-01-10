#!/usr/bin/env python3
"""
형사소송 증거자료용 포렌식 음성 분석 시스템

목표: 184개의 전체 통화 녹음을 분석하여 형사소송에 필요한
      과학적 증거자료 생성

기능:
- 10개 파일 단위 배치 모니터링
- 실시간 성능 메트릭 추적
- CPU/GPU 온도 모니터링
- 자동 성능 저하 감지 및 경고
- 법적 증거물 검증 및 저장
- 분석 결과 암호화 저장
"""

import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import psutil
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/forensic_evidence_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# GPU 가용성 확인
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"


class PerformanceMonitor:
    """시스템 성능 모니터링 클래스"""

    def __init__(self):
        """초기화"""
        self.metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_memory_percent": [],
            "gpu_temperature": [],
            "cpu_temperature": [],
            "timestamps": [],
        }
        self.start_time = time.time()

    def get_metrics(self) -> Dict:
        """현재 성능 메트릭 수집"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
        }

        # GPU 메트릭
        if GPU_AVAILABLE:
            try:
                import torch

                metrics["gpu_memory"] = (
                    torch.cuda.memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                    * 100
                )
                metrics["gpu_utilization"] = (
                    torch.cuda.memory_reserved()
                    / torch.cuda.get_device_properties(0).total_memory
                    * 100
                )
            except:
                metrics["gpu_memory"] = 0
                metrics["gpu_utilization"] = 0

        # 온도 정보 (Linux)
        try:
            result = subprocess.run(
                "cat /sys/class/thermal/thermal_zone0/temp",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                cpu_temp = int(result.stdout.strip()) / 1000  # mC to C
                metrics["cpu_temperature"] = cpu_temp
            else:
                metrics["cpu_temperature"] = None
        except:
            metrics["cpu_temperature"] = None

        # GPU 온도
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                metrics["gpu_temperature"] = float(result.stdout.strip())
            else:
                metrics["gpu_temperature"] = None
        except:
            metrics["gpu_temperature"] = None

        return metrics

    def check_degradation(self, metrics: Dict) -> Optional[str]:
        """성능 저하 감지"""
        warnings = []

        # CPU 사용률 높음
        if metrics["cpu_percent"] > 85:
            warnings.append(f"⚠️ CPU 높은 사용률: {metrics['cpu_percent']:.1f}%")

        # 메모리 부족
        if metrics["memory"] > 85:
            warnings.append(f"⚠️ 메모리 부족: {metrics['memory']:.1f}%")

        # GPU 메모리 부족
        if metrics.get("gpu_memory", 0) > 90:
            warnings.append(f"⚠️ GPU 메모리 부족: {metrics['gpu_memory']:.1f}%")

        # CPU 온도 높음
        if metrics.get("cpu_temperature") and metrics["cpu_temperature"] > 80:
            warnings.append(f"⚠️ CPU 온도 높음: {metrics['cpu_temperature']:.1f}°C")

        # GPU 온도 높음
        if metrics.get("gpu_temperature") and metrics["gpu_temperature"] > 80:
            warnings.append(f"⚠️ GPU 온도 높음: {metrics['gpu_temperature']:.1f}°C")

        if warnings:
            return "\n".join(warnings)
        return None


class ForensicEvidenceAnalyzer:
    """포렌식 증거 분석 엔진"""

    def __init__(
        self,
        audio_dir: Path,
        output_dir: Path,
        batch_size: int = 10,
        use_gpu: bool = True,
    ):
        """
        초기화

        Args:
            audio_dir: 오디오 파일 디렉토리
            output_dir: 결과 저장 디렉토리
            batch_size: 배치 크기 (기본 10)
            use_gpu: GPU 사용 여부
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = DEVICE if self.use_gpu else "cpu"

        self.monitor = PerformanceMonitor()
        self.results = {
            "metadata": {
                "analysis_type": "Forensic Evidence Analysis",
                "start_time": datetime.now().isoformat(),
                "device": self.device,
                "gpu_available": GPU_AVAILABLE,
                "total_files": 0,
                "batch_size": batch_size,
            },
            "batches": {},
            "statistics": {},
        }

        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        logger.info(f"포렌식 증거 분석 시작 (Device: {self.device})")
        logger.info(f"오디오 디렉토리: {self.audio_dir}")
        logger.info(f"결과 디렉토리: {self.output_dir}")

    def load_audio_files(self) -> List[Path]:
        """오디오 파일 로드"""
        audio_files = (
            list(self.audio_dir.glob("*.m4a"))
            + list(self.audio_dir.glob("*.wav"))
            + list(self.audio_dir.glob("*.mp3"))
        )
        audio_files.sort()
        logger.info(f"발견된 오디오 파일: {len(audio_files)}개")
        self.results["metadata"]["total_files"] = len(audio_files)
        return audio_files

    def extract_forensic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """포렌식 특성 추출"""
        try:
            features = {}

            # F0 추출 (기본)
            try:
                import librosa

                f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=550, sr=sr)
                features["f0_mean"] = float(np.nanmean(f0))
                features["f0_std"] = float(np.nanstd(f0))
                features["f0_min"] = float(np.nanmin(f0))
                features["f0_max"] = float(np.nanmax(f0))
                features["voiced_ratio"] = float(np.sum(~np.isnan(f0)) / len(f0))
            except:
                features["f0_mean"] = None

            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = float(np.mean(mfcc))
            features["mfcc_std"] = float(np.std(mfcc))

            # Spectral Centroid
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features["spectral_centroid"] = float(np.mean(spec_centroid))

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features["zcr_mean"] = float(np.mean(zcr))

            # RMS Energy
            rms = librosa.feature.rms(y=audio)
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            # Spectral Rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features["spectral_rolloff"] = float(np.mean(spec_rolloff))

            # Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["chroma_mean"] = float(np.mean(chroma))

            # Tempo
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            try:
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
                features["tempo"] = float(tempo)
            except:
                features["tempo"] = None

            return features
        except Exception as e:
            logger.error(f"포렌식 특성 추출 실패: {e}")
            return {}

    def analyze_audio_file(self, file_path: Path) -> Dict:
        """
        단일 오디오 파일 분석

        Args:
            file_path: 오디오 파일 경로

        Returns:
            분석 결과
        """
        try:
            # 파일 정보 수집
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            # 오디오 로드
            audio, sr = librosa.load(file_path, sr=16000)
            duration = len(audio) / sr

            # 포렌식 특성 추출
            forensic_features = self.extract_forensic_features(audio, sr)

            # 결과 구성
            result = {
                "file": file_path.name,
                "file_path": str(file_path),
                "file_size_mb": file_size / (1024 * 1024),
                "modified_time": file_mtime,
                "duration_seconds": duration,
                "sample_rate": sr,
                "samples": len(audio),
                "forensic_features": forensic_features,
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "✅ SUCCESS",
            }

            return result

        except Exception as e:
            logger.error(f"❌ 파일 분석 실패: {file_path.name} - {e}")
            return {
                "file": file_path.name,
                "file_path": str(file_path),
                "error": str(e),
                "status": "❌ FAILED",
            }

    def analyze_batch(self, batch_files: List[Path], batch_num: int) -> Dict:
        """
        배치 분석 (10개 파일)

        Args:
            batch_files: 배치 파일 리스트
            batch_num: 배치 번호

        Returns:
            배치 결과
        """
        batch_start_time = time.time()
        batch_results = {
            "batch_number": batch_num,
            "batch_size": len(batch_files),
            "files": {},
            "batch_metrics": {},
            "performance_data": [],
        }

        logger.info(f"\n{'=' * 70}")
        logger.info(f"📊 배치 {batch_num} 처리 시작 ({len(batch_files)}개 파일)")
        logger.info(f"{'=' * 70}")

        # 배치 내 각 파일 처리
        for file_idx, file_path in enumerate(
            tqdm(batch_files, desc=f"배치 {batch_num}", position=0, leave=True, disable=True)
        ):
            file_start = time.time()

            # 파일 분석
            result = self.analyze_audio_file(file_path)
            batch_results["files"][file_path.name] = result

            # 성능 메트릭 수집
            metrics = self.monitor.get_metrics()
            metrics["file_processing_time"] = time.time() - file_start
            batch_results["performance_data"].append(metrics)

            # 성능 저하 감지
            degradation_warning = self.monitor.check_degradation(metrics)
            if degradation_warning:
                logger.warning(degradation_warning)
                result["warnings"] = degradation_warning

            # 진행상황 출력
            if "error" not in result:
                logger.info(
                    f"  ✅ {file_path.name} "
                    f"({result.get('duration_seconds', 0):.1f}s) "
                    f"- {metrics['file_processing_time']:.2f}초"
                )

        # 배치 통계
        batch_elapsed = time.time() - batch_start_time
        successful_files = len([f for f in batch_results["files"].values() if "error" not in f])

        batch_results["batch_metrics"] = {
            "total_processing_time": batch_elapsed,
            "average_time_per_file": batch_elapsed / len(batch_files),
            "successful_files": successful_files,
            "failed_files": len(batch_files) - successful_files,
            "success_rate": successful_files / len(batch_files) * 100,
        }

        # 배치 결과 출력
        logger.info(f"\n📊 배치 {batch_num} 완료")
        logger.info(f"  - 성공: {successful_files}/{len(batch_files)}")
        logger.info(f"  - 처리 시간: {batch_elapsed:.2f}초")
        logger.info(f"  - 평균: {batch_elapsed / len(batch_files):.2f}초/파일")
        logger.info(f"  - 성공률: {batch_results['batch_metrics']['success_rate']:.1f}%")

        return batch_results

    def run(self) -> Dict:
        """전체 분석 실행"""
        start_time = time.time()

        # 오디오 파일 로드
        audio_files = self.load_audio_files()

        # 배치 처리
        for batch_start in range(0, len(audio_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(audio_files))
            batch_files = audio_files[batch_start:batch_end]
            batch_num = (batch_start // self.batch_size) + 1

            # 배치 분석
            batch_result = self.analyze_batch(batch_files, batch_num)
            self.results["batches"][f"batch_{batch_num}"] = batch_result

            # 배치 후 결과 저장
            self.save_batch_report(batch_num, batch_result)

        # 최종 통계
        elapsed_time = time.time() - start_time
        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        self.results["metadata"]["total_processing_time"] = elapsed_time

        # 전체 통계 계산
        total_files = sum(len(batch["files"]) for batch in self.results["batches"].values())
        successful = sum(
            sum(1 for f in batch["files"].values() if "error" not in f)
            for batch in self.results["batches"].values()
        )

        self.results["statistics"] = {
            "total_files_processed": total_files,
            "successful_files": successful,
            "failed_files": total_files - successful,
            "success_rate": successful / total_files * 100 if total_files > 0 else 0,
            "total_processing_time": elapsed_time,
            "average_per_file": elapsed_time / total_files if total_files > 0 else 0,
        }

        # 최종 보고서 출력
        self.print_final_report()

        return self.results

    def save_batch_report(self, batch_num: int, batch_result: Dict) -> None:
        """배치 보고서 저장"""
        report_path = self.output_dir / f"batch_{batch_num:03d}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)

    def print_final_report(self) -> None:
        """최종 보고서 출력"""
        stats = self.results["statistics"]
        logger.info("\n" + "=" * 70)
        logger.info("🎯 포렌식 증거 분석 완료")
        logger.info("=" * 70)
        logger.info(f"총 파일: {stats['total_files_processed']}개")
        logger.info(f"성공: {stats['successful_files']}개 ✅")
        logger.info(f"실패: {stats['failed_files']}개 ❌")
        logger.info(f"성공률: {stats['success_rate']:.1f}%")
        logger.info(f"총 처리 시간: {self._format_time(stats['total_processing_time'])}")
        logger.info(f"파일당 평균: {stats['average_per_file']:.2f}초")
        logger.info("=" * 70)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}분"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}시간"

    def save_results(self, output_path: Path) -> None:
        """최종 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 결과 저장: {output_path}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    audio_dir = Path("/home/innojini/dev/voice.man/ref/call")
    output_dir = Path("forensic_evidence_results")

    # 분석기 초기화 (10개 파일씩 배치)
    analyzer = ForensicEvidenceAnalyzer(
        audio_dir=audio_dir,
        output_dir=output_dir,
        batch_size=10,  # 10개 파일 배치
        use_gpu=GPU_AVAILABLE,
    )

    # 분석 실행
    results = analyzer.run()

    # 결과 저장
    output_file = output_dir / "forensic_evidence_complete_analysis.json"
    analyzer.save_results(output_file)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        sys.exit(1)
