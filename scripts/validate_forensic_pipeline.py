#!/usr/bin/env python3
"""
포렌식 파이프라인 엔드-투-엔드 검증 스크립트

목표: 183개의 전체 오디오 파일에 대해 포렌식 분석 파이프라인 실행 및 검증

프로세스:
1. 오디오 파일 로드 및 전처리
2. 1초 윈도우로 분할
3. GPU F0 추출 (CREPE 모델)
4. 포렌식 특성 추출 (MFCC, Spectral, etc.)
5. 신뢰도 검증
6. 포괄적 보고서 생성

성능 목표:
- 처리 시간: 2분 11초 (전체)
- 유효 F0: 99.0%
- 평균 신뢰도: 0.82
- 에러율: 0%
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from tqdm import tqdm

# 설정
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/forensic_validation.log"),
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


class ForensicPipelineValidator:
    """포렌식 파이프라인 검증 엔진"""

    def __init__(self, audio_dir: Path, use_gpu: bool = True):
        """
        초기화

        Args:
            audio_dir: 오디오 파일 디렉토리
            use_gpu: GPU 사용 여부
        """
        self.audio_dir = Path(audio_dir)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = DEVICE if self.use_gpu else "cpu"

        # 결과 저장소
        self.results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "device": self.device,
                "gpu_available": GPU_AVAILABLE,
            },
            "files": {},
            "statistics": {},
        }

        logger.info(f"포렌식 파이프라인 검증 시작 (Device: {self.device})")
        logger.info(f"오디오 디렉토리: {self.audio_dir}")

    def load_audio_files(self) -> List[Path]:
        """
        오디오 파일 로드

        Returns:
            오디오 파일 경로 리스트
        """
        audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))
        audio_files.sort()
        logger.info(f"발견된 오디오 파일: {len(audio_files)}개")
        return audio_files

    def extract_f0_features(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        F0 특성 추출

        Args:
            audio: 오디오 신호 (numpy array)
            sr: 샘플레이트

        Returns:
            (f0, confidence) 튜플
        """
        try:
            # GPU 백엔드 import (필요시)
            if self.use_gpu:
                try:
                    from src.voice_man.services.forensic.gpu.crepe_extractor import (
                        TorchCrepeExtractor,
                    )

                    extractor = TorchCrepeExtractor(model_name="full")
                    f0, confidence = extractor.extract_f0_batch(audio, sr, hop_length=160)
                except ImportError:
                    logger.warning("GPU 백엔드 사용 불가, CPU fallback 사용")
                    f0, confidence = self._extract_f0_librosa(audio, sr)
            else:
                f0, confidence = self._extract_f0_librosa(audio, sr)

            return f0, confidence
        except Exception as e:
            logger.error(f"F0 추출 실패: {e}")
            return np.array([]), np.array([])

    @staticmethod
    def _extract_f0_librosa(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Librosa를 사용한 F0 추출 (CPU fallback)

        Args:
            audio: 오디오 신호
            sr: 샘플레이트

        Returns:
            (f0, confidence) 튜플
        """
        # PYIN 알고리즘으로 F0 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=550, sr=sr)

        # voiced_probs를 confidence로 사용
        confidence = voiced_probs.astype(np.float32)

        return f0, confidence

    def extract_forensic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        포렌식 특성 추출

        Args:
            audio: 오디오 신호
            sr: 샘플레이트

        Returns:
            포렌식 특성 딕셔너리
        """
        try:
            features = {}

            # MFCC (Mel-Frequency Cepstral Coefficients)
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

            # Spectral Rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features["spectral_rolloff"] = float(np.mean(spec_rolloff))

            return features
        except Exception as e:
            logger.error(f"포렌식 특성 추출 실패: {e}")
            return {}

    def process_audio_file(self, file_path: Path) -> Dict:
        """
        단일 오디오 파일 처리

        Args:
            file_path: 오디오 파일 경로

        Returns:
            처리 결과 딕셔너리
        """
        try:
            # 오디오 로드
            audio, sr = librosa.load(file_path, sr=16000)

            # 1초 윈도우로 분할
            window_size = sr  # 16000 샘플 = 1초
            num_windows = len(audio) // window_size

            f0_values = []
            confidence_values = []
            window_results = []

            # 각 윈도우 처리
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                window_audio = audio[start:end]

                # F0 추출
                f0, confidence = self.extract_f0_features(window_audio, sr)

                if len(f0) > 0:
                    # 중앙 프레임의 F0 값 선택
                    mid_idx = len(f0) // 2
                    f0_val = f0[mid_idx]
                    conf_val = confidence[mid_idx] if mid_idx < len(confidence) else 0.0

                    if not np.isnan(f0_val):
                        f0_values.append(f0_val)
                        confidence_values.append(conf_val)

                window_results.append(
                    {
                        "window_id": i,
                        "f0": float(f0_val) if len(f0) > 0 else None,
                        "confidence": float(conf_val) if len(f0) > 0 else None,
                    }
                )

            # 포렌식 특성 추출
            forensic_features = self.extract_forensic_features(audio, sr)

            # 결과 계산
            valid_f0_count = len(f0_values)
            total_windows = num_windows

            result = {
                "file": file_path.name,
                "duration_seconds": len(audio) / sr,
                "total_windows": total_windows,
                "valid_f0_count": valid_f0_count,
                "valid_f0_rate": valid_f0_count / total_windows if total_windows > 0 else 0,
                "f0_mean": float(np.mean(f0_values)) if f0_values else None,
                "f0_std": float(np.std(f0_values)) if f0_values else None,
                "f0_min": float(np.min(f0_values)) if f0_values else None,
                "f0_max": float(np.max(f0_values)) if f0_values else None,
                "confidence_mean": float(np.mean(confidence_values)) if confidence_values else None,
                "confidence_std": float(np.std(confidence_values)) if confidence_values else None,
                "forensic_features": forensic_features,
                "windows": window_results[:5],  # 처음 5개 윈도우만 저장
            }

            logger.info(
                f"✅ {file_path.name}: "
                f"{valid_f0_count}/{total_windows} 윈도우 (유효율: {result['valid_f0_rate'] * 100:.1f}%)"
            )

            return result

        except Exception as e:
            logger.error(f"❌ 파일 처리 실패: {file_path.name} - {e}")
            return {
                "file": file_path.name,
                "error": str(e),
            }

    def validate_results(self) -> Dict:
        """
        결과 검증

        Returns:
            검증 결과 딕셔너리
        """
        files_processed = len([r for r in self.results["files"].values() if "error" not in r])
        total_windows = sum(
            r.get("total_windows", 0) for r in self.results["files"].values() if "error" not in r
        )
        valid_f0_total = sum(
            r.get("valid_f0_count", 0) for r in self.results["files"].values() if "error" not in r
        )

        valid_f0_rate = valid_f0_total / total_windows if total_windows > 0 else 0

        # 신뢰도 통계
        confidence_values = []
        for result in self.results["files"].values():
            if "error" not in result and result.get("confidence_mean"):
                confidence_values.append(result["confidence_mean"])

        validation = {
            "files_processed": files_processed,
            "total_windows": total_windows,
            "valid_f0_total": valid_f0_total,
            "valid_f0_rate": valid_f0_rate,
            "avg_confidence": (float(np.mean(confidence_values)) if confidence_values else None),
            "status": "✅ PASS" if valid_f0_rate >= 0.99 else "⚠️ WARNING",
            "target_valid_f0_rate": 0.99,
            "target_met": valid_f0_rate >= 0.99,
        }

        return validation

    def run(self) -> Dict:
        """
        포렌식 파이프라인 전체 실행

        Returns:
            최종 결과 딕셔너리
        """
        start_time = time.time()

        # 오디오 파일 로드
        audio_files = self.load_audio_files()

        # 각 파일 처리
        for file_path in tqdm(audio_files, desc="포렌식 분석 중"):
            result = self.process_audio_file(file_path)
            self.results["files"][file_path.name] = result

        # 결과 검증
        validation = self.validate_results()
        self.results["statistics"] = validation

        # 실행 시간 기록
        elapsed_time = time.time() - start_time
        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        self.results["metadata"]["elapsed_seconds"] = elapsed_time
        self.results["metadata"]["elapsed_formatted"] = self._format_elapsed(elapsed_time)

        logger.info("\n" + "=" * 60)
        logger.info("📊 포렌식 파이프라인 검증 완료")
        logger.info("=" * 60)
        logger.info(f"처리 파일: {validation['files_processed']}개")
        logger.info(f"총 윈도우: {validation['total_windows']}개")
        logger.info(
            f"유효 F0: {validation['valid_f0_total']}개 ({validation['valid_f0_rate'] * 100:.1f}%)"
        )
        logger.info(f"평균 신뢰도: {validation['avg_confidence']:.2f}")
        logger.info(f"처리 시간: {self.results['metadata']['elapsed_formatted']}")
        logger.info(f"상태: {validation['status']}")
        logger.info("=" * 60)

        return self.results

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """경과 시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}분"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}시간"

    def save_report(self, output_path: Path) -> None:
        """
        결과 보고서 저장

        Args:
            output_path: 보고서 저장 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"보고서 저장: {output_path}")


def main():
    """메인 실행 함수"""
    # 오디오 디렉토리 설정
    audio_dir = Path("/home/innojini/call_data/voice_segments")

    # 검증 실행
    validator = ForensicPipelineValidator(audio_dir, use_gpu=GPU_AVAILABLE)
    results = validator.run()

    # 보고서 저장
    report_path = Path("reports/forensic_validation_phase_11.json")
    validator.save_report(report_path)

    # 최종 상태 확인
    validation = results["statistics"]
    if validation["target_met"]:
        logger.info("✅ 모든 목표 달성!")
        return 0
    else:
        logger.warning("⚠️ 일부 목표 미달 - 자세한 결과는 보고서 참조")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        sys.exit(1)
