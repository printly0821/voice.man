#!/usr/bin/env python3
"""
포렌식 파이프라인 모니터링 스크립트

오디오 파일을 포렌식 파이프라인에 통과시키고 10개 단위로 결과를 표시합니다.
STT 정확도(신뢰도 점수, WER/CER)와 전사 통계를 포함합니다.

사용법:
    python scripts/forensic_monitor.py --input-dir ref/call
"""

# ============================================================================
# PyTorch 2.6+ 호환성 패치
# ============================================================================
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, weights_only=None, **kwargs):
    """호환성을 위한 torch.load 패치"""
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)


torch.load = _patched_torch_load
# ============================================================================

import argparse
import asyncio
import gc
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from voice_man.services.whisperx_service import WhisperXService
from voice_man.services.forensic.audio_feature_service import AudioFeatureService
from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
from voice_man.services.forensic.ser_service import SERService
from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
from voice_man.services.forensic.cross_validation_service import CrossValidationService
from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService
from voice_man.services.stt_accuracy_service import (
    STTAccuracyService,
    STTAccuracyResult,
)

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SharedModelPool:
    """GPU 메모리 효율을 위한 스레드 안전한 공유 모델 풀"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._whisper_service = None
            self._forensic_services = None
            self._stt_accuracy_service = None
            self._service_lock = threading.Lock()
            self._whisper_lock = threading.Lock()
            self._forensic_lock = threading.Lock()
            self._initialized = True

    def get_whisper_service(
        self, model_size: str = "large-v3", device: str = "cuda", language: str = "ko"
    ) -> WhisperXService:
        """공유 WhisperX 서비스 가져오기"""
        with self._whisper_lock:
            if self._whisper_service is None:
                logger.info(f"WhisperX 서비스 초기화 중: {model_size}")
                self._whisper_service = WhisperXService(
                    model_size=model_size,
                    device=device,
                    language=language,
                    compute_type="float16",
                )
                logger.info("WhisperX 서비스 초기화 완료")
            return self._whisper_service

    def get_forensic_services(self) -> Dict[str, Any]:
        """공유 포렌식 서비스 가져오기"""
        with self._forensic_lock:
            if self._forensic_services is None:
                logger.info("포렌식 서비스 초기화 중")

                audio_feature_service = AudioFeatureService()
                stress_analysis_service = StressAnalysisService()
                ser_service = SERService()
                crime_language_service = CrimeLanguageAnalysisService()
                cross_validation_service = CrossValidationService(
                    crime_language_service=crime_language_service,
                    ser_service=ser_service,
                )
                forensic_scoring_service = ForensicScoringService(
                    audio_feature_service=audio_feature_service,
                    stress_analysis_service=stress_analysis_service,
                    crime_language_service=crime_language_service,
                    ser_service=ser_service,
                    cross_validation_service=cross_validation_service,
                )

                self._forensic_services = {
                    "audio_feature": audio_feature_service,
                    "stress": stress_analysis_service,
                    "ser": ser_service,
                    "crime": crime_language_service,
                    "cross_validation": cross_validation_service,
                    "scoring": forensic_scoring_service,
                }
                logger.info("포렌식 서비스 초기화 완료")
            return self._forensic_services

    def get_stt_accuracy_service(self) -> STTAccuracyService:
        """STT 정확도 서비스 가져오기"""
        with self._forensic_lock:
            if self._stt_accuracy_service is None:
                self._stt_accuracy_service = STTAccuracyService()
            return self._stt_accuracy_service

    def cleanup_all(self):
        """모든 서비스 정리"""
        logger.info("모든 서비스 정리 중...")
        if self._whisper_service is not None:
            try:
                if hasattr(self._whisper_service, "unload"):
                    self._whisper_service.unload()
                del self._whisper_service
                self._whisper_service = None
            except Exception as e:
                logger.warning(f"Whisper 정리 중 오류: {e}")

        if self._forensic_services is not None:
            for name, service in self._forensic_services.items():
                try:
                    if hasattr(service, "unload"):
                        service.unload()
                    del service
                except Exception as e:
                    logger.warning(f"{name} 정리 중 오류: {e}")
            self._forensic_services = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("모든 서비스 정리 완료")


async def process_single_file(
    audio_path: Path,
    model_pool: SharedModelPool,
    model_size: str,
) -> Optional[Dict[str, Any]]:
    """
    단일 오디오 파일을 포렌식 분석으로 처리

    Args:
        audio_path: 오디오 파일 경로 (Path 객체)
        model_pool: 공유 모델 풀
        model_size: Whisper 모델 크기

    Returns:
        분석 결과 딕셔너리 또는 실패 시 None
    """
    audio_name = audio_path.name
    audio_path_str = str(audio_path.resolve())  # 절대 경로로 변환

    try:
        logger.info(f"처리 중: {audio_name}")

        # 서비스 가져오기
        whisper_service = model_pool.get_whisper_service(model_size=model_size)
        forensic_services = model_pool.get_forensic_services()
        stt_accuracy_service = model_pool.get_stt_accuracy_service()

        # 전사 실행
        logger.info(f"[{audio_name}] 전사 계산 중...")
        transcript_result = await whisper_service.process_audio(audio_path_str)

        # GPU 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[{audio_name}] 전사 완료")

        # STT 정확도 분석
        logger.info(f"[{audio_name}] STT 정확도 분석 중...")
        stt_accuracy = stt_accuracy_service.analyze_from_pipeline_result(
            file_path=str(audio_path),
            pipeline_result=transcript_result,
            reference_text=None,  # 참조 텍스트가 있는 경우 여기에 전달
        )

        # 포렌식 분석 실행
        logger.info(f"[{audio_name}] 포렌식 분석 실행 중...")
        forensic_score = await forensic_services["scoring"].analyze(
            audio_path=str(audio_path),
            transcript=transcript_result.text,
        )

        # 카테고리 점수 추출
        category_scores = {cs.category: cs for cs in forensic_score.category_scores}

        # 결과 준비 (STT 정확도 포함)
        result = {
            "file_name": audio_name,
            "file_path": str(audio_path),
            "duration_seconds": forensic_score.audio_duration_seconds,
            "transcript_text": transcript_result.text,
            "num_segments": len(transcript_result.segments),
            "num_speakers": len(transcript_result.speakers),
            # 포렌식 점수
            "overall_risk_score": forensic_score.overall_risk_score,
            "overall_risk_level": forensic_score.overall_risk_level,
            "confidence_level": getattr(forensic_score, "confidence_level", "N/A"),
            "gaslighting_score": category_scores.get(
                "gaslighting", type("obj", (object,), {"score": 0, "intensity": "N/A"})
            ).score,
            "gaslighting_intensity": getattr(
                category_scores.get("gaslighting"), "intensity", "N/A"
            ),
            "threat_score": category_scores.get(
                "threat", type("obj", (object,), {"score": 0, "intensity": "N/A"})
            ).score,
            "threat_intensity": getattr(category_scores.get("threat"), "intensity", "N/A"),
            "coercion_score": category_scores.get(
                "coercion", type("obj", (object,), {"score": 0, "intensity": "N/A"})
            ).score,
            "coercion_intensity": getattr(category_scores.get("coercion"), "intensity", "N/A"),
            "deception_score": category_scores.get(
                "deception", type("obj", (object,), {"score": 0, "intensity": "N/A"})
            ).score,
            "deception_intensity": getattr(category_scores.get("deception"), "intensity", "N/A"),
            "emotional_score": category_scores.get(
                "emotional_manipulation", type("obj", (object,), {"score": 0, "intensity": "N/A"})
            ).score,
            "emotional_intensity": getattr(
                category_scores.get("emotional_manipulation"), "intensity", "N/A"
            ),
            "voice_text_consistency": getattr(forensic_score, "voice_text_consistency", "N/A"),
            "cross_validation_consistency": getattr(
                forensic_score, "cross_validation_consistency", "N/A"
            ),
            "summary": forensic_score.summary,
            "recommendations": forensic_score.recommendations,
            "flags": forensic_score.flags,
            "processing_time": getattr(forensic_score, "processing_time_seconds", 0),
            # STT 정확도 메트릭
            "stt_confidence_avg": stt_accuracy.confidence.avg_word_confidence,
            "stt_confidence_min": stt_accuracy.confidence.min_word_confidence,
            "stt_confidence_max": stt_accuracy.confidence.max_word_confidence,
            "stt_confidence_grade": stt_accuracy.confidence.overall_confidence_grade,
            "stt_low_conf_words": stt_accuracy.confidence.low_confidence_words,
            "stt_low_conf_ratio": stt_accuracy.confidence.low_confidence_ratio,
            "stt_total_words": stt_accuracy.stats.total_words,
            "stt_unique_words": stt_accuracy.stats.unique_words,
            "stt_wer": stt_accuracy.errors.wer if stt_accuracy.errors else None,
            "stt_cer": stt_accuracy.errors.cer if stt_accuracy.errors else None,
            "stt_accuracy_grade": stt_accuracy.errors.accuracy_grade
            if stt_accuracy.errors
            else None,
            "stt_words_per_minute": stt_accuracy.stats.words_per_minute,
            "stt_korean_ratio": stt_accuracy.stats.korean_ratio,
            "stt_has_reference": stt_accuracy.has_reference,
            # 화자별 인식 정확도
            "stt_speaker_count": stt_accuracy.speaker_accuracy.speaker_count,
            "stt_speaker_words": stt_accuracy.speaker_accuracy.speaker_words,
            "stt_speaker_duration": stt_accuracy.speaker_accuracy.speaker_duration,
            "stt_speaker_confidence": stt_accuracy.speaker_accuracy.speaker_confidence,
            "stt_speaker_switches": stt_accuracy.speaker_accuracy.speaker_switches,
            "stt_speaker_switch_accuracy": stt_accuracy.speaker_accuracy.speaker_switch_accuracy,
            "stt_speaker_uniformity": stt_accuracy.speaker_accuracy.speaker_uniformity,
            # 타임스탬프 정확도
            "stt_avg_word_duration": stt_accuracy.timestamp_accuracy.avg_word_duration,
            "stt_min_word_duration": stt_accuracy.timestamp_accuracy.min_word_duration,
            "stt_max_word_duration": stt_accuracy.timestamp_accuracy.max_word_duration,
            "stt_avg_timestamp_gap": stt_accuracy.timestamp_accuracy.avg_timestamp_gap,
            "stt_max_timestamp_gap": stt_accuracy.timestamp_accuracy.max_timestamp_gap,
            "stt_timestamps_within_100ms": stt_accuracy.timestamp_accuracy.timestamps_within_100ms,
            "stt_timestamps_within_200ms": stt_accuracy.timestamp_accuracy.timestamps_within_200ms,
            "stt_timestamp_precision_ratio": stt_accuracy.timestamp_accuracy.timestamp_precision_ratio,
            # 문장 단위 분석
            "stt_total_sentences": stt_accuracy.sentence_metrics.total_sentences,
            "stt_avg_sentence_length": stt_accuracy.sentence_metrics.avg_sentence_length,
            "stt_avg_words_per_sentence": stt_accuracy.sentence_metrics.avg_words_per_sentence,
            "stt_complete_sentences": stt_accuracy.sentence_metrics.complete_sentences,
            "stt_sentence_completion_ratio": stt_accuracy.sentence_metrics.sentence_completion_ratio,
            "stt_punctuation_accuracy": stt_accuracy.sentence_metrics.punctuation_accuracy,
            "stt_missing_punctuation": stt_accuracy.sentence_metrics.missing_punctuation,
            # 음성-텍스트 일치도
            "stt_voice_text_correlation": stt_accuracy.voice_text_consistency.voice_text_correlation,
            "stt_acoustic_feature_match": stt_accuracy.voice_text_consistency.acoustic_feature_match,
            "stt_rhythm_consistency": stt_accuracy.voice_text_consistency.rhythm_consistency,
            "stt_pause_distribution_match": stt_accuracy.voice_text_consistency.pause_distribution_match,
            "stt_overall_consistency_score": stt_accuracy.voice_text_consistency.overall_consistency_score,
            "stt_consistency_grade": stt_accuracy.voice_text_consistency.consistency_grade,
        }

        # GPU 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[{audio_name}] 분석 완료 (위험도: {result['overall_risk_score']:.1f}/100, 신뢰도: {result['stt_confidence_grade']}등급)"
        )
        return result

    except Exception as e:
        logger.error(f"[{audio_name}] 분석 실패: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def display_results_table(results: List[Dict[str, Any]], title: str = "포렌식 분석 결과"):
    """
    결과를 깔끔한 CLI 테이블로 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
        title: 테이블 제목
    """
    if not results:
        print("\n" + "=" * 150)
        print(f"{title}")
        print("=" * 150)
        print("표시할 결과가 없습니다.")
        print("=" * 150)
        return

    print("\n" + "=" * 180)
    print(f"{title} ({len(results)}개 파일)")
    print("=" * 180)

    # 테이블 헤더 (한국어)
    headers = [
        "파일명",
        "길이",
        "화자",
        "위험도",
        "등급",
        "신뢰도",
        "단어수",
        "WPM",
        "가스라이팅",
        "협박",
        "강요",
        "사기",
        "감정조작",
        "처리시간(s)",
    ]

    rows = []
    for r in results:
        # 파일명 잘라서 표시
        filename = r["file_name"]
        if len(filename) > 25:
            filename = filename[:22] + "..."
        elif len(filename) < 25:
            filename = filename.ljust(25)

        rows.append(
            [
                filename,
                f"{r['duration_seconds']:.1f}s",
                str(r["num_speakers"]),
                f"{r['overall_risk_score']:.1f}",
                r["overall_risk_level"],
                f"{r['stt_confidence_avg']:.2f}({r['stt_confidence_grade']})",
                str(r["stt_total_words"]),
                f"{r['stt_words_per_minute']:.0f}",
                f"{r['gaslighting_score']:.1f}",
                f"{r['threat_score']:.1f}",
                f"{r['coercion_score']:.1f}",
                f"{r['deception_score']:.1f}",
                f"{r['emotional_score']:.1f}",
                f"{r['processing_time']:.1f}",
            ]
        )

    # 테이블 출력
    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        # 단순 테이블 형식 대체
        print("-" * 180)
        header_row = f"{headers[0]:<25} " + " ".join(f"{h:>12}" for h in headers[1:])
        print(header_row)
        print("-" * 180)
        for row in rows:
            row_str = f"{row[0]:<25} " + " ".join(f"{r:>12}" for r in row[1:])
            print(row_str)

    print("=" * 180)

    # 통계 계산 및 표시
    if len(results) > 0:
        avg_risk = sum(r["overall_risk_score"] for r in results) / len(results)
        high_risk_count = sum(1 for r in results if r["overall_risk_score"] >= 60)
        critical_count = sum(1 for r in results if r["overall_risk_score"] >= 80)

        # STT 신뢰도 통계
        avg_confidence = sum(r["stt_confidence_avg"] for r in results) / len(results)
        grade_dist = {}
        for r in results:
            g = r["stt_confidence_grade"]
            grade_dist[g] = grade_dist.get(g, 0) + 1

        print("\n📊 통계:")
        print(f"  평균 위험도 점수: {avg_risk:.1f}/100")
        print(f"  고위험 파일 (≥60): {high_risk_count}/{len(results)}")
        print(f"  심각 파일 (≥80): {critical_count}/{len(results)}")
        print(f"  평균 STT 신뢰도: {avg_confidence:.3f}")
        print(f"  신뢰도 등급 분포: {grade_dist}")

        # WER/CER 통계 (참조가 있는 파일만)
        results_with_ref = [r for r in results if r["stt_has_reference"]]
        if results_with_ref:
            avg_wer = sum(r["stt_wer"] for r in results_with_ref if r["stt_wer"]) / len(
                results_with_ref
            )
            avg_cer = sum(r["stt_cer"] for r in results_with_ref if r["stt_cer"]) / len(
                results_with_ref
            )
            print(f"\n📏 오류율 (참조 있는 {len(results_with_ref)}개 파일):")
            print(f"  평균 WER: {avg_wer:.3f}")
            print(f"  평균 CER: {avg_cer:.3f}")

        # 카테고리별 평균
        print("\n📈 카테고리별 평균:")
        print(f"  가스라이팅: {sum(r['gaslighting_score'] for r in results) / len(results):.1f}")
        print(f"  협박: {sum(r['threat_score'] for r in results) / len(results):.1f}")
        print(f"  강요: {sum(r['coercion_score'] for r in results) / len(results):.1f}")
        print(f"  사기: {sum(r['deception_score'] for r in results) / len(results):.1f}")
        print(f"  감정 조작: {sum(r['emotional_score'] for r in results) / len(results):.1f}")

    print("=" * 180 + "\n")


def display_stt_accuracy_table(results: List[Dict[str, Any]]):
    """
    STT 정확도 전용 테이블 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
    """
    if not results:
        return

    print("\n" + "=" * 140)
    print("🎯 STT 정확도 상세")
    print("=" * 140)

    headers = [
        "파일명",
        "신뢰도",
        "등급",
        "최저/최고",
        "저신뢰단어",
        "총단어",
        "고유단어",
        "WPM",
        "한글비율",
        "WER",
        "CER",
    ]

    rows = []
    for r in results:
        filename = r["file_name"]
        if len(filename) > 20:
            filename = filename[:17] + "..."

        rows.append(
            [
                filename,
                f"{r['stt_confidence_avg']:.3f}",
                r["stt_confidence_grade"],
                f"{r['stt_confidence_min']:.2f}/{r['stt_confidence_max']:.2f}",
                f"{r['stt_low_conf_words']}({r['stt_low_conf_ratio']:.1%})",
                str(r["stt_total_words"]),
                str(r["stt_unique_words"]),
                f"{r['stt_words_per_minute']:.0f}",
                f"{r['stt_korean_ratio']:.1%}",
                f"{r['stt_wer']:.3f}" if r["stt_wer"] else "N/A",
                f"{r['stt_cer']:.3f}" if r["stt_cer"] else "N/A",
            ]
        )

    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        print("-" * 140)
        header_row = f"{headers[0]:<20} " + " ".join(f"{h:>10}" for h in headers[1:])
        print(header_row)
        print("-" * 140)
        for row in rows:
            row_str = f"{row[0]:<20} " + " ".join(f"{r:>10}" for r in row[1:])
            print(row_str)

    print("=" * 140 + "\n")


def display_speaker_accuracy_table(results: List[Dict[str, Any]]):
    """
    화자별 인식 정확도 테이블 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
    """
    if not results:
        return

    print("\n" + "=" * 160)
    print("🎤 화자별 인식 정확도 상세")
    print("=" * 160)

    headers = [
        "파일명",
        "화자수",
        "화자별 단어",
        "화자별 발화시간(초)",
        "화자별 신뢰도",
        "화자전환",
        "전환정확도",
        "화자균형",
    ]

    rows = []
    for r in results:
        filename = r["file_name"]
        if len(filename) > 18:
            filename = filename[:15] + "..."

        # 화자별 단어/시간/신뢰도 포맷팅
        speaker_words = str(r["stt_speaker_words"])
        if len(speaker_words) > 25:
            speaker_words = speaker_words[:22] + "..."

        speaker_duration = str(r["stt_speaker_duration"])
        if len(speaker_duration) > 20:
            speaker_duration = speaker_duration[:17] + "..."

        speaker_conf = str(r["stt_speaker_confidence"])
        if len(speaker_conf) > 18:
            speaker_conf = speaker_conf[:15] + "..."

        rows.append(
            [
                filename,
                str(r["stt_speaker_count"]),
                speaker_words,
                speaker_duration,
                speaker_conf,
                str(r["stt_speaker_switches"]),
                f"{r['stt_speaker_switch_accuracy']:.2%}",
                f"{r['stt_speaker_uniformity']:.2f}",
            ]
        )

    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        print("-" * 160)
        header_row = f"{headers[0]:<18} " + " ".join(f"{h:>12}" for h in headers[1:])
        print(header_row)
        print("-" * 160)
        for row in rows:
            row_str = f"{row[0]:<18} " + " ".join(f"{r:>12}" for r in row[1:])
            print(row_str)

    print("=" * 160 + "\n")


def display_timestamp_accuracy_table(results: List[Dict[str, Any]]):
    """
    타임스탬프 정확도 테이블 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
    """
    if not results:
        return

    print("\n" + "=" * 160)
    print("⏱️  타임스탬프 정확도 상세")
    print("=" * 160)

    headers = [
        "파일명",
        "평균단어길이(ms)",
        "최소/최대(ms)",
        "평균타임갭(ms)",
        "최대갭(ms)",
        "100ms이내",
        "200ms이내",
        "정밀도비율",
    ]

    rows = []
    for r in results:
        filename = r["file_name"]
        if len(filename) > 18:
            filename = filename[:15] + "..."

        rows.append(
            [
                filename,
                f"{r['stt_avg_word_duration'] * 1000:.0f}",
                f"{r['stt_min_word_duration'] * 1000:.0f}/{r['stt_max_word_duration'] * 1000:.0f}",
                f"{r['stt_avg_timestamp_gap'] * 1000:.1f}",
                f"{r['stt_max_timestamp_gap'] * 1000:.0f}",
                f"{r['stt_timestamps_within_100ms']}개",
                f"{r['stt_timestamps_within_200ms']}개",
                f"{r['stt_timestamp_precision_ratio']:.2%}",
            ]
        )

    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        print("-" * 160)
        header_row = f"{headers[0]:<18} " + " ".join(f"{h:>12}" for h in headers[1:])
        print(header_row)
        print("-" * 160)
        for row in rows:
            row_str = f"{row[0]:<18} " + " ".join(f"{r:>12}" for r in row[1:])
            print(row_str)

    print("=" * 160 + "\n")


def display_sentence_metrics_table(results: List[Dict[str, Any]]):
    """
    문장 단위 분석 테이블 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
    """
    if not results:
        return

    print("\n" + "=" * 160)
    print("📝 문장 단위 분석 상세")
    print("=" * 160)

    headers = [
        "파일명",
        "문장수",
        "평균문장길이",
        "문장당단어",
        "완결문장",
        "완결비율",
        "문장부호정확도",
        "누락부호",
    ]

    rows = []
    for r in results:
        filename = r["file_name"]
        if len(filename) > 18:
            filename = filename[:15] + "..."

        rows.append(
            [
                filename,
                str(r["stt_total_sentences"]),
                f"{r['stt_avg_sentence_length']:.1f}자",
                f"{r['stt_avg_words_per_sentence']:.1f}개",
                f"{r['stt_complete_sentences']}/{r['stt_total_sentences']}",
                f"{r['stt_sentence_completion_ratio']:.1%}",
                f"{r['stt_punctuation_accuracy']:.1%}",
                str(r["stt_missing_punctuation"]),
            ]
        )

    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        print("-" * 160)
        header_row = f"{headers[0]:<18} " + " ".join(f"{h:>12}" for h in headers[1:])
        print(header_row)
        print("-" * 160)
        for row in rows:
            row_str = f"{row[0]:<18} " + " ".join(f"{r:>12}" for r in row[1:])
            print(row_str)

    print("=" * 160 + "\n")


def display_voice_text_consistency_table(results: List[Dict[str, Any]]):
    """
    음성-텍스트 일치도 테이블 표시

    Args:
        results: 분석 결과 딕셔너리 리스트
    """
    if not results:
        return

    print("\n" + "=" * 160)
    print("🎵 음성-텍스트 일치도 상세")
    print("=" * 160)

    headers = [
        "파일명",
        "음성-텍스트상관",
        "음향특성일치",
        "리듬일치도",
        "쉼분포일치",
        "종합일치점수",
        "일치등급",
    ]

    rows = []
    for r in results:
        filename = r["file_name"]
        if len(filename) > 18:
            filename = filename[:15] + "..."

        rows.append(
            [
                filename,
                f"{r['stt_voice_text_correlation']:.2f}",
                f"{r['stt_acoustic_feature_match']:.2f}",
                f"{r['stt_rhythm_consistency']:.2f}",
                f"{r['stt_pause_distribution_match']:.2f}",
                f"{r['stt_overall_consistency_score']:.2f}",
                r["stt_consistency_grade"],
            ]
        )

    if HAS_TABULATE:
        print(
            tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        )
    else:
        print("-" * 160)
        header_row = f"{headers[0]:<18} " + " ".join(f"{h:>12}" for h in headers[1:])
        print(header_row)
        print("-" * 160)
        for row in rows:
            row_str = f"{row[0]:<18} " + " ".join(f"{r:>12}" for r in row[1:])
            print(row_str)

    print("=" * 160 + "\n")


def display_detailed_report(result: Dict[str, Any]):
    """
    단일 파일에 대한 상세 분석 보고서 표시

    Args:
        result: 분석 결과 딕셔너리
    """
    print("\n" + "▓" * 80)
    print(f"상세 보고서: {result['file_name']}")
    print("▓" * 80)

    print("\n📁 파일 정보:")
    print(f"  경로: {result['file_path']}")
    print(f"  길이: {result['duration_seconds']:.1f}초")
    print(f"  화자: {result['num_speakers']}명")
    print(f"  세그먼트: {result['num_segments']}개")

    print("\n🎯 위험도 평가:")
    print(f"  전체 위험도 점수: {result['overall_risk_score']:.1f}/100")
    print(f"  위험도 등급: {result['overall_risk_level']}")
    print(f"  신뢰수준: {result['confidence_level']}")

    print("\n🎤 STT 정확도:")
    print(f"  평균 신뢰도: {result['stt_confidence_avg']:.3f}")
    print(f"  신뢰도 등급: {result['stt_confidence_grade']}등급")
    print(f"  신뢰도 범위: {result['stt_confidence_min']:.2f} ~ {result['stt_confidence_max']:.2f}")
    print(f"  저신뢰도 단어: {result['stt_low_conf_words']}개 ({result['stt_low_conf_ratio']:.1%})")
    print(f"  총 단어수: {result['stt_total_words']}개 (고유: {result['stt_unique_words']}개)")
    print(f"  말하기 속도: 분당 {result['stt_words_per_minute']:.0f}단어")
    print(f"  한글 비율: {result['stt_korean_ratio']:.1%}")

    if result["stt_has_reference"] and result["stt_wer"]:
        print(f"  단어 오류율 (WER): {result['stt_wer']:.3f} ({result['stt_accuracy_grade']}등급)")
        print(f"  문자 오류율 (CER): {result['stt_cer']:.3f}")

    print("\n🎤 화자별 인식 정확도:")
    print(f"  화자 수: {result['stt_speaker_count']}명")
    print(f"  화자별 단어 수: {result['stt_speaker_words']}")
    print(f"  화자별 발화 시간: {result['stt_speaker_duration']}")
    print(f"  화자별 신뢰도: {result['stt_speaker_confidence']}")
    print(f"  화자 전환 횟수: {result['stt_speaker_switches']}회")
    print(f"  화자 전환 정확도: {result['stt_speaker_switch_accuracy']:.2%}")
    print(f"  화자 균형도: {result['stt_speaker_uniformity']:.2f}")

    print("\n⏱️  타임스탬프 정확도:")
    print(f"  평균 단어 길이: {result['stt_avg_word_duration'] * 1000:.0f}ms")
    print(
        f"  단어 길이 범위: {result['stt_min_word_duration'] * 1000:.0f}ms ~ {result['stt_max_word_duration'] * 1000:.0f}ms"
    )
    print(f"  평균 타임스탬프 갭: {result['stt_avg_timestamp_gap'] * 1000:.1f}ms")
    print(f"  최대 타임스탬프 갭: {result['stt_max_timestamp_gap'] * 1000:.0f}ms")
    print(f"  100ms 이내 타임스탬프: {result['stt_timestamps_within_100ms']}개")
    print(f"  200ms 이내 타임스탬프: {result['stt_timestamps_within_200ms']}개")
    print(f"  타임스탬프 정밀도 비율: {result['stt_timestamp_precision_ratio']:.2%}")

    print("\n📝 문장 단위 분석:")
    print(f"  총 문장 수: {result['stt_total_sentences']}개")
    print(f"  평균 문장 길이: {result['stt_avg_sentence_length']:.1f}자")
    print(f"  문장당 평균 단어: {result['stt_avg_words_per_sentence']:.1f}개")
    print(
        f"  완결 문장: {result['stt_complete_sentences']}/{result['stt_total_sentences']} ({result['stt_sentence_completion_ratio']:.1%})"
    )
    print(f"  문장 부호 정확도: {result['stt_punctuation_accuracy']:.1%}")
    print(f"  누락된 문장 부호: {result['stt_missing_punctuation']}개")

    print("\n🎵 음성-텍스트 일치도:")
    print(f"  음성-텍스트 상관계수: {result['stt_voice_text_correlation']:.3f}")
    print(f"  음향 특성 일치도: {result['stt_acoustic_feature_match']:.3f}")
    print(f"  리듬 일치도: {result['stt_rhythm_consistency']:.3f}")
    print(f"  쉼 분포 일치도: {result['stt_pause_distribution_match']:.3f}")
    print(f"  종합 일치 점수: {result['stt_overall_consistency_score']:.3f}")
    print(f"  일치 등급: {result['stt_consistency_grade']}")

    print("\n📊 카테고리 점수:")
    print(
        f"  가스라이팅: {result['gaslighting_score']:.1f}/100 ({result['gaslighting_intensity']})"
    )
    print(f"  협박: {result['threat_score']:.1f}/100 ({result['threat_intensity']})")
    print(f"  강요: {result['coercion_score']:.1f}/100 ({result['coercion_intensity']})")
    print(f"  사기: {result['deception_score']:.1f}/100 ({result['deception_intensity']})")
    print(f"  감정 조작: {result['emotional_score']:.1f}/100 ({result['emotional_intensity']})")

    print("\n🔍 교차 검증:")
    print(f"  음성-텍스트 일치성: {result['voice_text_consistency']}")
    print(f"  교차 검증 일치성: {result['cross_validation_consistency']}")

    print("\n📝 요약:")
    for line in result["summary"].split("\n"):
        print(f"  {line}")

    if result["flags"]:
        print("\n⚠️  플래그:")
        for flag in result["flags"]:
            print(f"  - {flag}")

    if result["recommendations"]:
        print("\n💡 권장사항:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")

    print("\n" + "▓" * 80 + "\n")


async def process_batch_async(
    audio_files: List[Path],
    model_size: str,
    update_interval: int = 10,
) -> List[Dict[str, Any]]:
    """
    오디오 파일을 배치로 처리하고 주기적으로 결과 표시

    Args:
        audio_files: 오디오 파일 경로 리스트
        model_size: Whisper 모델 크기
        update_interval: N개 파일마다 결과 표시

    Returns:
        모든 결과 리스트
    """
    model_pool = SharedModelPool()
    results = []

    start_time = datetime.now()
    total_count = len(audio_files)

    logger.info("=" * 60)
    logger.info("포렌식 파이프라인 모니터링")
    logger.info("=" * 60)
    logger.info(f"처리할 파일: {total_count}개")
    logger.info(f"업데이트 주기: {update_interval}개")
    logger.info(f"모델: {model_size}")
    logger.info("=" * 60)

    for i, audio_path in enumerate(audio_files, 1):
        file_start = datetime.now()
        result = await process_single_file(audio_path, model_pool, model_size)

        if result:
            result["processing_time"] = (datetime.now() - file_start).total_seconds()
            results.append(result)
            logger.info(
                f"[{i}/{total_count}] 완료: {result['file_name']} (위험도: {result['overall_risk_score']:.1f}, 신뢰도: {result['stt_confidence_grade']})"
            )
        else:
            logger.error(f"[{i}/{total_count}] 실패: {audio_path.name}")

        # update_interval마다 또는 마지막에 진행 업데이트 표시
        if i % update_interval == 0 or i == total_count:
            display_results_table(results, title=f"진행 상황 ({i}/{total_count}개 파일 처리 완료)")
            display_stt_accuracy_table(results)
            # 확장된 STT 정밀도 메트릭 표시
            display_speaker_accuracy_table(results)
            display_timestamp_accuracy_table(results)
            display_sentence_metrics_table(results)
            display_voice_text_consistency_table(results)

    # 정리
    model_pool.cleanup_all()

    # 전체 처리 시간 표시
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"총 처리 시간: {total_time:.1f}초")

    return results


async def main_async(args):
    """메인 비동기 진입점"""
    # 오디오 파일 재귀적으로 찾기
    audio_files = []
    for ext in ["*.m4a", "*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(Path(args.input_dir).rglob(ext))

    # 일관된 처리를 위해 파일 정렬
    audio_files = sorted(audio_files)

    if args.limit:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        print(f"\n❌ {args.input_dir}에서 오디오 파일을 찾지 못했습니다")
        return 1

    # 파일 처리
    results = await process_batch_async(
        audio_files=audio_files,
        model_size=args.model,
        update_interval=args.update_interval,
    )

    # 최종 요약 테이블
    display_results_table(results, title="최종 결과 - 포렌식 분석 완료")
    display_stt_accuracy_table(results)
    # 확장된 STT 정밀도 메트릭 최종 요약
    display_speaker_accuracy_table(results)
    display_timestamp_accuracy_table(results)
    display_sentence_metrics_table(results)
    display_voice_text_consistency_table(results)

    # 요청 시 상세 보고서 표시
    if args.verbose and results:
        for result in results:
            display_detailed_report(result)

    # JSON으로 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장됨: {output_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="포렌식 파이프라인 처리를 CLI 테이블로 모니터링 (STT 정확도 포함)"
    )
    parser.add_argument(
        "--input-dir",
        default="ref/call",
        help="오디오 파일이 있는 디렉토리 (기본값: ref/call)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="결과 출력 JSON 파일",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="처리할 파일 수 제한",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10,
        help="N개 파일마다 결과 표시 (기본값: 10)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "distil-large-v3",
        ],
        help="Whisper 모델 크기 (기본값: large-v3)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="각 파일에 대한 상세 보고서 표시",
    )

    args = parser.parse_args()

    if not HAS_TABULATE:
        print("\n⚠️  'tabulate' 패키지를 찾지 못했습니다. 더 나은 테이블 형식을 위해 설치하세요:")
        print("   pip install tabulate\n")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
