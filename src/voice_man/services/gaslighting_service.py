"""
가스라이팅 패턴 감지 서비스

부정, 전가, 축소, 혼란 유발 패턴을 감지하고 분석하는 서비스
"""

from datetime import datetime
from typing import List, Dict
from voice_man.models.gaslighting import (
    GaslightingPattern,
    GaslightingPatternType,
    GaslightingAnalysisResult,
)


class GaslightingService:
    """
    가스라이팅 패턴 감지 서비스

    4가지 주요 패턴 감지:
    1. 부정 (Denial): 상대방의 경험/기억 부정
    2. 전가 (Blame-shifting): 책임을 상대방에게 전가
    3. 축소 (Minimizing): 상대방의 감정/경험 축소
    4. 혼란 (Confusion): 의도적 혼란 유발
    """

    # 패턴별 키워드 사전
    PATTERN_KEYWORDS = {
        GaslightingPatternType.DENIAL: [
            "그런 적 없어",
            "그런 거 없었는데",
            "네가 잘못 기억하는 거야",
            "기억 안 나",
            "상상한 거지",
            "상상한 거겠지",
            "그건 없었던 일이야",
            "내가 언제 그랬어",
            "거짓말하지 마",
            "넌 계속 그런다고 해",
            "말도 안 되는 소리해",
        ],
        GaslightingPatternType.BLAME_SHIFTING: [
            "네 때문에",
            "네가 먼저 그랬어",
            "네가 그렇게 해서",
            "너부터 잘해",
            "네 탓이야",
            "네가 문제야",
            "너 때문에 이렇게 됐어",
            "네 행동 때문이야",
            "네가 원인이야",
        ],
        GaslightingPatternType.MINIMIZING: [
            "별거 아닌데",
            "과민반응하지 마",
            "그 정도로 화내?",
            "크게 문제 아닌데",
            "너 심각하게 생각하네",
            "그냥 농담이었어",
            "왜 그렇게 예민해",
            "별일 아니야",
            "그게 다 뭐야",
            "너너 괜찮을 것 같아",
        ],
        GaslightingPatternType.CONFUSION: [
            "아까는 그랬는데",
            "지금은 말이 안 되네",
            "뭐가 진짜야?",
            "너 이상하네",
            "왜 갑자기 그래",
            "말이 바뀌네",
            "앞뒤가 안 맞아",
            "너 헷갈리나?",
            "무슨 소리하는 거야",
            "이해가 안 돼",
            "그럴 리가",
            "있을 수가",
            "말이 안 돼",
        ],
    }

    def __init__(self):
        """가스라이팅 서비스 초기화"""
        pass

    def detect_patterns(self, text: str) -> List[GaslightingPattern]:
        """
        텍스트에서 가스라이팅 패턴 감지

        Args:
            text: 분석할 텍스트

        Returns:
            감지된 패턴 리스트
        """
        if not text or not text.strip():
            return []

        patterns = []

        # 각 패턴 유형별로 키워드 매칭
        for pattern_type, keywords in self.PATTERN_KEYWORDS.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword in text:
                    matched_keywords.append(keyword)

            if matched_keywords:
                # 신뢰도 및 강도 계산
                confidence = self._calculate_confidence(text, matched_keywords, pattern_type)
                intensity = self._calculate_intensity(text, matched_keywords)

                # 패턴 생성 (화자와 타임스탬프는 기본값)
                pattern = GaslightingPattern(
                    type=pattern_type,
                    confidence=confidence,
                    intensity=intensity,
                    speaker="Unknown",  # 대화 분석 시 설정
                    text=text,
                    timestamp="00:00:00",  # 대화 분석 시 설정
                    matched_keywords=matched_keywords,
                )
                patterns.append(pattern)

        return patterns

    def _calculate_confidence(
        self, text: str, matched_keywords: List[str], pattern_type: GaslightingPatternType
    ) -> float:
        """
        신뢰도 점수 계산

        Args:
            text: 전체 텍스트
            matched_keywords: 매칭된 키워드
            pattern_type: 패턴 유형

        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        base_confidence = 0.7

        # 키워드 빈도에 따른 가중치
        keyword_count = len(matched_keywords)
        if keyword_count >= 2:
            return min(base_confidence + 0.15, 1.0)
        elif keyword_count == 1:
            # 명확한 키워드 확인
            explicit_keywords = [
                "그런 적 없어",
                "네 때문에",
                "별거 아닌데",
                "말이 안 되네",
            ]
            if any(kw in matched_keywords[0] for kw in explicit_keywords):
                return 0.85
            return base_confidence

        return base_confidence

    def _calculate_intensity(self, text: str, matched_keywords: List[str]) -> float:
        """
        패턴 강도 계산

        Args:
            text: 전체 텍스트
            matched_keywords: 매칭된 키워드

        Returns:
            강도 점수 (0.0 ~ 1.0)
        """
        base_intensity = 0.5

        # 강한 표현 감지
        strong_indicators = ["절대", "계속", "항상", "!", "?"]
        text_lower = text.lower()

        for indicator in strong_indicators:
            if indicator in text_lower:
                base_intensity += 0.15

        return min(base_intensity, 1.0)

    def analyze_conversation(self, conversation: List[Dict]) -> GaslightingAnalysisResult:
        """
        대화 전체를 분석하여 가스라이팅 패턴 추적

        Args:
            conversation: 대화 리스트
                [{"speaker": str, "text": str, "time": str}, ...]

        Returns:
            가스라이팅 분석 결과
        """
        all_patterns = []
        speaker_patterns: Dict[str, List[GaslightingPattern]] = {}
        pattern_frequency: Dict[GaslightingPatternType, int] = {
            pattern_type: 0 for pattern_type in GaslightingPatternType
        }

        # 각 발언 분석
        for utterance in conversation:
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "")
            time = utterance.get("time", "00:00:00")

            # 패턴 감지
            detected_patterns = self.detect_patterns(text)

            # 화자 및 타임스탬프 정보 업데이트
            for pattern in detected_patterns:
                pattern.speaker = speaker
                pattern.timestamp = time
                all_patterns.append(pattern)

                # 화자별 패턴 그룹화
                if speaker not in speaker_patterns:
                    speaker_patterns[speaker] = []
                speaker_patterns[speaker].append(pattern)

                # 패턴 빈도 계산
                pattern_frequency[pattern.type] += 1

        # 위험도 평가
        risk_level = self._assess_risk_level(all_patterns, pattern_frequency)

        # 권고 조치 생성
        recommendations = self._generate_recommendations(risk_level, pattern_frequency)

        # 분석 결과 생성
        result = GaslightingAnalysisResult(
            patterns=all_patterns,
            pattern_frequency=pattern_frequency,
            speaker_patterns=speaker_patterns,
            risk_level=risk_level,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat(),
        )

        return result

    def _assess_risk_level(
        self, patterns: List[GaslightingPattern], frequency: Dict[GaslightingPatternType, int]
    ) -> str:
        """
        종합 위험도 평가

        Args:
            patterns: 모든 패턴 리스트
            frequency: 패턴 빈도수

        Returns:
            위험도 레벨 (낮음, 중간, 높음, 매우 높음)
        """
        total_patterns = len(patterns)

        if total_patterns == 0:
            return "낮음"

        # 빈도 기반 평가
        high_frequency_patterns = sum(1 for count in frequency.values() if count >= 3)
        medium_frequency_patterns = sum(1 for count in frequency.values() if count >= 2)

        # 다양한 패턴 유형이 감지된 경우
        unique_pattern_types = sum(1 for count in frequency.values() if count > 0)

        if high_frequency_patterns >= 2 or unique_pattern_types >= 4:
            return "매우 높음"
        elif (
            high_frequency_patterns >= 1
            or medium_frequency_patterns >= 3
            or unique_pattern_types >= 3
        ):
            return "높음"
        elif total_patterns >= 4 or unique_pattern_types >= 2:
            return "중간"
        elif total_patterns >= 2:
            return "중간"
        else:
            return "낮음"

    def _generate_recommendations(
        self, risk_level: str, frequency: Dict[GaslightingPatternType, int]
    ) -> List[str]:
        """
        권고 조치 생성

        Args:
            risk_level: 위험도 레벨
            frequency: 패턴 빈도수

        Returns:
            권고 조치 리스트
        """
        recommendations = []

        if risk_level in ["높음", "매우 높음"]:
            recommendations.append("전문가 상담 권장")
            recommendations.append("대화 기록 보존")

        # 부정 패턴이 많은 경우
        if frequency[GaslightingPatternType.DENIAL] >= 3:
            recommendations.append("상대방의 기억을 지속적으로 부정하는 패턴 확인")

        # 전가 패턴이 많은 경우
        if frequency[GaslightingPatternType.BLAME_SHIFTING] >= 3:
            recommendations.append("책임 전가 패턴 확인: 피해자가 자책할 가능성")

        # 축소 패턴이 많은 경우
        if frequency[GaslightingPatternType.MINIMIZING] >= 3:
            recommendations.append("감정 축소 패턴 확인: 피해자의 감정 무시")

        # 혼란 패턴이 많은 경우
        if frequency[GaslightingPatternType.CONFUSION] >= 3:
            recommendations.append("혼란 유발 패턴 확인: 가스라이팅 징후 강력")

        if not recommendations:
            recommendations.append("현재 가스라이팅 패턴 미감지 또는 낮은 수준")

        return recommendations
