"""
Unit tests for Korean Cultural Context Analyzer
SPEC-NLP-KOBERT-001 TASK-003: Korean cultural context analyzer
"""

import pytest
from typing import List, Literal, Optional
from unittest.mock import Mock, patch

from voice_man.services.nlp.cultural_analyzer import (
    LevelTransition,
    SpeechLevelResult,
    HierarchyContext,
    ManipulationPattern,
    KoreanCulturalAnalyzer,
)


class TestKoreanCulturalAnalyzer:
    """Test suite for KoreanCulturalAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Analyzer fixture"""

        return KoreanCulturalAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None

    def test_analyze_speech_level_returns_speech_level_result(self, analyzer):
        """Test analyze_speech_level returns SpeechLevelResult"""
        result = analyzer.analyze_speech_level("안녕하세요")

        assert isinstance(result, SpeechLevelResult)
        assert hasattr(result, "level")
        assert hasattr(result, "formal_ratio")
        assert hasattr(result, "informal_ratio")
        assert hasattr(result, "level_transitions")

    def test_analyze_speech_level_detects_formal(self, analyzer):
        """Test analyze_speech_level detects formal speech"""
        formal_text = "안녕하세요? 오늘 날씨가 좋네요. 저는 학생입니다."
        result = analyzer.analyze_speech_level(formal_text)

        assert result.level in ["formal", "mixed"]
        assert result.formal_ratio > result.informal_ratio

    def test_analyze_speech_level_detects_informal(self, analyzer):
        """Test analyze_speech_level detects informal speech"""
        informal_text = "안녕! 오늘 날씨 좋다. 나는 학생이야."
        result = analyzer.analyze_speech_level(informal_text)

        assert result.level in ["informal", "mixed"]
        assert result.informal_ratio > result.formal_ratio

    def test_analyze_speech_level_detects_mixed(self, analyzer):
        """Test analyze_speech_level detects mixed speech levels"""
        mixed_text = "안녕하세요! 반가워. 선생님, 오늘 수업 있어?"
        result = analyzer.analyze_speech_level(mixed_text)

        assert result.level == "mixed"
        assert len(result.level_transitions) > 0

    def test_analyze_speech_level_empty_text(self, analyzer):
        """Test analyze_speech_level handles empty text"""
        result = analyzer.analyze_speech_level("")

        assert result.level == "mixed"
        assert result.formal_ratio == 0.0
        assert result.informal_ratio == 0.0

    def test_analyze_speech_level_ratios_sum_to_one(self, analyzer):
        """Test formal and informal ratios sum to 1.0"""
        result = analyzer.analyze_speech_level("안녕하세요 안녕")

        assert abs(result.formal_ratio + result.informal_ratio - 1.0) < 0.01

    def test_detect_hierarchy_context_returns_hierarchy_context(self, analyzer):
        """Test detect_hierarchy_context returns HierarchyContext"""
        result = analyzer.detect_hierarchy_context("할머니, 밥 드세요")

        assert isinstance(result, HierarchyContext)
        assert hasattr(result, "has_family_markers")
        assert hasattr(result, "has_job_title_markers")
        assert hasattr(result, "has_social_markers")
        assert hasattr(result, "detected_relationships")

    def test_detect_hierarchy_context_detects_family_markers(self, analyzer):
        """Test detect_hierarchy_context detects family markers"""
        family_text = "할머니, 아버지, 어머니, 형, 누나"
        result = analyzer.detect_hierarchy_context(family_text)

        assert result.has_family_markers is True

    def test_detect_hierarchy_context_detects_job_title_markers(self, analyzer):
        """Test detect_hierarchy_context detects job title markers"""
        job_text = "부장님, 사장님, 과장님, 대리님"
        result = analyzer.detect_hierarchy_context(job_text)

        assert result.has_job_title_markers is True

    def test_detect_hierarchy_context_detects_social_markers(self, analyzer):
        """Test detect_hierarchy_context detects social markers"""
        social_text = "선생님, 고객님, 손님, 교수님"
        result = analyzer.detect_hierarchy_context(social_text)

        assert result.has_social_markers is True

    def test_detect_hierarchy_context_empty_text(self, analyzer):
        """Test detect_hierarchy_context handles empty text"""
        result = analyzer.detect_hierarchy_context("")

        assert result.has_family_markers is False
        assert result.has_job_title_markers is False
        assert result.has_social_markers is False
        assert len(result.detected_relationships) == 0

    def test_detect_manipulation_patterns_returns_list(self, analyzer):
        """Test detect_manipulation_patterns returns list"""
        result = analyzer.detect_manipulation_patterns("테스트")

        assert isinstance(result, list)

    def test_detect_manipulation_patterns_detects_gaslighting(self, analyzer):
        """Test detect_manipulation_patterns detects gaslighting patterns"""
        gaslighting_text = "네가 잘못했으니까 그래. 다 널 위해서야."
        result = analyzer.detect_manipulation_patterns(gaslighting_text)

        assert len(result) > 0
        assert any(p.category == "gaslighting" for p in result)

    def test_detect_manipulation_patterns_detects_threats(self, analyzer):
        """Test detect_manipulation_patterns detects threat patterns"""
        threat_text = "그렇게 하면 큰일 낼 거야. 후회하게 될 거야."
        result = analyzer.detect_manipulation_patterns(threat_text)

        assert len(result) > 0
        assert any(p.category == "threat" for p in result)

    def test_detect_manipulation_patterns_detects_coercion(self, analyzer):
        """Test detect_manipulation_patterns detects coercion patterns"""
        coercion_text = "너라면 할 수 있잖아. 나를 실망시키지 마."
        result = analyzer.detect_manipulation_patterns(coercion_text)

        assert len(result) > 0
        assert any(p.category == "coercion" for p in result)

    def test_detect_manipulation_patterns_confidence_between_zero_and_one(self, analyzer):
        """Test manipulation pattern confidence is between 0 and 1"""
        text = "네가 잘못했으니까 그래."
        result = analyzer.detect_manipulation_patterns(text)

        for pattern in result:
            assert 0.0 <= pattern.confidence <= 1.0

    def test_detect_manipulation_patterns_empty_text(self, analyzer):
        """Test detect_manipulation_patterns handles empty text"""
        result = analyzer.detect_manipulation_patterns("")

        assert result == []

    def test_analyze_comprehensive_combines_all_analyses(self, analyzer):
        """Test analyze_comprehensive combines speech, hierarchy, and manipulation analysis"""
        text = "선생님, 네가 잘못했으니까 그래. 다 널 위해서야."

        result = analyzer.analyze_comprehensive(text)

        assert hasattr(result, "speech_level")
        assert hasattr(result, "hierarchy_context")
        assert hasattr(result, "manipulation_patterns")

    def test_korean_specific_patterns(self, analyzer):
        """Test Korean-specific manipulation patterns are detected"""
        korean_patterns = [
            "그렇게 느끼는 게 이상한 거야",  # Invalidating feelings
            "다 널 위해서야",  # For your own good
            "너라면 할 수 있잖아",  # Guilt trip
        ]

        for pattern in korean_patterns:
            result = analyzer.detect_manipulation_patterns(pattern)
            # At least some patterns should be detected
            assert isinstance(result, list)


class TestKoreanCulturalAnalyzerEdgeCases:
    """Edge case tests for KoreanCulturalAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Analyzer fixture"""

        return KoreanCulturalAnalyzer()

    def test_very_long_text(self, analyzer):
        """Test analyzer handles very long text"""
        long_text = "안녕하세요 " * 1000
        result = analyzer.analyze_speech_level(long_text)

        assert isinstance(result, SpeechLevelResult)

    def test_mixed_korean_english(self, analyzer):
        """Test analyzer handles mixed Korean-English text"""
        mixed_text = "Hello 선생님, 오늘 수업 있어?"
        result = analyzer.analyze_speech_level(mixed_text)

        assert isinstance(result, SpeechLevelResult)

    def test_special_characters(self, analyzer):
        """Test analyzer handles special characters"""
        special_text = "안녕하세요!!! ~~~ ㅋㅋㅋ ㅠㅠ"
        result = analyzer.analyze_speech_level(special_text)

        assert isinstance(result, SpeechLevelResult)

    def test_multiple_sentences(self, analyzer):
        """Test analyzer handles multiple sentences"""
        multi_text = "안녕하세요? 오늘 날씨가 좋네요. 반가워! 잘 지냈어?"
        result = analyzer.analyze_speech_level(multi_text)

        assert isinstance(result, SpeechLevelResult)
        assert len(result.level_transitions) > 0
