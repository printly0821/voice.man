"""
Tests for Forensic Score Models
SPEC-FORENSIC-001 Phase 2-D: TDD tests for forensic scoring Pydantic models

RED Phase: These tests define expected behavior for the forensic scoring system.
"""

import pytest
from datetime import datetime

from voice_man.models.forensic.forensic_score import (
    RiskLevel,
    ForensicCategory,
    CategoryScore,
    DeceptionAnalysis,
    GaslightingAnalysis,
    ThreatAssessment,
    ForensicEvidence,
    ForensicScoreResult,
)


class TestRiskLevelEnum:
    """Test RiskLevel enum values and ranges."""

    def test_risk_level_values(self):
        """Test that all risk levels are defined."""
        assert RiskLevel.MINIMAL == "minimal"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MODERATE == "moderate"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_risk_level_count(self):
        """Test that there are exactly 5 risk levels."""
        assert len(RiskLevel) == 5


class TestForensicCategoryEnum:
    """Test ForensicCategory enum values."""

    def test_forensic_category_values(self):
        """Test that all forensic categories are defined."""
        assert ForensicCategory.GASLIGHTING == "gaslighting"
        assert ForensicCategory.THREAT == "threat"
        assert ForensicCategory.COERCION == "coercion"
        assert ForensicCategory.DECEPTION == "deception"
        assert ForensicCategory.EMOTIONAL_MANIPULATION == "emotional_manipulation"

    def test_forensic_category_count(self):
        """Test that there are exactly 5 categories."""
        assert len(ForensicCategory) == 5


class TestCategoryScore:
    """Test CategoryScore model validation."""

    def test_valid_category_score(self):
        """Test creating a valid category score."""
        score = CategoryScore(
            category=ForensicCategory.GASLIGHTING,
            score=75.5,
            confidence=0.85,
            evidence_count=5,
            key_indicators=["denial pattern", "blame shifting"],
        )
        assert score.category == ForensicCategory.GASLIGHTING
        assert score.score == 75.5
        assert score.confidence == 0.85
        assert score.evidence_count == 5
        assert len(score.key_indicators) == 2

    def test_score_bounds_lower(self):
        """Test that score cannot be below 0."""
        with pytest.raises(ValueError):
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=-1.0,
                confidence=0.5,
                evidence_count=1,
                key_indicators=["test"],
            )

    def test_score_bounds_upper(self):
        """Test that score cannot exceed 100."""
        with pytest.raises(ValueError):
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=101.0,
                confidence=0.5,
                evidence_count=1,
                key_indicators=["test"],
            )

    def test_confidence_bounds_lower(self):
        """Test that confidence cannot be below 0."""
        with pytest.raises(ValueError):
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=50.0,
                confidence=-0.1,
                evidence_count=1,
                key_indicators=["test"],
            )

    def test_confidence_bounds_upper(self):
        """Test that confidence cannot exceed 1."""
        with pytest.raises(ValueError):
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=50.0,
                confidence=1.1,
                evidence_count=1,
                key_indicators=["test"],
            )

    def test_evidence_count_non_negative(self):
        """Test that evidence count cannot be negative."""
        with pytest.raises(ValueError):
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=50.0,
                confidence=0.5,
                evidence_count=-1,
                key_indicators=["test"],
            )


class TestDeceptionAnalysis:
    """Test DeceptionAnalysis model validation."""

    def test_valid_deception_analysis(self):
        """Test creating a valid deception analysis."""
        analysis = DeceptionAnalysis(
            deception_probability=0.75,
            voice_text_consistency=0.4,
            emotional_authenticity=0.3,
            linguistic_markers_count=8,
            behavioral_indicators=["hedging", "distancing", "inconsistent timing"],
        )
        assert analysis.deception_probability == 0.75
        assert analysis.voice_text_consistency == 0.4
        assert analysis.emotional_authenticity == 0.3
        assert analysis.linguistic_markers_count == 8
        assert len(analysis.behavioral_indicators) == 3

    def test_deception_probability_bounds(self):
        """Test deception probability bounds."""
        with pytest.raises(ValueError):
            DeceptionAnalysis(
                deception_probability=1.5,
                voice_text_consistency=0.5,
                emotional_authenticity=0.5,
                linguistic_markers_count=0,
                behavioral_indicators=[],
            )

    def test_voice_text_consistency_bounds(self):
        """Test voice-text consistency bounds."""
        with pytest.raises(ValueError):
            DeceptionAnalysis(
                deception_probability=0.5,
                voice_text_consistency=-0.1,
                emotional_authenticity=0.5,
                linguistic_markers_count=0,
                behavioral_indicators=[],
            )


class TestGaslightingAnalysis:
    """Test GaslightingAnalysis model validation."""

    def test_valid_gaslighting_analysis(self):
        """Test creating a valid gaslighting analysis."""
        analysis = GaslightingAnalysis(
            intensity_score=65.0,
            patterns_detected=["denial", "countering", "trivializing"],
            manipulation_techniques=["reality distortion", "blame shifting"],
            victim_impact_level=RiskLevel.HIGH,
            recurring_phrases=["you're imagining things", "that never happened"],
        )
        assert analysis.intensity_score == 65.0
        assert len(analysis.patterns_detected) == 3
        assert len(analysis.manipulation_techniques) == 2
        assert analysis.victim_impact_level == RiskLevel.HIGH
        assert len(analysis.recurring_phrases) == 2

    def test_intensity_score_bounds(self):
        """Test intensity score bounds."""
        with pytest.raises(ValueError):
            GaslightingAnalysis(
                intensity_score=105.0,
                patterns_detected=[],
                manipulation_techniques=[],
                victim_impact_level=RiskLevel.LOW,
                recurring_phrases=[],
            )


class TestThreatAssessment:
    """Test ThreatAssessment model validation."""

    def test_valid_threat_assessment(self):
        """Test creating a valid threat assessment."""
        assessment = ThreatAssessment(
            threat_level=RiskLevel.HIGH,
            threat_types=["direct", "conditional"],
            immediacy="near-term",
            specificity="specific",
            credibility_score=0.8,
        )
        assert assessment.threat_level == RiskLevel.HIGH
        assert len(assessment.threat_types) == 2
        assert assessment.immediacy == "near-term"
        assert assessment.specificity == "specific"
        assert assessment.credibility_score == 0.8

    def test_credibility_score_bounds(self):
        """Test credibility score bounds."""
        with pytest.raises(ValueError):
            ThreatAssessment(
                threat_level=RiskLevel.LOW,
                threat_types=[],
                immediacy="long-term",
                specificity="vague",
                credibility_score=1.5,
            )


class TestForensicEvidence:
    """Test ForensicEvidence model validation."""

    def test_valid_forensic_evidence(self):
        """Test creating valid forensic evidence."""
        evidence = ForensicEvidence(
            timestamp=125.5,
            evidence_type="gaslighting_pattern",
            description="Denial of victim's experience detected",
            severity=RiskLevel.MODERATE,
            supporting_data={"pattern": "denial", "confidence": 0.9},
        )
        assert evidence.timestamp == 125.5
        assert evidence.evidence_type == "gaslighting_pattern"
        assert evidence.severity == RiskLevel.MODERATE
        assert evidence.supporting_data["confidence"] == 0.9


class TestForensicScoreResult:
    """Test ForensicScoreResult model validation."""

    def test_valid_forensic_score_result(self):
        """Test creating a valid forensic score result."""
        result = ForensicScoreResult(
            analysis_id="forensic-001",
            analyzed_at=datetime.now(),
            audio_duration_seconds=180.5,
            overall_risk_score=72.5,
            overall_risk_level=RiskLevel.HIGH,
            confidence_level=0.85,
            category_scores=[
                CategoryScore(
                    category=ForensicCategory.GASLIGHTING,
                    score=80.0,
                    confidence=0.9,
                    evidence_count=5,
                    key_indicators=["denial"],
                ),
                CategoryScore(
                    category=ForensicCategory.THREAT,
                    score=60.0,
                    confidence=0.8,
                    evidence_count=3,
                    key_indicators=["conditional threat"],
                ),
            ],
            deception_analysis=DeceptionAnalysis(
                deception_probability=0.7,
                voice_text_consistency=0.4,
                emotional_authenticity=0.3,
                linguistic_markers_count=5,
                behavioral_indicators=["hedging"],
            ),
            gaslighting_analysis=GaslightingAnalysis(
                intensity_score=75.0,
                patterns_detected=["denial", "blame_shifting"],
                manipulation_techniques=["reality distortion"],
                victim_impact_level=RiskLevel.HIGH,
                recurring_phrases=["that never happened"],
            ),
            threat_assessment=ThreatAssessment(
                threat_level=RiskLevel.MODERATE,
                threat_types=["conditional"],
                immediacy="near-term",
                specificity="specific",
                credibility_score=0.75,
            ),
            evidence_items=[],
            summary="High risk of gaslighting behavior detected",
            recommendations=["Seek professional consultation"],
            flags=["GASLIGHTING_PATTERN", "DECEPTION_DETECTED"],
        )

        assert result.analysis_id == "forensic-001"
        assert result.overall_risk_score == 72.5
        assert result.overall_risk_level == RiskLevel.HIGH
        assert len(result.category_scores) == 2
        assert result.deception_analysis.deception_probability == 0.7

    def test_overall_risk_score_bounds_lower(self):
        """Test that overall risk score cannot be below 0."""
        with pytest.raises(ValueError):
            ForensicScoreResult(
                analysis_id="test",
                analyzed_at=datetime.now(),
                audio_duration_seconds=60.0,
                overall_risk_score=-5.0,
                overall_risk_level=RiskLevel.LOW,
                confidence_level=0.5,
                category_scores=[],
                deception_analysis=DeceptionAnalysis(
                    deception_probability=0.1,
                    voice_text_consistency=0.9,
                    emotional_authenticity=0.9,
                    linguistic_markers_count=0,
                    behavioral_indicators=[],
                ),
                gaslighting_analysis=GaslightingAnalysis(
                    intensity_score=10.0,
                    patterns_detected=[],
                    manipulation_techniques=[],
                    victim_impact_level=RiskLevel.MINIMAL,
                    recurring_phrases=[],
                ),
                threat_assessment=ThreatAssessment(
                    threat_level=RiskLevel.MINIMAL,
                    threat_types=[],
                    immediacy="long-term",
                    specificity="vague",
                    credibility_score=0.1,
                ),
                evidence_items=[],
                summary="",
                recommendations=[],
                flags=[],
            )

    def test_overall_risk_score_bounds_upper(self):
        """Test that overall risk score cannot exceed 100."""
        with pytest.raises(ValueError):
            ForensicScoreResult(
                analysis_id="test",
                analyzed_at=datetime.now(),
                audio_duration_seconds=60.0,
                overall_risk_score=105.0,
                overall_risk_level=RiskLevel.CRITICAL,
                confidence_level=0.5,
                category_scores=[],
                deception_analysis=DeceptionAnalysis(
                    deception_probability=0.9,
                    voice_text_consistency=0.1,
                    emotional_authenticity=0.1,
                    linguistic_markers_count=10,
                    behavioral_indicators=["all indicators"],
                ),
                gaslighting_analysis=GaslightingAnalysis(
                    intensity_score=95.0,
                    patterns_detected=["all"],
                    manipulation_techniques=["all"],
                    victim_impact_level=RiskLevel.CRITICAL,
                    recurring_phrases=["all"],
                ),
                threat_assessment=ThreatAssessment(
                    threat_level=RiskLevel.CRITICAL,
                    threat_types=["all"],
                    immediacy="immediate",
                    specificity="detailed",
                    credibility_score=0.95,
                ),
                evidence_items=[],
                summary="",
                recommendations=[],
                flags=[],
            )

    def test_confidence_level_bounds(self):
        """Test confidence level bounds."""
        with pytest.raises(ValueError):
            ForensicScoreResult(
                analysis_id="test",
                analyzed_at=datetime.now(),
                audio_duration_seconds=60.0,
                overall_risk_score=50.0,
                overall_risk_level=RiskLevel.MODERATE,
                confidence_level=1.5,  # Invalid
                category_scores=[],
                deception_analysis=DeceptionAnalysis(
                    deception_probability=0.5,
                    voice_text_consistency=0.5,
                    emotional_authenticity=0.5,
                    linguistic_markers_count=2,
                    behavioral_indicators=[],
                ),
                gaslighting_analysis=GaslightingAnalysis(
                    intensity_score=50.0,
                    patterns_detected=[],
                    manipulation_techniques=[],
                    victim_impact_level=RiskLevel.MODERATE,
                    recurring_phrases=[],
                ),
                threat_assessment=ThreatAssessment(
                    threat_level=RiskLevel.MODERATE,
                    threat_types=[],
                    immediacy="long-term",
                    specificity="vague",
                    credibility_score=0.5,
                ),
                evidence_items=[],
                summary="",
                recommendations=[],
                flags=[],
            )
