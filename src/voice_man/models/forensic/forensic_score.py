"""
Forensic Score Models
SPEC-FORENSIC-001 Phase 2-D: Pydantic models for forensic scoring system

These models define the data structures for the integrated forensic scoring
system that combines all analysis results into a comprehensive risk assessment.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """
    Risk level classification based on score ranges.

    Score Ranges:
        MINIMAL: 0-20
        LOW: 21-40
        MODERATE: 41-60
        HIGH: 61-80
        CRITICAL: 81-100
    """

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ForensicCategory(str, Enum):
    """
    Forensic analysis categories for scoring.

    Categories:
        GASLIGHTING: Psychological manipulation patterns
        THREAT: Direct or indirect threat expressions
        COERCION: Forced compliance behaviors
        DECEPTION: Dishonesty and misleading patterns
        EMOTIONAL_MANIPULATION: Emotional control tactics
    """

    GASLIGHTING = "gaslighting"
    THREAT = "threat"
    COERCION = "coercion"
    DECEPTION = "deception"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"


class CategoryScore(BaseModel):
    """
    Score for a specific forensic category.

    Attributes:
        category: The forensic category being scored.
        score: Risk score from 0.0 to 100.0.
        confidence: Confidence level of the score (0.0-1.0).
        evidence_count: Number of evidence items supporting this score.
        key_indicators: List of key indicators detected.
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    category: ForensicCategory = Field(..., description="Forensic category being scored")
    score: float = Field(..., ge=0.0, le=100.0, description="Risk score for this category (0-100)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level of the score (0.0-1.0)"
    )
    evidence_count: int = Field(..., ge=0, description="Number of supporting evidence items")
    key_indicators: List[str] = Field(
        default_factory=list, description="Key indicators detected for this category"
    )


class DeceptionAnalysis(BaseModel):
    """
    Deception analysis result combining voice and text analysis.

    Weights for calculation:
        - Text-voice inconsistency: 35%
        - Deception language markers: 30%
        - Emotional inconsistency: 25%
        - Stress patterns: 10%

    Attributes:
        deception_probability: Overall probability of deception (0.0-1.0).
        voice_text_consistency: Consistency between voice and text (0.0-1.0, higher=more consistent).
        emotional_authenticity: Authenticity of emotional expression (0.0-1.0).
        linguistic_markers_count: Count of deception linguistic markers.
        behavioral_indicators: List of behavioral indicators of deception.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    deception_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Overall probability of deception (0.0-1.0)"
    )
    voice_text_consistency: float = Field(
        ..., ge=0.0, le=1.0, description="Voice-text consistency score (higher=more consistent)"
    )
    emotional_authenticity: float = Field(
        ..., ge=0.0, le=1.0, description="Emotional authenticity score (higher=more authentic)"
    )
    linguistic_markers_count: int = Field(
        ..., ge=0, description="Count of deception linguistic markers"
    )
    behavioral_indicators: List[str] = Field(
        default_factory=list, description="Behavioral indicators of deception"
    )


class GaslightingAnalysis(BaseModel):
    """
    Gaslighting analysis result.

    Weights for calculation:
        - Gaslighting pattern detection: 40%
        - Emotional manipulation indicators: 30%
        - Recurring patterns: 20%
        - Cross-validation inconsistency: 10%

    Attributes:
        intensity_score: Overall gaslighting intensity (0.0-100.0).
        patterns_detected: List of gaslighting patterns detected.
        manipulation_techniques: Specific manipulation techniques identified.
        victim_impact_level: Assessed impact level on victim.
        recurring_phrases: Phrases that recur as manipulation tools.
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    intensity_score: float = Field(
        ..., ge=0.0, le=100.0, description="Gaslighting intensity score (0-100)"
    )
    patterns_detected: List[str] = Field(
        default_factory=list, description="Gaslighting patterns detected"
    )
    manipulation_techniques: List[str] = Field(
        default_factory=list, description="Manipulation techniques identified"
    )
    victim_impact_level: RiskLevel = Field(..., description="Assessed victim impact level")
    recurring_phrases: List[str] = Field(
        default_factory=list, description="Recurring manipulation phrases"
    )


class ThreatAssessment(BaseModel):
    """
    Threat assessment result.

    Assessment criteria:
        - Direct threat expressions
        - Conditional threats
        - Implicit/veiled threats
        - Stress pattern correlation

    Attributes:
        threat_level: Overall threat level.
        threat_types: Types of threats detected.
        immediacy: How immediate the threat is (immediate, near-term, long-term).
        specificity: How specific the threat is (vague, specific, detailed).
        credibility_score: Credibility assessment of the threat (0.0-1.0).
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    threat_level: RiskLevel = Field(..., description="Overall threat level")
    threat_types: List[str] = Field(default_factory=list, description="Types of threats detected")
    immediacy: str = Field(..., description="Threat immediacy (immediate, near-term, long-term)")
    specificity: str = Field(..., description="Threat specificity (vague, specific, detailed)")
    credibility_score: float = Field(
        ..., ge=0.0, le=1.0, description="Threat credibility score (0.0-1.0)"
    )


class ForensicEvidence(BaseModel):
    """
    Individual forensic evidence item.

    Attributes:
        timestamp: Audio timestamp in seconds where evidence was found.
        evidence_type: Type of evidence (e.g., gaslighting_pattern, threat_expression).
        description: Human-readable description of the evidence.
        severity: Severity level of this evidence.
        supporting_data: Additional supporting data as key-value pairs.
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    timestamp: float = Field(..., ge=0.0, description="Audio timestamp in seconds")
    evidence_type: str = Field(..., description="Type of evidence")
    description: str = Field(..., description="Human-readable description")
    severity: RiskLevel = Field(..., description="Severity level of this evidence")
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional supporting data"
    )


class ForensicScoreResult(BaseModel):
    """
    Complete forensic scoring result.

    This model aggregates all analysis results into a comprehensive
    forensic assessment with risk scores, detailed analyses, and
    actionable recommendations.

    Overall Score Calculation (weighted average):
        - Gaslighting: 25%
        - Threat: 25%
        - Coercion: 20%
        - Deception: 20%
        - Emotional Manipulation: 10%

    Risk Level Mapping:
        - 0-20: MINIMAL
        - 21-40: LOW
        - 41-60: MODERATE
        - 61-80: HIGH
        - 81-100: CRITICAL

    Confidence Calculation:
        confidence = (
            avg_category_confidence * 0.4 +
            evidence_count_factor * 0.3 +
            cross_validation_consistency * 0.3
        )

    Attributes:
        analysis_id: Unique identifier for this analysis.
        analyzed_at: Timestamp when analysis was performed.
        audio_duration_seconds: Duration of analyzed audio in seconds.
        overall_risk_score: Combined risk score (0-100).
        overall_risk_level: Risk level classification.
        confidence_level: Overall confidence in the analysis (0.0-1.0).
        category_scores: Scores for each forensic category.
        deception_analysis: Detailed deception analysis.
        gaslighting_analysis: Detailed gaslighting analysis.
        threat_assessment: Detailed threat assessment.
        evidence_items: List of forensic evidence items.
        summary: Human-readable summary of findings.
        recommendations: List of recommended actions.
        flags: Warning flags for critical findings.
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    # Metadata
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    analyzed_at: datetime = Field(..., description="Timestamp of analysis")
    audio_duration_seconds: float = Field(..., gt=0.0, description="Audio duration in seconds")

    # Overall Scores
    overall_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Combined risk score (0-100)"
    )
    overall_risk_level: RiskLevel = Field(..., description="Risk level classification")
    confidence_level: float = Field(
        ..., ge=0.0, le=1.0, description="Overall analysis confidence (0.0-1.0)"
    )

    # Category Scores
    category_scores: List[CategoryScore] = Field(
        default_factory=list, description="Scores for each forensic category"
    )

    # Detailed Analyses
    deception_analysis: DeceptionAnalysis = Field(..., description="Detailed deception analysis")
    gaslighting_analysis: GaslightingAnalysis = Field(
        ..., description="Detailed gaslighting analysis"
    )
    threat_assessment: ThreatAssessment = Field(..., description="Detailed threat assessment")

    # Evidence
    evidence_items: List[ForensicEvidence] = Field(
        default_factory=list, description="List of forensic evidence items"
    )

    # Summary and Recommendations
    summary: str = Field(..., description="Human-readable summary of findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    flags: List[str] = Field(
        default_factory=list, description="Warning flags for critical findings"
    )
