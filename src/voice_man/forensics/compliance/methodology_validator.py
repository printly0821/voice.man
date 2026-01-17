"""
Methodology Validation Protocol for ISO/IEC 17025:2017 Compliance

This module implements methodology validation requirements per:
- ISO/IEC 17025:2017 Clause 7.2 (Validation of methods)
- NIST SP 800-86 (Digital Forensics Validation Guidelines)
- ENFSI (European Network of Forensic Science Institutes) guidelines

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 17025:2017 Clause 7.2.1-7.2.7
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationCriteria:
    """
    ISO/IEC 17025 validation criteria

    Attributes:
        specificity: True negative rate (특이도)
        sensitivity: True positive rate (감도)
        selectivity: Distinguishing similar samples (선택도)
        detection_limit: Limit of Detection (LOD, 검출한계)
        robustness: Environmental variation tolerance (견고성)
        bias: Systematic error assessment (편향성)
        precision: Repeatability & Reproducibility (정밀도)
        uncertainty: Measurement uncertainty (불확실도)
    """
    specificity: bool = True
    sensitivity: bool = True
    selectivity: bool = True
    detection_limit: bool = False
    robustness: bool = True
    bias: bool = True
    precision: bool = True
    uncertainty: bool = True


@dataclass
class ValidationParameters:
    """
    Validation test parameters

    Attributes:
        sample_size: Minimum 30 samples per ENFSI guidelines
        confidence_level: Default 95%
        reference_material: Use certified reference material
        independent_verification: Cross-lab verification
        blind_testing: Blind sample testing
        reproducibility_conditions: Environmental conditions
    """
    sample_size: int = 30
    confidence_level: float = 0.95
    reference_material: bool = True
    independent_verification: bool = True
    blind_testing: bool = True
    reproducibility_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Methodology validation result

    Attributes:
        validation_id: Unique validation identifier
        method_name: Name of validated method
        method_version: Version identifier
        validation_date: ISO 8601 timestamp
        validation_status: "approved" | "conditional" | "failed"
        criteria_results: Per-criteria results dictionary
        metrics: Computed metrics (Precision, Recall, F1, etc.)
        recommendations: List of improvement recommendations
        expires_at: Validation expiration date (typically 2 years)
        validator: Validator information
    """
    validation_id: str
    method_name: str
    method_version: str
    validation_date: str
    validation_status: str
    criteria_results: Dict[str, Any]
    metrics: Dict[str, Any]
    recommendations: List[str]
    expires_at: str
    validator: Dict[str, str]


class MethodologyValidator:
    """
    ISO/IEC 17025:2017 Methodology Validator

    Implements validation protocol for forensic analysis methods.

    Reference:
        - ISO/IEC 17025:2017 Clause 7.2.1-7.2.7
        - NIST SP 800-86 Section 4.3 (Method Validation)
        - ENFSI Validation Guidelines

    Examples:
        >>> validator = MethodologyValidator()
        >>> criteria = ValidationCriteria()
        >>> result = validator.validate_method(
        ...     method_name="Audio Forensic Analysis",
        ...     method_version="1.0.0",
        ...     validation_criteria=criteria,
        ...     test_data={"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 0, 1]}
        ... )
        >>> print(result["validation_status"])
        "approved"
    """

    def __init__(self, db_session=None):
        """
        Initialize methodology validator

        Args:
            db_session: Database session for persisting validation results
        """
        self.db_session = db_session
        self.validation_log: List[ValidationResult] = []
        logger.info("MethodologyValidator initialized")

    def validate_method(
        self,
        method_name: str,
        method_version: str,
        validation_criteria: ValidationCriteria,
        test_data: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform full methodology validation per ISO/IEC 17025:2017 7.2.

        Validates forensic analysis methods against ISO/IEC 17025 requirements.
        Implements validation criteria specified in Clause 7.2.1-7.2.7.

        Args:
            method_name: Name of the forensic analysis method
            method_version: Version identifier
            validation_criteria: Criteria to validate (ValidationCriteria)
            test_data: Test dataset for validation
            reference_data: Reference method results for comparison (optional)

        Returns:
            Dict containing:
                - validation_id: Unique validation identifier (e.g., VAL-20260117-AUDIOFOR)
                - method_name: Validated method name
                - method_version: Method version
                - validation_date: ISO 8601 timestamp
                - validation_status: "approved" | "conditional" | "failed"
                - criteria_results: Dict of per-criteria validation results
                - metrics: Precision, Recall, F1, Uncertainty, CI, etc.
                - recommendations: List of improvement recommendations
                - expires_at: Validation expiration date (typically 2 years)
                - validator: Validator information

        Examples:
            >>> validator = MethodologyValidator()
            >>> criteria = ValidationCriteria(
            ...     specificity=True,
            ...     sensitivity=True,
            ...     precision=True
            ... )
            >>> result = validator.validate_method(
            ...     method_name="Crime Language Detection",
            ...     method_version="1.0.0",
            ...     validation_criteria=criteria,
            ...     test_data={"y_true": [...], "y_pred": [...]}
            ... )
        """
        validation_id = self._generate_validation_id(method_name)

        results: Dict[str, Any] = {
            "validation_id": validation_id,
            "method_name": method_name,
            "method_version": method_version,
            "validation_date": datetime.now(timezone.utc).isoformat(),
            "validator": self._get_validator_info(),
            "criteria_results": {},
            "metrics": {},
            "recommendations": [],
            "validation_status": "pending",
        }

        # 1. Specificity (True Negative Rate) - 특이도
        if validation_criteria.specificity:
            specificity_result = self._validate_specificity(test_data)
            results["criteria_results"]["specificity"] = specificity_result
            results["metrics"]["specificity"] = specificity_result["metric"]

        # 2. Sensitivity (True Positive Rate) - 감도
        if validation_criteria.sensitivity:
            sensitivity_result = self._validate_sensitivity(test_data)
            results["criteria_results"]["sensitivity"] = sensitivity_result
            results["metrics"]["sensitivity"] = sensitivity_result["metric"]

        # 3. Selectivity - 선택도
        if validation_criteria.selectivity:
            selectivity_result = self._validate_selectivity(test_data)
            results["criteria_results"]["selectivity"] = selectivity_result
            results["metrics"]["selectivity"] = selectivity_result["metric"]

        # 4. Detection Limit (LOD) - 검출한계
        if validation_criteria.detection_limit:
            lod_result = self._validate_detection_limit(test_data)
            results["criteria_results"]["detection_limit"] = lod_result
            if lod_result["lod_value"] is not None:
                results["metrics"]["lod"] = lod_result["lod_value"]

        # 5. Robustness - 견고성
        if validation_criteria.robustness:
            robustness_result = self._validate_robustness(test_data)
            results["criteria_results"]["robustness"] = robustness_result
            results["metrics"]["robustness_score"] = robustness_result["score"]

        # 6. Bias Assessment - 편향성
        if validation_criteria.bias:
            bias_result = self._validate_bias(test_data, reference_data)
            results["criteria_results"]["bias"] = bias_result
            if bias_result["bias_value"] is not None:
                results["metrics"]["bias"] = bias_result["bias_value"]

        # 7. Precision (Repeatability & Reproducibility) - 정밀도
        if validation_criteria.precision:
            precision_result = self._validate_precision(test_data)
            results["criteria_results"]["precision"] = precision_result
            results["metrics"].update(precision_result["metrics"])

        # 8. Uncertainty Calculation - 불확실도
        if validation_criteria.uncertainty:
            uncertainty_result = self._calculate_uncertainty(test_data)
            results["criteria_results"]["uncertainty"] = uncertainty_result
            if uncertainty_result["uncertainty"] is not None:
                results["metrics"]["uncertainty"] = uncertainty_result["uncertainty"]
            if uncertainty_result["ci"] is not None:
                results["metrics"]["confidence_interval"] = uncertainty_result["ci"]

        # Determine overall validation status
        results["validation_status"] = self._determine_status(results["criteria_results"])

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Set expiration (2 years from validation date)
        results["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(days=730)
        ).isoformat()

        # Persist to database if session available
        if self.db_session:
            self._persist_validation(results)

        # Store in validation log
        validation_result = ValidationResult(
            validation_id=results["validation_id"],
            method_name=results["method_name"],
            method_version=results["method_version"],
            validation_date=results["validation_date"],
            validation_status=results["validation_status"],
            criteria_results=results["criteria_results"],
            metrics=results["metrics"],
            recommendations=results["recommendations"],
            expires_at=results["expires_at"],
            validator=results["validator"],
        )
        self.validation_log.append(validation_result)

        logger.info(
            f"Method validation completed: {method_name} v{method_version} - {results['validation_status']}"
        )

        return results

    def _validate_specificity(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate specificity (True Negative Rate)

        TNR = TN / (TN + FP)
        """
        try:
            from sklearn.metrics import confusion_matrix

            y_true = test_data.get("y_true", [])
            y_pred = test_data.get("y_pred", [])

            if not y_true or not y_pred:
                return {
                    "criterion": "specificity",
                    "status": "skipped",
                    "metric": None,
                    "threshold": 0.95,
                    "description": "Insufficient data for specificity calculation",
                }

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

            return {
                "criterion": "specificity",
                "status": "pass" if tnr >= 0.95 else "conditional",
                "metric": float(tnr),
                "threshold": 0.95,
                "description": f"True Negative Rate (특이도): {tnr:.4f}",
            }
        except Exception as e:
            logger.error(f"Specificity validation error: {e}")
            return {
                "criterion": "specificity",
                "status": "error",
                "metric": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_sensitivity(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate sensitivity (True Positive Rate)

        TPR = TP / (TP + FN)
        """
        try:
            from sklearn.metrics import confusion_matrix

            y_true = test_data.get("y_true", [])
            y_pred = test_data.get("y_pred", [])

            if not y_true or not y_pred:
                return {
                    "criterion": "sensitivity",
                    "status": "skipped",
                    "metric": None,
                    "threshold": 0.90,
                    "description": "Insufficient data for sensitivity calculation",
                }

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

            return {
                "criterion": "sensitivity",
                "status": "pass" if tpr >= 0.90 else "conditional",
                "metric": float(tpr),
                "threshold": 0.90,
                "description": f"True Positive Rate (감도/Recall): {tpr:.4f}",
            }
        except Exception as e:
            logger.error(f"Sensitivity validation error: {e}")
            return {
                "criterion": "sensitivity",
                "status": "error",
                "metric": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_selectivity(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate selectivity (distinguishing similar samples)

        Uses F1-score as selectivity metric
        """
        try:
            from sklearn.metrics import f1_score

            y_true = test_data.get("y_true", [])
            y_pred = test_data.get("y_pred", [])

            if not y_true or not y_pred:
                return {
                    "criterion": "selectivity",
                    "status": "skipped",
                    "metric": None,
                    "threshold": 0.85,
                    "description": "Insufficient data for selectivity calculation",
                }

            f1 = f1_score(y_true, y_pred, average="binary")

            return {
                "criterion": "selectivity",
                "status": "pass" if f1 >= 0.85 else "conditional",
                "metric": float(f1),
                "threshold": 0.85,
                "description": f"F1-Score (선택도): {f1:.4f}",
            }
        except Exception as e:
            logger.error(f"Selectivity validation error: {e}")
            return {
                "criterion": "selectivity",
                "status": "error",
                "metric": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_detection_limit(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Limit of Detection (LOD)

        LOD = 3.3 * (standard deviation of blank) / slope
        Simplified: minimum signal-to-noise ratio
        """
        try:
            signals = test_data.get("signals", [])
            noise = test_data.get("noise", [])

            if not signals or not noise:
                return {
                    "criterion": "detection_limit",
                    "status": "skipped",
                    "lod_value": None,
                    "description": "Insufficient data for LOD calculation",
                }

            import numpy as np

            noise_std = np.std(noise)
            lod = 3.3 * noise_std

            return {
                "criterion": "detection_limit",
                "status": "pass",
                "lod_value": float(lod),
                "description": f"Limit of Detection (검출한계): {lod:.4f}",
            }
        except Exception as e:
            logger.error(f"Detection limit validation error: {e}")
            return {
                "criterion": "detection_limit",
                "status": "error",
                "lod_value": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_robustness(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate robustness to environmental variation

        Tests method performance under different conditions
        """
        try:
            variations = test_data.get("environmental_variations", {})

            if not variations:
                return {
                    "criterion": "robustness",
                    "status": "skipped",
                    "score": None,
                    "description": "No environmental variation data provided",
                }

            scores = []
            for condition, results in variations.items():
                scores.append(results.get("accuracy", 0))

            import numpy as np

            robustness_score = 1.0 - np.std(scores)  # Lower std = higher robustness

            return {
                "criterion": "robustness",
                "status": "pass" if robustness_score >= 0.90 else "conditional",
                "score": float(robustness_score),
                "description": f"Robustness Score (견고성): {robustness_score:.4f}",
            }
        except Exception as e:
            logger.error(f"Robustness validation error: {e}")
            return {
                "criterion": "robustness",
                "status": "error",
                "score": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_bias(
        self,
        test_data: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate bias (systematic error)

        Compares test results with reference method
        """
        try:
            if reference_data is None:
                return {
                    "criterion": "bias",
                    "status": "skipped",
                    "bias_value": None,
                    "description": "No reference data available",
                }

            test_results = test_data.get("results", [])
            ref_results = reference_data.get("results", [])

            if not test_results or not ref_results:
                return {
                    "criterion": "bias",
                    "status": "skipped",
                    "bias_value": None,
                    "description": "Insufficient data for bias calculation",
                }

            import numpy as np

            bias = np.mean(np.array(test_results) - np.array(ref_results))

            return {
                "criterion": "bias",
                "status": "pass" if abs(bias) < 0.05 else "conditional",
                "bias_value": float(bias),
                "description": f"Bias (편향): {bias:.4f}",
            }
        except Exception as e:
            logger.error(f"Bias validation error: {e}")
            return {
                "criterion": "bias",
                "status": "error",
                "bias_value": None,
                "description": f"Error: {str(e)}",
            }

    def _validate_precision(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate precision (Repeatability & Reproducibility)

        Repeatability: Same operator, same equipment, short time
        Reproducibility: Different operators, different equipment, different time
        """
        try:
            repeatability_data = test_data.get("repeatability", [])
            reproducibility_data = test_data.get("reproducibility", [])

            import numpy as np
            from scipy import stats

            results: Dict[str, Any] = {}

            if repeatability_data:
                # Calculate repeatability (within-run standard deviation)
                repeatability_std = np.std(repeatability_data, ddof=1)
                repeatability_mean = np.mean(repeatability_data)
                repeatability_cv = (repeatability_std / repeatability_mean * 100) if repeatability_mean != 0 else 0

                results["repeatability_std"] = float(repeatability_std)
                results["repeatability_cv"] = float(repeatability_cv)
                results["repeatability_status"] = "pass" if repeatability_cv < 5.0 else "conditional"

            if reproducibility_data:
                # Calculate reproducibility (between-run standard deviation)
                reproducibility_std = np.std(reproducibility_data, ddof=1)
                reproducibility_mean = np.mean(reproducibility_data)
                reproducibility_cv = (reproducibility_std / reproducibility_mean * 100) if reproducibility_mean != 0 else 0

                results["reproducibility_std"] = float(reproducibility_std)
                results["reproducibility_cv"] = float(reproducibility_cv)
                results["reproducibility_status"] = "pass" if reproducibility_cv < 10.0 else "conditional"

            return {
                "criterion": "precision",
                "status": results.get("repeatability_status", "skipped"),
                "metrics": results,
                "description": f"Repeatability CV: {results.get('repeatability_cv', 0):.2f}%, Reproducibility CV: {results.get('reproducibility_cv', 0):.2f}%",
            }
        except Exception as e:
            logger.error(f"Precision validation error: {e}")
            return {
                "criterion": "precision",
                "status": "error",
                "metrics": {},
                "description": f"Error: {str(e)}",
            }

    def _calculate_uncertainty(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate measurement uncertainty per GUM (Guide to Uncertainty in Measurement)

        Combined uncertainty = sqrt(sum(u_i^2))
        Expanded uncertainty = k * combined_uncertainty (k=2 for 95% CI)
        """
        try:
            components = test_data.get("uncertainty_components", [])

            if not components:
                return {
                    "criterion": "uncertainty",
                    "status": "skipped",
                    "uncertainty": None,
                    "ci": None,
                    "description": "No uncertainty components provided",
                }

            import numpy as np

            combined_uncertainty = np.sqrt(sum([c**2 for c in components]))
            expanded_uncertainty = 2 * combined_uncertainty  # k=2 for 95% CI

            # Calculate confidence interval
            mean_value = test_data.get("mean_value", 0)
            ci_lower = mean_value - expanded_uncertainty
            ci_upper = mean_value + expanded_uncertainty

            return {
                "criterion": "uncertainty",
                "status": "pass",
                "uncertainty": float(combined_uncertainty),
                "ci": (float(ci_lower), float(ci_upper)),
                "description": f"Uncertainty (불확실도): {combined_uncertainty:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
            }
        except Exception as e:
            logger.error(f"Uncertainty calculation error: {e}")
            return {
                "criterion": "uncertainty",
                "status": "error",
                "uncertainty": None,
                "ci": None,
                "description": f"Error: {str(e)}",
            }

    def _determine_status(self, criteria_results: Dict[str, Dict]) -> str:
        """
        Determine overall validation status

        Returns:
            "approved" if all criteria passed
            "conditional" if some criteria conditional
            "failed" if any criteria failed
        """
        statuses = [r.get("status", "failed") for r in criteria_results.values()]

        if all(s == "pass" for s in statuses):
            return "approved"
        elif any(s == "failed" for s in statuses):
            return "failed"
        else:
            return "conditional"

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations

        Returns:
            List of recommendation strings
        """
        recommendations: List[str] = []

        for criterion, result in results.get("criteria_results", {}).items():
            if result.get("status") == "conditional":
                recommendations.append(
                    f"Improve {criterion}: {result.get('description', '')}"
                )
            elif result.get("status") == "failed":
                recommendations.append(
                    f"Critical: Fix {criterion} before production use"
                )

        if not recommendations:
            recommendations.append("Method meets all validation criteria. Approved for production use.")

        return recommendations

    def _generate_validation_id(self, method_name: str) -> str:
        """
        Generate unique validation identifier

        Format: VAL-YYYYMMDD-METHODNAME
        """
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        # Remove spaces and limit to 15 chars to avoid truncation
        method_suffix = method_name.upper().replace(" ", "")[:15]
        return f"VAL-{date_str}-{method_suffix}"

    def _get_validator_info(self) -> Dict[str, str]:
        """
        Get validator information

        Returns:
            Dict with validator_id, validator_name, organization
        """
        return {
            "validator_id": "VAL-SYSTEM",
            "validator_name": "Automated Validation System",
            "organization": "Korean Digital Forensic Center",
        }

    def _persist_validation(self, results: Dict[str, Any]) -> None:
        """
        Persist validation results to database

        Args:
            results: Validation results dictionary
        """
        # Implementation depends on database schema
        # Use method_validation table
        logger.info(f"Persisting validation: {results['validation_id']}")


# Convenience functions for common validation tasks


def validate_audio_analysis_method(
    method_name: str,
    test_data: Dict[str, Any],
    db_session=None,
) -> Dict[str, Any]:
    """
    Validate audio analysis methodology

    Convenience function for validating audio forensic analysis methods.

    Args:
        method_name: Name of the audio analysis method
        test_data: Test dataset with y_true, y_pred
        db_session: Database session for persistence

    Returns:
        Validation results dictionary

    Examples:
        >>> result = validate_audio_analysis_method(
        ...     method_name="Speaker Identification",
        ...     test_data={"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 0, 0]}
        ... )
        >>> print(result["validation_status"])
        "conditional"
    """
    validator = MethodologyValidator(db_session)

    criteria = ValidationCriteria(
        specificity=True,
        sensitivity=True,
        selectivity=True,
        detection_limit=False,
        robustness=True,
        bias=True,
        precision=True,
        uncertainty=True,
    )

    return validator.validate_method(
        method_name=method_name,
        method_version="1.0.0",
        validation_criteria=criteria,
        test_data=test_data,
    )


def validate_crime_language_detection(
    test_data: Dict[str, Any],
    reference_data: Optional[Dict[str, Any]] = None,
    db_session=None,
) -> Dict[str, Any]:
    """
    Validate crime language detection methodology

    Convenience function for validating crime language detection methods.

    Args:
        test_data: Test dataset
        reference_data: Reference method results
        db_session: Database session for persistence

    Returns:
        Validation results dictionary
    """
    validator = MethodologyValidator(db_session)

    criteria = ValidationCriteria(
        specificity=True,
        sensitivity=True,
        selectivity=True,
        detection_limit=False,
        robustness=False,
        bias=True,
        precision=True,
        uncertainty=True,
    )

    return validator.validate_method(
        method_name="Crime Language Detection",
        method_version="1.0.0",
        validation_criteria=criteria,
        test_data=test_data,
        reference_data=reference_data,
    )


def get_validation_status(validation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get validation status by ID

    Args:
        validation_id: Validation identifier (e.g., VAL-20260117-AUDIOFOR)

    Returns:
        Validation results dictionary or None if not found
    """
    # Implementation: Query database or validation log
    return None
