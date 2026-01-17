"""
Performance Metrics for Forensic Classification Validation.

Implements precision, recall, F1 score, and confusion matrix
for forensic analysis quality assessment.

TAG: [FORENSIC-EVIDENCE-001]
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    confusion_matrix as sklearn_confusion_matrix,
    accuracy_score,
)


class PerformanceMetrics:
    """
    Performance metrics calculator for forensic classification validation.

    Provides precision, recall, F1 score, and confusion matrix
    for evaluating forensic analysis accuracy.
    """

    def __init__(self):
        """Initialize performance metrics calculator."""
        pass

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision score.

        Precision = TP / (TP + FP)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            float: Precision score (0.0 to 1.0)
        """
        return float(precision_score(y_true, y_pred, zero_division=0.0))

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall score.

        Recall = TP / (TP + FN)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            float: Recall score (0.0 to 1.0)
        """
        return float(recall_score(y_true, y_pred, zero_division=0.0))

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate F1 score (harmonic mean of precision and recall).

        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            float: F1 score (0.0 to 1.0)
        """
        return float(sklearn_f1_score(y_true, y_pred, zero_division=0.0))

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Generate confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dict with TP, TN, FP, FN counts
        """
        # For binary classification
        cm = sklearn_confusion_matrix(y_true, y_pred)

        # Extract values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            if len(cm) == 1:
                if y_true[0] == 1:
                    # All true positives or false negatives
                    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                    tn = 0
                    fp = 0
                else:
                    # All true negatives or false positives
                    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                    tp = 0
                    fn = 0
            else:
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.

        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        return float(accuracy_score(y_true, y_pred))

    def multiclass_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics for multiclass classification.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dict containing precision, recall, f1_score, and support per class
        """
        from sklearn.metrics import classification_report

        # Get classification report as dictionary
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)

        # Extract macro averages
        metrics = {
            "precision": report.get("macro avg", {}).get("precision", 0.0),
            "recall": report.get("macro avg", {}).get("recall", 0.0),
            "f1_score": report.get("macro avg", {}).get("f1-score", 0.0),
            "support": report.get("macro avg", {}).get("support", 0),
        }

        return metrics

    def get_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Get all metrics at once.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dict containing all metrics
        """
        return {
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1_score": self.f1_score(y_true, y_pred),
            "accuracy": self.accuracy(y_true, y_pred),
            "confusion_matrix": self.confusion_matrix(y_true, y_pred),
        }
