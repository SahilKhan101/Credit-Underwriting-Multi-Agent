"""
Credit evaluation metrics.
All functions accept numpy arrays or plain Python lists.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def gini_coefficient(y_true: list[int], scores: list[float]) -> float:
    """
    Gini = 2 * AUC - 1  (discriminatory power of the composite score).
    y_true: 1 = bad (rejected/default), 0 = good (approved).
    scores: composite credit score 0–1000 (higher = better credit → INVERT for AUC).
    """
    y = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    # Higher score → lower default probability → invert for AUC
    auc = roc_auc_score(y, -s)
    return round(2 * auc - 1, 4)


def ks_statistic(y_true: list[int], scores: list[float]) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between good and bad score CDFs.
    """
    y  = np.array(y_true, dtype=int)
    s  = np.array(scores, dtype=float)
    goods = np.sort(s[y == 0])
    bads  = np.sort(s[y == 1])

    if len(goods) == 0 or len(bads) == 0:
        return float("nan")

    all_scores = np.sort(np.unique(s))
    cdf_good   = np.searchsorted(goods, all_scores, side="right") / len(goods)
    cdf_bad    = np.searchsorted(bads,  all_scores, side="right") / len(bads)
    return round(float(np.max(np.abs(cdf_good - cdf_bad))), 4)


def approval_rate(recommendations: list[str]) -> float:
    approved = sum(
        1 for r in recommendations
        if r in ("AUTO_APPROVE", "APPROVE_WITH_CONDITIONS")
    )
    return round(approved / max(len(recommendations), 1), 4)


def auto_decision_rate(recommendations: list[str]) -> float:
    auto = sum(
        1 for r in recommendations
        if r in ("AUTO_APPROVE", "AUTO_REJECT")
    )
    return round(auto / max(len(recommendations), 1), 4)


def decision_agreement_rate(predicted: list[str], expected: list[str]) -> float:
    """How often the model matches ground-truth labels (eval set only)."""
    matches = sum(p == e for p, e in zip(predicted, expected))
    return round(matches / max(len(predicted), 1), 4)


def foir_compliance_rate(
    recommendations: list[str],
    foiars: list[float],
    limit: float = 0.50,
) -> float:
    """
    Fraction of approved cases where FOIR ≤ limit (policy adherence check).
    A value < 1.0 means we approved borrowers who violate the FOIR cap.
    """
    approved_foiars = [
        f for rec, f in zip(recommendations, foiars)
        if rec in ("AUTO_APPROVE", "APPROVE_WITH_CONDITIONS")
    ]
    if not approved_foiars:
        return float("nan")
    compliant = sum(1 for f in approved_foiars if f <= limit)
    return round(compliant / len(approved_foiars), 4)


def bad_rate_by_band(
    scores: list[float],
    y_true: list[int],
    bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    Default rate within each score band.
    Monotonically decreasing bad rate = well-calibrated score.
    """
    if bands is None:
        bands = [(0, 550), (550, 650), (650, 750), (750, 1001)]

    s = np.array(scores, dtype=float)
    y = np.array(y_true, dtype=int)
    result = {}
    for lo, hi in bands:
        mask = (s >= lo) & (s < hi)
        if mask.sum() == 0:
            result[f"{lo:.0f}-{hi:.0f}"] = float("nan")
        else:
            result[f"{lo:.0f}-{hi:.0f}"] = round(float(y[mask].mean()), 4)
    return result


def summary_report(
    scores: list[float],
    recommendations: list[str],
    y_true: list[int],
    foiars: list[float],
    expected: list[str] | None = None,
) -> dict:
    """Return all metrics as a single dict for easy display."""
    report: dict = {
        "n": len(scores),
        "approval_rate":      approval_rate(recommendations),
        "auto_decision_rate": auto_decision_rate(recommendations),
        "gini_coefficient":   gini_coefficient(y_true, scores),
        "ks_statistic":       ks_statistic(y_true, scores),
        "foir_compliance_rate": foir_compliance_rate(recommendations, foiars),
        "bad_rate_by_band":   bad_rate_by_band(scores, y_true),
    }
    if expected:
        report["decision_agreement_rate"] = decision_agreement_rate(recommendations, expected)
    return report
