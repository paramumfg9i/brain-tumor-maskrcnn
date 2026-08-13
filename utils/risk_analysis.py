# =============================================================================
# utils/risk_analysis.py — Risk metrics computation
# =============================================================================

import numpy as np
from config import RISK_MAP, RISK_COLOR


def compute_affected_area(mask: np.ndarray) -> float:
    """
    Compute the percentage of brain area covered by the tumor mask.
    Assumes non-black pixels = brain region; mask = tumor.
    """
    if mask is None or mask.sum() == 0:
        return 0.0

    h, w    = mask.shape[:2]
    total   = h * w
    tumor_px = int((mask > 127).sum())
    return round((tumor_px / total) * 100, 2)


def compute_risk_score(confidence: float,
                       affected_area: float,
                       predicted_class: str) -> int:
    """
    Composite risk score on a 0–100 scale.
      - Base from classifier confidence (0–60 pts)
      - Area contribution (0–30 pts)
      - Class weight (0–10 pts)
    """
    if predicted_class == "No Tumor":
        return 0

    class_weight = {"Glioma Tumor": 10, "Meningioma Tumor": 5,
                    "Pituitary Tumor": 5}.get(predicted_class, 0)
    conf_score   = confidence * 60
    area_score   = min(affected_area / 25, 1.0) * 30   # cap at 25% area → 30 pts

    raw  = conf_score + area_score + class_weight
    return min(100, int(raw))


def get_risk_level(predicted_class: str) -> str:
    return RISK_MAP.get(predicted_class, "Low")


def get_risk_color(risk_level: str) -> str:
    return RISK_COLOR.get(risk_level, "#22c55e")


def full_risk_report(predicted_class: str,
                     confidence: float,
                     mask: np.ndarray) -> dict:
    """Return a complete risk summary dict."""
    area  = compute_affected_area(mask)
    score = compute_risk_score(confidence, area, predicted_class)
    level = get_risk_level(predicted_class)
    color = get_risk_color(level)

    return {
        "risk_level":    level,
        "risk_color":    color,
        "affected_area": area,
        "risk_score":    score,
    }
