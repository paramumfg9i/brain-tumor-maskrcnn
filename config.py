# =============================================================================
# config.py — Global configuration for Brain Tumor MRI Analysis App
# =============================================================================

import os

# ── Project Metadata ──────────────────────────────────────────────────────────
APP_TITLE    = "ADVANCE BRAIN TUMOR CLASSIFICATION AND SEGMENTATION USING MASK R-CNN"
APP_SUBTITLE = "AI-powered MRI analysis  •  localization  •  risk assessment  •  report generation"
APP_VERSION  = "2.0.0"

# ── Class Definitions (4 classes — complete mapping) ─────────────────────────
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
CLASS_IDS   = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)   # 4

# ── Risk Mapping per Tumor Type ───────────────────────────────────────────────
RISK_MAP = {
    "Glioma Tumor":     "High",
    "Meningioma Tumor": "Medium",
    "Pituitary Tumor":  "Medium",
    "No Tumor":         "Low",
}

RISK_COLOR = {
    "High":   "#ef4444",   # red
    "Medium": "#f59e0b",   # amber
    "Low":    "#22c55e",   # green
}

# ── Brain Region Estimation (heuristic from bounding-box centroid) ───────────
BRAIN_REGIONS = [
    "Frontal Lobe",
    "Parietal Lobe",
    "Temporal Lobe",
    "Occipital Lobe",
    "Cerebellum",
    "Brain Stem",
    "Thalamus",
]

# ── Classifier / Segmenter hyper-params ──────────────────────────────────────
CLASSIFIER_INPUT_SIZE = (224, 224)
SEGMENTER_INPUT_SIZE  = (512, 512)
CONFIDENCE_THRESHOLD  = 0.50          # minimum score to show mask

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier_model.h5")
MASKRCNN_PATH   = os.path.join(MODELS_DIR, "mask_rcnn_weights.pth")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
