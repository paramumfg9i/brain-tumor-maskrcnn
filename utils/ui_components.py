# =============================================================================
# utils/ui_components.py — Streamlit UI helper components
# =============================================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image


def render_highlighted_image(img_array     : np.ndarray,
                              mask          : np.ndarray,
                              bbox          : tuple,
                              label         : str,
                              confidence    : float) -> np.ndarray:
    """
    Overlay red semi-transparent mask + green bounding box + label on the MRI.
    Returns an RGB uint8 numpy array.
    """
    out = img_array.copy().astype(np.uint8)

    # Red overlay
    if mask is not None and mask.sum() > 0:
        red_layer         = np.zeros_like(out)
        red_layer[:, :, 0] = 200
        alpha_mask         = (mask > 127).astype(np.float32) * 0.45
        for c in range(3):
            out[:, :, c] = (
                out[:, :, c] * (1 - alpha_mask) +
                red_layer[:, :, c] * alpha_mask
            ).astype(np.uint8)

    # Bounding box
    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background
        lbl_text = f"{label} - {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = x1, max(y1 - 8, th + 4)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(out, lbl_text, (lx + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)

    return out


def metric_card_html(label: str, value: str,
                     color: str = "#f5f5f5",
                     sub: str = "") -> str:
    """Return HTML for a glowing metric card."""
    return f"""
    <div style="
        background: linear-gradient(135deg, #111 0%, #1a1a1a 100%);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 0 18px rgba(255,255,255,0.04);
        transition: transform 0.2s;
    ">
        <div style="font-size:11px; color:#777; text-transform:uppercase;
                    letter-spacing:1.5px; margin-bottom:8px;">{label}</div>
        <div style="font-size:26px; font-weight:700; color:{color};
                    font-family:'Courier New', monospace;">{value}</div>
        {"<div style='font-size:10px;color:#555;margin-top:4px;'>" + sub + "</div>" if sub else ""}
    </div>
    """


def probability_bar_html(class_name: str, prob: float, is_top: bool) -> str:
    """Animated horizontal progress bar for class probabilities."""
    pct        = prob * 100
    bar_color  = "#ffffff" if is_top else "#444"
    glow       = "0 0 12px rgba(255,255,255,0.3)" if is_top else "none"
    border     = "1px solid #333" if is_top else "1px solid #222"
    text_color = "#fff" if is_top else "#888"

    return f"""
    <div style="margin-bottom: 12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-size:12px; color:{text_color}; font-weight:{'700' if is_top else '400'};">
                {class_name}
            </span>
            <span style="font-size:12px; color:{text_color}; font-family:'Courier New';">
                {pct:.1f}%
            </span>
        </div>
        <div style="background:#111; border-radius:6px; height:10px;
                    border:{border}; overflow:hidden;">
            <div style="
                width:{pct:.1f}%;
                height:100%;
                background:{bar_color};
                border-radius:6px;
                box-shadow:{glow};
                transition: width 0.8s ease;
            "></div>
        </div>
    </div>
    """
