# =============================================================================
# app.py — ADVANCE BRAIN TUMOR CLASSIFICATION AND SEGMENTATION USING MASK R-CNN
#
# Run:  python -m streamlit run app.py
# =============================================================================

import os
import sys
import datetime
import time

# ── Streamlit MUST be imported and configured first ───────────────────────────
import streamlit as st

# ── Path fix so sub-modules resolve correctly ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must come before any other st.* call)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Brain Tumor AI | Mask R-CNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Heavy imports AFTER page config ──────────────────────────────────────────
try:
    import numpy as np
    from PIL import Image
    from config import APP_TITLE, APP_SUBTITLE, CLASS_NAMES, OUTPUTS_DIR
    from utils.preprocess    import load_image, enhance_mri, prepare_for_classifier
    from utils.inference     import classify_tumor, get_classifier
    from utils.segmentation  import segment_tumor
    from utils.risk_analysis import full_risk_report
    from utils.ui_components import (render_highlighted_image,
                                      metric_card_html,
                                      probability_bar_html)
    from utils.report_generator import generate_pdf_report
    _IMPORTS_OK = True
except Exception as _import_err:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_import_err)

if not _IMPORTS_OK:
    st.error("❌ Import error — check that all dependencies are installed.")
    st.code("pip install -r requirements.txt" + chr(10) + chr(10) + "Error: " + str(_IMPORT_ERROR))
    st.stop()


# ── Inject CSS ────────────────────────────────────────────────────────────────
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


# ── Additional inline styles ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

/* Animated gradient title */
@keyframes shimmer {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.title-gradient {
    background: linear-gradient(270deg, #fff 0%, #aaa 40%, #fff 70%, #ccc 100%);
    background-size: 400% 400%;
    animation: shimmer 5s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Space Mono', monospace;
    font-size: clamp(16px, 2.1vw, 26px);
    font-weight: 700;
    letter-spacing: 2px;
    text-align: center;
    line-height: 1.3;
    margin-bottom: 6px;
}
.subtitle-text {
    text-align: center;
    color: #555;
    font-size: 13px;
    letter-spacing: 1px;
    margin-bottom: 32px;
    font-family: 'DM Sans', sans-serif;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 28px 0 12px 0;
    padding-left: 12px;
    border-left: 2px solid #333;
}
.glow-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #333, transparent);
    margin: 24px 0;
}
.report-block {
    background: #0e0e0e;
    border: 1px solid #1f1f1f;
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    font-family: 'DM Sans', sans-serif;
}
.location-card {
    background: linear-gradient(135deg, #111 0%, #151515 100%);
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
/* Pulsing dot for live indicator */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f'<div class="title-gradient">🧠 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle-text">{APP_SUBTITLE}</div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — model status
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 System Status")
    st.markdown('<span class="live-dot"></span> **Model loaded successfully**',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("- Classifier: EfficientNetB0 (4-class)")
    st.markdown("- Segmenter: Mask R-CNN ResNet-50 FPN")
    st.markdown("- Classes: Glioma · Meningioma · Pituitary · No Tumor")
    st.markdown("---")
    st.markdown("**Version:** 2.0.0")
    st.caption("For research & screening purposes only")


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-LOAD CLASSIFIER (warms up TF session)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def warm_up_classifier():
    return get_classifier()

with st.spinner("🔧 Initialising model pipeline…"):
    _clf = warm_up_classifier()

st.success("✅  Model loaded successfully")


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Upload MRI Scan</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop your MRI image here — JPG, JPEG, PNG accepted (T1 / T2 weighted)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    run_btn = st.button("🚀 Classify & Analyze", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

if uploaded_file and run_btn:

    # ── 1. Load & pre-process ─────────────────────────────────────────────────
    with st.spinner("🔬 Processing MRI scan…"):
        original_arr  = load_image(uploaded_file)
        enhanced_arr  = enhance_mri(original_arr)
        clf_input     = prepare_for_classifier(original_arr)

    # ── 2. Classification (4-class) ───────────────────────────────────────────
    with st.spinner("🤖 Running 4-class tumor classifier…"):
        clf_result = classify_tumor(clf_input)

    predicted_class = clf_result["predicted_class"]
    confidence      = clf_result["confidence"]
    probabilities   = clf_result["probabilities"]

    # ── 3. Segmentation (Mask R-CNN) ──────────────────────────────────────────
    with st.spinner("🎯 Running Mask R-CNN segmentation…"):
        seg_result = segment_tumor(original_arr, predicted_class, confidence)

    mask          = seg_result["mask"]
    bbox          = seg_result["bbox"]
    has_tumor     = seg_result["has_tumor"]
    tumor_region  = seg_result["region_label"]

    # ── 4. Risk analysis ──────────────────────────────────────────────────────
    risk = full_risk_report(predicted_class, confidence, mask)

    # ── 5. Highlighted image ──────────────────────────────────────────────────
    highlighted_arr = render_highlighted_image(
        original_arr, mask, bbox,
        label=predicted_class, confidence=confidence
    )

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS UI
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # ── A. Three MRI Panels ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">MRI Analysis Panels</div>',
                unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="medium")

    with p1:
        st.image(original_arr, caption="ORIGINAL MRI", use_container_width=True)
    with p2:
        st.image(enhanced_arr, caption="ENHANCED MRI  (CLAHE + sharpening)",
                 use_container_width=True)
    with p3:
        st.image(highlighted_arr, caption="TUMOR HIGHLIGHTED  (Mask R-CNN)",
                 use_container_width=True)

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # ── B. Four Metric Cards ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Diagnosis Dashboard</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")

    with mc1:
        st.markdown(metric_card_html(
            "Diagnosis", predicted_class,
            color="#ffffff",
            sub="4-class EfficientNetB0"
        ), unsafe_allow_html=True)

    with mc2:
        st.markdown(metric_card_html(
            "Risk Level", risk["risk_level"],
            color=risk["risk_color"],
            sub=f"Confidence {confidence*100:.1f}%"
        ), unsafe_allow_html=True)

    with mc3:
        area_str = f"{risk['affected_area']:.2f}%"
        st.markdown(metric_card_html(
            "Affected Area", area_str,
            color="#a0a0a0",
            sub="of brain region"
        ), unsafe_allow_html=True)

    with mc4:
        st.markdown(metric_card_html(
            "Risk Score", f"{risk['risk_score']}/100",
            color=risk["risk_color"],
            sub="composite score"
        ), unsafe_allow_html=True)

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # ── C. Tumor Location + Probabilities ─────────────────────────────────────
    st.markdown('<div class="section-header">Spatial Analysis & Class Probabilities</div>',
                unsafe_allow_html=True)

    loc_col, prob_col = st.columns([1, 2], gap="large")

    with loc_col:
        bbox_str = (f"({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})"
                    if has_tumor else "N/A")
        st.markdown(f"""
        <div class="location-card">
            <div style="font-size:11px;color:#555;text-transform:uppercase;
                        letter-spacing:2px;margin-bottom:12px;">Tumor Location</div>
            <div style="font-size:28px;margin-bottom:4px;">📍</div>
            <div style="font-size:18px;font-weight:700;color:#fff;
                        font-family:'Space Mono',monospace;margin-bottom:8px;">
                {tumor_region}
            </div>
            <div style="font-size:10px;color:#555;margin-bottom:14px;">
                Bounding Box: <span style="color:#777;font-family:'Courier New';">
                {bbox_str}</span>
            </div>
            <div style="font-size:10px;color:#444;">
                {'🔴 Tumor detected — segmentation active' if has_tumor
                 else '🟢 No tumor — segmentation skipped'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown("""
        <div style="background:#0e0e0e;border:1px solid #1f1f1f;
                    border-radius:14px;padding:22px;">
            <div style="font-size:11px;color:#555;text-transform:uppercase;
                        letter-spacing:2px;margin-bottom:18px;">Class Probabilities</div>
        """, unsafe_allow_html=True)

        top_class = max(probabilities, key=probabilities.get)
        for cls in CLASS_NAMES:
            is_top = (cls == top_class)
            st.markdown(probability_bar_html(cls, probabilities[cls], is_top),
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # ── D. Analysis Report Viewer ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Analysis Report Viewer</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="report-block">', unsafe_allow_html=True)

    rpt_col1, rpt_col2 = st.columns([1, 2], gap="large")

    with rpt_col1:
        st.image(highlighted_arr, caption="Tumor Highlighted", use_container_width=True)

    with rpt_col2:
        badge_color = risk["risk_color"]
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;">
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;width:40%;">Report ID</td>
                <td style="padding:8px 0;color:#888;font-size:12px;font-family:'Courier New';">
                    BTR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Timestamp</td>
                <td style="padding:8px 0;color:#888;font-size:12px;">{ts}</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Diagnosis</td>
                <td style="padding:8px 0;color:#fff;font-size:14px;font-weight:700;">{predicted_class}</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Confidence</td>
                <td style="padding:8px 0;color:#aaa;font-size:12px;">{confidence*100:.2f}%</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Risk Level</td>
                <td style="padding:8px 0;">
                    <span class="badge"
                          style="background:{badge_color}22;color:{badge_color};
                                 border:1px solid {badge_color}44;">
                        {risk['risk_level']}
                    </span>
                </td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Risk Score</td>
                <td style="padding:8px 0;color:#aaa;font-size:12px;">{risk['risk_score']} / 100</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Affected Area</td>
                <td style="padding:8px 0;color:#aaa;font-size:12px;">{risk['affected_area']:.2f}%</td>
            </tr>
            <tr>
                <td style="padding:8px 0;color:#555;font-size:12px;">Tumor Location</td>
                <td style="padding:8px 0;color:#aaa;font-size:12px;">{tumor_region}</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        # Mini prob table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:11px;color:#444;text-transform:uppercase;
                    letter-spacing:2px;margin-bottom:8px;">Class Probabilities</div>""",
                    unsafe_allow_html=True)
        prob_rows = "".join([
            f"<tr><td style='padding:4px 8px;color:#555;font-size:11px;'>{c}</td>"
            f"<td style='padding:4px 8px;color:#777;font-size:11px;"
            f"font-family:Courier New;text-align:right;'>{p*100:.1f}%</td></tr>"
            for c, p in sorted(probabilities.items(), key=lambda x: -x[1])
        ])
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;
                      border:1px solid #1f1f1f;border-radius:8px;overflow:hidden;">
            {prob_rows}
        </table>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── E. PDF Download ───────────────────────────────────────────────────────
    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Download Report</div>',
                unsafe_allow_html=True)

    with st.spinner("📄 Generating PDF report…"):
        pdf_bytes = generate_pdf_report(
            original_img    = original_arr,
            highlighted_img = highlighted_arr,
            diagnosis       = predicted_class,
            confidence      = confidence,
            risk_level      = risk["risk_level"],
            risk_score      = risk["risk_score"],
            affected_area   = risk["affected_area"],
            tumor_location  = tumor_region,
            probabilities   = probabilities,
            timestamp       = ts,
        )

    dl_col, _ = st.columns([1, 5])
    with dl_col:
        st.download_button(
            label      = "⬇  Download PDF Report",
            data       = pdf_bytes,
            file_name  = f"BrainTumor_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime       = "application/pdf",
            use_container_width=True,
        )

    st.caption("⚠ For research & screening use only. Consult a qualified neurologist for clinical decisions.")

elif uploaded_file and not run_btn:
    # Preview uploaded image
    st.markdown('<div class="section-header">Uploaded Scan Preview</div>',
                unsafe_allow_html=True)
    prev_col, _ = st.columns([2, 3])
    with prev_col:
        st.image(uploaded_file, caption="Awaiting analysis — press 🚀 Classify & Analyze",
                 use_container_width=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#333;">
        <div style="font-size:48px;margin-bottom:16px;">🧠</div>
        <div style="font-size:14px;font-family:'Space Mono',monospace;
                    letter-spacing:2px;color:#3a3a3a;">
            UPLOAD AN MRI SCAN TO BEGIN ANALYSIS
        </div>
        <div style="font-size:11px;color:#282828;margin-top:8px;">
            Supports T1 / T2 weighted MRI • JPG · JPEG · PNG
        </div>
    </div>
    """, unsafe_allow_html=True)
