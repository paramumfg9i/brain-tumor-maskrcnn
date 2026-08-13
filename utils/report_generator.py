# =============================================================================
# utils/report_generator.py — PDF medical report via ReportLab
# =============================================================================

import io
import os
import datetime
import numpy as np
from PIL import Image as PILImage

from reportlab.lib              import colors
from reportlab.lib.pagesizes    import A4
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units        import mm, cm
from reportlab.platypus         import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image as RLImage,
                                        HRFlowable)
from reportlab.lib.enums        import TA_CENTER, TA_LEFT
from reportlab.pdfgen           import canvas


# ── Colour palette ─────────────────────────────────────────────────────────────
BLACK  = colors.HexColor("#0a0a0a")
WHITE  = colors.HexColor("#f5f5f5")
GRAY   = colors.HexColor("#555555")
ACCENT = colors.HexColor("#e0e0e0")
RED    = colors.HexColor("#ef4444")
AMBER  = colors.HexColor("#f59e0b")
GREEN  = colors.HexColor("#22c55e")


def _risk_color(level: str):
    return {"High": RED, "Medium": AMBER, "Low": GREEN}.get(level, GREEN)


def _pil_to_rl_image(arr: np.ndarray, max_w_cm: float = 8.0) -> RLImage:
    """Convert a numpy RGB array to a ReportLab Image flowable."""
    buf = io.BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    img = RLImage(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth  = max_w_cm * cm
    img.drawHeight = max_w_cm * cm * aspect
    return img


def generate_pdf_report(original_img   : np.ndarray,
                         highlighted_img: np.ndarray,
                         diagnosis      : str,
                         confidence     : float,
                         risk_level     : str,
                         risk_score     : int,
                         affected_area  : float,
                         tumor_location : str,
                         probabilities  : dict,
                         timestamp      : str | None = None) -> bytes:
    """
    Build a professional PDF medical report and return raw bytes.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title="Brain Tumor Analysis Report",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        fontSize=15, fontName="Helvetica-Bold",
        alignment=TA_CENTER, textColor=BLACK,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "SubTitle",
        fontSize=9, fontName="Helvetica",
        alignment=TA_CENTER, textColor=GRAY,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=BLACK, spaceBefore=12, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#222222"), spaceAfter=3,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "ADVANCE BRAIN TUMOR CLASSIFICATION AND SEGMENTATION<br/>USING MASK R-CNN",
        title_style))
    story.append(Paragraph(
        "AI-powered MRI Analysis  •  Tumor Localization  •  Risk Assessment",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLACK))
    story.append(Spacer(1, 6))

    # Timestamp + report ID
    report_id = f"BTR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    meta_data = [
        ["Report ID:", report_id, "Timestamp:", timestamp],
        ["Diagnosis:", diagnosis, "Confidence:", f"{confidence*100:.1f}%"],
    ]
    meta_table = Table(meta_data, colWidths=[35*mm, 65*mm, 30*mm, 45*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0),(2,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0),(-1,-1), GRAY),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── MRI Images ────────────────────────────────────────────────────────────
    story.append(Paragraph("MRI Scan Images", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
    story.append(Spacer(1, 4))

    orig_rl = _pil_to_rl_image(original_img, max_w_cm=7.5)
    high_rl = _pil_to_rl_image(highlighted_img, max_w_cm=7.5)

    img_table = Table(
        [[orig_rl, high_rl],
         [Paragraph("Original MRI", body_style),
          Paragraph("Tumor Highlighted", body_style)]],
        colWidths=[85*mm, 85*mm],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",    (0,0),(-1,-1), "CENTER"),
        ("VALIGN",   (0,0),(-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # ── Diagnosis & Risk ──────────────────────────────────────────────────────
    story.append(Paragraph("Diagnosis & Risk Assessment", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
    story.append(Spacer(1, 4))

    rc = _risk_color(risk_level)
    diag_data = [
        ["Diagnosis",      "Risk Level",  "Risk Score",       "Affected Area"],
        [diagnosis,        risk_level,    f"{risk_score}/100", f"{affected_area:.2f}%"],
    ]
    diag_table = Table(diag_data, colWidths=[50*mm, 40*mm, 40*mm, 40*mm])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1, 0),  colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",    (0,0),(-1, 0),  WHITE),
        ("FONTNAME",     (0,0),(-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1),  9),
        ("ALIGN",        (0,0),(-1,-1),  "CENTER"),
        ("VALIGN",       (0,0),(-1,-1),  "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f0f0f0")]),
        ("TEXTCOLOR",    (1,1),(1, 1),   rc),
        ("FONTNAME",     (1,1),(1, 1),   "Helvetica-Bold"),
        ("GRID",         (0,0),(-1,-1),  0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0,0),(-1,-1),  6),
        ("BOTTOMPADDING",(0,0),(-1,-1),  6),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Estimated Tumor Location: <b>{tumor_location}</b>", body_style))
    story.append(Spacer(1, 10))

    # ── Class Probabilities ───────────────────────────────────────────────────
    story.append(Paragraph("Class Probabilities", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
    story.append(Spacer(1, 4))

    prob_header = [["Tumor Class", "Probability (%)", "Indicator"]]
    prob_rows   = []
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar_len = int(prob * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        prob_rows.append([cls, f"{prob*100:.1f}%", bar[:20]])   # truncate for PDF

    prob_table = Table(prob_header + prob_rows,
                       colWidths=[65*mm, 35*mm, 65*mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0),  colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",     (0,0),(-1, 0),  WHITE),
        ("FONTNAME",      (0,0),(-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1),  8),
        ("ALIGN",         (1,0),(1,-1),   "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [WHITE, colors.HexColor("#f5f5f5")]),
        ("GRID",          (0,0),(-1,-1),  0.4, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0,0),(-1,-1),  5),
        ("BOTTOMPADDING", (0,0),(-1,-1),  5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
    story.append(Spacer(1, 4))
    disclaimer = (
        "<b>Medical Disclaimer:</b> This report is generated by an AI-assisted "
        "system for research and screening purposes only. It does not constitute "
        "a medical diagnosis. Please consult a qualified radiologist or "
        "neurologist for clinical decision-making."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer", fontSize=7, textColor=GRAY, fontName="Helvetica-Oblique")))

    doc.build(story)
    return buf.getvalue()
