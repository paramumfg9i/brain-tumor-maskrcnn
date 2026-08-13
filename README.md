# 🧠 ADVANCE BRAIN TUMOR CLASSIFICATION AND SEGMENTATION USING MASK R-CNN

> AI-powered MRI analysis · Tumor localisation · Risk assessment · Report generation

---

## 📁 Project Structure

```
brain_tumor_maskrcnn_app/
├── app.py                   # Main Streamlit application
├── config.py                # Global configuration (classes, paths, thresholds)
├── requirements.txt         # Python dependencies
├── models/
│   ├── class_names.py       # Authoritative 4-class label list
│   ├── classifier_model.h5  # (place trained weights here)
│   └── mask_rcnn_weights.pth# (place trained Mask R-CNN weights here)
├── utils/
│   ├── preprocess.py        # MRI loading, CLAHE enhancement, tensor prep
│   ├── inference.py         # 4-class EfficientNetB0 classifier
│   ├── segmentation.py      # Mask R-CNN tumor segmentation
│   ├── risk_analysis.py     # Risk level, affected area, risk score
│   ├── report_generator.py  # ReportLab PDF medical report
│   └── ui_components.py     # HTML card / bar helpers for Streamlit
├── assets/
│   └── styles.css           # Premium dark-theme CSS
└── outputs/                 # Auto-created output directory
```

---

## ⚙️ Local Setup & Run

### 1. Prerequisites
- Python 3.10 or 3.11 (recommended)
- pip ≥ 23.0

### 2. Install dependencies
```bash
cd brain_tumor_maskrcnn_app
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment:
> ```bash
> python -m venv venv && source venv/bin/activate   # Linux/macOS
> venv\Scripts\activate                              # Windows
> pip install -r requirements.txt
> ```

### 3. (Optional) Add pre-trained weights
| File | Where to get it |
|------|----------------|
| `models/classifier_model.h5` | Fine-tune EfficientNetB0 on the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 4 classes |
| `models/mask_rcnn_weights.pth` | Fine-tune torchvision Mask R-CNN on a brain-MRI instance-segmentation dataset |

Without weights, the app runs in **demo mode** — still produces all four classes, segmentation masks, risk metrics, and PDF reports.

### 4. Launch the app
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🏗️ Architecture

```
MRI Image
    │
    ├─ Preprocessing (CLAHE + sharpening)
    │
    ├─ [Classifier]  EfficientNetB0 → 4-class softmax
    │       Glioma Tumor | Meningioma Tumor | Pituitary Tumor | No Tumor
    │
    └─ [Segmenter]   Mask R-CNN ResNet-50 FPN
            → binary mask, bounding box, confidence
```

The two-branch design **eliminates** the old binary-collapse bug where only
Glioma or No Tumor were ever predicted.

---

## 🧪 Four Classes — Complete Mapping

| Class | Risk | Notes |
|-------|------|-------|
| Glioma Tumor | 🔴 High | Most aggressive; originates in glial cells |
| Meningioma Tumor | 🟡 Medium | Usually benign; arises from meninges |
| Pituitary Tumor | 🟡 Medium | Affects hormone regulation |
| No Tumor | 🟢 Low | Clean scan — full report still generated |

---

## 📄 PDF Report Contents
- Project title + report ID + timestamp  
- Original MRI & Tumor-highlighted image (side by side)  
- Diagnosis, Risk Level, Risk Score, Affected Area  
- Estimated Tumor Location (bounding-box centroid heuristic)  
- Class Probabilities table  
- Medical disclaimer  

# 🧠 Advanced Brain Tumor Classification and Segmentation using Mask R-CNN

## 🚀 Live Demo

👉 [Try the Brain Tumor AI Web App](https://brain-tumor-maskrcnn-jr9omnjjjr6uxroxebusau.streamlit.app/)
---

## 🩺 Disclaimer
This tool is intended for **research and educational purposes only**.  
It does not constitute a medical diagnosis. Always consult a qualified  
radiologist or neurologist for clinical decisions.
