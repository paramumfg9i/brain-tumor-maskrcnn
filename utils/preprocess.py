# =============================================================================
# utils/preprocess.py — MRI image pre-processing helpers
# =============================================================================

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


def load_image(uploaded_file) -> np.ndarray:
    """Load a Streamlit UploadedFile / PIL Image into an RGB numpy array."""
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def enhance_mri(img_array: np.ndarray) -> np.ndarray:
    """
    Apply MRI-style enhancement:
      1. CLAHE for local contrast normalisation
      2. Slight sharpening
      3. Minor brightness / contrast lift
    Returns an RGB uint8 numpy array.
    """
    # Convert to LAB for CLAHE on luminance channel
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    lab_eq  = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Sharpening kernel
    kernel  = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def prepare_for_classifier(img_array: np.ndarray,
                            target_size=(224, 224)) -> np.ndarray:
    """
    Resize → normalise to [0, 1] → add batch dimension.
    Returns float32 array of shape (1, H, W, 3).
    """
    pil  = Image.fromarray(img_array).resize(target_size, Image.LANCZOS)
    arr  = np.array(pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prepare_for_segmenter(img_array: np.ndarray,
                           target_size=(512, 512)) -> np.ndarray:
    """
    Resize to segmenter input size.
    Returns a uint8 array (no batch dim — handled by segmentation module).
    """
    pil = Image.fromarray(img_array).resize(target_size, Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)
