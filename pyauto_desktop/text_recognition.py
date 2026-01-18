import cv2
import numpy as np
import re
from rapidocr import RapidOCR



_ENGINE_CACHE = {}


def get_engine():
    """Returns the cached RapidOCR engine."""
    if 'fast' not in _ENGINE_CACHE:
        _ENGINE_CACHE['fast'] = RapidOCR()
    return _ENGINE_CACHE['fast']


def preprocess_image(img, mode='clean'):
    """
    Preprocessing modes:
    1. 'clean'    -> Upscale, Denoise, CLAHE (General purpose)
    2. 'binarize' -> Color split, High Contrast, Thresholding (Colored/Low contrast text)
    3. 'restore'  -> Sharpening, Gamma Correction, Bilateral Filter (Faded/Damaged text)
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot preprocess empty image.")

    processed = None

    if mode == 'clean':
        scale_factor = 4
        img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        if len(img_large.shape) == 3:
            gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_large

        gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    elif mode == 'binarize':
        scale_factor = 4
        img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        if len(img_large.shape) == 3:
            lab = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            hsv = cv2.cvtColor(img_large, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
            channels = [l_channel, v_channel, gray]
            contrasts = [np.std(ch) for ch in channels]
            best_channel = channels[np.argmax(contrasts)]
        else:
            best_channel = img_large

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(best_channel)
        _, processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, processed_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if np.sum(processed_inv == 0) < np.sum(processed == 0):
            processed = processed_inv

        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    elif mode == 'restore':
        scale_factor = 3
        img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        if len(img_large.shape) == 3:
            gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_large

        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharp = cv2.addWeighted(gray, 2.5, gaussian, -1.5, 0)
        norm = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)

        gamma = 0.6
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(norm, table)

        bilateral = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)
        _, processed = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    else:
        # Fallback/Error for unknown modes
        raise ValueError(f"Unknown mode: {mode}. Use 'clean', 'binarize', or 'restore'")

    padding = 30
    return cv2.copyMakeBorder(
        processed,
        top=padding, bottom=padding, left=padding, right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )


def filter_english_only(text):
    """
    Removes non-ASCII characters. Keeps English letters, numbers, and standard symbols.
    """
    return re.sub(r'[^\x20-\x7E]', '', text).strip()


def get_text_from_image(image_data, mode='clean', use_det=False):
    """
    Main entry point for text extraction.
    Args:
        image_data: cv2 image array
        lang: language code (default 'en')
        mode: preprocessing mode ('clean', 'binarize', 'restore')
        use_det: boolean to enable/disable detection model (default False)
    """
    if image_data is None or image_data.size == 0:
        raise ValueError("Input image_data is None or empty.")

    try:
        processed_img = preprocess_image(image_data, mode=mode)
        engine = get_engine()

        result = engine(processed_img, use_det=use_det, use_cls=True, use_rec=True)

        if not result:
            return []

        raw_lines = []
        if hasattr(result, 'txts') and result.txts:
            raw_lines = list(result.txts)
        elif isinstance(result, list):
            raw_lines = result

        extracted_texts = []
        for line in raw_lines:
            text = str(line)
            cleaned_text = filter_english_only(text)
            if cleaned_text:
                extracted_texts.append(cleaned_text)

        return extracted_texts

    except Exception as e:
        print(f"OCR Error: {e}")
        return []