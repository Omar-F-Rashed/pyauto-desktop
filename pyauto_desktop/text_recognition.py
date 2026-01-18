import asyncio
import numpy as np
import cv2
from PIL import Image

try:
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
    from winrt.windows.storage.streams import DataReader
    WINRT_AVAILABLE = True
except ImportError:
    WINRT_AVAILABLE = False


def _numpy_to_software_bitmap(numpy_img):
    """Safely converts any Numpy Image (Gray/BGR/BGRA) to the required WinRT format."""
    if len(numpy_img.shape) == 2:  # Grayscale
        rgb_img = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2RGBA)
    elif numpy_img.shape[2] == 3:  # BGR
        rgb_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGBA)
    elif numpy_img.shape[2] == 4:  # BGRA
        rgb_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported image shape: {numpy_img.shape}. Expected Gray, BGR, or BGRA.")

    img_pil = Image.fromarray(rgb_img)
    pixel_data = img_pil.tobytes()

    bitmap = SoftwareBitmap(
        BitmapPixelFormat.RGBA8,
        img_pil.width,
        img_pil.height,
        BitmapAlphaMode.PREMULTIPLIED
    )

    reader = DataReader.from_buffer(pixel_data)
    buffer = reader.read_buffer(len(pixel_data))
    bitmap.copy_from_buffer(buffer)

    return bitmap


async def _execute_ocr_async(bitmap, lang_tag):
    """Async worker for OCR."""
    if not WINRT_AVAILABLE:
        raise ImportError("WinRT dependencies are missing. Please install winrt-Windows.Media.Ocr")

    lang_found = None
    possible_langs = OcrEngine.available_recognizer_languages

    for l in possible_langs:
        if lang_tag.lower() in l.language_tag.lower():
            lang_found = l
            break

    if lang_found:
        engine = OcrEngine.try_create_from_language(lang_found)
    else:
        engine = OcrEngine.try_create_from_user_profile_languages()

    if not engine:
        raise RuntimeError(
            f"Failed to create OcrEngine. Language '{lang_tag}' might not be installed on this Windows system.")

    result = await engine.recognize_async(bitmap)
    return [line.text for line in result.lines]


def preprocess_image(img):
    """
    Standardizes input to high-contrast Black Text on White Background.
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot preprocess empty image.")

    scale_factor = 3
    img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    if len(img_large.shape) == 3:
        if img_large.shape[2] == 4:
            gray = cv2.cvtColor(img_large, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_large

    mean_brightness = np.mean(gray)
    if mean_brightness < 127:
        gray = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    padding = 20
    padded = cv2.copyMakeBorder(
        binary,
        top=padding, bottom=padding, left=padding, right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def get_text_from_image(image_data, lang='en-US'):
    """
    Main entry point for OCR. Raises exceptions if processing fails.
    """
    if not WINRT_AVAILABLE:
        raise ImportError(
            "WinRT libraries not found. Run: pip install winrt-Windows.Media.Ocr winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams")

    if image_data is None or image_data.size == 0:
        raise ValueError("Input image_data is None or empty.")

    try:
        processed_img = preprocess_image(image_data)
        bitmap = _numpy_to_software_bitmap(processed_img)
        return asyncio.run(_execute_ocr_async(bitmap, lang))

    except Exception as e:
        raise RuntimeError(f"OCR Execution Failed: {str(e)}") from e