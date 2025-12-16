import cv2
import numpy as np
from PIL import Image, ImageGrab


def _load_image(img):
    """Helper to load image from path or PIL Image."""
    if isinstance(img, str):
        return Image.open(img)
    return img


def _non_max_suppression(boxes, overlap_thresh):
    """
    Standard Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
    boxes: List of (x, y, w, h)
    overlap_thresh: Float (0.0 to 1.0)
    """
    if len(boxes) == 0:
        return []

    # Convert to float numpy array for calculations
    boxes = np.array(boxes, dtype=np.float32)

    # Convert to [x1, y1, x2, y2] for NMS logic
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # Return as integers (x, y, w, h)
    return boxes[pick].astype("int").tolist()


def locateAllOnScreen(image, region=None, grayscale=False, confidence=0.9, overlap_threshold=0.5):
    """
    Locate all instances of 'image' on the screen.
    """
    # 1. Capture Screenshot
    if region:
        x, y, w, h = region
        haystack_pil = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    else:
        haystack_pil = ImageGrab.grab()

    haystack_np = np.array(haystack_pil)

    # 2. Prepare Haystack (Screenshot)
    haystack = cv2.cvtColor(haystack_np, cv2.COLOR_RGB2BGR)

    # 3. Prepare Needle (Template)
    needle_pil = _load_image(image)
    needle_np = np.array(needle_pil)

    needle = None
    mask = None

    # Handle Alpha Channel in Template
    if needle_pil.mode == 'RGBA':
        needle_bgra = cv2.cvtColor(needle_np, cv2.COLOR_RGBA2BGRA)
        needle = needle_bgra[:, :, :3]
        mask = needle_bgra[:, :, 3]
    else:
        needle = cv2.cvtColor(needle_np, cv2.COLOR_RGB2BGR)

    # 4. Grayscale Option
    if grayscale:
        haystack = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY)
        needle = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY)

    # 5. Match
    if mask is not None and not grayscale:
        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)

    # 6. Filter by Confidence
    loc = np.where(res >= confidence)

    rects = []
    h, w = needle.shape[:2]

    # loc is (y_indices, x_indices)
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    # 7. Non-Max Suppression
    if overlap_threshold < 1.0 and len(rects) > 1:
        rects = _non_max_suppression(rects, overlap_threshold)

    # Apply region offset if necessary
    if region:
        rx, ry, _, _ = region
        final_rects = []
        for (x, y, w, h) in rects:
            final_rects.append((x + rx, y + ry, w, h))
        return final_rects

    return [tuple(r) for r in rects]


def locateOnScreen(image, region=None, grayscale=False, confidence=0.9, overlap_threshold=0.5):
    """
    Locate the first instance of 'image' on the screen.
    Returns (x, y, w, h) or None.
    """
    results = locateAllOnScreen(image, region, grayscale, confidence, overlap_threshold)
    if results:
        return results[0]
    return None