from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from core.bubbles import sample_bubble_score
from core.ocr_engine import detect_with_fallback

def extract_student_id_handwritten_ocr(img_bgr: np.ndarray) -> tuple[str | None, bool]:
    h, w = img_bgr.shape[:2]
    x0 = int(0.02 * w)
    x1 = int(0.48 * w)
    y0 = int(0.13 * h)
    y1 = int(0.21 * h)

    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    variants: list[np.ndarray] = []
    variants.append(gray)
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(th1)
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    variants.append(th2)
    variants.append(cv2.bitwise_not(th1))

    match = detect_with_fallback(
        variants,
        pattern=r"\[\s*\d{5}\s*\]|\d{5}",
        min_len=5,
        max_len=7,
        tesseract_configs=[
            "--psm 6 -c tessedit_char_whitelist=0123456789[]:",
            "--psm 7 -c tessedit_char_whitelist=0123456789[]:",
        ],
        min_tesseract_confidence=45.0,
    )
    if match:
        normalized = re.sub(r"\D", "", match.text)
        if len(normalized) == 5:
            return normalized, match.confidence < 55.0
        low_conf = match.confidence < 55.0
        return match.text, low_conf

    return None, True


def _cluster_positions(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def extract_student_id_bubble(img_bgr: np.ndarray) -> tuple[str | None, bool]:
    h, w = img_bgr.shape[:2]
    x0 = int(0.02 * w)
    x1 = int(0.48 * w)
    y0 = int(0.14 * h)
    y1 = int(0.44 * h)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, True

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 60 or area > 1800:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if min(cw, ch) < 8 or max(cw, ch) > 70:
            continue
        candidates.append((x + (cw / 2.0), y + (ch / 2.0), max(cw, ch) / 2.0))

    if len(candidates) < 30:
        return None, True

    x_clusters = _cluster_positions([c[0] for c in candidates], tolerance=max(8.0, roi.shape[1] * 0.03))
    y_clusters = _cluster_positions([c[1] for c in candidates], tolerance=max(6.0, roi.shape[0] * 0.02))
    if len(x_clusters) < 5 or len(y_clusters) < 10:
        return None, True

    columns = sorted(x_clusters)[:5]
    rows = sorted(y_clusters)[:10]
    scores_by_col: list[list[float]] = [[] for _ in range(5)]
    for col_idx, cx in enumerate(columns):
        for cy in rows:
            score = sample_bubble_score(gray, int(cx), int(cy), radius=10)
            scores_by_col[col_idx].append(score)

    digits: list[str] = []
    low_conf = False
    for col_scores in scores_by_col:
        if len(col_scores) != 10:
            return None, True
        best_idx = int(np.argmax(col_scores))
        sorted_scores = sorted(col_scores, reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        if top < 35.0 or (top - second) < 4.0:
            low_conf = True
        digits.append(str(best_idx))

    return "".join(digits), low_conf


def resolve_student_id(
    qr_id: str | None,
    bubble_id: str | None,
    handwritten_id: str | None,
    mode: str,
) -> dict[str, Any]:
    if mode == "qr":
        return {
            "student_id": qr_id,
            "student_id_all": qr_id if qr_id else None,
            "student_id_qr": qr_id,
            "student_id_text": None,
            "student_id_bubble": None,
            "student_id_handwritten": None,
            "student_id_source": "qr",
            "id_conflict": False,
            "needs_review": qr_id is None,
        }

    if mode == "text":
        detected: list[str] = []
        if bubble_id and bubble_id not in detected:
            detected.append(bubble_id)
        if handwritten_id and handwritten_id not in detected:
            detected.append(handwritten_id)

        conflict = len(set(detected)) > 1
        chosen = bubble_id or handwritten_id
        source = "text_bubble" if bubble_id else "text_handwritten" if handwritten_id else "missing"

        return {
            "student_id": chosen,
            "student_id_all": ",".join(detected) if detected else None,
            "student_id_qr": qr_id,
            "student_id_text": None,
            "student_id_bubble": bubble_id,
            "student_id_handwritten": handwritten_id,
            "student_id_source": source,
            "id_conflict": conflict,
            "needs_review": True,
        }

    detected = []
    if qr_id:
        detected.append(qr_id)
    if bubble_id and bubble_id not in detected:
        detected.append(bubble_id)
    if handwritten_id and handwritten_id not in detected:
        detected.append(handwritten_id)

    conflict = len(set(detected)) > 1

    if qr_id:
        chosen = qr_id
        source = "qr"
    elif bubble_id:
        chosen = bubble_id
        source = "text_bubble"
    elif handwritten_id:
        chosen = handwritten_id
        source = "text_handwritten"
    else:
        chosen = None
        source = "missing"

    return {
        "student_id": chosen,
        "student_id_all": ",".join(detected) if detected else None,
        "student_id_qr": qr_id,
        "student_id_text": None,
        "student_id_bubble": bubble_id,
        "student_id_handwritten": handwritten_id,
        "student_id_source": source,
        "id_conflict": conflict,
        "needs_review": chosen is None or conflict or source.startswith("text"),
    }
