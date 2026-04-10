from __future__ import annotations

import re
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract


def get_page_text(page: fitz.Page) -> str:
    return page.get_text("text", sort=True)


def extract_student_id_printed_text(page: fitz.Page) -> str | None:
    text = get_page_text(page)
    m = re.search(r"Student ID:\s*(\d+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_student_id_handwritten_ocr(img_bgr: np.ndarray) -> tuple[str | None, bool]:
    h, w = img_bgr.shape[:2]
    x0 = int(0.02 * w)
    x1 = int(0.48 * w)
    y0 = int(0.16 * h)
    y1 = int(0.30 * h)

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

    best_digits: str | None = None
    best_conf = -1.0

    for variant in variants:
        data = pytesseract.image_to_data(
            variant,
            config="--psm 6 -c tessedit_char_whitelist=0123456789[]:",
            output_type=pytesseract.Output.DICT,
        )

        tokens: list[str] = []
        confs: list[float] = []

        for txt, conf_str in zip(data["text"], data["conf"]):
            txt = (txt or "").strip()
            try:
                conf_val = float(conf_str)
            except Exception:
                conf_val = -1.0

            if txt:
                tokens.append(txt)
                confs.append(conf_val)

        joined = " ".join(tokens)
        digit_groups = re.findall(r"\d+", joined)

        if digit_groups:
            candidate = max(digit_groups, key=len)
            avg_conf = np.mean([c for c in confs if c >= 0]) if confs else -1.0
            if len(candidate) >= 1 and avg_conf > best_conf:
                best_digits = candidate
                best_conf = avg_conf

    if best_digits:
        low_conf = best_conf < 45
        return best_digits, low_conf

    return None, True


def resolve_student_id(
    qr_id: str | None,
    printed_id: str | None,
    handwritten_id: str | None,
    mode: str,
) -> dict[str, Any]:
    if mode == "qr":
        return {
            "student_id": qr_id,
            "student_id_all": qr_id if qr_id else None,
            "student_id_qr": qr_id,
            "student_id_text": None,
            "student_id_handwritten": None,
            "student_id_source": "qr",
            "id_conflict": False,
            "needs_review": qr_id is None,
        }

    if mode == "text":
        detected: list[str] = []
        if printed_id:
            detected.append(printed_id)
        if handwritten_id and handwritten_id not in detected:
            detected.append(handwritten_id)

        conflict = len(set(detected)) > 1
        chosen = printed_id or handwritten_id

        return {
            "student_id": chosen,
            "student_id_all": ",".join(detected) if detected else None,
            "student_id_qr": qr_id,
            "student_id_text": printed_id,
            "student_id_handwritten": handwritten_id,
            "student_id_source": "text",
            "id_conflict": conflict,
            "needs_review": True,
        }

    detected = []
    if qr_id:
        detected.append(qr_id)
    if printed_id and printed_id not in detected:
        detected.append(printed_id)
    if handwritten_id and handwritten_id not in detected:
        detected.append(handwritten_id)

    conflict = len(set(detected)) > 1

    if qr_id:
        chosen = qr_id
        source = "qr"
    elif printed_id:
        chosen = printed_id
        source = "text_printed"
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
        "student_id_text": printed_id,
        "student_id_handwritten": handwritten_id,
        "student_id_source": source,
        "id_conflict": conflict,
        "needs_review": chosen is None or conflict or source.startswith("text"),
    }
