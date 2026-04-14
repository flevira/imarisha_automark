from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def decode_qr_robust(img_bgr: np.ndarray) -> str | None:
    detector = cv2.QRCodeDetector()
    attempts: list[np.ndarray] = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    attempts.append(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    attempts.append(clahe.apply(gray))

    attempts.append(
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    )

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(otsu)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    attempts.append(cv2.filter2D(gray, -1, sharpen_kernel))

    attempts.append(
        cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    )

    h, w = gray.shape
    qr_rois = (
        # Legacy location: header top-right.
        (0.10, 0.38, 0.72, 0.98),
        # New layout: below student info (name + student id), still on right side.
        (0.26, 0.56, 0.60, 0.98),
        # Slightly lower fallback for scan variance.
        (0.34, 0.68, 0.58, 0.98),
    )
    for y0, y1, x0, x1 in qr_rois:
        qr_crop = gray[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        if qr_crop.size == 0:
            continue
        attempts.append(
            cv2.resize(qr_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        )

    for attempt in attempts:
        data, _, _ = detector.detectAndDecode(attempt)
        if data:
            return data

    return None


def parse_qr_data(data: str | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "type": None,
        "studentid": None,
        "examid": None,
        "testid": None,
        "assessmentid": None,
        "studentname": None,
    }
    if not data:
        return parsed

    for part in data.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            parsed[k.strip().lower()] = v.strip()
    return parsed
