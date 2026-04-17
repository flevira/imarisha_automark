from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
import pytesseract

from core.bubbles import sample_bubble_score
from core.ocr_engine import detect_with_fallback


def _parse_bracket_token(text: str) -> str | None:
    m = re.search(r"\[(.*?)\]", text)
    if not m:
        return None
    inside = m.group(1)
    digits = [ch if ch.isdigit() else "0" for ch in inside]
    if len(digits) < 5:
        digits.extend(["0"] * (5 - len(digits)))
    if not digits:
        return "00000"
    return "".join(digits[:5])


def _find_studentnumber_bracket_bounds(roi_gray: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        data = pytesseract.image_to_data(
            roi_gray,
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return None

    tokens: list[dict[str, float | str]] = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 20.0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        tokens.append({"text": text, "norm": re.sub(r"[^A-Z0-9\[\]]", "", text.upper()), "x": x, "y": y, "w": w, "h": h})

    if not tokens:
        return None

    heading = next((t for t in tokens if "STUDENTNUMBER" in str(t["norm"])), None)
    if heading is None:
        return None

    h_y = int(heading["y"])
    h_h = int(heading["h"])
    min_row_y = h_y + int(0.5 * h_h)

    # Best-case: OCR returns the full bracket token in one piece under STUDENTNUMBER.
    for tok in tokens:
        if int(tok["y"]) < min_row_y:
            continue
        parsed = _parse_bracket_token(str(tok["text"]))
        if parsed is None:
            continue
        x = int(tok["x"])
        y = int(tok["y"])
        w = int(tok["w"])
        h = int(tok["h"])
        return x, y, x + w, y + h

    # Next: find separate [ and ] tokens on a row under STUDENTNUMBER.
    lefts = [t for t in tokens if "[" in str(t["text"])]
    rights = [t for t in tokens if "]" in str(t["text"])]
    lefts = [t for t in lefts if int(t["y"]) >= min_row_y]
    rights = [t for t in rights if int(t["y"]) >= min_row_y]

    for left in lefts:
        lx = int(left["x"])
        ly = int(left["y"])
        lh = int(left["h"])
        candidates = [r for r in rights if int(r["x"]) > lx + 20 and abs(int(r["y"]) - ly) <= max(12, lh)]
        if not candidates:
            continue
        right = min(candidates, key=lambda r: int(r["x"]))
        rx = int(right["x"]) + int(right["w"])
        y0 = max(0, min(ly, int(right["y"])) - 2)
        y1 = min(roi_gray.shape[0], max(ly + lh, int(right["y"]) + int(right["h"])) + 2)
        return lx, y0, rx, y1

    return None


def _extract_bracket_digits_by_slots(roi_gray: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[str, bool]:
    x0, y0, x1, y1 = bounds
    x0 = max(0, min(x0, roi_gray.shape[1] - 1))
    x1 = max(x0 + 1, min(x1, roi_gray.shape[1]))
    y0 = max(0, min(y0, roi_gray.shape[0] - 1))
    y1 = max(y0 + 1, min(y1, roi_gray.shape[0]))
    bracket_roi = roi_gray[y0:y1, x0:x1]
    if bracket_roi.size == 0:
        return "00000", True

    digits: list[str] = []
    low_conf = False
    for slot in range(5):
        sx0 = int(round(slot * bracket_roi.shape[1] / 5.0))
        sx1 = int(round((slot + 1) * bracket_roi.shape[1] / 5.0))
        if sx1 <= sx0:
            digits.append("0")
            low_conf = True
            continue
        cell = bracket_roi[:, sx0:sx1]
        if cell.size == 0:
            digits.append("0")
            low_conf = True
            continue

        cell = cv2.copyMakeBorder(cell, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
        _, th = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(th)
        hit = detect_with_fallback(
            [cell, th, inv],
            pattern=r"\d",
            min_len=1,
            max_len=1,
            tesseract_configs=[
                "--psm 10 -c tessedit_char_whitelist=0123456789",
                "--psm 13 -c tessedit_char_whitelist=0123456789",
            ],
            min_tesseract_confidence=45.0,
        )
        if hit and hit.text.isdigit():
            digits.append(hit.text)
            if hit.confidence < 55.0:
                low_conf = True
        else:
            digits.append("0")
            low_conf = True
    return "".join(digits), low_conf


def extract_student_id_handwritten_ocr(img_bgr: np.ndarray) -> tuple[str | None, bool]:
    h, w = img_bgr.shape[:2]
    x0 = int(0.02 * w)
    x1 = int(0.48 * w)
    y0 = int(0.10 * h)
    y1 = int(0.24 * h)

    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bracket_bounds = _find_studentnumber_bracket_bounds(gray)
    if bracket_bounds is not None:
        digits, low_conf = _extract_bracket_digits_by_slots(gray, bracket_bounds)
        return digits, low_conf

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
        bracket_parse = _parse_bracket_token(match.text)
        if bracket_parse is not None:
            return bracket_parse, match.confidence < 55.0

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


def _ocr_tokens(gray: np.ndarray) -> list[dict[str, float | str]]:
    try:
        data = pytesseract.image_to_data(
            gray,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    tokens: list[dict[str, float | str]] = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 25.0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        tokens.append(
            {
                "text": text,
                "norm": re.sub(r"[^A-Z0-9#]", "", text.upper()),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2.0),
                "cy": y + (h / 2.0),
            }
        )
    return tokens


def _detect_bubble_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
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
    return candidates


def _nearest_cluster(target: float, clusters: list[float], max_dist: float) -> float | None:
    if not clusters:
        return None
    nearest = min(clusters, key=lambda c: abs(c - target))
    if abs(nearest - target) > max_dist:
        return None
    return float(nearest)


def _resolve_grid_from_anchors(
    gray: np.ndarray,
    candidates: list[tuple[float, float, float]],
) -> tuple[list[float], list[float], bool]:
    x_clusters = _cluster_positions(
        [c[0] for c in candidates], tolerance=max(8.0, gray.shape[1] * 0.03)
    )
    y_clusters = _cluster_positions(
        [c[1] for c in candidates], tolerance=max(6.0, gray.shape[0] * 0.02)
    )
    if len(x_clusters) < 5 or len(y_clusters) < 10:
        return [], [], False

    columns = sorted(x_clusters)[:5]
    rows = sorted(y_clusters)[:10]

    tokens = _ocr_tokens(gray)
    if not tokens:
        return columns, rows, False

    heading_found = any("STUDENTNUMBER" in str(t["norm"]) for t in tokens)
    hash_tok = next((t for t in tokens if str(t["norm"]) == "#"), None)

    digit_tokens = [t for t in tokens if str(t["norm"]).isdigit()]
    col_label_tokens = [t for t in digit_tokens if str(t["norm"]) in {"1", "2", "3", "4", "5"}]

    if hash_tok and len(col_label_tokens) >= 3:
        hash_y = float(hash_tok["cy"])
        same_line = [
            t for t in col_label_tokens if abs(float(t["cy"]) - hash_y) <= max(18.0, gray.shape[0] * 0.04)
        ]
        if len(same_line) >= 3:
            by_digit = {
                int(str(t["norm"])): float(t["cx"])
                for t in sorted(same_line, key=lambda t: float(t["cx"]))
            }
            anchored_cols: list[float] = []
            for idx in range(1, 6):
                target = by_digit.get(idx)
                if target is None:
                    anchored_cols.append(columns[idx - 1])
                    continue
                snapped = _nearest_cluster(target, x_clusters, max_dist=max(16.0, gray.shape[1] * 0.05))
                anchored_cols.append(snapped if snapped is not None else columns[idx - 1])
            columns = anchored_cols

    row_label_tokens = [t for t in digit_tokens if str(t["norm"]) in {str(i) for i in range(10)}]
    if len(row_label_tokens) >= 6 and hash_tok:
        hash_y = float(hash_tok["cy"])
        below_hash = [t for t in row_label_tokens if float(t["cy"]) > hash_y + 8]
        if below_hash:
            by_digit_row = {int(str(t["norm"])): float(t["cy"]) for t in below_hash}
            anchored_rows: list[float] = []
            for idx in range(10):
                target_y = by_digit_row.get(idx)
                if target_y is None:
                    anchored_rows.append(rows[idx])
                    continue
                snapped = _nearest_cluster(target_y, y_clusters, max_dist=max(16.0, gray.shape[0] * 0.05))
                anchored_rows.append(snapped if snapped is not None else rows[idx])
            rows = anchored_rows

    return columns, rows, heading_found


def _resolve_grid_from_ocr_labels(gray: np.ndarray) -> tuple[list[float], list[float], bool]:
    tokens = _ocr_tokens(gray)
    if not tokens:
        return [], [], False

    heading = next((t for t in tokens if "STUDENTNUMBER" in str(t["norm"])), None)
    hash_tok = next((t for t in tokens if str(t["norm"]) == "#"), None)
    if heading is None or hash_tok is None:
        return [], [], False

    hash_y = float(hash_tok["cy"])
    col_tokens = [
        t
        for t in tokens
        if str(t["norm"]) in {"1", "2", "3", "4", "5"}
        and abs(float(t["cy"]) - hash_y) <= max(22.0, gray.shape[0] * 0.05)
    ]
    if len(col_tokens) < 5:
        return [], [], True
    col_by_digit = {
        int(str(t["norm"])): float(t["cx"]) for t in sorted(col_tokens, key=lambda t: float(t["cx"]))
    }
    if any(i not in col_by_digit for i in range(1, 6)):
        return [], [], True
    columns = [col_by_digit[i] for i in range(1, 6)]

    row_tokens = [t for t in tokens if str(t["norm"]) in {str(i) for i in range(10)}]
    row_tokens = [t for t in row_tokens if float(t["cy"]) > hash_y + max(8.0, gray.shape[0] * 0.015)]
    by_digit_row: dict[int, float] = {}
    for t in sorted(row_tokens, key=lambda t: float(t["cy"])):
        digit = int(str(t["norm"]))
        by_digit_row.setdefault(digit, float(t["cy"]))

    if any(i not in by_digit_row for i in range(10)):
        return [], [], True
    rows = [by_digit_row[i] for i in range(10)]
    return columns, rows, True


def _extract_student_id_from_grid(
    gray: np.ndarray,
    columns: list[float],
    rows: list[float],
) -> tuple[str | None, bool]:
    if len(columns) < 5 or len(rows) < 10:
        return None, True

    col_gap = float(np.median(np.diff(sorted(columns)))) if len(columns) >= 2 else 36.0
    row_gap = float(np.median(np.diff(sorted(rows)))) if len(rows) >= 2 else 22.0
    radius = int(round(max(6.0, min(12.0, min(col_gap, row_gap) * 0.28))))

    scores_by_col: list[list[float]] = [[] for _ in range(5)]
    for col_idx, cx in enumerate(columns):
        for cy in rows:
            score = sample_bubble_score(gray, int(round(cx)), int(round(cy)), radius=radius)
            scores_by_col[col_idx].append(score)

    digits: list[str] = []
    low_conf = False
    for col_scores in scores_by_col:
        if len(col_scores) != 10:
            return None, True
        sorted_scores = sorted(col_scores, reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        if top < 18.0:
            digits.append("0")
            low_conf = True
            continue

        best_idx = int(np.argmax(col_scores))
        if (top - second) < 2.5:
            low_conf = True
        digits.append(str(best_idx))
    return "".join(digits), low_conf


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
    candidates = _detect_bubble_candidates(gray)
    columns: list[float] = []
    rows: list[float] = []
    if len(candidates) >= 12:
        columns, rows, _ = _resolve_grid_from_anchors(gray, candidates)

    if len(columns) < 5 or len(rows) < 10:
        columns, rows, heading_found = _resolve_grid_from_ocr_labels(gray)
        if not heading_found:
            return None, True

    return _extract_student_id_from_grid(gray, columns, rows)


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

    if mode == "bubble":
        return {
            "student_id": bubble_id,
            "student_id_all": bubble_id if bubble_id else None,
            "student_id_qr": qr_id,
            "student_id_text": None,
            "student_id_bubble": bubble_id,
            "student_id_handwritten": handwritten_id,
            "student_id_source": "text_bubble" if bubble_id else "missing",
            "id_conflict": False,
            "needs_review": bubble_id is None,
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
