from __future__ import annotations

import re
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract

from core.bubbles import append_output_rows, extract_row_answers
from core.ocr_engine import detect_with_fallback
from core.qr import decode_qr_robust, parse_qr_data
from core.rows import extract_rows_for_section
from core.sections import detect_section_blocks, find_option_centers
from core.student_id import (
    extract_student_id_handwritten_ocr,
    extract_student_id_printed_text,
    resolve_student_id,
)

RENDER_ZOOM = 3.5
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def has_usable_text_layer(page: fitz.Page) -> bool:
    text = page.get_text("text", sort=True) or ""
    return len(re.sub(r"[^A-Za-z0-9]", "", text)) >= 20


def _ocr_data(img_bgr: np.ndarray, *, psm: int = 6) -> dict[str, list[Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_data(
        th,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )


def _estimate_skew_correction_degrees(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    foreground = cv2.countNonZero(bw)
    total = bw.shape[0] * bw.shape[1]
    if total <= 0 or (foreground / total) < 0.005:
        return 0.0

    ys, xs = np.where(bw > 0)
    if len(xs) < 300:
        return 0.0

    coords = np.column_stack((xs, ys)).astype(np.float32)
    _, _, raw_angle = cv2.minAreaRect(coords)
    if raw_angle < -45:
        correction = -(90 + raw_angle)
    else:
        correction = -raw_angle

    if abs(correction) < 0.35:
        return 0.0
    return float(np.clip(correction, -60.0, 60.0))


def _rotate_image_keep_size(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return img_bgr

    h, w = img_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def normalize_scanned_page_rotation(img_bgr: np.ndarray) -> np.ndarray:
    correction = _estimate_skew_correction_degrees(img_bgr)
    if abs(correction) < 1e-6:
        return img_bgr
    return _rotate_image_keep_size(img_bgr, correction)


def _largest_quadrilateral(
    contours: list[np.ndarray], min_area_ratio: float, img_area: float
) -> np.ndarray | None:
    best: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area <= best_area or area < (img_area * min_area_ratio):
            continue
        best = approx
        best_area = area
    return best


def _order_quad_points(quad: np.ndarray) -> np.ndarray:
    pts = quad.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def align_scanned_page_to_template(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h <= 20 or w <= 20:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quad = _largest_quadrilateral(contours, min_area_ratio=0.35, img_area=float(h * w))
    if quad is None:
        return img_bgr

    src = _order_quad_points(quad)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    aligned = cv2.warpPerspective(
        img_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned


def detect_section_blocks_scanned(img_bgr: np.ndarray) -> list[dict[str, Any]]:
    data = _ocr_data(img_bgr, psm=6)
    lines: dict[tuple[int, int, int, int], list[str]] = {}
    first_idx_by_line: dict[tuple[int, int, int, int], int] = {}

    for i, txt in enumerate(data["text"]):
        token = (txt or "").strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 20:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
            int(data["page_num"][i]),
        )
        lines.setdefault(key, []).append(token)
        first_idx_by_line.setdefault(key, i)

    def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            row_min = curr[0]
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + cost,
                    )
                )
                row_min = min(row_min, curr[-1])
            if row_min > max_dist:
                return max_dist + 1
            prev = curr
        return prev[-1]

    def _fuzzy_substring_match(text: str, pattern: str, max_dist: int = 2) -> bool:
        if pattern in text or text in pattern:
            return True
        if len(text) < 4:
            return False

        min_len = max(4, len(pattern) - max_dist)
        max_len = min(len(text), len(pattern) + max_dist)
        for win_len in range(min_len, max_len + 1):
            for start in range(0, len(text) - win_len + 1):
                part = text[start : start + win_len]
                if _levenshtein(part, pattern, max_dist=max_dist) <= max_dist:
                    return True
        return False

    heading_specs = [
        ("MC", ("MULTIPLECHOICE", "MULTIPLE", "CHOICE")),
        ("TF", ("TRUEFALSE", "TRUE", "FALSE")),
        ("MATCH", ("MATCHING", "MATCH")),
        ("OPEN_ENDED", ("OPENENDED", "OPEN", "ENDED")),
    ]

    found: list[tuple[str, int]] = []
    for key, toks in lines.items():
        normalized = re.sub(r"[^A-Z]", "", " ".join(toks).upper())
        for section_type, patterns in heading_specs:
            if any(_fuzzy_substring_match(normalized, pat) for pat in patterns):
                found.append((section_type, int(data["top"][first_idx_by_line[key]])))
                break

    found.sort(key=lambda x: x[1])
    if not found:
        return []

    h = img_bgr.shape[0]
    blocks: list[dict[str, Any]] = []
    for idx, (section_type, y_top) in enumerate(found):
        y0 = max(0, y_top + 20)
        y1 = found[idx + 1][1] - 8 if idx + 1 < len(found) else h - 10
        if y1 > y0 + 10:
            blocks.append({"section_type": section_type, "y0_px": y0, "y1_px": y1})
    print(f"[extractor][scanned] detected sections: {len(blocks)}")
    return blocks


def extract_question_ids_from_roi(roi_bgr: np.ndarray) -> str | None:
    if roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)

    match = detect_with_fallback(
        [upscaled, th, inv],
        pattern=r"\d{5,6}",
        min_len=5,
        max_len=6,
        tesseract_configs=[
            "--psm 6 -c tessedit_char_whitelist=0123456789",
            "--psm 7 -c tessedit_char_whitelist=0123456789",
        ],
        min_tesseract_confidence=58.0,
    )
    return match.text if match else None


def extract_item_numbers_from_roi(roi_bgr: np.ndarray) -> int | None:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)

    match = detect_with_fallback(
        [gray, th, inv],
        pattern=r"[1-4]",
        min_len=1,
        max_len=1,
        tesseract_configs=[
            "--psm 6 -c tessedit_char_whitelist=1234",
            "--psm 7 -c tessedit_char_whitelist=1234",
        ],
        min_tesseract_confidence=50.0,
    )
    return int(match.text) if match else None


def extract_rows_for_section_scanned(
    img_bgr: np.ndarray, section: dict[str, Any]
) -> list[dict[str, Any]]:
    y0 = max(0, int(section["y0_px"]))
    y1 = min(img_bgr.shape[0], int(section["y1_px"]))
    if y1 <= y0:
        return []

    section_roi = img_bgr[y0:y1, :]
    gray = cv2.cvtColor(section_roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(bw > 0, axis=1).astype(np.float32)
    smooth_window = max(5, min(21, (bw.shape[0] // 30) | 1))
    smooth_kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    projection_smoothed = np.convolve(projection, smooth_kernel, mode="same")

    threshold = max(8.0, float(np.percentile(projection_smoothed, 45)))
    active = projection_smoothed > threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= 6:
                bands.append((start, i))
            start = None
    if start is not None and (len(active) - start) >= 6:
        bands.append((start, len(active) - 1))

    if not bands:
        print(f"[extractor][scanned] rows in {section['section_type']}: 0 (missing question_id: 0)")
        return []

    section_h = max(1, y1 - y0)
    min_row_h = max(10, int(section_h * 0.015))
    max_row_h = max(min_row_h + 8, int(section_h * 0.18))
    merge_gap = max(3, min_row_h // 2)

    merged_bands: list[tuple[int, int]] = []
    for b0, b1 in bands:
        if not merged_bands:
            merged_bands.append((b0, b1))
            continue
        prev0, prev1 = merged_bands[-1]
        if b0 - prev1 <= merge_gap:
            merged_bands[-1] = (prev0, b1)
        else:
            merged_bands.append((b0, b1))

    refined_bands: list[tuple[int, int]] = []
    for b0, b1 in merged_bands:
        height = b1 - b0
        if height < min_row_h:
            continue
        if height > max_row_h:
            split_count = max(2, int(round(height / max_row_h)))
            split_h = max(min_row_h, height // split_count)
            split_start = b0
            while split_start < b1:
                split_end = min(b1, split_start + split_h)
                if split_end - split_start >= min_row_h:
                    refined_bands.append((split_start, split_end))
                split_start = split_end
            continue
        refined_bands.append((b0, b1))

    rows: list[dict[str, Any]] = []
    w = img_bgr.shape[1]
    qid_x1 = int(w * 0.35)
    item_x0 = int(w * 0.18)

    missing_qid = 0
    for yb0, yb1 in refined_bands:
        y_abs0 = y0 + yb0
        y_abs1 = y0 + yb1
        if y_abs1 <= y_abs0:
            continue

        left_roi = img_bgr[y_abs0:y_abs1, :qid_x1]
        qid = extract_question_ids_from_roi(left_roi)
        needs_review = False
        if not qid:
            missing_qid += 1
            needs_review = True

        item_number = None
        if section["section_type"] == "MATCH":
            item_roi = img_bgr[y_abs0:y_abs1, item_x0:qid_x1]
            item_number = extract_item_numbers_from_roi(item_roi)
            if item_number is None:
                continue

        rows.append(
            {
                "section_type": section["section_type"],
                "question_id": qid,
                "item_number": item_number,
                "y_center_px": (y_abs0 + y_abs1) / 2.0,
                "row_bbox_px": (0, y_abs0, w, y_abs1),
                "needs_review": needs_review,
            }
        )

    print(
        f"[extractor][scanned] rows in {section['section_type']}: {len(rows)} "
        f"(missing question_id: {missing_qid})"
    )
    return rows


def _find_option_centers_scanned(
    section_type: str, page: fitz.Page, img_bgr: np.ndarray
) -> dict[str, float]:
    labels_by_section = {
        "MC": ["A", "B", "C", "D"],
        "TF": ["T", "F"],
        "MATCH": ["A", "B", "C", "D"],
    }
    labels = labels_by_section.get(section_type, [])
    if not labels:
        return {}

    section_layouts: dict[str, tuple[int, int]] = {
        "MC": (6, 3),
        "TF": (4, 3),
        "MATCH": (7, 4),
    }
    total_cols, answer_col_start = section_layouts.get(section_type, (0, 0))
    if total_cols == 0:
        return {}

    left_margin = 0.06
    usable_width = 0.90
    col_width = usable_width / total_cols
    x_ratios = [
        left_margin + ((answer_col_start - 1) + idx + 0.5) * col_width
        for idx in range(len(labels))
    ]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = img_bgr.shape[1] / page.rect.width
    centers: dict[str, float] = {}

    for lbl, ratio in zip(labels, x_ratios):
        x_px = int(img_bgr.shape[1] * ratio)
        x_candidates = [x_px + shift for shift in (-12, -8, -4, 0, 4, 8, 12)]
        x_candidates = [x for x in x_candidates if 5 <= x < (gray.shape[1] - 5)]
        if not x_candidates:
            centers[lbl] = x_px / sx
            continue

        best_x = x_px
        best_score = float("inf")
        for xc in x_candidates:
            col = gray[:, xc - 2 : xc + 3]
            score = float(np.mean(col))
            if score < best_score:
                best_score = score
                best_x = xc
        centers[lbl] = best_x / sx

    return centers


def _rows_px_to_page(
    rows_px: list[dict[str, Any]], page: fitz.Page, img_bgr: np.ndarray
) -> list[dict[str, Any]]:
    sx = img_bgr.shape[1] / page.rect.width
    sy = img_bgr.shape[0] / page.rect.height

    rows: list[dict[str, Any]] = []
    for row in rows_px:
        x0, y0, x1, y1 = row["row_bbox_px"]
        rows.append(
            {
                "section_type": row["section_type"],
                "question_id": row["question_id"],
                "item_number": row["item_number"],
                "y_center": row["y_center_px"] / sy,
                "row_bbox": (x0 / sx, y0 / sy, x1 / sx, y1 / sy),
            }
        )
    return rows


def render_page_to_bgr(page: fitz.Page, zoom: float = RENDER_ZOOM) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 1) Grayscale base + slight blue-ink boost for pen-filled bubbles.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue_boost = cv2.subtract(img[:, :, 0], img[:, :, 2])
    blue_boost = cv2.normalize(blue_boost, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.addWeighted(gray, 0.85, blue_boost, 0.15, 0)
    # 2) Denoise while preserving edges.
    denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    # 3) Correct uneven illumination, then enhance local contrast.
    background = cv2.GaussianBlur(denoised, (0, 0), 21)
    normalized = cv2.divide(denoised, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(normalized)
    # 4) Sharpen to make bubble/line boundaries clearer.
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(contrast, ddepth=-1, kernel=sharpen_kernel)
    # 5) Adaptive thresholding to handle scan brightness variation.
    thresh = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    # 6) Morphological cleanup for small isolated noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Preserve grayscale details while selectively strengthening detected marks.
    base = cv2.addWeighted(normalized, 0.65, sharp, 0.35, 0)
    preprocessed = np.where(cleaned > 0, sharp, base).astype(np.uint8)

    # Keep downstream contracts unchanged (3-channel BGR image).
    return cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sort_cols = ["student_id", "page", "section", "question_id", "item_number", "answer"]

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        key: list[Any] = []
        for col in sort_cols:
            val = row.get(col)
            key.append((val is None, "" if val is None else str(val)))
        return tuple(key)

    return sorted(rows, key=sort_key)


def _open_file_as_document(file_path: str) -> fitz.Document:
    try:
        return fitz.open(file_path)
    except Exception:
        lower_path = file_path.lower()
        if not any(lower_path.endswith(ext) for ext in IMAGE_SUFFIXES):
            raise

    img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Unsupported or unreadable image file: {file_path}")

    h, w = img_bgr.shape[:2]
    success, encoded = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError(f"Failed to encode image file: {file_path}")

    doc = fitz.open()
    page = doc.new_page(width=float(w), height=float(h))
    page.insert_image(page.rect, stream=encoded.tobytes())
    return doc


def extract_file(file_path: str, student_id_mode: str) -> list[dict[str, Any]]:
    doc = _open_file_as_document(file_path)
    all_rows: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        img_bgr = render_page_to_bgr(page)
        page_number = page_idx + 1
        use_text_layer = has_usable_text_layer(page)
        if not use_text_layer:
            img_bgr = normalize_scanned_page_rotation(img_bgr)
            img_bgr = align_scanned_page_to_template(img_bgr)
        print(
            f"[extractor] page {page_number}: "
            f"{'text-layer mode' if use_text_layer else 'scanned-image mode'}"
        )

        qr_raw = decode_qr_robust(img_bgr)
        if not qr_raw:
            all_rows.append(
                {
                    "page": page_number,
                    "student_id": None,
                    "student_id_all": None,
                    "student_id_qr": None,
                    "student_id_text": None,
                    "student_id_handwritten": None,
                    "student_id_source": "missing",
                    "id_conflict": False,
                    "sheet_type": None,
                    "exam_id": None,
                    "test_id": None,
                    "section": None,
                    "question_id": None,
                    "item_number": None,
                    "answer": None,
                    "matching_pair": None,
                    "manual_marking": False,
                    "needs_review": True,
                    "score_debug": None,
                    "qr_raw": None,
                }
            )
            continue

        qr_parsed = parse_qr_data(qr_raw)
        qr_student_id = qr_parsed.get("studentid")
        qr_type = (qr_parsed.get("type") or "").upper() or None

        exam_id = None
        test_id = None
        if qr_type == "EXAM":
            exam_id = qr_parsed.get("examid") or qr_parsed.get("assessmentid")
        elif qr_type == "TEST":
            test_id = qr_parsed.get("testid") or qr_parsed.get("assessmentid")

        printed_id = extract_student_id_printed_text(page)
        handwritten_id, _ = extract_student_id_handwritten_ocr(img_bgr)

        id_info = resolve_student_id(
            qr_id=qr_student_id,
            printed_id=printed_id,
            handwritten_id=handwritten_id,
            mode=student_id_mode,
        )

        base = {
            "page": page_number,
            "student_id": id_info["student_id"],
            "student_id_all": id_info["student_id_all"],
            "student_id_qr": id_info["student_id_qr"],
            "student_id_text": id_info["student_id_text"],
            "student_id_handwritten": id_info["student_id_handwritten"],
            "student_id_source": id_info["student_id_source"],
            "id_conflict": id_info["id_conflict"],
            "sheet_type": qr_type,
            "exam_id": exam_id if qr_type == "EXAM" else None,
            "test_id": test_id if qr_type == "TEST" else None,
            "qr_raw": qr_raw,
        }

        sections = (
            detect_section_blocks(page)
            if use_text_layer
            else detect_section_blocks_scanned(img_bgr)
        )

        for section in sections:
            if section["section_type"] == "OPEN_ENDED":
                continue

            if use_text_layer:
                option_centers = find_option_centers(page, section)
                rows = extract_rows_for_section(page, section)
            else:
                option_centers = _find_option_centers_scanned(
                    section["section_type"], page, img_bgr
                )
                rows = _rows_px_to_page(
                    extract_rows_for_section_scanned(img_bgr, section), page, img_bgr
                )

            for row in rows:
                answers, row_review, scores = extract_row_answers(
                    page=page,
                    img_bgr=img_bgr,
                    section=section,
                    row=row,
                    option_centers=option_centers,
                )

                review_flag = (
                    row_review or id_info["needs_review"] or (row["question_id"] is None)
                )

                append_output_rows(
                    output_rows=all_rows,
                    base=base,
                    row=row,
                    answers=answers,
                    needs_review=review_flag,
                    scores=scores,
                )

    doc.close()
    return _sorted_rows(all_rows)
