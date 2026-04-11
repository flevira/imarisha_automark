from __future__ import annotations

import re
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract

from core.bubbles import append_output_rows, extract_row_answers
from core.qr import decode_qr_robust, parse_qr_data
from core.rows import extract_rows_for_section
from core.sections import detect_section_blocks, find_option_centers
from core.student_id import (
    extract_student_id_handwritten_ocr,
    extract_student_id_printed_text,
    resolve_student_id,
)

RENDER_ZOOM = 3.5
INCLUDE_OPEN_ENDED = True


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

    heading_specs = [
        ("MC", ("MULTIPLECHOICE",)),
        ("TF", ("TRUEFALSE",)),
        ("MATCH", ("MATCHING",)),
        ("OPEN_ENDED", ("OPENENDED",)),
    ]

    found: list[tuple[str, int]] = []
    for key, toks in lines.items():
        normalized = re.sub(r"[^A-Z]", "", " ".join(toks).upper())
        for section_type, patterns in heading_specs:
            if any(pat in normalized for pat in patterns):
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
    return blocks


def extract_question_ids_from_roi(roi_bgr: np.ndarray) -> str | None:
    txt = pytesseract.image_to_string(
        roi_bgr,
        config="--psm 6 -c tessedit_char_whitelist=0123456789",
    )
    m = re.search(r"\d{5,6}", txt or "")
    return m.group(0) if m else None


def extract_item_numbers_from_roi(roi_bgr: np.ndarray) -> int | None:
    txt = pytesseract.image_to_string(
        roi_bgr,
        config="--psm 6 -c tessedit_char_whitelist=1234",
    )
    m = re.search(r"[1-4]", txt or "")
    return int(m.group(0)) if m else None


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

    projection = np.sum(bw > 0, axis=1)
    active = projection > max(10, int(0.01 * bw.shape[1]))

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= 10:
                bands.append((start, i))
            start = None
    if start is not None and (len(active) - start) >= 10:
        bands.append((start, len(active) - 1))

    rows: list[dict[str, Any]] = []
    w = img_bgr.shape[1]
    qid_x1 = int(w * 0.35)
    item_x0 = int(w * 0.18)

    for yb0, yb1 in bands:
        y_abs0 = y0 + yb0
        y_abs1 = y0 + yb1
        if y_abs1 <= y_abs0:
            continue

        left_roi = img_bgr[y_abs0:y_abs1, :qid_x1]
        qid = extract_question_ids_from_roi(left_roi)
        if not qid:
            continue

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
            }
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

    if section_type == "TF":
        x_ratios = [0.78, 0.88]
    else:
        x_ratios = [0.70, 0.78, 0.86, 0.94]

    sx = img_bgr.shape[1] / page.rect.width
    return {lbl: (img_bgr.shape[1] * ratio) / sx for lbl, ratio in zip(labels, x_ratios)}


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


def extract_file(file_path: str, student_id_mode: str) -> list[dict[str, Any]]:
    doc = fitz.open(file_path)
    all_rows: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        img_bgr = render_page_to_bgr(page)
        page_number = page_idx + 1
        use_text_layer = has_usable_text_layer(page)
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

            if section["section_type"] == "OPEN_ENDED":
                if INCLUDE_OPEN_ENDED:
                    for row in rows:
                        append_output_rows(
                            output_rows=all_rows,
                            base=base,
                            row=row,
                            answers=[],
                            needs_review=True,
                            scores={},
                        )
                continue

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
