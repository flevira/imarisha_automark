from __future__ import annotations

import re
from pathlib import Path
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
    extract_student_id_bubble,
    extract_student_id_handwritten_ocr,
    resolve_student_id,
)

RENDER_ZOOM = 4.0
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_LAYER_SKEW_FALLBACK_DEG = 2.5
OCR_OVERLAY_MIN_ALNUM = 20
OCR_OVERLAY_PSM_MODES = (6, 11)
OCR_OVERLAY_IOU_THRESHOLD = 0.3
OCR_OVERLAY_CENTER_DISTANCE_RATIO = 0.65
OCR_OVERLAY_MIN_CONF = 40.0
OCR_OVERLAY_MIN_CONF_SINGLE_CHAR = 65.0

_OCR_OVERLAY_CACHE: dict[str, bytes] = {}


def has_usable_text_layer(page: fitz.Page) -> bool:
    text = page.get_text("text", sort=True) or ""
    return len(re.sub(r"[^A-Za-z0-9]", "", text)) >= OCR_OVERLAY_MIN_ALNUM


def _bbox_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    l1, t1, w1, h1 = b1
    l2, t2, w2, h2 = b2
    r1, btm1 = l1 + w1, t1 + h1
    r2, btm2 = l2 + w2, t2 + h2

    inter_w = max(0, min(r1, r2) - max(l1, l2))
    inter_h = max(0, min(btm1, btm2) - max(t1, t2))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _center_distance_ratio(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    l1, t1, w1, h1 = b1
    l2, t2, w2, h2 = b2
    c1x, c1y = l1 + (w1 / 2.0), t1 + (h1 / 2.0)
    c2x, c2y = l2 + (w2 / 2.0), t2 + (h2 / 2.0)
    distance = float(np.hypot(c1x - c2x, c1y - c2y))
    normalizer = max(1.0, min(w1, h1, w2, h2))
    return distance / normalizer


def _aggregate_ocr_token_candidates(
    candidates: list[dict[str, Any]],
    *,
    iou_threshold: float = OCR_OVERLAY_IOU_THRESHOLD,
    center_distance_ratio: float = OCR_OVERLAY_CENTER_DISTANCE_RATIO,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: float(item["conf"]), reverse=True)
    groups: list[list[dict[str, Any]]] = []

    for candidate in sorted_candidates:
        bbox = candidate["bbox"]
        matched_group: list[dict[str, Any]] | None = None
        for group in groups:
            group_best = group[0]
            group_bbox = group_best["bbox"]
            if _bbox_iou(bbox, group_bbox) >= iou_threshold:
                matched_group = group
                break
            if _center_distance_ratio(bbox, group_bbox) <= center_distance_ratio:
                matched_group = group
                break
        if matched_group is None:
            groups.append([candidate])
            continue
        matched_group.append(candidate)

    return [group[0] for group in groups]


def _ocr_data_from_gray(gray: np.ndarray, *, psm: int = 6) -> dict[str, list[Any]]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inverted = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    variants = {
        "gray": gray,
        "otsu_binary": otsu_binary,
        "otsu_inverted": otsu_inverted,
        "adaptive": adaptive,
    }

    psm_modes = tuple(dict.fromkeys((psm, *OCR_OVERLAY_PSM_MODES)))
    candidates: list[dict[str, Any]] = []
    for variant_name, img in variants.items():
        for mode in psm_modes:
            data = pytesseract.image_to_data(
                img,
                config=f"--psm {mode}",
                output_type=pytesseract.Output.DICT,
            )
            for i, raw in enumerate(data.get("text", [])):
                token = (raw or "").strip()
                if not token:
                    continue
                conf_raw = str(data.get("conf", ["-1"])[i]).strip()
                try:
                    conf = float(conf_raw)
                except ValueError:
                    continue
                if conf < 0:
                    continue

                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])
                if width <= 0 or height <= 0:
                    continue
                candidates.append(
                    {
                        "text": token,
                        "conf": conf,
                        "bbox": (left, top, width, height),
                        "variant": variant_name,
                        "psm": mode,
                    }
                )

    best_tokens = _aggregate_ocr_token_candidates(candidates)
    return {
        "text": [item["text"] for item in best_tokens],
        "conf": [item["conf"] for item in best_tokens],
        "left": [item["bbox"][0] for item in best_tokens],
        "top": [item["bbox"][1] for item in best_tokens],
        "width": [item["bbox"][2] for item in best_tokens],
        "height": [item["bbox"][3] for item in best_tokens],
    }


def create_ocr_overlay_pdf(doc: fitz.Document) -> tuple[fitz.Document, bool, int]:
    overlay_doc = fitz.open()
    applied = False
    total_chars = 0

    for page_idx in range(len(doc)):
        src_page = doc[page_idx]
        page_rect = src_page.rect
        dst_page = overlay_doc.new_page(width=page_rect.width, height=page_rect.height)

        if has_usable_text_layer(src_page):
            dst_page.show_pdf_page(dst_page.rect, doc, page_idx)
            continue

        applied = True
        pix = src_page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), alpha=False)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        dst_page.insert_image(dst_page.rect, stream=pix.tobytes("png"))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        data = _ocr_data_from_gray(gray, psm=6)
        chars_on_page = 0
        inserted_tokens = 0
        inserted_conf_total = 0.0
        scale_x = page_rect.width / float(pix.width)
        scale_y = page_rect.height / float(pix.height)

        for i, raw in enumerate(data.get("text", [])):
            token = (raw or "").strip()
            if not token:
                continue
            conf_raw = data.get("conf", [-1])[i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                continue
            min_conf = OCR_OVERLAY_MIN_CONF_SINGLE_CHAR if len(token) <= 1 else OCR_OVERLAY_MIN_CONF
            if conf < min_conf:
                continue

            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])
            if width <= 0 or height <= 0:
                continue

            x = float(left) * scale_x
            y_bottom = float(top + height) * scale_y
            font_size = max(6.0, float(height) * scale_y * 0.95)
            dst_page.insert_text(
                fitz.Point(x, y_bottom),
                token,
                fontsize=font_size,
                render_mode=3,
                overlay=True,
            )
            chars_on_page += len(token)
            inserted_tokens += 1
            inserted_conf_total += conf

        total_chars += chars_on_page
        mean_conf = inserted_conf_total / inserted_tokens if inserted_tokens else 0.0
        print(
            f"[extractor] OCR overlay page {page_idx + 1}: tokens={inserted_tokens}, "
            f"chars={chars_on_page}, avg_conf={mean_conf:.1f}"
        )

    return overlay_doc, applied, total_chars


def _doc_cache_key(file_path: str) -> str:
    p = Path(file_path)
    try:
        stat = p.stat()
    except OSError:
        return file_path
    return f"{p.resolve()}::{int(stat.st_mtime_ns)}::{stat.st_size}"


def _prepare_document_for_extraction(doc: fitz.Document, file_path: str) -> tuple[fitz.Document, bool]:
    if all(has_usable_text_layer(doc[i]) for i in range(len(doc))):
        print("[extractor] Using native text layer")
        return doc, False

    print("[extractor] Applying OCR overlay (scanned mode)")
    cache_key = _doc_cache_key(file_path)
    cached = _OCR_OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        return fitz.open(stream=cached, filetype="pdf"), True

    overlay_doc, applied, ocr_chars = create_ocr_overlay_pdf(doc)
    print(f"[extractor] OCR characters extracted: {ocr_chars}")
    if applied:
        overlay_bytes = overlay_doc.tobytes()
        _OCR_OVERLAY_CACHE[cache_key] = overlay_bytes
        return fitz.open(stream=overlay_bytes, filetype="pdf"), True
    return overlay_doc, False


def _ocr_data(img_bgr: np.ndarray, *, psm: int = 6) -> dict[str, list[Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return _ocr_data_from_gray(gray, psm=psm)


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


def detect_section_blocks_scanned(
    img_bgr: np.ndarray,
    *,
    return_debug: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _ocr_data(img_bgr, psm=6)
    h = img_bgr.shape[0]
    w = img_bgr.shape[1]

    def _normalize_heading_token(token: str) -> str:
        cleaned = re.sub(r"[^A-Z]", "", token.upper())
        if not cleaned:
            return ""
        # OCR around bubble rows often pads tokens with repeated O/0 at edges.
        cleaned = re.sub(r"^[O0]{2,}", "", cleaned)
        cleaned = re.sub(r"[O0]{2,}$", "", cleaned)
        return cleaned

    def _is_heading_token_noise(token: str) -> bool:
        if len(token) < 4:
            return True
        if token == "STUDENTNUMBER":
            return True
        distinct = len(set(token))
        if distinct <= 1:
            return True
        o_ratio = sum(1 for ch in token if ch == "O") / float(len(token))
        return o_ratio > 0.6

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
        ("MC", ("MULTIPLE",)),
        ("TF", ("TRUEFALSE",)),
        ("MATCH", ("MATCHING",)),
        ("OPEN_ENDED", ("OPEN",)),
    ]

    def _find_sections(
        min_conf: float,
        *,
        max_dist: int = 2,
        positional_constraints: bool = False,
    ) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
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
            if conf < min_conf:
                continue
            left = int(data["left"][i])
            width = int(data["width"][i])
            top = int(data["top"][i])
            x_center = left + max(1, width) // 2
            if positional_constraints:
                in_left_band = x_center <= int(w * 0.45)
                in_right_band = x_center >= int(w * 0.55)
                if not (in_left_band or in_right_band):
                    continue
                if top < int(h * 0.10):
                    continue

            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
                int(data["page_num"][i]),
            )
            lines.setdefault(key, []).append(token)
            first_idx_by_line.setdefault(key, i)

        heading_candidates: list[tuple[str, int, int]] = []
        found_sections: list[tuple[str, int, int]] = []
        for key, toks in lines.items():
            normalized = re.sub(r"[^A-Z]", "", " ".join(toks).upper())
            candidate_tokens = [
                _normalize_heading_token(tok)
                for tok in toks
                if _normalize_heading_token(tok)
            ]
            candidate_tokens = [tok for tok in candidate_tokens if not _is_heading_token_noise(tok)]
            candidate_name = "".join(candidate_tokens)
            if len(candidate_name) >= 4:
                idx = first_idx_by_line[key]
                left = int(data["left"][idx])
                width = int(data["width"][idx])
                x_center = left + max(1, width) // 2
                heading_candidates.append((candidate_name, int(data["top"][idx]), x_center))

            for section_type, patterns in heading_specs:
                if any(
                    _fuzzy_substring_match(normalized, pat, max_dist=max_dist)
                    for pat in patterns
                ):
                    idx = first_idx_by_line[key]
                    left = int(data["left"][idx])
                    width = int(data["width"][idx])
                    x_center = left + max(1, width) // 2
                    found_sections.append(
                        (section_type, int(data["top"][idx]), x_center)
                    )
                    break
        return found_sections, heading_candidates

    found, heading_candidates = _find_sections(min_conf=20.0)
    if not found:
        found, heading_candidates = _find_sections(min_conf=10.0)
    if not found:
        found, heading_candidates = _find_sections(
            min_conf=7.0,
            max_dist=3,
            positional_constraints=True,
        )

    def _find_studentnumber_anchor_y() -> int | None:
        line_tokens: dict[tuple[int, int, int, int], list[tuple[str, int, int]]] = {}
        for i, txt in enumerate(data["text"]):
            token = (txt or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 12.0:
                continue

            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
                int(data["page_num"][i]),
            )
            top = int(data["top"][i])
            height = int(data["height"][i])
            line_tokens.setdefault(key, []).append((token.upper(), top, height))

        best_anchor: int | None = None
        for words in line_tokens.values():
            normalized = re.sub(
                r"[^A-Z]",
                "",
                "".join(token for token, *_ in words),
            )
            if "STUDENTNUMBER" not in normalized:
                continue
            line_bottom = max(top + h for _, top, h in words)
            if best_anchor is None or line_bottom > best_anchor:
                best_anchor = line_bottom
        return best_anchor

    studentnumber_anchor_y = _find_studentnumber_anchor_y()
    if studentnumber_anchor_y is not None and found:
        filtered = [item for item in found if item[1] >= (studentnumber_anchor_y - 20)]
        if len(filtered) >= 2:
            found = filtered
    if studentnumber_anchor_y is not None and heading_candidates:
        filtered_candidates = [
            item for item in heading_candidates if item[1] >= (studentnumber_anchor_y - 20)
        ]
        if filtered_candidates:
            heading_candidates = filtered_candidates

    mid_x = w // 2
    candidate_left_count = sum(1 for _, _, x in heading_candidates if x <= mid_x)
    candidate_right_count = sum(1 for _, _, x in heading_candidates if x > mid_x)
    heading_candidate_names = [name for name, _, _ in sorted(heading_candidates, key=lambda t: t[1])]
    if not found:
        empty_debug = {
            "raw_headings_detected": len(heading_candidates),
            "column_left_headings": candidate_left_count,
            "column_right_headings": candidate_right_count,
            "heading_candidates": heading_candidate_names,
            "left_column_sections": 0,
            "right_column_sections": 0,
        }
        return ([], empty_debug) if return_debug else []

    left_margin = max(10, int(w * 0.03))
    right_margin = max(4, int(w * 0.012))
    blocks: list[dict[str, Any]] = []

    # Use geometric page-width split only (50/50) to keep behavior deterministic.
    split_x = mid_x

    # Use one deterministic split strategy only:
    # headings below STUDENTNUMBER (when present), then split by divider/midpoint.
    left_col = [item for item in found if item[2] <= split_x]
    right_col = [item for item in found if item[2] > split_x]
    split_source = "midpoint"

    def _build_column_blocks(
        column_items: list[tuple[str, int, int]], x0: int, x1: int
    ) -> list[dict[str, Any]]:
        if not column_items:
            return []
        column_items.sort(key=lambda x: x[1])
        deduped: list[tuple[str, int, int]] = []
        for section_type, y_top, x_left in column_items:
            if (
                deduped
                and section_type == deduped[-1][0]
                and abs(y_top - deduped[-1][1]) <= 16
            ):
                continue
            deduped.append((section_type, y_top, x_left))

        col_blocks: list[dict[str, Any]] = []
        for idx, (section_type, y_top, _) in enumerate(deduped):
            y0 = max(0, y_top + 20)
            y1 = deduped[idx + 1][1] - 8 if idx + 1 < len(deduped) else h - 10
            if y1 > y0 + 10:
                col_blocks.append(
                    {
                        "section_type": section_type,
                        "y0_px": y0,
                        "y1_px": y1,
                        "x0_px": max(0, x0),
                        "x1_px": min(w, x1),
                    }
                )
        return col_blocks

    left_blocks = _build_column_blocks(left_col, 0, split_x - left_margin)
    right_blocks = _build_column_blocks(right_col, split_x + right_margin, w)
    blocks.extend(left_blocks)
    blocks.extend(right_blocks)
    print(f"[extractor][scanned] detected sections: {len(blocks)}")
    debug = {
        "raw_headings_detected": len(heading_candidates),
        "column_left_headings": candidate_left_count,
        "column_right_headings": candidate_right_count,
        "heading_candidates": heading_candidate_names,
        "left_column_sections": len(left_blocks),
        "right_column_sections": len(right_blocks),
        "column_split_x_px": int(split_x),
        "column_split_source": split_source,
    }
    return (blocks, debug) if return_debug else blocks


def _detect_vertical_column_divider_x(
    *,
    img_bgr: np.ndarray,
    y_start: int,
) -> int | None:
    if img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]
    y0 = max(0, min(h - 1, int(y_start)))
    roi = img_bgr[y0:h, :]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = max(25, int(roi.shape[0] * 0.08))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    col_counts = np.sum(vertical > 0, axis=0).astype(np.float32)
    if col_counts.size < 10:
        return None

    x_lo = int(w * 0.4)
    x_hi = int(w * 0.6)
    if x_hi <= x_lo:
        return None

    search = col_counts[x_lo:x_hi]
    if search.size == 0:
        return None

    best_idx = int(np.argmax(search))
    best_strength = float(search[best_idx])
    min_strength = max(20.0, float(roi.shape[0] * 0.28))
    if best_strength < min_strength:
        return None

    return x_lo + best_idx


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
        min_tesseract_confidence=50.0,
    )
    return match.text if match else None


def extract_question_id_with_fallbacks(
    row_roi_bgr: np.ndarray,
    *,
    section_type: str,
) -> str | None:
    if row_roi_bgr.size == 0:
        return None

    h, w = row_roi_bgr.shape[:2]
    if h <= 2 or w <= 2:
        return None

    def _slice(x_start_ratio: float, x_end_ratio: float) -> np.ndarray:
        x0 = max(0, min(w - 1, int(round(w * x_start_ratio))))
        x1 = max(x0 + 1, min(w, int(round(w * x_end_ratio))))
        return row_roi_bgr[:, x0:x1]

    qid = extract_question_ids_from_roi(_slice(0.0, 0.35))
    if qid:
        return qid

    # Scanned pages often have slight section x-boundary drift. Probe a wider
    # area to recover the QID when the narrow left crop misses the printed text.
    fallback_windows = [
        (0.0, 0.50),
        (0.05, 0.60),
        (0.0, 0.70),
    ]
    if section_type == "MATCH":
        fallback_windows.extend([(0.10, 0.65), (0.15, 0.70)])

    for start, end in fallback_windows:
        qid = extract_question_ids_from_roi(_slice(start, end))
        if qid:
            return qid
    return None


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
        min_tesseract_confidence=44.0,
    )
    return int(match.text) if match else None


def extract_rows_for_section_scanned(
    img_bgr: np.ndarray, section: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    debug: dict[str, Any] = {
        "header_offset_px": 0,
        "bands_raw_count": 0,
        "bands_merged_count": 0,
        "bands_refined_count": 0,
        "rows_candidate_count": 0,
        "rows_dropped_missing_item_number": 0,
        "rows_missing_question_id": 0,
    }
    y0 = max(0, int(section["y0_px"]))
    y1 = min(img_bgr.shape[0], int(section["y1_px"]))
    if y1 <= y0:
        return [], debug

    x0 = max(0, int(section.get("x0_px", 0)))
    x1 = min(img_bgr.shape[1], int(section.get("x1_px", img_bgr.shape[1])))
    if x1 <= x0:
        return [], debug

    section_roi = img_bgr[y0:y1, x0:x1]
    data_start_offset = _find_scanned_table_header_bottom(section_roi)
    if data_start_offset is not None and data_start_offset > 0:
        debug["header_offset_px"] = int(data_start_offset)
        y0 = min(y1, y0 + data_start_offset)
        section_roi = img_bgr[y0:y1, x0:x1]
        if section_roi.size == 0:
            return [], debug
    gray = cv2.cvtColor(section_roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(bw > 0, axis=1).astype(np.float32)
    smooth_window = max(5, min(21, (bw.shape[0] // 30) | 1))
    smooth_kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    projection_smoothed = np.convolve(projection, smooth_kernel, mode="same")

    threshold = max(6.0, float(np.percentile(projection_smoothed, 35)))
    active = projection_smoothed > threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= 5:
                bands.append((start, i))
            start = None
    if start is not None and (len(active) - start) >= 5:
        bands.append((start, len(active) - 1))

    debug["bands_raw_count"] = len(bands)
    if not bands:
        print(f"[extractor][scanned] rows in {section['section_type']}: 0 (missing question_id: 0)")
        return [], debug

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

    debug["bands_merged_count"] = len(merged_bands)
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

    debug["bands_refined_count"] = len(refined_bands)
    row_candidates: list[dict[str, Any]] = []
    section_w = max(1, x1 - x0)
    qid_x1_local = int(section_w * 0.35)
    item_x0_local = int(section_w * 0.40)
    item_x1_local = int(section_w * 0.62)

    missing_qid = 0
    rows_dropped_missing_item_number = 0
    qid_candidates = _extract_scanned_qid_candidates(
        section_roi_bgr=section_roi,
        section_type=str(section["section_type"]),
        x_offset=x0,
        y_offset=y0,
    )
    for yb0, yb1 in refined_bands:
        y_abs0 = y0 + yb0
        y_abs1 = y0 + yb1
        if y_abs1 <= y_abs0:
            continue

        qid = _pick_qid_for_row_from_candidates(
            qid_candidates=qid_candidates,
            y_abs0=y_abs0,
            y_abs1=y_abs1,
        )
        row_roi = img_bgr[y_abs0:y_abs1, x0:x1]
        if not qid:
            qid = extract_question_id_with_fallbacks(
                row_roi,
                section_type=str(section["section_type"]),
            )
        needs_review = False
        if not qid:
            missing_qid += 1
            needs_review = True

        item_number = None
        if section["section_type"] == "MATCH":
            item_roi = img_bgr[
                y_abs0:y_abs1, x0 + item_x0_local : x0 + item_x1_local
            ]
            item_number = extract_item_numbers_from_roi(item_roi)
            if item_number is None:
                rows_dropped_missing_item_number += 1
                continue

        row_candidates.append(
            {
                "section_type": section["section_type"],
                "question_id": qid,
                "item_number": item_number,
                "y_center_px": (y_abs0 + y_abs1) / 2.0,
                "row_bbox_px": (x0, y_abs0, x1, y_abs1),
                "needs_review": needs_review,
            }
        )

    rows = _dedupe_scanned_row_candidates(row_candidates)
    debug["rows_candidate_count"] = len(row_candidates)
    debug["rows_dropped_missing_item_number"] = rows_dropped_missing_item_number
    debug["rows_missing_question_id"] = missing_qid

    print(
        f"[extractor][scanned] rows in {section['section_type']}: {len(rows)} "
        f"(missing question_id: {missing_qid})"
    )
    return rows, debug


def _extract_scanned_qid_candidates(
    section_roi_bgr: np.ndarray,
    *,
    section_type: str,
    x_offset: int,
    y_offset: int,
) -> list[dict[str, float | str]]:
    if section_roi_bgr.size == 0:
        return []

    h, w = section_roi_bgr.shape[:2]
    x_limit_ratio = 0.72 if section_type == "MATCH" else 0.62
    x_limit = int(max(10, w * x_limit_ratio))

    candidates: list[dict[str, float | str]] = []
    for psm in (6, 11):
        data = _ocr_data(section_roi_bgr, psm=psm)
        for i, txt in enumerate(data["text"]):
            token = re.sub(r"[^0-9]", "", (txt or ""))
            # Relax scanned debug capture: keep any numeric token so row matching can
            # expose where OCR is reading IDs, even when digit count is imperfect.
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 15:
                continue

            left = int(data["left"][i])
            width = int(data["width"][i])
            if left > x_limit:
                continue

            top = int(data["top"][i])
            height = int(data["height"][i])
            y_center = y_offset + top + (height / 2.0)
            x_center = x_offset + left + (width / 2.0)
            candidates.append(
                {
                    "question_id": token,
                    "y_center": float(y_center),
                    "x_center": float(x_center),
                }
            )

    if not candidates:
        return []

    deduped: dict[tuple[str, int], dict[str, float | str]] = {}
    for cand in candidates:
        qid = str(cand["question_id"])
        y_bucket = int(round(float(cand["y_center"]) / 8.0))
        deduped[(qid, y_bucket)] = cand

    return sorted(deduped.values(), key=lambda c: float(c["y_center"]))


def _pick_qid_for_row_from_candidates(
    *,
    qid_candidates: list[dict[str, float | str]],
    y_abs0: int,
    y_abs1: int,
) -> str | None:
    if not qid_candidates:
        return None

    row_center = (y_abs0 + y_abs1) / 2.0
    row_h = max(6.0, float(y_abs1 - y_abs0))
    max_dist = max(20.0, row_h * 1.35)

    best_qid: str | None = None
    best_dist = float("inf")
    for cand in qid_candidates:
        y = float(cand["y_center"])
        dist = abs(y - row_center)
        if dist <= max_dist and dist < best_dist:
            best_qid = str(cand["question_id"])
            best_dist = dist

    return best_qid


def _extract_scanned_item_candidates(
    *,
    section_roi_bgr: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> list[dict[str, float | int]]:
    if section_roi_bgr.size == 0:
        return []

    h, w = section_roi_bgr.shape[:2]
    x_min = int(w * 0.33)
    x_max = int(w * 0.72)
    y_max = int(h * 0.97)
    candidates: list[dict[str, float | int]] = []

    for psm in (6, 11):
        data = _ocr_data(section_roi_bgr, psm=psm)
        for i, txt in enumerate(data["text"]):
            token = re.sub(r"[^0-9]", "", (txt or ""))
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 12.0:
                continue

            left = int(data["left"][i])
            width = int(data["width"][i])
            x_center_local = left + (width / 2.0)
            if x_center_local < x_min or x_center_local > x_max:
                continue

            top = int(data["top"][i])
            height = int(data["height"][i])
            y_center_local = top + (height / 2.0)
            if y_center_local > y_max:
                continue

            candidates.append(
                {
                    "item_number": int(token),
                    "y_center": float(y_offset + y_center_local),
                    "x_center": float(x_offset + x_center_local),
                }
            )

    if not candidates:
        return []

    deduped: dict[tuple[int, int], dict[str, float | int]] = {}
    for cand in candidates:
        item_num = int(cand["item_number"])
        y_bucket = int(round(float(cand["y_center"]) / 6.0))
        deduped[(item_num, y_bucket)] = cand

    return sorted(deduped.values(), key=lambda c: float(c["y_center"]))


def _pick_item_number_for_row_from_candidates(
    *,
    item_candidates: list[dict[str, float | int]],
    y_abs0: int,
    y_abs1: int,
) -> int | None:
    if not item_candidates:
        return None

    row_center = (y_abs0 + y_abs1) / 2.0
    row_height = max(1.0, float(y_abs1 - y_abs0))
    max_delta = max(8.0, row_height * 0.9)
    best_item: int | None = None
    best_delta = float("inf")
    for cand in item_candidates:
        delta = abs(float(cand["y_center"]) - row_center)
        if delta <= max_delta and delta < best_delta:
            best_item = int(cand["item_number"])
            best_delta = delta
    return best_item


def _find_scanned_table_header_bottom(section_roi_bgr: np.ndarray) -> int | None:
    if section_roi_bgr.size == 0:
        return None

    data = _ocr_data(section_roi_bgr, psm=6)
    lines: dict[tuple[int, int, int, int], list[tuple[str, int, int]]] = {}

    for i, txt in enumerate(data["text"]):
        token = (txt or "").strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 15:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
            int(data["page_num"][i]),
        )
        top = int(data["top"][i])
        height = int(data["height"][i])
        lines.setdefault(key, []).append((token.upper(), top, height))

    best_bottom: int | None = None
    for words in lines.values():
        tokens = [re.sub(r"[^A-Z0-9]", "", word) for word, *_ in words]
        normalized = " ".join(t for t in tokens if t)
        has_qid = ("QID" in normalized) or ("NO" in normalized)
        has_option = any(opt in normalized for opt in ("A", "B", "C", "D", "T", "F", "ITEM"))
        if not (has_qid and has_option):
            continue

        line_bottom = max(top + h for _, top, h in words)
        if best_bottom is None or line_bottom > best_bottom:
            best_bottom = line_bottom

    if best_bottom is None:
        return None

    return max(0, best_bottom + 8)


def _dedupe_scanned_row_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []

    qid_present = [row for row in rows if row.get("question_id")]
    should_drop_missing_qid = len(qid_present) >= 2

    deduped: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []

    for row in rows:
        section_type = str(row.get("section_type") or "")
        qid = row.get("question_id")
        item_number = row.get("item_number")

        if should_drop_missing_qid and not qid:
            continue

        if not qid:
            passthrough.append(row)
            continue

        if section_type in {"MC", "TF"}:
            dedupe_key = (section_type, str(qid), None)
        else:
            dedupe_key = (section_type, str(qid), item_number)

        current = deduped.get(dedupe_key)
        if current is None:
            deduped[dedupe_key] = row
            continue

        curr_h = float(current["row_bbox_px"][3] - current["row_bbox_px"][1])
        new_h = float(row["row_bbox_px"][3] - row["row_bbox_px"][1])
        if new_h > curr_h:
            deduped[dedupe_key] = row

    combined = [*deduped.values(), *passthrough]
    return sorted(combined, key=lambda r: float(r.get("y_center_px", 0.0)))


def _find_option_centers_scanned(
    section: dict[str, Any], page: fitz.Page, img_bgr: np.ndarray
) -> dict[str, float]:
    section_type = section["section_type"]
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

    x0_px = int(section.get("x0_px", 0))
    x1_px = int(section.get("x1_px", img_bgr.shape[1]))
    if x1_px <= x0_px:
        return {}

    ocr_centers, ocr_center_debug = _find_option_centers_scanned_from_header_ocr(
        img_bgr=img_bgr,
        section=section,
        labels=labels,
        sx=(img_bgr.shape[1] / page.rect.width),
    )
    if ocr_center_debug.get("direct_count", 0) or ocr_center_debug.get("inferred_count", 0):
        print(
            "[extractor][scanned] option centers header OCR stats: "
            f"direct={ocr_center_debug.get('direct_count', 0)}, "
            f"inferred={ocr_center_debug.get('inferred_count', 0)}"
        )
    if ocr_centers:
        return ocr_centers

    left_margin = 0.06
    usable_width = 0.88
    col_width = usable_width / total_cols
    x_ratios = [
        left_margin + ((answer_col_start - 1) + idx + 0.5) * col_width
        for idx in range(len(labels))
    ]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = img_bgr.shape[1] / page.rect.width
    centers: dict[str, float] = {}

    for lbl, ratio in zip(labels, x_ratios):
        x_px = int(x0_px + (x1_px - x0_px) * ratio)
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


def _find_option_centers_scanned_from_header_ocr(
    img_bgr: np.ndarray,
    section: dict[str, Any],
    labels: list[str],
    sx: float,
) -> tuple[dict[str, float], dict[str, int]]:
    x0_px = max(0, int(section.get("x0_px", 0)))
    x1_px = min(img_bgr.shape[1], int(section.get("x1_px", img_bgr.shape[1])))
    y0_px = max(0, int(section.get("y0_px", 0)))
    y1_px = min(img_bgr.shape[0], int(section.get("y1_px", img_bgr.shape[0])))
    if x1_px <= x0_px or y1_px <= y0_px:
        return {}, {"direct_count": 0, "inferred_count": 0}

    header_h = max(40, min(180, int((y1_px - y0_px) * 0.22)))
    header_roi = img_bgr[y0_px : min(y1_px, y0_px + header_h), x0_px:x1_px]
    if header_roi.size == 0:
        return {}, {"direct_count": 0, "inferred_count": 0}

    data = _ocr_data(header_roi, psm=6)
    candidates: dict[str, list[tuple[float, float]]] = {label: [] for label in labels}

    def _resolve_header_label(token: str) -> str | None:
        normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
        if not normalized:
            return None
        if normalized in labels:
            return normalized
        # Permit partial but unambiguous matches ("(A)", "B.", "C-").
        if len(normalized) <= 3:
            hits = [label for label in labels if label in normalized]
            if len(hits) == 1:
                return hits[0]
        return None

    for i, txt in enumerate(data["text"]):
        token = (txt or "").strip()
        label = _resolve_header_label(token)
        if label is None or label not in candidates:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 20:
            continue
        left = int(data["left"][i])
        width = int(data["width"][i])
        center_px = x0_px + left + (width / 2.0)
        candidates[label].append((center_px / sx, conf))

    resolved: dict[str, float] = {}
    conf_by_label: dict[str, float] = {}
    for label in labels:
        vals = candidates.get(label) or []
        if vals:
            x_vals = [x for x, _ in vals]
            conf_vals = [c for _, c in vals]
            resolved[label] = float(np.median(x_vals))
            conf_by_label[label] = float(np.max(conf_vals))

    high_conf_pairs: list[tuple[int, float, float]] = []
    for idx, label in enumerate(labels):
        if label not in resolved:
            continue
        max_conf = conf_by_label.get(label, 0.0)
        if max_conf >= 65.0:
            high_conf_pairs.append((idx, resolved[label], max_conf))

    inferred_count = 0
    if len(resolved) < len(labels) and len(high_conf_pairs) >= 2:
        idxs = np.array([idx for idx, _, _ in high_conf_pairs], dtype=np.float64)
        xs = np.array([x for _, x, _ in high_conf_pairs], dtype=np.float64)
        weights = np.array([conf for _, _, conf in high_conf_pairs], dtype=np.float64)
        slope, intercept = np.polyfit(idxs, xs, deg=1, w=weights)
        if slope > 0:
            for idx, label in enumerate(labels):
                if label in resolved:
                    continue
                resolved[label] = float((slope * idx) + intercept)
                inferred_count += 1

    direct_count = len(resolved) - inferred_count
    if len(resolved) == len(labels):
        return resolved, {"direct_count": direct_count, "inferred_count": inferred_count}
    return {}, {"direct_count": direct_count, "inferred_count": inferred_count}


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


def render_page_raw_to_bgr(page: fitz.Page, zoom: float = RENDER_ZOOM) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def preprocess_scanned_for_omr(img: np.ndarray) -> np.ndarray:
    # 1) Grayscale base + scan-brightness normalization.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    print(f"[preprocess] brightness_before={brightness:.2f}")

    adjustment_applied = False
    alpha = 1.0
    beta = 0.0
    if brightness > 245.0:
        # Overexposed scans: reduce brightness slightly and increase contrast.
        alpha = 1.2
        beta = -20.0
        adjustment_applied = True
    elif brightness < 180.0:
        # Underexposed scans: raise brightness and slightly reduce contrast.
        alpha = 1.1
        beta = 20.0
        adjustment_applied = True

    if adjustment_applied:
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        print(f"[preprocess] brightness_adjustment=applied alpha={alpha:.2f} beta={beta:.1f}")
    else:
        print("[preprocess] brightness_adjustment=none")

    # Slight blue-ink boost for pen-filled bubbles.
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


def render_page_to_bgr(page: fitz.Page, zoom: float = RENDER_ZOOM) -> np.ndarray:
    return preprocess_scanned_for_omr(render_page_raw_to_bgr(page, zoom=zoom))


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sort_cols = ["student_id", "page", "section", "question_id", "item_number", "answer"]

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        key: list[Any] = []
        for col in sort_cols:
            val = row.get(col)
            key.append((val is None, "" if val is None else str(val)))
        return tuple(key)

    return sorted(rows, key=sort_key)


def _compute_scan_quality_metrics(
    img_bgr: np.ndarray,
    estimated_skew: float,
    used_text_layer: bool,
) -> dict[str, Any]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    score = 100.0
    if brightness < 80.0:
        score -= min(25.0, (80.0 - brightness) * 0.5)
    elif brightness > 220.0:
        score -= min(25.0, (brightness - 220.0) * 0.5)

    if contrast < 35.0:
        score -= min(20.0, (35.0 - contrast) * 0.7)

    if blur_laplacian_var < 120.0:
        score -= min(25.0, (120.0 - blur_laplacian_var) * 0.15)

    score -= min(12.0, abs(estimated_skew) * 4.0)
    score = max(0.0, min(100.0, score))

    if score >= 80.0:
        quality_band = "good"
    elif score >= 60.0:
        quality_band = "fair"
    else:
        quality_band = "poor"

    return {
        "scan_quality_score": round(score, 1),
        "scan_quality_band": quality_band,
        "scan_brightness_mean": round(brightness, 1),
        "scan_contrast_std": round(contrast, 1),
        "scan_blur_laplacian_var": round(blur_laplacian_var, 1),
        "scan_estimated_skew_deg": round(float(estimated_skew), 2),
        "scan_mode": "text-layer" if used_text_layer else "scanned-image",
    }


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


def _format_extraction_report(
    file_path: str,
    page_reports: list[dict[str, Any]],
    total_rows: int,
) -> str:
    lines: list[str] = [
        f"Workflow report for: {file_path}",
        f"- Pages processed: {len(page_reports)}",
        f"- Output rows: {total_rows}",
        "",
    ]
    for page_report in page_reports:
        page_no = page_report["page"]
        lines.extend(
            [
                f"Page {page_no}",
                f"  1) Mode: {page_report['mode']}",
                f"  2) Sections detected: {page_report['sections_detected_total']} "
                f"(usable: {page_report['sections_used']})",
                f"     - scanned heading OCR: raw={page_report.get('scan_headings_raw', 0)}, "
                f"left={page_report.get('scan_headings_left', 0)}, "
                f"right={page_report.get('scan_headings_right', 0)}",
                "     - scanned heading candidates: "
                + (
                    ", ".join(page_report.get("scan_heading_candidates", []))
                    if page_report.get("scan_heading_candidates")
                    else "none"
                ),
                f"  3) Rows detected before OMR: {page_report['rows_detected_total']}",
                f"  4) OMR answers detected: {page_report['answers_detected_total']}",
                f"  5) Unanswered rows after OMR: {page_report['rows_unanswered_total']}",
                f"  6) Rows flagged for review: {page_report['rows_needing_review_total']}",
            ]
        )
        scan_heading_guard_warning = page_report.get("scan_heading_guard_warning")
        if scan_heading_guard_warning:
            lines.append(f"     - ⚠️ {scan_heading_guard_warning}")

        section_debug = page_report.get("section_debug", [])
        if section_debug:
            lines.append("  Section breakdown:")
            for item in section_debug:
                lines.append(
                    "    - "
                    f"{item['section_type']}: rows={item['rows_detected']}, "
                    f"answered={item['answers_detected']}, "
                    f"missing_qid={item['rows_missing_question_id']}, "
                    f"missing_option_centers={item['missing_option_centers']}, "
                    f"bands(raw/merged/refined)="
                    f"{item.get('bands_raw_count', 0)}/"
                    f"{item.get('bands_merged_count', 0)}/"
                    f"{item.get('bands_refined_count', 0)}, "
                    f"header_skip_px={item.get('header_offset_px', 0)}, "
                    f"rows_dropped_missing_item_number="
                    f"{item.get('rows_dropped_missing_item_number', 0)}"
                )
        lines.append("")

    return "\n".join(lines).strip()


def extract_file_with_report(
    file_path: str, student_id_mode: str
) -> tuple[list[dict[str, Any]], str]:
    source_doc = _open_file_as_document(file_path)
    doc, force_text_layer = _prepare_document_for_extraction(source_doc, file_path)
    if doc is not source_doc:
        source_doc.close()
    all_rows: list[dict[str, Any]] = []
    page_reports: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            rendered_raw_bgr = render_page_raw_to_bgr(page)
            img_raw_bgr = rendered_raw_bgr.copy()
            img_bgr = preprocess_scanned_for_omr(img_raw_bgr)
        except AttributeError:
            # Test doubles may only provide the preprocessed rendering hook.
            img_bgr = render_page_to_bgr(page)
            img_raw_bgr = img_bgr.copy()
            rendered_raw_bgr = img_raw_bgr.copy()
        page_number = page_idx + 1
        use_text_layer = has_usable_text_layer(page)
        estimated_skew = _estimate_skew_correction_degrees(img_raw_bgr)
        if (
            not force_text_layer
            and use_text_layer
            and abs(estimated_skew) >= TEXT_LAYER_SKEW_FALLBACK_DEG
        ):
            # Some scanned PDFs include an OCR text layer. Their word coordinates are
            # often unreliable when the underlying scan is skewed, so force scanned
            # mode so deskew/alignment can be applied consistently.
            use_text_layer = False
        if not use_text_layer:
            img_raw_bgr = normalize_scanned_page_rotation(img_raw_bgr)
            img_raw_bgr = align_scanned_page_to_template(img_raw_bgr)
            img_bgr = preprocess_scanned_for_omr(img_raw_bgr)
        quality_metrics = _compute_scan_quality_metrics(
            img_bgr=img_raw_bgr,
            estimated_skew=estimated_skew,
            used_text_layer=use_text_layer,
        )
        print(
            f"[extractor] page {page_number}: "
            f"{'text-layer mode' if use_text_layer else 'scanned-image mode'}"
        )

        qr_raw = decode_qr_robust(img_raw_bgr)
        qr_parsed = parse_qr_data(qr_raw) if qr_raw else {}
        qr_student_id = qr_parsed.get("studentid")
        qr_student_name = qr_parsed.get("studentname")
        qr_type = (qr_parsed.get("type") or "").upper() or None

        exam_id = None
        test_id = None
        if qr_type == "EXAM":
            exam_id = qr_parsed.get("examid") or qr_parsed.get("assessmentid")
        elif qr_type == "TEST":
            test_id = qr_parsed.get("testid") or qr_parsed.get("assessmentid")

        # Student-ID bubbles are image-based marks. Even when answer extraction uses
        # text-layer mode, use a normalized/aligned scan view for ID detection.
        if use_text_layer:
            id_raw_bgr = normalize_scanned_page_rotation(rendered_raw_bgr)
            id_raw_bgr = align_scanned_page_to_template(id_raw_bgr)
            id_bgr = preprocess_scanned_for_omr(id_raw_bgr)
        else:
            id_raw_bgr = img_raw_bgr
            id_bgr = img_bgr

        bubble_id, _ = extract_student_id_bubble(id_bgr)
        if bubble_id is None:
            bubble_id, _ = extract_student_id_bubble(id_raw_bgr)
        handwritten_id, _ = extract_student_id_handwritten_ocr(img_raw_bgr)

        id_info = resolve_student_id(
            qr_id=qr_student_id,
            bubble_id=bubble_id,
            handwritten_id=handwritten_id,
            mode=student_id_mode,
        )

        base = {
            "page": page_number,
            "student_id": id_info["student_id"],
            "student_id_all": id_info["student_id_all"],
            "student_id_qr": id_info["student_id_qr"],
            "student_id_text": id_info["student_id_text"],
            "student_id_bubble": id_info["student_id_bubble"],
            "student_id_handwritten": id_info["student_id_handwritten"],
            "student_id_source": id_info["student_id_source"],
            "id_conflict": id_info["id_conflict"],
            "sheet_type": qr_type,
            "exam_id": exam_id if qr_type == "EXAM" else None,
            "test_id": test_id if qr_type == "TEST" else None,
            "studentName": qr_student_name,
            "qr_raw": qr_raw,
            "source_file": file_path,
            **quality_metrics,
        }

        scan_section_debug: dict[str, Any] = {}
        if use_text_layer:
            sections = detect_section_blocks(page)
        else:
            sections, scan_section_debug = detect_section_blocks_scanned(
                img_raw_bgr, return_debug=True
            )
        page_report: dict[str, Any] = {
            "page": page_number,
            "mode": "text-layer" if use_text_layer else "scanned-image",
            "sections_detected_total": len(sections),
            "sections_used": 0,
            "rows_detected_total": 0,
            "answers_detected_total": 0,
            "rows_unanswered_total": 0,
            "rows_needing_review_total": 0,
            "scan_headings_raw": int(scan_section_debug.get("raw_headings_detected", 0)),
            "scan_headings_left": int(scan_section_debug.get("column_left_headings", 0)),
            "scan_headings_right": int(scan_section_debug.get("column_right_headings", 0)),
            "scan_heading_candidates": list(scan_section_debug.get("heading_candidates", [])),
            "scan_heading_guard_warning": None,
            "section_debug": [],
        }
        if not use_text_layer:
            left_count = page_report["scan_headings_left"]
            right_count = page_report["scan_headings_right"]
            raw_count = page_report["scan_headings_raw"]
            if raw_count > 0 and (left_count == 0 or right_count == 0):
                side = "left" if right_count == 0 else "right"
                page_report["scan_heading_guard_warning"] = (
                    "Heading OCR appears one-sided "
                    f"(left={left_count}, right={right_count}); "
                    f"section split may be unreliable ({side}-only detection)."
                )

        for section in sections:
            if section["section_type"] == "OPEN_ENDED":
                continue
            page_report["sections_used"] += 1

            if use_text_layer:
                option_centers = find_option_centers(page, section)
                rows = extract_rows_for_section(page, section)
                scanned_row_debug = {}
            else:
                option_centers = _find_option_centers_scanned(section, page, img_raw_bgr)
                scanned_rows_px, scanned_row_debug = extract_rows_for_section_scanned(
                    img_raw_bgr, section
                )
                rows = _rows_px_to_page(scanned_rows_px, page, img_raw_bgr)

            section_answers = 0
            section_unanswered = 0
            section_review = 0
            rows_missing_question_id = 0
            if section["section_type"] == "MC":
                expected_labels = {"A", "B", "C", "D"}
            elif section["section_type"] == "TF":
                expected_labels = {"T", "F"}
            elif section["section_type"] == "MATCH":
                expected_labels = {"A", "B", "C", "D"}
            else:
                expected_labels = set()
            missing_option_centers = sorted(expected_labels - set(option_centers.keys()))

            page_report["rows_detected_total"] += len(rows)
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
                if row["question_id"] is None:
                    rows_missing_question_id += 1
                if answers:
                    section_answers += 1
                    page_report["answers_detected_total"] += 1
                else:
                    section_unanswered += 1
                    page_report["rows_unanswered_total"] += 1
                if review_flag:
                    section_review += 1
                    page_report["rows_needing_review_total"] += 1

                append_output_rows(
                    output_rows=all_rows,
                    base=base,
                    row=row,
                    answers=answers,
                    needs_review=review_flag,
                    scores=scores,
                )
            page_report["section_debug"].append(
                {
                    "section_type": section["section_type"],
                    "rows_detected": len(rows),
                    "answers_detected": section_answers,
                    "rows_unanswered": section_unanswered,
                    "rows_missing_question_id": rows_missing_question_id,
                    "rows_needing_review": section_review,
                    "missing_option_centers": ",".join(missing_option_centers)
                    if missing_option_centers
                    else "none",
                    "bands_raw_count": int(scanned_row_debug.get("bands_raw_count", 0)),
                    "bands_merged_count": int(scanned_row_debug.get("bands_merged_count", 0)),
                    "bands_refined_count": int(scanned_row_debug.get("bands_refined_count", 0)),
                    "header_offset_px": int(scanned_row_debug.get("header_offset_px", 0)),
                    "rows_dropped_missing_item_number": int(
                        scanned_row_debug.get("rows_dropped_missing_item_number", 0)
                    ),
                }
            )
        page_reports.append(page_report)

    doc.close()
    sorted_rows = _sorted_rows(all_rows)
    report_text = _format_extraction_report(
        file_path=file_path,
        page_reports=page_reports,
        total_rows=len(sorted_rows),
    )
    return sorted_rows, report_text


def extract_file(file_path: str, student_id_mode: str) -> list[dict[str, Any]]:
    rows, _ = extract_file_with_report(file_path=file_path, student_id_mode=student_id_mode)
    return rows
