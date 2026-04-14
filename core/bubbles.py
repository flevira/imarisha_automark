from __future__ import annotations

from typing import Any

import cv2
import fitz
import numpy as np

from core.confidence_model import BubbleConfidenceModel

MATCH_LEFT_LABELS = {1: "A", 2: "B", 3: "C", 4: "D"}
BUBBLE_MODEL = BubbleConfidenceModel()


def page_to_image_scale(page: fitz.Page, img_bgr: np.ndarray) -> tuple[float, float]:
    sx = img_bgr.shape[1] / page.rect.width
    sy = img_bgr.shape[0] / page.rect.height
    return sx, sy


def sample_bubble_score(gray: np.ndarray, cx: int, cy: int, radius: int) -> float:
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(gray.shape[1], cx + radius)
    y1 = min(gray.shape[0], cy + radius)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    rr = min(roi.shape[0], roi.shape[1]) / 2.2
    cy0 = roi.shape[0] / 2.0
    cx0 = roi.shape[1] / 2.0
    mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= rr**2

    pixels = roi[mask]
    if pixels.size == 0:
        pixels = roi.flatten()

    return 255.0 - float(np.mean(pixels))


def _column_mapping_fallback(
    gray: np.ndarray,
    row_bbox_px: tuple[int, int, int, int],
    section_type: str,
) -> dict[str, float]:
    layout_map: dict[str, tuple[int, dict[str, int], list[str]]] = {
        "MC": (6, {"A": 3, "B": 4, "C": 5, "D": 6}, ["A", "B", "C", "D"]),
        "TF": (4, {"T": 3, "F": 4}, ["T", "F"]),
        "MATCH": (7, {"A": 4, "B": 5, "C": 6, "D": 7}, ["A", "B", "C", "D"]),
    }
    layout = layout_map.get(section_type)
    if not layout:
        return {}

    total_cols, label_to_col, labels = layout
    x0, y0, x1, y1 = row_bbox_px
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(gray.shape[1], x1)
    y1 = min(gray.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return {}

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return {}

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        6,
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_h = max(1, y1 - y0)
    min_area = max(18.0, (row_h * 0.10) ** 2)
    max_area = max(min_area + 20.0, (row_h * 0.80) ** 2)

    left_margin = 0.06
    usable_width = 0.88
    row_w = max(1, x1 - x0)
    col_w = (usable_width * row_w) / float(total_cols)
    x_start = x0 + (left_margin * row_w)

    per_label_score: dict[str, float] = {label: 0.0 for label in labels}
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx_local = float(moments["m10"] / moments["m00"])
        cy_local = float(moments["m01"] / moments["m00"])
        cx = x0 + cx_local
        cy = y0 + cy_local

        col_idx = int(round(((cx - x_start) / max(col_w, 1e-6)) + 0.5))
        matched_label = next(
            (label for label, col in label_to_col.items() if col == col_idx), None
        )
        if matched_label is None:
            continue

        score = sample_bubble_score(gray, int(cx), int(cy), max(6, int(row_h * 0.34)))
        per_label_score[matched_label] = max(per_label_score[matched_label], score)

    return per_label_score


def extract_row_answers(
    page: fitz.Page,
    img_bgr: np.ndarray,
    section: dict[str, Any],
    row: dict[str, Any],
    option_centers: dict[str, float],
) -> tuple[list[str], bool, dict[str, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx, sy = page_to_image_scale(page, img_bgr)

    if section["section_type"] == "MC":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    elif section["section_type"] == "TF":
        ordered_labels = ["T", "F"]
        output_map = {"T": "True", "F": "False"}
    elif section["section_type"] == "MATCH":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    else:
        return [], True, {}

    expected_label_set = set(ordered_labels)
    if not option_centers or not expected_label_set.issubset(set(option_centers.keys())):
        x0 = int(row["row_bbox"][0] * sx)
        x1 = int(row["row_bbox"][2] * sx)
        section_x0 = section.get("x0")
        section_x1 = section.get("x1")
        if section_x0 is not None and section_x1 is not None:
            x0 = int(float(section_x0) * sx)
            x1 = int(float(section_x1) * sx)
        elif section.get("x0_px") is not None and section.get("x1_px") is not None:
            x0 = int(section["x0_px"])
            x1 = int(section["x1_px"])

        y0 = int(row["row_bbox"][1] * sy)
        y1 = int(row["row_bbox"][3] * sy)
        fallback_scores = _column_mapping_fallback(
            gray=gray,
            row_bbox_px=(x0, y0, x1, y1),
            section_type=section["section_type"],
        )
        if fallback_scores:
            filled_labels, model_review, probabilities = BUBBLE_MODEL.classify(
                ordered_labels=ordered_labels,
                scores=fallback_scores,
                allow_multiple_answers=(section["section_type"] == "MC"),
            )
            needs_review = model_review
            if section["section_type"] in {"TF", "MATCH"} and len(filled_labels) != 1:
                needs_review = True
            answers = [output_map[label] for label in filled_labels if label in output_map]
            debug_scores = {f"score_{key}": value for key, value in fallback_scores.items()}
            debug_scores.update({f"prob_{key}": value for key, value in probabilities.items()})
            if not answers:
                return [], needs_review, debug_scores
            return answers, needs_review, debug_scores

    row_h_page = max(10.0, row["row_bbox"][3] - row["row_bbox"][1])
    radius = max(10, int((row_h_page * sy) * 0.38))
    cy = int(row["y_center"] * sy)

    scores: dict[str, float] = {}
    for label in ordered_labels:
        if label not in option_centers:
            scores[label] = 0.0
            continue
        cx = int(option_centers[label] * sx)
        scores[label] = sample_bubble_score(gray, cx, cy, radius)

    if not scores:
        return [], True, scores

    filled_labels, model_review, probabilities = BUBBLE_MODEL.classify(
        ordered_labels=ordered_labels,
        scores=scores,
        allow_multiple_answers=(section["section_type"] == "MC"),
    )

    needs_review = model_review
    if section["section_type"] == "TF" and len(filled_labels) != 1:
        needs_review = True

    if section["section_type"] == "MATCH" and len(filled_labels) != 1:
        needs_review = True

    answers = [output_map[label] for label in filled_labels if label in output_map]
    if not answers:
        return [], needs_review, {}

    debug_scores = {f"score_{key}": value for key, value in scores.items()}
    debug_scores.update({f"prob_{key}": value for key, value in probabilities.items()})
    return answers, needs_review, debug_scores


def build_matching_pair(item_number: int | None, answer: str | None) -> str | None:
    if item_number is None or answer is None:
        return None
    left = MATCH_LEFT_LABELS.get(item_number)
    if not left:
        return None
    return f"{left}={answer}"


def append_output_rows(
    output_rows: list[dict[str, Any]],
    base: dict[str, Any],
    row: dict[str, Any],
    answers: list[str],
    needs_review: bool,
    scores: dict[str, float],
) -> None:
    if row["section_type"] == "OPEN_ENDED":
        output_rows.append(
            {
                **base,
                "section": "OPEN_ENDED",
                "question_id": row["question_id"],
                "item_number": None,
                "answer": None,
                "answered": "No",
                "matching_pair": None,
                "manual_marking": True,
                "needs_review": True,
                "score_debug": None,
            }
        )
        return

    if not answers:
        output_rows.append(
            {
                **base,
                "section": row["section_type"],
                "question_id": row["question_id"],
                "item_number": row["item_number"],
                "answer": None,
                "answered": "No",
                "matching_pair": None,
                "manual_marking": False,
                "needs_review": bool(needs_review),
                "score_debug": None,
            }
        )
        return

    for ans in answers:
        output_rows.append(
            {
                **base,
                "section": row["section_type"],
                "question_id": row["question_id"],
                "item_number": row["item_number"],
                "answer": ans,
                "answered": "Yes",
                "matching_pair": build_matching_pair(row["item_number"], ans)
                if row["section_type"] == "MATCH"
                else None,
                "manual_marking": False,
                "needs_review": bool(needs_review),
                "score_debug": str(scores),
            }
        )
