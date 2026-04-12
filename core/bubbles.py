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
    )

    needs_review = model_review
    if section["section_type"] == "TF" and len(filled_labels) != 1:
        needs_review = True

    if section["section_type"] == "MATCH" and len(filled_labels) != 1:
        needs_review = True

    answers = [output_map[label] for label in filled_labels if label in output_map]
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
                "needs_review": True,
                "score_debug": str(scores),
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
