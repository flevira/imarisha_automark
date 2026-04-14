from __future__ import annotations

import fitz
import numpy as np
import cv2

from core.bubbles import extract_row_answers


class _DummyPage:
    def __init__(self) -> None:
        self.rect = fitz.Rect(0, 0, 700, 220)


def _column_center_x(width: int, total_cols: int, column_idx: int) -> int:
    left_margin = 0.06
    usable_width = 0.88
    col_w = (usable_width * width) / float(total_cols)
    x_start = left_margin * width
    return int(round(x_start + ((column_idx - 0.5) * col_w)))


def test_tf_fallback_maps_column_3_to_true_when_option_centers_missing() -> None:
    page = _DummyPage()
    img = np.full((220, 700, 3), 255, dtype=np.uint8)
    cy = 80
    cx_true = _column_center_x(width=700, total_cols=4, column_idx=3)
    cv2.circle(img, (cx_true, cy), 10, (0, 0, 0), thickness=-1)

    section = {"section_type": "TF", "x0": 0.0, "x1": 700.0}
    row = {
        "section_type": "TF",
        "question_id": "82023",
        "item_number": None,
        "y_center": 80.0,
        "row_bbox": (0.0, 60.0, 700.0, 100.0),
    }

    answers, needs_review, _scores = extract_row_answers(
        page=page, img_bgr=img, section=section, row=row, option_centers={}
    )
    assert answers == ["True"]
    assert needs_review is False


def test_match_fallback_maps_column_7_to_d_when_option_centers_missing() -> None:
    page = _DummyPage()
    img = np.full((220, 700, 3), 255, dtype=np.uint8)
    cy = 120
    cx_d = _column_center_x(width=700, total_cols=7, column_idx=7)
    cv2.circle(img, (cx_d, cy), 9, (0, 0, 0), thickness=-1)

    section = {"section_type": "MATCH", "x0": 0.0, "x1": 700.0}
    row = {
        "section_type": "MATCH",
        "question_id": "82587",
        "item_number": 1,
        "y_center": 120.0,
        "row_bbox": (0.0, 100.0, 700.0, 140.0),
    }

    answers, needs_review, _scores = extract_row_answers(
        page=page, img_bgr=img, section=section, row=row, option_centers={}
    )
    assert answers == ["D"]
    assert needs_review is False
