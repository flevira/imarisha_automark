from __future__ import annotations

import re
from typing import Any

import fitz

from core.sections import group_words_into_lines


RowData = dict[str, Any]


def get_page_words(page: fitz.Page) -> list[tuple[Any, ...]]:
    return page.get_text("words", sort=True)


def extract_rows_for_section(page: fitz.Page, section: dict[str, Any]) -> list[RowData]:
    words = get_page_words(page)
    section_x0 = float(section.get("x0", page.rect.x0))
    section_x1 = float(section.get("x1", page.rect.x1))
    section_words = [
        w
        for w in words
        if w[1] >= section["y0"]
        and w[3] <= section["y1"]
        and w[0] >= section_x0
        and w[2] <= section_x1
    ]
    lines = group_words_into_lines(section_words)

    rows: list[RowData] = []

    header_markers = (
        "NO QID A B C D",
        "QID A B C D",
        "NO QID T F",
        "QID T F",
        "NO QID ITEM A B C D",
        "QID ITEM A B C D",
        "QUESTION STUDENT ANSWER",
        "NO QID STUDENT ANSWER",
    )

    header_idx: int | None = None
    for idx, line in enumerate(lines):
        texts = [w[4].strip() for w in line if w[4].strip()]
        if not texts:
            continue
        joined = " ".join(texts).upper()
        if any(marker in joined for marker in header_markers):
            header_idx = idx
            break

    candidate_lines = lines[header_idx + 1 :] if header_idx is not None else lines

    for line in candidate_lines:
        texts = [w[4].strip() for w in line if w[4].strip()]
        if not texts:
            continue

        joined = " ".join(texts).upper()

        if any(marker in joined for marker in header_markers):
            continue

        xs = [w[0] for w in line]
        ys0 = [w[1] for w in line]
        xs1 = [w[2] for w in line]
        ys1 = [w[3] for w in line]
        bbox = (min(xs), min(ys0), max(xs1), max(ys1))
        y_center = (bbox[1] + bbox[3]) / 2.0

        if section["section_type"] in {"MC", "TF", "OPEN_ENDED"}:
            qids = [t for t in texts if re.fullmatch(r"\d{5,6}", t)]
            if qids:
                rows.append(
                    {
                        "section_type": section["section_type"],
                        "question_id": qids[0],
                        "item_number": None,
                        "y_center": y_center,
                        "row_bbox": bbox,
                    }
                )

        elif section["section_type"] == "MATCH":
            qids = [t for t in texts if re.fullmatch(r"\d{5,6}", t)]
            items = [t for t in texts if re.fullmatch(r"[1-4]", t)]
            if qids and items:
                rows.append(
                    {
                        "section_type": section["section_type"],
                        "question_id": qids[0],
                        "item_number": int(items[0]),
                        "y_center": y_center,
                        "row_bbox": bbox,
                    }
                )

    return rows
