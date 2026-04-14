from __future__ import annotations

from typing import Any

import fitz
import numpy as np


SectionBlock = dict[str, Any]


def get_page_words(page: fitz.Page) -> list[tuple[Any, ...]]:
    return page.get_text("words", sort=True)


def search_text_rects(page: fitz.Page, phrase: str) -> list[fitz.Rect]:
    return page.search_for(phrase, quads=False)


def detect_section_blocks(page: fitz.Page) -> list[SectionBlock]:
    heading_specs = [
        ("MC", "MULTIPLE"),
        ("TF", "TRUEFALSE"),
        ("MATCH", "MATCHING"),
        ("OPEN_ENDED", "OPEN"),
    ]

    found: list[tuple[str, fitz.Rect]] = []
    for section_type, phrase in heading_specs:
        rects = search_text_rects(page, phrase)
        for rect in rects:
            found.append((section_type, rect))

    if not found:
        return []

    page_bottom = page.rect.y1
    page_mid_x = page.rect.width / 2.0
    margin_x = page.rect.width * 0.03

    left_col = [item for item in found if ((item[1].x0 + item[1].x1) / 2.0) <= page_mid_x]
    right_col = [item for item in found if ((item[1].x0 + item[1].x1) / 2.0) > page_mid_x]

    blocks: list[SectionBlock] = []

    def _build_column_blocks(
        column_items: list[tuple[str, fitz.Rect]], x0: float, x1: float
    ) -> list[SectionBlock]:
        if not column_items:
            return []

        column_items.sort(key=lambda x: x[1].y0)
        deduped: list[tuple[str, fitz.Rect]] = []
        for section_type, rect in column_items:
            if deduped and section_type == deduped[-1][0] and abs(rect.y0 - deduped[-1][1].y0) <= 12:
                continue
            deduped.append((section_type, rect))

        col_blocks: list[SectionBlock] = []
        for idx, (section_type, rect) in enumerate(deduped):
            y0 = rect.y1 + 6
            if idx + 1 < len(deduped):
                y1 = deduped[idx + 1][1].y0 - 6
            else:
                y1 = page_bottom - 10
            if y1 > y0 + 10:
                col_blocks.append(
                    {"section_type": section_type, "y0": y0, "y1": y1, "x0": x0, "x1": x1}
                )
        return col_blocks

    blocks.extend(_build_column_blocks(left_col, 0.0, page_mid_x - margin_x))
    blocks.extend(_build_column_blocks(right_col, page_mid_x + margin_x, page.rect.x1))

    return blocks


def find_option_centers(page: fitz.Page, section: SectionBlock) -> dict[str, float]:
    words = get_page_words(page)
    top_zone_y1 = section["y0"] + 80

    labels_by_section = {
        "MC": ["A", "B", "C", "D"],
        "TF": ["T", "F"],
        "MATCH": ["A", "B", "C", "D"],
    }

    labels = labels_by_section.get(section["section_type"], [])
    centers: dict[str, float] = {}

    section_x0 = float(section.get("x0", 0.0))
    section_x1 = float(section.get("x1", page.rect.x1))

    for x0, y0, x1, y1, text, *_ in words:
        if y0 < section["y0"] or y1 > top_zone_y1:
            continue
        if x0 < section_x0 or x1 > section_x1:
            continue
        clean = text.strip().upper()
        if clean in labels and clean not in centers:
            centers[clean] = (x0 + x1) / 2.0

    return centers


def group_words_into_lines(
    words: list[tuple[Any, ...]], y_tolerance: float = 6.0
) -> list[list[tuple[Any, ...]]]:
    lines: list[list[tuple[Any, ...]]] = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        if not lines:
            lines.append([word])
            continue

        current_y = np.mean([w[1] for w in lines[-1]])
        if abs(word[1] - current_y) <= y_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines
