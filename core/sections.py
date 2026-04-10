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
        ("MC", "MULTIPLE CHOICE"),
        ("TF", "TRUE / FALSE"),
        ("MATCH", "MATCHING"),
        ("OPEN_ENDED", "OPEN ENDED"),
    ]

    found: list[tuple[str, fitz.Rect]] = []
    for section_type, phrase in heading_specs:
        rects = search_text_rects(page, phrase)
        for rect in rects:
            found.append((section_type, rect))

    found.sort(key=lambda x: x[1].y0)

    if not found:
        return []

    page_bottom = page.rect.y1
    blocks: list[SectionBlock] = []

    for idx, (section_type, rect) in enumerate(found):
        y0 = rect.y1 + 6
        if idx + 1 < len(found):
            y1 = found[idx + 1][1].y0 - 6
        else:
            y1 = page_bottom - 10

        if y1 > y0 + 10:
            blocks.append({"section_type": section_type, "y0": y0, "y1": y1})

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

    for x0, y0, x1, y1, text, *_ in words:
        if y0 < section["y0"] or y1 > top_zone_y1:
            continue
        clean = text.strip().upper()
        if clean in labels and clean not in centers:
            centers[clean] = (x0 + x1) / 2.0

    return centers


def group_words_into_lines(
    words: list[tuple[Any, ...]], y_tolerance: float = 4.0
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
