from __future__ import annotations

from typing import Any

import cv2
import fitz
import numpy as np

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



def render_page_to_bgr(page: fitz.Page, zoom: float = RENDER_ZOOM) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


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

        sections = detect_section_blocks(page)

        for section in sections:
            option_centers = find_option_centers(page, section)
            rows = extract_rows_for_section(page, section)

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
