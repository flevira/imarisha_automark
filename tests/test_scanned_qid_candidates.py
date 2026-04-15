from __future__ import annotations

import numpy as np

from core import extractor as extractor_module


def test_extract_scanned_qid_candidates_reads_ids_from_section_ocr(monkeypatch) -> None:
    roi = np.full((200, 300, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(_img, *, psm: int):
        if psm == 6:
            return {
                "text": ["81593", "X", "82587"],
                "conf": ["92", "90", "88"],
                "left": [20, 140, 40],
                "top": [50, 100, 150],
                "width": [40, 10, 40],
                "height": [14, 10, 14],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 1, 1],
                "page_num": [1, 1, 1],
            }
        return {
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "block_num": [],
            "par_num": [],
            "line_num": [],
            "page_num": [],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)

    out = extractor_module._extract_scanned_qid_candidates(
        section_roi_bgr=roi,
        section_type="MC",
        x_offset=100,
        y_offset=200,
    )

    assert [c["question_id"] for c in out] == ["81593", "82587"]
    assert out[0]["y_center"] < out[1]["y_center"]


def test_pick_qid_for_row_from_candidates_returns_nearest_match() -> None:
    candidates = [
        {"question_id": "81593", "y_center": 120.0, "x_center": 80.0},
        {"question_id": "81555", "y_center": 170.0, "x_center": 80.0},
    ]

    qid = extractor_module._pick_qid_for_row_from_candidates(
        qid_candidates=candidates,
        y_abs0=160,
        y_abs1=178,
    )

    assert qid == "81555"

