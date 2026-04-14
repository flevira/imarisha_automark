from __future__ import annotations

import numpy as np

from core import extractor as extractor_module


def test_detect_section_blocks_scanned_handles_two_column_layout(monkeypatch) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["MULTIPLE", "TRUEFALSE", "TRUEFALSE", "MATCHING"],
            "conf": ["95", "95", "95", "95"],
            "left": [40, 560, 40, 560],
            "top": [620, 120, 1100, 320],
            "width": [120, 130, 130, 120],
            "height": [24, 24, 24, 24],
            "block_num": [1, 2, 3, 4],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "page_num": [1, 1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)

    blocks = extractor_module.detect_section_blocks_scanned(img)
    section_types = [block["section_type"] for block in sorted(blocks, key=lambda b: b["y0_px"])]

    assert section_types.count("MC") == 1
    assert section_types.count("MATCH") == 1
    assert section_types.count("TF") == 2


def test_detect_section_blocks_scanned_dedupes_nearby_duplicate_headings(monkeypatch) -> None:
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["MULTIPLE", "MULTIPLE", "TRUEFALSE"],
            "conf": ["95", "95", "95"],
            "left": [20, 24, 24],
            "top": [120, 130, 520],
            "width": [100, 100, 120],
            "height": [20, 20, 20],
            "block_num": [1, 2, 3],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
            "page_num": [1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)

    blocks = extractor_module.detect_section_blocks_scanned(img)
    section_types = [block["section_type"] for block in blocks]

    assert section_types.count("MC") == 1
    assert section_types.count("TF") == 1
