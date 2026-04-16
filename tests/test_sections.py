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


def test_detect_section_blocks_scanned_uses_studentnumber_anchor_for_column_split(
    monkeypatch,
) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["STUDENTNUMBER", "TRUEFALSE", "MATCHING", "TRUEFALSE", "MULTIPLE"],
            "conf": ["95", "95", "95", "95", "95"],
            "left": [60, 580, 620, 80, 70],
            "top": [220, 250, 520, 1080, 620],
            "width": [230, 130, 140, 130, 140],
            "height": [30, 24, 24, 24, 24],
            "block_num": [1, 2, 3, 4, 5],
            "par_num": [1, 1, 1, 1, 1],
            "line_num": [1, 1, 1, 1, 1],
            "page_num": [1, 1, 1, 1, 1],
        }

    # Bad divider x that would put all headings on one side.
    def _fake_divider(*, img_bgr, y_start):
        return img_bgr.shape[1] - 5

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)
    monkeypatch.setattr(extractor_module, "_detect_vertical_column_divider_x", _fake_divider)

    _blocks, debug = extractor_module.detect_section_blocks_scanned(img, return_debug=True)

    assert debug["column_split_source"] == "midpoint"
    assert debug["column_left_headings"] == 2
    assert debug["column_right_headings"] == 2


def test_detect_section_blocks_scanned_uses_geometric_page_split(
    monkeypatch,
) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["STUDENTNUMBER", "TRUEFALSE", "MULTIPLE", "MATCHING"],
            "conf": ["95", "95", "95", "95"],
            "left": [55, 610, 70, 620],
            "top": [200, 260, 620, 520],
            "width": [220, 130, 120, 130],
            "height": [30, 24, 24, 24],
            "block_num": [1, 2, 3, 4],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "page_num": [1, 1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)
    _blocks, debug = extractor_module.detect_section_blocks_scanned(img, return_debug=True)

    assert debug["column_split_source"] == "midpoint"
    assert debug["column_left_headings"] == 1
    assert debug["column_right_headings"] == 2


def test_detect_section_blocks_scanned_splits_columns_using_heading_center_x(
    monkeypatch,
) -> None:
    img = np.full((1200, 1000, 3), 255, dtype=np.uint8)

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["MULTIPLE", "TRUEFALSE", "MATCHING"],
            "conf": ["95", "95", "95"],
            # TRUEFALSE starts slightly left of the page midpoint, but most of the
            # heading sits on the right column.
            "left": [70, 470, 560],
            "top": [620, 220, 420],
            "width": [120, 110, 120],
            "height": [24, 24, 24],
            "block_num": [1, 2, 3],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
            "page_num": [1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)
    _blocks, debug = extractor_module.detect_section_blocks_scanned(img, return_debug=True)

    assert debug["column_left_headings"] == 1
    assert debug["column_right_headings"] == 2
