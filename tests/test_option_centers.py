from __future__ import annotations

import fitz
import numpy as np
import pytest

from core import extractor as extractor_module


class _DummyPage:
    def __init__(self) -> None:
        self.rect = fitz.Rect(0, 0, 1000, 1400)


def test_scanned_option_centers_use_header_ocr_labels(monkeypatch) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    page = _DummyPage()
    section = {
        "section_type": "MC",
        "x0_px": 0,
        "x1_px": 500,
        "y0_px": 100,
        "y1_px": 800,
    }

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["A", "B", "C", "D"],
            "conf": ["95", "95", "95", "95"],
            "left": [300, 340, 380, 420],
            "width": [10, 10, 10, 10],
            "top": [20, 20, 20, 20],
            "height": [10, 10, 10, 10],
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "page_num": [1, 1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)

    centers = extractor_module._find_option_centers_scanned(section, page, img)
    assert centers["A"] == 305.0
    assert centers["B"] == pytest.approx(345.0)
    assert centers["C"] == 385.0
    assert centers["D"] == pytest.approx(425.0)


def test_scanned_option_centers_infer_missing_labels_from_partial_header(monkeypatch) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    page = _DummyPage()
    section = {
        "section_type": "MC",
        "x0_px": 0,
        "x1_px": 500,
        "y0_px": 100,
        "y1_px": 800,
    }

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["A", "C"],
            "conf": ["92", "93"],
            "left": [300, 380],
            "width": [10, 10],
            "top": [20, 20],
            "height": [10, 10],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "page_num": [1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)
    centers = extractor_module._find_option_centers_scanned(section, page, img)

    assert centers["A"] == 305.0
    assert centers["B"] == pytest.approx(345.0)
    assert centers["C"] == 385.0
    assert centers["D"] == pytest.approx(425.0)


def test_scanned_option_centers_allow_unambiguous_partial_label_tokens(monkeypatch) -> None:
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    page = _DummyPage()
    section = {
        "section_type": "MC",
        "x0_px": 0,
        "x1_px": 500,
        "y0_px": 100,
        "y1_px": 800,
    }

    def _fake_ocr_data(*_args, **_kwargs):
        return {
            "text": ["A.", "(B)", "C", "D-"],
            "conf": ["95", "95", "95", "95"],
            "left": [300, 340, 380, 420],
            "width": [10, 10, 10, 10],
            "top": [20, 20, 20, 20],
            "height": [10, 10, 10, 10],
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "page_num": [1, 1, 1, 1],
        }

    monkeypatch.setattr(extractor_module, "_ocr_data", _fake_ocr_data)
    centers = extractor_module._find_option_centers_scanned(section, page, img)

    assert centers["A"] == 305.0
    assert centers["B"] == 345.0
    assert centers["C"] == 385.0
    assert centers["D"] == 425.0
