from __future__ import annotations

import fitz

from core import rows as rows_module


class _DummyPage:
    def __init__(self) -> None:
        self.rect = fitz.Rect(0, 0, 600, 800)


def test_extract_rows_for_section_starts_after_header(monkeypatch) -> None:
    page = _DummyPage()
    section = {"section_type": "MC", "y0": 100.0, "y1": 300.0, "x0": 0.0, "x1": 300.0}

    words = [
        (10.0, 110.0, 30.0, 120.0, "99999"),  # should be ignored (above header)
        (10.0, 130.0, 25.0, 140.0, "NO"),
        (30.0, 130.0, 45.0, 140.0, "QID"),
        (50.0, 130.0, 60.0, 140.0, "A"),
        (65.0, 130.0, 75.0, 140.0, "B"),
        (80.0, 130.0, 90.0, 140.0, "C"),
        (95.0, 130.0, 105.0, 140.0, "D"),
        (10.0, 150.0, 35.0, 160.0, "81593"),
    ]

    monkeypatch.setattr(rows_module, "get_page_words", lambda _: words)

    extracted = rows_module.extract_rows_for_section(page, section)
    assert len(extracted) == 1
    assert extracted[0]["question_id"] == "81593"
