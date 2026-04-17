from __future__ import annotations

from types import SimpleNamespace

from core import extractor


class _MockPage:
    def __init__(self, text: str, words: list[tuple[float, ...]]) -> None:
        self._text = text
        self._words = words
        self.rect = SimpleNamespace(width=1000.0, height=1400.0)

    def get_text(self, mode: str, sort: bool = True):  # noqa: ARG002
        if mode == "text":
            return self._text
        if mode == "words":
            return self._words
        raise ValueError(mode)


def test_has_usable_text_layer_accepts_valid_text_payload() -> None:
    text = (
        "STUDENT NUMBER 00530 MULTIPLE CHOICE SECTION QUESTION 1 ANSWER A "
        "QUESTION 2 ANSWER B TRUE FALSE SECTION QUESTION 3 ANSWER T"
    )
    words = [
        (40.0, 40.0, 140.0, 65.0, "STUDENT", 0, 0, 0),
        (160.0, 40.0, 250.0, 65.0, "NUMBER", 0, 0, 1),
        (70.0, 220.0, 200.0, 245.0, "MULTIPLE", 1, 0, 0),
        (230.0, 220.0, 330.0, 245.0, "CHOICE", 1, 0, 1),
        (80.0, 510.0, 210.0, 535.0, "QUESTION", 2, 0, 0),
        (240.0, 510.0, 295.0, 535.0, "ANSWER", 2, 0, 1),
        (80.0, 820.0, 190.0, 845.0, "TRUE", 3, 0, 0),
        (220.0, 820.0, 300.0, 845.0, "FALSE", 3, 0, 1),
        (520.0, 220.0, 650.0, 245.0, "SECTION", 4, 0, 0),
        (520.0, 510.0, 620.0, 535.0, "QUESTION", 5, 0, 0),
        (520.0, 820.0, 620.0, 845.0, "ANSWER", 6, 0, 0),
        (700.0, 820.0, 740.0, 845.0, "T", 6, 0, 1),
    ]
    page = _MockPage(text=text, words=words)

    usable, reason, metrics = extractor.evaluate_text_layer_quality(page)

    assert usable is True
    assert reason == "quality_ok"
    assert metrics["alnum_count"] >= extractor.OCR_OVERLAY_MIN_ALNUM


def test_has_usable_text_layer_rejects_noisy_ocr_payload() -> None:
    noisy_text = " ".join(
        [
            "AAAAAA11",
            "11111111",
            "@@@A@@@1",
            "B#######2",
            "CCCCCC33",
            "3$$$$$$$",
            "DDDDDD44",
            "4%%%%%%%",
            "EEEEEE55",
            "5^^^^^^^",
        ]
    )
    words = [
        (50.0 + i * 70.0, 120.0, 110.0 + i * 70.0, 145.0, tok, 0, 0, i)
        for i, tok in enumerate(noisy_text.split())
    ]
    page = _MockPage(text=noisy_text, words=words)

    usable, reason, metrics = extractor.evaluate_text_layer_quality(page)

    assert metrics["alnum_count"] >= extractor.OCR_OVERLAY_MIN_ALNUM
    assert usable is False
    assert reason in {"low_char_diversity", "high_garbage_ratio"}


def test_has_usable_text_layer_rejects_bbox_spread_mismatch() -> None:
    text = " ".join(f"QUESTION{i} ANSWER{i}" for i in range(1, 9))
    words = [
        (100.0, 100.0 + i * 2.0, 150.0, 112.0 + i * 2.0, f"T{i}", 0, 0, i)
        for i in range(16)
    ]
    page = _MockPage(text=text, words=words)

    usable, reason, _ = extractor.evaluate_text_layer_quality(page, spot_check_bbox=True)

    assert usable is False
    assert reason == "bbox_spread_mismatch"


def test_format_extraction_report_includes_text_layer_gate_reason() -> None:
    report = extractor._format_extraction_report(
        file_path="dummy.pdf",
        page_reports=[
            {
                "page": 1,
                "mode": "scanned-image",
                "text_layer_gate_decision": "reject",
                "text_layer_gate_reason": "high_garbage_ratio",
                "sections_detected_total": 0,
                "sections_used": 0,
                "rows_detected_total": 0,
                "answers_detected_total": 0,
                "rows_unanswered_total": 0,
                "rows_needing_review_total": 0,
                "scan_headings_raw": 0,
                "scan_headings_left": 0,
                "scan_headings_right": 0,
                "scan_heading_candidates": [],
                "section_debug": [],
            }
        ],
        total_rows=0,
    )

    assert "text-layer gate: reject (high_garbage_ratio)" in report
