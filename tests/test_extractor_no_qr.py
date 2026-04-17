from __future__ import annotations

import fitz
import numpy as np

from core import extractor as extractor_module


class _DummyPage:
    def __init__(self) -> None:
        self.rect = fitz.Rect(0, 0, 1000, 1400)


class _DummyDoc:
    def __init__(self) -> None:
        self._page = _DummyPage()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> _DummyPage:
        assert idx == 0
        return self._page

    def close(self) -> None:
        return None


def test_extract_file_continues_answer_extraction_when_qr_missing(monkeypatch) -> None:
    monkeypatch.setattr(extractor_module, "_open_file_as_document", lambda _path: _DummyDoc())
    monkeypatch.setattr(
        extractor_module, "render_page_to_bgr", lambda _page: np.full((1200, 900, 3), 255, dtype=np.uint8)
    )
    monkeypatch.setattr(extractor_module, "has_usable_text_layer", lambda _page: True)
    monkeypatch.setattr(extractor_module, "_estimate_skew_correction_degrees", lambda _img: 0.0)
    monkeypatch.setattr(extractor_module, "decode_qr_robust", lambda _img: None)
    monkeypatch.setattr(
        extractor_module,
        "extract_student_id_bubble",
        lambda _img: ("00530", False),
    )
    monkeypatch.setattr(
        extractor_module,
        "extract_student_id_handwritten_ocr",
        lambda _img: (None, True),
    )
    monkeypatch.setattr(
        extractor_module,
        "detect_section_blocks",
        lambda _page: [{"section_type": "MC", "y0": 100.0, "y1": 500.0, "x0": 0.0, "x1": 450.0}],
    )
    monkeypatch.setattr(extractor_module, "find_option_centers", lambda *_args, **_kwargs: {"A": 100.0})
    monkeypatch.setattr(
        extractor_module,
        "extract_rows_for_section",
        lambda *_args, **_kwargs: [
            {
                "section_type": "MC",
                "question_id": "81593",
                "item_number": None,
                "y_center": 180.0,
                "row_bbox": (10.0, 170.0, 300.0, 190.0),
            }
        ],
    )
    monkeypatch.setattr(
        extractor_module,
        "extract_row_answers",
        lambda **_kwargs: (["A"], False, {"score_A": 90.0}),
    )

    rows = extractor_module.extract_file("dummy.pdf", student_id_mode="bubble")
    assert len(rows) == 1
    assert rows[0]["question_id"] == "81593"
    assert rows[0]["answer"] == "A"
    assert rows[0]["student_id"] == "00530"
    assert rows[0]["qr_raw"] is None


def test_extract_file_retries_bubble_id_with_raw_image_when_preprocessed_misses(monkeypatch) -> None:
    monkeypatch.setattr(extractor_module, "_open_file_as_document", lambda _path: _DummyDoc())
    raw_img = np.full((1200, 900, 3), 255, dtype=np.uint8)
    processed_img = np.zeros((1200, 900, 3), dtype=np.uint8)
    monkeypatch.setattr(extractor_module, "render_page_raw_to_bgr", lambda _page: raw_img.copy())
    monkeypatch.setattr(extractor_module, "render_page_to_bgr", lambda _page: raw_img.copy())
    monkeypatch.setattr(extractor_module, "preprocess_scanned_for_omr", lambda _img: processed_img.copy())
    monkeypatch.setattr(extractor_module, "has_usable_text_layer", lambda _page: True)
    monkeypatch.setattr(extractor_module, "_estimate_skew_correction_degrees", lambda _img: 0.0)
    monkeypatch.setattr(extractor_module, "decode_qr_robust", lambda _img: None)

    seen_means: list[float] = []

    def _bubble_stub(img: np.ndarray) -> tuple[str | None, bool]:
        mean_val = float(img.mean())
        seen_means.append(mean_val)
        if mean_val < 10.0:
            return None, True
        return "00530", False

    monkeypatch.setattr(extractor_module, "extract_student_id_bubble", _bubble_stub)
    monkeypatch.setattr(extractor_module, "extract_student_id_handwritten_ocr", lambda _img: (None, True))
    monkeypatch.setattr(
        extractor_module,
        "detect_section_blocks",
        lambda _page: [{"section_type": "MC", "y0": 100.0, "y1": 500.0, "x0": 0.0, "x1": 450.0}],
    )
    monkeypatch.setattr(extractor_module, "find_option_centers", lambda *_args, **_kwargs: {"A": 100.0})
    monkeypatch.setattr(
        extractor_module,
        "extract_rows_for_section",
        lambda *_args, **_kwargs: [
            {
                "section_type": "MC",
                "question_id": "81593",
                "item_number": None,
                "y_center": 180.0,
                "row_bbox": (10.0, 170.0, 300.0, 190.0),
            }
        ],
    )
    monkeypatch.setattr(extractor_module, "extract_row_answers", lambda **_kwargs: (["A"], False, {"score_A": 90.0}))

    rows = extractor_module.extract_file("dummy.pdf", student_id_mode="bubble")
    assert len(rows) == 1
    assert rows[0]["student_id_bubble"] == "00530"
    assert rows[0]["student_id_source"] == "text_bubble"
    assert len(seen_means) == 2
    assert seen_means[0] < 10.0
    assert seen_means[1] > 200.0
