from __future__ import annotations

import fitz
import numpy as np

from core import bubbles as bubbles_module


class _DummyTfBubbleModel:
    def classify(self, **_kwargs):  # type: ignore[no-untyped-def]
        return ["T", "F"], False, {"T": 0.82, "F": 0.79}


class _DummyMatchBubbleModel:
    def classify(self, **_kwargs):  # type: ignore[no-untyped-def]
        return ["A", "B"], True, {"A": 0.82, "B": 0.79, "C": 0.11, "D": 0.07}


def test_tf_with_both_options_detected_is_forced_unanswered(monkeypatch) -> None:
    monkeypatch.setattr(bubbles_module, "BUBBLE_MODEL", _DummyTfBubbleModel())

    page = type("DummyPage", (), {"rect": fitz.Rect(0, 0, 1000, 1400)})()
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    section = {"section_type": "TF"}
    row = {"row_bbox": (0.0, 100.0, 900.0, 130.0), "y_center": 115.0}
    option_centers = {"T": 700.0, "F": 820.0}

    answers, needs_review, scores = bubbles_module.extract_row_answers(
        page=page,
        img_bgr=img,
        section=section,
        row=row,
        option_centers=option_centers,
    )

    assert answers == []
    assert needs_review is False
    assert scores == {}


def test_match_with_multiple_options_detected_uses_strongest_and_flags_review(monkeypatch) -> None:
    monkeypatch.setattr(bubbles_module, "BUBBLE_MODEL", _DummyMatchBubbleModel())

    page = type("DummyPage", (), {"rect": fitz.Rect(0, 0, 1000, 1400)})()
    img = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    section = {"section_type": "MATCH"}
    row = {"row_bbox": (0.0, 100.0, 900.0, 130.0), "y_center": 115.0}
    option_centers = {"A": 640.0, "B": 760.0, "C": 840.0, "D": 920.0}

    answers, needs_review, scores = bubbles_module.extract_row_answers(
        page=page,
        img_bgr=img,
        section=section,
        row=row,
        option_centers=option_centers,
    )

    assert answers == ["A"]
    assert needs_review is True
    assert "score_A" in scores
    assert "prob_A" in scores
