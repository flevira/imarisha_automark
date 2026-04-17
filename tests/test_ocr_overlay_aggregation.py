from __future__ import annotations

import numpy as np

from core import extractor


def test_aggregate_ocr_candidates_prefers_highest_confidence_for_overlap() -> None:
    candidates = [
        {"text": "A", "conf": 55.0, "bbox": (100, 100, 40, 20)},
        {"text": "4", "conf": 92.0, "bbox": (102, 100, 39, 20)},
        {"text": "B", "conf": 80.0, "bbox": (260, 210, 35, 18)},
    ]

    merged = extractor._aggregate_ocr_token_candidates(candidates, iou_threshold=0.2)

    assert len(merged) == 2
    assert merged[0]["text"] == "4"
    assert merged[0]["conf"] == 92.0
    assert merged[1]["text"] == "B"


def test_aggregate_ocr_candidates_uses_center_distance_for_nearby_boxes() -> None:
    candidates = [
        {"text": "I", "conf": 91.0, "bbox": (50, 50, 8, 28)},
        {"text": "1", "conf": 87.0, "bbox": (58, 50, 8, 28)},
        {"text": "X", "conf": 70.0, "bbox": (140, 55, 8, 28)},
    ]

    merged = extractor._aggregate_ocr_token_candidates(candidates, iou_threshold=0.95, center_distance_ratio=1.2)

    assert len(merged) == 2
    assert merged[0]["text"] == "I"
    assert merged[1]["text"] == "X"


def test_ocr_data_from_gray_runs_multi_pass_and_deduplicates(monkeypatch) -> None:
    calls: list[str] = []

    def fake_image_to_data(_img, *, config: str, output_type):
        assert output_type == extractor.pytesseract.Output.DICT
        calls.append(config)
        if "--psm 6" in config:
            return {
                "text": ["Math", "12"],
                "conf": ["71", "83"],
                "left": [10, 120],
                "top": [20, 20],
                "width": [60, 24],
                "height": [18, 18],
            }
        return {
            "text": ["Mαth", "12"],
            "conf": ["93", "65"],
            "left": [11, 121],
            "top": [20, 20],
            "width": [61, 24],
            "height": [18, 18],
        }

    monkeypatch.setattr(extractor.pytesseract, "image_to_data", fake_image_to_data)

    gray = np.full((80, 200), 200, dtype=np.uint8)
    data = extractor._ocr_data_from_gray(gray, psm=6)

    # 4 variants x 2 PSMs
    assert len(calls) == 8
    assert "--psm 6" in calls[0]
    assert any("--psm 11" in call for call in calls)

    # highest-confidence duplicate survives aggregation
    assert data["text"] == ["Mαth", "12"]
    assert data["conf"] == [93.0, 83.0]
