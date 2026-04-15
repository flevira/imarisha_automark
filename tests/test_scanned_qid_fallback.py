from __future__ import annotations

import numpy as np

from core import extractor as extractor_module


def test_extract_question_id_with_fallbacks_uses_wider_windows(monkeypatch) -> None:
    roi = np.full((40, 100, 3), 255, dtype=np.uint8)

    observed_widths: list[int] = []

    def _fake_extract(crop: np.ndarray) -> str | None:
        observed_widths.append(crop.shape[1])
        return "81976" if crop.shape[1] >= 50 else None

    monkeypatch.setattr(extractor_module, "extract_question_ids_from_roi", _fake_extract)

    qid = extractor_module.extract_question_id_with_fallbacks(roi, section_type="MC")

    assert qid == "81976"
    assert observed_widths[0] == 35
    assert any(width >= 50 for width in observed_widths[1:])

