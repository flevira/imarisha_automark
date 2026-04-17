from __future__ import annotations

import numpy as np

from core import ocr_engine


def test_detect_with_fallback_returns_tesseract_only_hit(monkeypatch) -> None:
    variants = [np.full((20, 20, 3), 255, dtype=np.uint8), np.full((20, 20, 3), 240, dtype=np.uint8)]
    seen_configs: list[str] = []

    monkeypatch.setattr(ocr_engine, "_extract_tokens_paddle", lambda _img: [])
    monkeypatch.setattr(ocr_engine, "_extract_tokens_easyocr", lambda _img: [])

    def _fake_tesseract(_img: np.ndarray, config: str) -> list[ocr_engine.OCRCandidate]:
        seen_configs.append(config)
        return [ocr_engine.OCRCandidate(text="qid=12345", confidence=58.0, engine="tesseract")]

    monkeypatch.setattr(ocr_engine, "_extract_tokens_tesseract", _fake_tesseract)

    hit = ocr_engine.detect_with_fallback(
        variants,
        pattern=r"\d+",
        tesseract_configs=["--psm 6", "--psm 7"],
        min_len=5,
        max_len=5,
    )

    assert hit is not None
    assert hit.text == "12345"
    assert hit.engine == "tesseract"
    assert seen_configs == ["--psm 6", "--psm 7", "--psm 6", "--psm 7"]


def test_detect_with_fallback_prefers_numeric_tesseract_over_lower_ranked_primary(monkeypatch) -> None:
    variants = [np.full((20, 20, 3), 255, dtype=np.uint8)]

    monkeypatch.setattr(
        ocr_engine,
        "_extract_tokens_paddle",
        lambda _img: [ocr_engine.OCRCandidate(text="12345", confidence=66.0, engine="paddleocr")],
    )
    monkeypatch.setattr(ocr_engine, "_extract_tokens_easyocr", lambda _img: [])
    monkeypatch.setattr(
        ocr_engine,
        "_extract_tokens_tesseract",
        lambda _img, _config: [ocr_engine.OCRCandidate(text="12345", confidence=62.0, engine="tesseract")],
    )

    hit = ocr_engine.detect_with_fallback(
        variants,
        pattern=r"\d+",
        tesseract_configs=["--oem 1 --psm 7"],
        min_len=5,
        max_len=5,
    )

    assert hit is not None
    assert hit.text == "12345"
    assert hit.engine == "tesseract"
