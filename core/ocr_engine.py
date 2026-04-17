from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    confidence: float
    engine: str


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _extract_tokens_tesseract(img: np.ndarray, config: str) -> list[OCRCandidate]:
    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    out: list[OCRCandidate] = []
    for txt, conf in zip(data["text"], data["conf"]):
        token = (txt or "").strip()
        if not token:
            continue
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1.0
        if conf_val < 0:
            continue
        out.append(OCRCandidate(text=token, confidence=conf_val, engine="tesseract"))
    return out


@lru_cache(maxsize=1)
def _get_paddle_reader() -> Any | None:
    try:
        from paddleocr import PaddleOCR  # type: ignore

        return PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
        )
    except Exception:
        return None


def _extract_tokens_paddle(img: np.ndarray) -> list[OCRCandidate]:
    reader = _get_paddle_reader()
    if reader is None:
        return []
    try:
        result = reader.ocr(img, cls=False)
    except Exception:
        return []

    out: list[OCRCandidate] = []
    if not result:
        return out

    for line in result[0] or []:
        if len(line) < 2:
            continue
        text, conf = line[1][0], line[1][1]
        token = (text or "").strip()
        if not token:
            continue
        try:
            conf_val = float(conf) * 100.0
        except Exception:
            conf_val = 0.0
        out.append(OCRCandidate(text=token, confidence=conf_val, engine="paddleocr"))
    return out


@lru_cache(maxsize=1)
def _get_easyocr_reader() -> Any | None:
    try:
        import easyocr  # type: ignore

        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None


def _extract_tokens_easyocr(img: np.ndarray) -> list[OCRCandidate]:
    reader = _get_easyocr_reader()
    if reader is None:
        return []
    try:
        result = reader.readtext(img, detail=1)
    except Exception:
        return []

    out: list[OCRCandidate] = []
    for item in result:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        try:
            conf = float(item[2]) * 100.0
        except Exception:
            conf = 0.0
        out.append(OCRCandidate(text=text, confidence=conf, engine="easyocr"))
    return out


def _match_candidates(
    candidates: list[OCRCandidate],
    pattern: str,
    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> list[OCRCandidate]:
    matched: list[OCRCandidate] = []
    for candidate in candidates:
        for hit in re.findall(pattern, candidate.text):
            token = str(hit).strip()
            if min_len is not None and len(token) < min_len:
                continue
            if max_len is not None and len(token) > max_len:
                continue
            matched.append(
                OCRCandidate(text=token, confidence=candidate.confidence, engine=candidate.engine)
            )
    return matched


def detect_with_fallback(
    variants: list[np.ndarray],
    *,
    pattern: str,
    tesseract_configs: list[str],
    min_len: int | None = None,
    max_len: int | None = None,
    min_tesseract_confidence: float = 55.0,
) -> OCRCandidate | None:
    variant_hits: list[str] = []
    primary_hits: list[OCRCandidate] = []
    # Historical name retained for compatibility. This is now a generic
    # confidence gate for the primary OCR pool (Paddle + Tesseract).
    min_primary_confidence = min_tesseract_confidence

    def _consistency_boost(text: str) -> float:
        count = variant_hits.count(text)
        return min(18.0, float(count - 1) * 6.0)

    def _engine_bias(candidate: OCRCandidate) -> float:
        if candidate.engine == "tesseract":
            # Numeric/whitelisted Tesseract hits are often highly precise even
            # when confidence is slightly lower than neural engines.
            return 10.0 if candidate.text.isdigit() else 6.0
        if candidate.engine == "easyocr":
            return 2.0
        return 0.0

    def _score(candidate: OCRCandidate) -> float:
        return candidate.confidence + _consistency_boost(candidate.text) + _engine_bias(candidate)

    # Primary OCR path: PaddleOCR + Tesseract
    for variant in variants:
        gray = _to_gray(variant)
        paddle_hits = _match_candidates(
            _extract_tokens_paddle(gray),
            pattern,
            min_len=min_len,
            max_len=max_len,
        )
        primary_hits.extend(paddle_hits)
        variant_hits.extend([hit.text for hit in paddle_hits])
        for config in tesseract_configs:
            tesseract_hits = _match_candidates(
                _extract_tokens_tesseract(gray, config),
                pattern,
                min_len=min_len,
                max_len=max_len,
            )
            primary_hits.extend(tesseract_hits)
            variant_hits.extend([hit.text for hit in tesseract_hits])

    if primary_hits:
        best_primary = max(
            primary_hits,
            key=_score,
        )
        boosted_conf = _score(best_primary)
        if boosted_conf >= min_primary_confidence:
            return OCRCandidate(
                text=best_primary.text,
                confidence=boosted_conf,
                engine=best_primary.engine,
            )

    # Fallback OCR path: EasyOCR
    fallback_hits: list[OCRCandidate] = []
    for variant in variants:
        gray = _to_gray(variant)
        fallback_hits.extend(
            _match_candidates(
                _extract_tokens_easyocr(gray),
                pattern,
                min_len=min_len,
                max_len=max_len,
            )
        )

    if not fallback_hits and not primary_hits:
        return None

    combined = primary_hits + fallback_hits
    best = max(
        combined,
        key=_score,
    )
    return OCRCandidate(
        text=best.text,
        confidence=_score(best),
        engine=best.engine,
    )
