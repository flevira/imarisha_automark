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
def _get_doctr_predictor() -> Any | None:
    try:
        from doctr.io import DocumentFile  # type: ignore
        from doctr.models import ocr_predictor  # type: ignore

        predictor = ocr_predictor(pretrained=True)
        return (DocumentFile, predictor)
    except Exception:
        return None


def _extract_tokens_doctr(img: np.ndarray) -> list[OCRCandidate]:
    doctr = _get_doctr_predictor()
    if doctr is None:
        return []

    document_file_cls, predictor = doctr
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        doc = document_file_cls.from_images([rgb])
        result = predictor(doc).export()
    except Exception:
        return []

    out: list[OCRCandidate] = []
    try:
        pages = result.get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    for word in line.get("words", []):
                        token = str(word.get("value", "")).strip()
                        if not token:
                            continue
                        conf = float(word.get("confidence", 0.0)) * 100.0
                        out.append(OCRCandidate(text=token, confidence=conf, engine="doctr"))
    except Exception:
        return []
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
    all_hits: list[OCRCandidate] = []

    for variant in variants:
        gray = _to_gray(variant)
        tess_tokens: list[OCRCandidate] = []
        for cfg in tesseract_configs:
            tess_tokens.extend(_extract_tokens_tesseract(gray, cfg))
        tess_hits = _match_candidates(
            tess_tokens,
            pattern,
            min_len=min_len,
            max_len=max_len,
        )
        all_hits.extend(tess_hits)
        variant_hits.extend([hit.text for hit in tess_hits])

    def _consistency_boost(text: str) -> float:
        count = variant_hits.count(text)
        return min(18.0, float(count - 1) * 6.0)

    if all_hits:
        best_tesseract = max(
            all_hits,
            key=lambda c: c.confidence + _consistency_boost(c.text),
        )
        boosted_conf = best_tesseract.confidence + _consistency_boost(best_tesseract.text)
        if boosted_conf >= min_tesseract_confidence:
            return OCRCandidate(
                text=best_tesseract.text,
                confidence=boosted_conf,
                engine="tesseract",
            )

    fallback_hits: list[OCRCandidate] = []
    for variant in variants:
        gray = _to_gray(variant)
        fallback_hits.extend(
            _match_candidates(
                _extract_tokens_paddle(gray),
                pattern,
                min_len=min_len,
                max_len=max_len,
            )
        )
        fallback_hits.extend(
            _match_candidates(
                _extract_tokens_doctr(gray),
                pattern,
                min_len=min_len,
                max_len=max_len,
            )
        )

    if not fallback_hits and not all_hits:
        return None

    combined = all_hits + fallback_hits
    best = max(
        combined,
        key=lambda c: c.confidence + _consistency_boost(c.text) + (4.0 if c.engine != "tesseract" else 0.0),
    )
    return OCRCandidate(
        text=best.text,
        confidence=best.confidence + _consistency_boost(best.text),
        engine=best.engine,
    )

