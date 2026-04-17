import cv2
import numpy as np
from core.ocr_engine import OCRCandidate

from core import student_id as student_id_module
from core.student_id import extract_student_id_bubble, extract_student_id_handwritten_ocr, resolve_student_id


def _build_bubble_grid_image(student_id: str) -> np.ndarray:
    img = np.full((1200, 900, 3), 255, dtype=np.uint8)
    x0 = int(0.02 * img.shape[1])
    y0 = int(0.14 * img.shape[0])
    roi_w = int((0.48 - 0.02) * img.shape[1])
    roi_h = int((0.44 - 0.14) * img.shape[0])

    cols = np.linspace(x0 + 50, x0 + roi_w - 50, 5).astype(int)
    rows = np.linspace(y0 + 30, y0 + roi_h - 30, 10).astype(int)
    radius = 14

    for cx in cols:
        for cy in rows:
            cv2.circle(img, (int(cx), int(cy)), radius, (0, 0, 0), 2)

    for col_idx, ch in enumerate(student_id):
        row_idx = int(ch)
        cx = int(cols[col_idx])
        cy = int(rows[row_idx])
        cv2.circle(img, (cx, cy), radius - 4, (0, 0, 0), -1)

    return img


def test_extract_student_id_bubble_reads_filled_rows() -> None:
    img = _build_bubble_grid_image("00530")
    detected, low_conf = extract_student_id_bubble(img)
    assert detected == "00530"
    assert low_conf is False




def test_extract_student_id_bubble_uses_zero_for_unfilled_column() -> None:
    img = _build_bubble_grid_image("94567")

    x0 = int(0.02 * img.shape[1])
    y0 = int(0.14 * img.shape[0])
    roi_w = int((0.48 - 0.02) * img.shape[1])
    roi_h = int((0.44 - 0.14) * img.shape[0])
    cols = np.linspace(x0 + 50, x0 + roi_w - 50, 5).astype(int)
    rows = np.linspace(y0 + 30, y0 + roi_h - 30, 10).astype(int)

    # Clear the first filled bubble so column 1 is empty and should read as 0.
    cv2.circle(img, (int(cols[0]), int(rows[9])), 12, (255, 255, 255), -1)

    detected, low_conf = extract_student_id_bubble(img)
    assert detected is not None
    assert detected[0] == "0"
    assert low_conf is True

def test_resolve_student_id_prefers_bubble_when_no_qr() -> None:
    resolved = resolve_student_id(
        qr_id=None,
        bubble_id="00530",
        handwritten_id="00530",
        mode="auto",
    )
    assert resolved["student_id"] == "00530"
    assert resolved["student_id_source"] == "text_bubble"
    assert resolved["student_id_bubble"] == "00530"
    assert resolved["id_conflict"] is False


def test_resolve_student_id_bubble_mode_uses_only_bubble() -> None:
    resolved = resolve_student_id(
        qr_id="12345",
        bubble_id="00530",
        handwritten_id="88888",
        mode="bubble",
    )
    assert resolved["student_id"] == "00530"
    assert resolved["student_id_all"] == "00530"
    assert resolved["student_id_source"] == "text_bubble"
    assert resolved["needs_review"] is False
    assert resolved["id_conflict"] is False


def test_extract_student_id_handwritten_ocr_uses_bracket_slots_and_zero_for_empty(
    monkeypatch,
) -> None:
    img = np.full((1200, 900, 3), 255, dtype=np.uint8)

    # Force bracket detection and control slot OCR outputs:
    monkeypatch.setattr(
        student_id_module,
        "_find_studentnumber_bracket_bounds",
        lambda _gray: (20, 20, 220, 80),
    )
    slot_hits = iter(
        [
            OCRCandidate(text="9", confidence=92.0, engine="paddleocr"),
            OCRCandidate(text="1", confidence=87.0, engine="paddleocr"),
            None,
            OCRCandidate(text="4", confidence=93.0, engine="paddleocr"),
            None,
        ]
    )
    monkeypatch.setattr(student_id_module, "detect_with_fallback", lambda *_args, **_kwargs: next(slot_hits))

    detected, low_conf = extract_student_id_handwritten_ocr(img)
    assert detected == "91040"
    assert low_conf is True


def test_find_studentnumber_bracket_bounds_requires_studentnumber_heading(monkeypatch) -> None:
    roi = np.full((120, 300), 255, dtype=np.uint8)
    fake_data = {
        "text": ["[", "]"],
        "conf": [90, 90],
        "left": [50, 240],
        "top": [40, 40],
        "width": [8, 8],
        "height": [16, 16],
    }
    monkeypatch.setattr(student_id_module.pytesseract, "image_to_data", lambda *_args, **_kwargs: fake_data)

    bounds = student_id_module._find_studentnumber_bracket_bounds(roi)
    assert bounds is None


def test_extract_student_id_handwritten_ocr_returns_zeros_when_slots_empty(monkeypatch) -> None:
    img = np.full((1200, 900, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        student_id_module,
        "_find_studentnumber_bracket_bounds",
        lambda _gray: (20, 20, 220, 80),
    )
    monkeypatch.setattr(student_id_module, "detect_with_fallback", lambda *_args, **_kwargs: None)

    detected, low_conf = extract_student_id_handwritten_ocr(img)
    assert detected == "00000"
    assert low_conf is True


def test_extract_student_id_bubble_falls_back_to_ocr_grid_when_bubble_contours_missing(monkeypatch) -> None:
    img = np.full((1200, 900, 3), 255, dtype=np.uint8)
    x0 = int(0.02 * img.shape[1])
    y0 = int(0.14 * img.shape[0])
    roi_w = int((0.48 - 0.02) * img.shape[1])
    roi_h = int((0.44 - 0.14) * img.shape[0])

    # ROI-local grid coordinates used by the OCR-label fallback.
    cols = np.linspace(70, roi_w - 70, 5).astype(int)
    rows = np.linspace(35, roi_h - 25, 10).astype(int)
    wanted = "91530"

    for col_idx, ch in enumerate(wanted):
        cx = int(x0 + cols[col_idx])
        cy = int(y0 + rows[int(ch)])
        cv2.circle(img, (cx, cy), 10, (0, 0, 0), -1)

    tokens = [
        {"norm": "STUDENTNUMBER", "cx": float(140), "cy": float(8), "x": 80, "y": 2, "w": 180, "h": 18},
        {"norm": "#", "cx": float(12), "cy": float(24), "x": 8, "y": 18, "w": 8, "h": 14},
    ]
    for idx, cx in enumerate(cols, start=1):
        tokens.append(
            {"norm": str(idx), "cx": float(cx), "cy": float(24), "x": int(cx) - 5, "y": 18, "w": 10, "h": 14}
        )
    for idx, cy in enumerate(rows):
        tokens.append({"norm": str(idx), "cx": float(12), "cy": float(cy), "x": 8, "y": int(cy) - 5, "w": 10, "h": 12})

    monkeypatch.setattr(student_id_module, "_detect_bubble_candidates", lambda _gray: [])
    monkeypatch.setattr(student_id_module, "_ocr_tokens", lambda _gray: tokens)

    detected, low_conf = extract_student_id_bubble(img)
    assert detected == wanted
    assert low_conf is False
