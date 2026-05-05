from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core import extractor


def _load_four_image() -> np.ndarray:
    img_path = Path(__file__).resolve().parents[1] / "FOUR.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load fixture: {img_path}"
    return img


def _ordered_centers(diag: dict) -> list[np.ndarray]:
    selected = diag["selected"]
    return [
        np.array(selected["tl"]["center"], dtype=np.float32),
        np.array(selected["tr"]["center"], dtype=np.float32),
        np.array(selected["br"]["center"], dtype=np.float32),
        np.array(selected["bl"]["center"], dtype=np.float32),
    ]


def test_marker_detector_returns_corners_in_tl_tr_br_bl_order() -> None:
    img = _load_four_image()

    _aligned, ok, diag = extractor._align_scanned_page_with_markers_internal(img)

    assert ok, diag
    assert diag["markers_found"] == 4

    tl, tr, br, bl = _ordered_centers(diag)
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]
    assert tl[0] < br[0] and tl[1] < br[1]


def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _mild_perspective_warp(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = int(w * 0.03)
    dy = int(h * 0.03)
    dst = np.float32([[dx, dy], [w - 1 - dx, 0], [w - 1, h - 1 - dy], [0, h - 1]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _brightness_shift(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def test_marker_alignment_robust_to_synthetic_perturbations() -> None:
    base = _load_four_image()
    variants = [
        _rotate(base, 1.0),
        _rotate(base, 0.8),
        _mild_perspective_warp(base),
        _brightness_shift(base, alpha=1.08, beta=12),
        _brightness_shift(base, alpha=0.9, beta=-8),
    ]

    for variant in variants:
        aligned, ok, diag = extractor._align_scanned_page_with_markers_internal(variant)
        assert ok, diag
        assert aligned.shape == variant.shape

        h, w = aligned.shape[:2]
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        _, binary_inv = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

        x_band = int(w * 0.2)
        y_band = int(h * 0.2)
        corners = {
            "tl": binary_inv[:y_band, :x_band],
            "tr": binary_inv[:y_band, w - x_band :],
            "br": binary_inv[h - y_band :, w - x_band :],
            "bl": binary_inv[h - y_band :, :x_band],
        }

        for roi in corners.values():
            # Marker should still occupy at least a small area in each corner band.
            assert int(cv2.countNonZero(roi)) > 150


def test_missing_one_marker_uses_fallback_without_crash() -> None:
    img = _load_four_image().copy()
    h, w = img.shape[:2]

    # Remove the top-left marker area to force marker-alignment fallback.
    cv2.rectangle(img, (0, 0), (int(w * 0.2), int(h * 0.2)), (255, 255, 255), thickness=-1)

    aligned, ok, diag = extractor._align_scanned_page_with_markers_internal(img)

    assert not ok
    assert diag["fallback_used"] is True
    assert aligned.shape == img.shape

    fallback = extractor.align_scanned_page_to_template(img)
    assert fallback is not None
    assert fallback.shape == img.shape
