from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.qr import _build_qr_rois, parse_qr_data


def test_parse_qr_data_includes_student_name_field() -> None:
    parsed = parse_qr_data(
        "type=EXAM;studentId=12345;studentName=Jane Doe;examId=ENG-01"
    )

    assert parsed["studentid"] == "12345"
    assert parsed["studentname"] == "Jane Doe"
    assert parsed["examid"] == "ENG-01"


def test_build_qr_rois_includes_new_right_side_zone() -> None:
    gray = np.zeros((1000, 800), dtype=np.uint8)
    rois = _build_qr_rois(gray)

    assert len(rois) >= 2

    # First ROI should target the new right-side placement (08-42% height, 72-99% width).
    assert rois[0].shape == (340, 216)

    # Second ROI extends lower in case scans push the QR further down.
    assert rois[1].shape == (440, 248)
