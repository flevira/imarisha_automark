import cv2
import numpy as np

from core.student_id import extract_student_id_bubble, resolve_student_id


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
