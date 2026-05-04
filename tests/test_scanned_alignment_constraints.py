from core import extractor


def test_option_center_monotonic_mc():
    labels = ["A", "B", "C", "D"]
    centers = {"A": 100.0, "B": 99.0, "C": 130.0, "D": 129.0}
    out = extractor._enforce_monotonic_option_centers(centers, labels)
    assert out["A"] < out["B"] < out["C"] < out["D"]


def test_row_consistency_flags_spacing():
    rows = [
        {"y_center_px": 100.0, "row_bbox_px": (20, 90, 300, 110)},
        {"y_center_px": 130.0, "row_bbox_px": (20, 120, 300, 140)},
        {"y_center_px": 260.0, "row_bbox_px": (20, 250, 300, 270)},
    ]
    flags = extractor._row_consistency_flags(rows, "MC")
    assert 1 in flags or 2 in flags


def test_row_consistency_deterministic_for_known_qids():
    rows = [
        {"question_id": "82575", "y_center_px": 100.0, "row_bbox_px": (20, 90, 300, 110)},
        {"question_id": "81904", "y_center_px": 130.0, "row_bbox_px": (20, 120, 300, 140)},
        {"question_id": "81912", "y_center_px": 160.0, "row_bbox_px": (20, 150, 300, 170)},
        {"question_id": "81918", "y_center_px": 190.0, "row_bbox_px": (20, 180, 300, 200)},
        {"question_id": "81942", "y_center_px": 220.0, "row_bbox_px": (20, 210, 300, 230)},
    ]
    flags1 = extractor._row_consistency_flags(rows, "MC")
    flags2 = extractor._row_consistency_flags(rows, "MC")
    assert flags1 == flags2
