from core.extractor import _normalize_detected_question_id, _question_id_roi_bounds


def test_normalize_detected_question_id_handles_common_ocr_substitutions() -> None:
    assert _normalize_detected_question_id("8IS9G") == "81596"
    assert _normalize_detected_question_id("B229B") == "82298"


def test_normalize_detected_question_id_trims_prefixed_row_numbers() -> None:
    assert _normalize_detected_question_id("10 82298") == "82298"


def test_question_id_roi_bounds_focus_on_qid_band() -> None:
    assert _question_id_roi_bounds(1000) == (100, 350)
