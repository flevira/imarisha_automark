from core.rows import _dedupe_text_row_candidates


def _row(section: str, qid: str | None, item: int | None, y0: float, y1: float) -> dict[str, object]:
    return {
        "section_type": section,
        "question_id": qid,
        "item_number": item,
        "y_center": (y0 + y1) / 2.0,
        "row_bbox": (0.0, y0, 100.0, y1),
    }


def test_dedupe_tf_rows_by_question_id() -> None:
    rows = [
        _row("TF", "81596", None, 100, 126),
        _row("TF", "81596", None, 101, 121),
        _row("TF", "82298", None, 140, 164),
    ]

    out = _dedupe_text_row_candidates(rows)

    assert len(out) == 2
    assert [row["question_id"] for row in out] == ["81596", "82298"]


def test_dedupe_match_rows_by_question_and_item() -> None:
    rows = [
        _row("MATCH", "82987", 1, 200, 226),
        _row("MATCH", "82987", 1, 201, 220),
        _row("MATCH", "82987", 2, 226, 250),
    ]

    out = _dedupe_text_row_candidates(rows)

    assert len(out) == 2
    assert [(row["question_id"], row["item_number"]) for row in out] == [
        ("82987", 1),
        ("82987", 2),
    ]
