from core.extractor import _dedupe_scanned_row_candidates


def _row(section: str, qid: str | None, item: int | None, y0: int, y1: int) -> dict[str, object]:
    return {
        "section_type": section,
        "question_id": qid,
        "item_number": item,
        "y_center_px": (y0 + y1) / 2,
        "row_bbox_px": (0, y0, 100, y1),
        "needs_review": qid is None,
    }


def test_dedupe_mc_rows_by_question_id() -> None:
    rows = [
        _row("MC", "81596", None, 100, 125),
        _row("MC", "81596", None, 101, 120),
        _row("MC", "82298", None, 140, 164),
    ]

    out = _dedupe_scanned_row_candidates(rows)

    assert len(out) == 2
    assert [row["question_id"] for row in out] == ["81596", "82298"]


def test_dedupe_match_rows_by_question_and_item() -> None:
    rows = [
        _row("MATCH", "82987", 1, 200, 225),
        _row("MATCH", "82987", 1, 201, 220),
        _row("MATCH", "82987", 2, 226, 250),
    ]

    out = _dedupe_scanned_row_candidates(rows)

    assert len(out) == 2
    assert [(row["question_id"], row["item_number"]) for row in out] == [
        ("82987", 1),
        ("82987", 2),
    ]


def test_drop_missing_qid_rows_when_real_rows_exist() -> None:
    rows = [
        _row("MC", "81596", None, 100, 125),
        _row("MC", "82298", None, 140, 164),
        _row("MC", None, None, 170, 190),
    ]

    out = _dedupe_scanned_row_candidates(rows)

    assert len(out) == 2
    assert all(row["question_id"] is not None for row in out)
