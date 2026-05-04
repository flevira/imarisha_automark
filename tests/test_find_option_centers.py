"""Tests for find_option_centers() in core/sections.py.

Verifies that:
- MC sections use an 80 px search zone (labels in a compact header row).
- MATCH sections use a 150 px search zone (labels spread vertically, one per item row).
- Option C is detected in MATCH sections even when its label falls beyond 80 px.
"""
from __future__ import annotations

import fitz

from core import sections as sections_module


class _DummyPage:
    def __init__(self) -> None:
        self.rect = fitz.Rect(0, 0, 600, 800)


# ---------------------------------------------------------------------------
# MC section – 80 px zone is sufficient; all four labels sit in a single row
# ---------------------------------------------------------------------------

def test_mc_detects_all_labels_within_80px(monkeypatch) -> None:
    """All four MC option labels appear in a single header row at y≈10 px."""
    section = {"section_type": "MC", "y0": 100.0, "y1": 500.0, "x0": 0.0, "x1": 600.0}

    words = [
        # Option labels in a tight header row (y0=108, y1=118 → within 80 px of y0=100)
        (50.0, 108.0, 60.0, 118.0, "A"),
        (80.0, 108.0, 90.0, 118.0, "B"),
        (110.0, 108.0, 120.0, 118.0, "C"),
        (140.0, 108.0, 150.0, 118.0, "D"),
    ]
    monkeypatch.setattr(sections_module, "get_page_words", lambda _: words)

    page = _DummyPage()
    centers = sections_module.find_option_centers(page, section)

    assert set(centers.keys()) == {"A", "B", "C", "D"}
    assert centers["A"] == 55.0
    assert centers["C"] == 115.0


def test_mc_ignores_labels_beyond_80px(monkeypatch) -> None:
    """Labels placed beyond 80 px from section y0 must not be picked up for MC."""
    section = {"section_type": "MC", "y0": 100.0, "y1": 500.0, "x0": 0.0, "x1": 600.0}

    words = [
        # Within zone
        (50.0, 108.0, 60.0, 118.0, "A"),
        (80.0, 108.0, 90.0, 118.0, "B"),
        # Beyond 80 px (y0=100+80=180; these words end at y1=195 > 180)
        (110.0, 185.0, 120.0, 195.0, "C"),
        (140.0, 185.0, 150.0, 195.0, "D"),
    ]
    monkeypatch.setattr(sections_module, "get_page_words", lambda _: words)

    page = _DummyPage()
    centers = sections_module.find_option_centers(page, section)

    assert "A" in centers
    assert "B" in centers
    assert "C" not in centers
    assert "D" not in centers


# ---------------------------------------------------------------------------
# MATCH section – 150 px zone; items are spread vertically (~25-30 px each)
# ---------------------------------------------------------------------------

def test_match_detects_option_c_beyond_80px(monkeypatch) -> None:
    """Option C in a MATCH section falls at ~90 px – beyond the old 80 px limit."""
    section = {"section_type": "MATCH", "y0": 200.0, "y1": 700.0, "x0": 0.0, "x1": 600.0}

    # Simulate four vertically-stacked item rows, each ~25 px tall.
    # y offsets from section y0: A≈5, B≈30, C≈55, D≈80 (all within 150 px)
    words = [
        (50.0, 205.0, 60.0, 220.0, "A"),   # offset  5–20 px  ✓
        (50.0, 230.0, 60.0, 245.0, "B"),   # offset 30–45 px  ✓
        (50.0, 255.0, 60.0, 270.0, "C"),   # offset 55–70 px  ✓ (was missed with 80 px)
        (50.0, 280.0, 60.0, 295.0, "D"),   # offset 80–95 px  ✓
    ]
    monkeypatch.setattr(sections_module, "get_page_words", lambda _: words)

    page = _DummyPage()
    centers = sections_module.find_option_centers(page, section)

    assert set(centers.keys()) == {"A", "B", "C", "D"}
    assert centers["C"] == 55.0


def test_match_detects_option_c_at_edge_of_old_limit(monkeypatch) -> None:
    """Option C label whose y1 falls just past 80 px is now captured by the 150 px zone."""
    section = {"section_type": "MATCH", "y0": 200.0, "y1": 700.0, "x0": 0.0, "x1": 600.0}

    # C label: y0=275, y1=290 → y1 offset = 90 px (> 80, ≤ 150)
    words = [
        (50.0, 210.0, 60.0, 225.0, "A"),
        (50.0, 235.0, 60.0, 250.0, "B"),
        (50.0, 275.0, 60.0, 290.0, "C"),   # y1 offset = 90 px – previously missed
        (50.0, 300.0, 60.0, 315.0, "D"),
    ]
    monkeypatch.setattr(sections_module, "get_page_words", lambda _: words)

    page = _DummyPage()
    centers = sections_module.find_option_centers(page, section)

    assert "C" in centers, "Option C must be detected within the 150 px MATCH zone"
    assert centers["C"] == 55.0


def test_match_ignores_labels_beyond_150px(monkeypatch) -> None:
    """Labels beyond 150 px from section y0 are still excluded for MATCH sections."""
    section = {"section_type": "MATCH", "y0": 200.0, "y1": 700.0, "x0": 0.0, "x1": 600.0}

    words = [
        (50.0, 205.0, 60.0, 220.0, "A"),
        (50.0, 230.0, 60.0, 245.0, "B"),
        # C and D placed beyond 150 px (y1 offset > 150)
        (50.0, 355.0, 60.0, 370.0, "C"),   # y1 offset = 170 px
        (50.0, 380.0, 60.0, 395.0, "D"),   # y1 offset = 195 px
    ]
    monkeypatch.setattr(sections_module, "get_page_words", lambda _: words)

    page = _DummyPage()
    centers = sections_module.find_option_centers(page, section)

    assert "A" in centers
    assert "B" in centers
    assert "C" not in centers
    assert "D" not in centers
