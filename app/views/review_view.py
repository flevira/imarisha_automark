import flet as ft


def _section_label(section_name: str) -> str:
    aliases = {
        "MC": "Multiple Choice",
        "TF": "True / False",
        "MATCH": "Matching",
        "OPEN_ENDED": "Open Ended",
    }
    return aliases.get(section_name, section_name)


def build_review_view(page: ft.Page, extracted_rows: list[dict] | None = None) -> None:
    page.controls.clear()
    rows = extracted_rows or []

    total_rows = len(rows)
    needs_review_rows = sum(1 for row in rows if row.get("needs_review"))
    manual_marking_rows = sum(1 for row in rows if row.get("manual_marking"))
    section_stats: dict[str, int] = {}

    for row in rows:
        section = str(row.get("section") or "UNKNOWN")
        section_stats[section] = section_stats.get(section, 0) + 1

    section_items = (
        [
            ft.Text(
                f"• {_section_label(section)}: {count}",
                size=14,
            )
            for section, count in sorted(section_stats.items())
        ]
        if section_stats
        else [ft.Text("• No sections available", size=14, italic=True)]
    )

    from app.views.home_view import build_home_view

    page.add(
        ft.Text("Extraction Review", size=28, weight=ft.FontWeight.BOLD),
        ft.Text(f"Total extracted rows: {total_rows}", size=16),
        ft.Text(f"Rows flagged for review: {needs_review_rows}", size=16),
        ft.Text(f"Rows requiring manual marking: {manual_marking_rows}", size=16),
        ft.Container(height=8),
        ft.Text("Section breakdown", size=18, weight=ft.FontWeight.W_600),
        ft.Column(controls=section_items, spacing=6),
        ft.Container(height=12),
        ft.ElevatedButton("Back", on_click=lambda _: build_home_view(page)),
    )
    page.update()
