from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.extractor import extract_file


def print_stats(rows: list[dict[str, object]]) -> None:
    total_rows = len(rows)
    pages = sorted({row.get("page") for row in rows if row.get("page") is not None})
    with_answers = sum(1 for row in rows if row.get("answer") not in (None, ""))
    needs_review = sum(1 for row in rows if bool(row.get("needs_review")))
    manual_marking = sum(1 for row in rows if bool(row.get("manual_marking")))
    conflicts = sum(1 for row in rows if bool(row.get("id_conflict")))

    section_counts = Counter((row.get("section") or "UNKNOWN") for row in rows)
    sheet_type_counts = Counter((row.get("sheet_type") or "UNKNOWN") for row in rows)

    unique_students = sorted(
        {sid for sid in (row.get("student_id") for row in rows) if sid is not None}
    )

    print("Statistics:")
    print(f"- Total rows: {total_rows}")
    print(f"- Pages seen: {len(pages)} ({', '.join(map(str, pages)) if pages else 'none'})")
    print(f"- Rows with answers: {with_answers}")
    print(f"- Rows missing answers: {total_rows - with_answers}")
    print(f"- Needs review: {needs_review}")
    print(f"- Manual marking rows: {manual_marking}")
    print(f"- ID conflicts: {conflicts}")
    print(f"- Unique student IDs: {len(unique_students)}")
    if unique_students:
        print(f"  IDs: {', '.join(map(str, unique_students))}")

    print("- Rows by section:")
    for section, count in sorted(section_counts.items()):
        print(f"  - {section}: {count}")

    print("- Rows by sheet type:")
    for sheet_type, count in sorted(sheet_type_counts.items()):
        print(f"  - {sheet_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for extract_file")
    parser.add_argument("file_path", help="Path to a PDF answer sheet")
    parser.add_argument(
        "--student-id-mode",
        default="auto",
        choices=["qr", "text", "auto"],
        help="Student ID resolution mode",
    )
    args = parser.parse_args()

    rows = extract_file(args.file_path, args.student_id_mode)
    print(f"Extracted rows: {len(rows)}")
    print_stats(rows)
    if rows:
        print(rows[0])


if __name__ == "__main__":
    main()
