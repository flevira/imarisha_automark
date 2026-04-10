from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.extractor import extract_file


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
    if rows:
        print(rows[0])


if __name__ == "__main__":
    main()
