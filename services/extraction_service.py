from __future__ import annotations

from typing import Any

from core.extractor import extract_file


def run_extraction(file_paths: list[str], student_id_mode: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        rows = extract_file(file_path=file_path, student_id_mode=student_id_mode)
        results.extend(rows)
    return results
