from __future__ import annotations

from typing import Any

from core.extractor import extract_file


def run_extraction(file_path: str, student_id_mode: str) -> list[dict[str, Any]]:
    return extract_file(file_path=file_path, student_id_mode=student_id_mode)

from docs.prototype_extractor import process_pdf  # we will define this

def run_extraction(file_paths, student_id_mode):
    results = []

    for file_path in file_paths:
        rows = process_pdf(file_path, student_id_mode)
        results.extend(rows)

    return results
