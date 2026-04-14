from __future__ import annotations

from typing import Any

from core.extractor import extract_file, extract_file_with_report


def run_extraction(file_paths: list[str], student_id_mode: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        rows = extract_file(file_path=file_path, student_id_mode=student_id_mode)
        results.extend(rows)
    return results


def run_extraction_with_report(
    file_paths: list[str], student_id_mode: str
) -> tuple[list[dict[str, Any]], str]:
    results: list[dict[str, Any]] = []
    report_parts: list[str] = []
    for file_path in file_paths:
        rows, report_text = extract_file_with_report(
            file_path=file_path,
            student_id_mode=student_id_mode,
        )
        results.extend(rows)
        report_parts.append(report_text)
    return results, "\n\n".join(part for part in report_parts if part.strip())
