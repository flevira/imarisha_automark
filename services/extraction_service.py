from __future__ import annotations

from typing import Any

from core.extractor import extract_file


def run_extraction(file_path: str, student_id_mode: str) -> list[dict[str, Any]]:
    return extract_file(file_path=file_path, student_id_mode=student_id_mode)
