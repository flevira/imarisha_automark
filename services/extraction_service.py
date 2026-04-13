from __future__ import annotations

from typing import Any

from core.extractor import extract_file
from services.pdf_ocr_enhancer import cleanup_prepared_input, prepare_extraction_input


def run_extraction(file_paths: list[str], student_id_mode: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        prepared = prepare_extraction_input(file_path=file_path)
        try:
            rows = extract_file(file_path=prepared.path, student_id_mode=student_id_mode)
            if prepared.path != file_path:
                for row in rows:
                    row["source_file"] = file_path
                    row["ocr_enhanced_input"] = prepared.enhanced
            else:
                for row in rows:
                    row["ocr_enhanced_input"] = prepared.enhanced
            results.extend(rows)
        finally:
            cleanup_prepared_input(prepared.cleanup_path)
    return results
