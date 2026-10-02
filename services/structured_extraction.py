"""M3 document adapter. Preparation failures never enter OMR."""
from dataclasses import dataclass, replace
import time

from core.structured_extraction import StructuredExtractor, StructuredPage
from core.written_extraction import capture_written
from core.manifest_layout import PreparationError
from services.page_preparation import PreparedDocument


@dataclass
class StructuredDocument:
    pages: dict[int, StructuredPage]
    issues: list[dict]
    extraction_seconds: float


def extract_prepared_document(document: PreparedDocument) -> StructuredDocument:
    start = time.perf_counter()
    pages, groups = {}, {}
    issues = [i for i in document.issues if i['code'] != 'student-identity-pending']
    for prepared in document.pages:
        if prepared.status != 'prepared' or prepared.page is None:
            issues.append(dict(code='page-not-extracted', source_page=prepared.source_page_number, reason=prepared.reason))
            continue
        page = prepared.page
        try:
            result = capture_written(page, StructuredExtractor().extract_page(page))
        except PreparationError as exc:
            issues.append(dict(code=exc.code, source_page=prepared.source_page_number, reason=str(exc)))
            continue
        pages[prepared.source_page_number] = result
        if result.identity_reasons:
            issues.append(dict(code='student-identity-review', source_page=prepared.source_page_number, reasons=result.identity_reasons))
        if page.student_id is None and result.student_id is not None:
            key = (page.alignment.layout.layout_id, result.student_id)
            groups.setdefault(key, []).append((prepared, result))
    # Shared pages can only be audited after reading their grids.
    for (layout, student), entries in groups.items():
        numbers = {}
        for prepared, result in entries:
            numbers.setdefault(prepared.page.alignment.layout_page_number, []).append((prepared, result))
        expected = entries[0][0].page.geometry['layoutPageCount']
        missing = sorted(set(range(1, expected+1))-numbers.keys())
        if missing:
            issues.append(dict(code='missing-pages', layout_id=layout, student_id=student, pages=missing))
        for number, duplicates in numbers.items():
            if len(duplicates) > 1:
                issues.append(dict(code='duplicate-page', layout_id=layout, student_id=student, page=number))
                for _, result in duplicates:
                    result.responses = tuple(replace(r, review_reasons=r.review_reasons+('duplicate-page',)) for r in result.responses)
    return StructuredDocument(pages, issues, time.perf_counter()-start)
