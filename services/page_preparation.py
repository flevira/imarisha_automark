"""M2 preparation service. The legacy extraction service remains unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time

import cv2
import fitz
import numpy as np

from core.contracts import ExtractionPage, ManifestResolver, page_geometry
from core.manifest_layout import PreparationError, bind_sheet
from core.page_alignment import MarkerAligner, decode_sheet


@dataclass
class PreparedPage:
    source_page_number: int
    status: str
    reason: str | None = None
    message: str | None = None
    page: ExtractionPage | None = None
    seconds: float = 0

    def summary(self) -> dict:
        result = dict(source_page=self.source_page_number, status=self.status, reason=self.reason, message=self.message, seconds=self.seconds)
        if self.page:
            p = self.page
            result.update(layout_id=p.alignment.layout.layout_id, layout_page=p.alignment.layout_page_number,
                          student_id=p.student_id, assessment_group=p.alignment.layout.assessment_group_id,
                          alignment_error_pt=p.alignment.error_pt, pixel_to_page=list(p.alignment.pixel_to_page))
        return result


@dataclass
class PreparedDocument:
    source_id: str
    pages: list[PreparedPage]
    issues: list[dict] = field(default_factory=list)
    total_seconds: float = 0

    def summary(self) -> dict:
        return dict(source_id=self.source_id, pages=[p.summary() for p in self.pages], issues=self.issues, total_seconds=self.total_seconds,
                    scope="Page preparation only; no bubble reading, handwriting or marking")


def prepare_image(image: np.ndarray, source_id: str, source_page_number: int, resolver: ManifestResolver) -> PreparedPage:
    start = time.perf_counter()
    result = PreparedPage(source_page_number, "review")
    try:
        decoded = decode_sheet(image)
        identity = decoded.identity
        manifest = resolver.resolve(identity.layout_id, identity.assessment_type, identity.assessment_id)
        layout, _ = bind_sheet(identity, manifest)
        geometry = page_geometry(manifest, identity.page_number)
        geometry["layoutPageCount"] = len(manifest["pages"])
        alignment = MarkerAligner().align_decoded(image, geometry, layout, decoded)
        image_png = cv2.imencode(".png", image)[1].tobytes()
        result.status = "prepared"
        result.page = ExtractionPage(source_id, source_page_number, image_png, alignment, geometry, identity.student_id)
    except PreparationError as exc:
        result.reason, result.message = exc.code, str(exc)
        if exc.code == "legacy-sheet":
            result.status = "legacy"
    result.seconds = time.perf_counter()-start
    return result


def audit_pages(document: PreparedDocument, expected_counts: dict[str, int]) -> None:
    groups = {}
    for prepared in document.pages:
        if not prepared.page:
            continue
        page = prepared.page
        if page.student_id is None:
            # Shared sheets cannot be grouped by student until M3 reads the grid.
            document.issues.append(dict(code="student-identity-pending", source_page=prepared.source_page_number))
            continue
        key = (page.alignment.layout.layout_id, page.student_id)
        groups.setdefault(key, []).append(prepared)
    for (layout, student), pages in groups.items():
        by_number = {}
        for prepared in pages:
            by_number.setdefault(prepared.page.alignment.layout_page_number, []).append(prepared)
        missing = sorted(set(range(1, expected_counts[layout]+1))-set(by_number))
        if missing:
            document.issues.append(dict(code="missing-pages", layout_id=layout, student_id=student, pages=missing))
        for number, duplicates in by_number.items():
            if len(duplicates) > 1:
                document.issues.append(dict(code="duplicate-page", layout_id=layout, student_id=student, page=number,
                                           source_pages=[p.source_page_number for p in duplicates]))
                for prepared in duplicates:
                    prepared.status = "review"
                    prepared.reason = "duplicate-page"
                    prepared.message = "Duplicate student/layout/page identity; do not extract twice"
                    # Keep evidence/transform for review; downstream must honor status.


def prepare_document(path: Path, resolver: ManifestResolver, *, dpi: int = 150) -> PreparedDocument:
    start = time.perf_counter()
    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise PreparationError("invalid-dpi", "Preparation DPI must be between 72 and 300")
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    result = PreparedDocument("sha256:"+digest.hexdigest(), [])
    with fitz.open(path) as doc:
        if not 1 <= len(doc) <= 200:
            raise PreparationError("page-limit", "Prepare between 1 and 200 pages per document")
        for number, page in enumerate(doc, 1):
            if page.rect.width*page.rect.height*(dpi/72)**2 > 20_000_000:
                result.pages.append(PreparedPage(number, "review", "image-too-large", "Page exceeds preparation pixel limit"))
                continue
            pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
            image = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
            prepared = prepare_image(image, result.source_id, number, resolver)
            result.pages.append(prepared)
    # The internal page count is copied from the resolved manifest by the preparation helper.
    audit_pages(result, {p.page.alignment.layout.layout_id: p.page.geometry["layoutPageCount"] for p in result.pages if p.page})
    result.total_seconds = time.perf_counter()-start
    return result
