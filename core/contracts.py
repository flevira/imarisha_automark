"""M1 boundaries only. No OCR, database writes, network access or marking here.

Wire version 1 belongs to AutoMark records, independently of PDF layout versions.
Positions use PDF points from the top-left of the normalized page, never pixels.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class ResponseType(str, Enum):
    MC = "MC"
    MA = "MA"
    TF = "TF"
    MATCH = "MATCH"
    WRITTEN = "WRITTEN"


class ReadStatus(str, Enum):
    READ = "read"
    BLANK = "blank"
    UNCERTAIN = "uncertain"
    ERROR = "error"


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string; do not coerce numeric IDs")


@dataclass(frozen=True)
class LayoutRef:
    layout_id: str
    version: int
    assessment_type: str
    assessment_id: str
    assessment_group_id: str

    def __post_init__(self) -> None:
        for name in ("layout_id", "assessment_id", "assessment_group_id"):
            identifier(getattr(self, name), name)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Invalid layout version")
        if self.assessment_type not in ("EXAM", "TEST"):
            raise ValueError("Invalid assessment type")


@dataclass(frozen=True)
class SourceRegion:
    source_id: str  # content digest or managed source ID; not a transient upload path
    page_number: int  # one-based source-document page, not QR page identity
    bbox: tuple[float, float, float, float]  # x, y, width, height
    crop_ref: str  # managed crop reference, retained for later review

    def __post_init__(self) -> None:
        identifier(self.source_id, "source_id")
        identifier(self.crop_ref, "crop_ref")
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("Source page must be one-based")
        if len(self.bbox) != 4 or any(type(n) not in (int, float) or not math.isfinite(n) for n in self.bbox):
            raise ValueError("Region requires four finite coordinates")
        if min(self.bbox[:2]) < 0 or min(self.bbox[2:]) <= 0:
            raise ValueError("Region must have nonnegative origin and positive size")


@dataclass(frozen=True)
class ExtractedResponse:
    response_id: str
    layout: LayoutRef
    student_id: str | None
    question_id: str
    display_number: int
    layout_page_number: int
    response_type: ResponseType
    regions: tuple[SourceRegion, ...]
    raw_text: str | None = None
    selected_options: tuple[str, ...] = ()
    item_number: int | None = None
    status: ReadStatus = ReadStatus.READ
    review_reasons: tuple[str, ...] = ()
    engine: str = "fixture"
    revision: int = 1
    schema_version: int = 1
    part_number: int | None = None  # WRITTEN region reading order across pages

    def __post_init__(self) -> None:
        for name in ("response_id", "question_id", "engine"):
            identifier(getattr(self, name), name)
        if self.student_id is not None:
            identifier(self.student_id, "student_id")
        for value in (self.display_number, self.layout_page_number, self.revision):
            if type(value) is not int or value < 1:
                raise ValueError("Numbering and revisions must be positive integers")
        if self.schema_version != 1 or not self.regions:
            raise ValueError("Unsupported record version or missing source evidence")
        if not isinstance(self.response_type, ResponseType) or not isinstance(self.status, ReadStatus):
            raise ValueError("Use explicit response type and read status")
        if self.part_number is not None and (self.response_type != ResponseType.WRITTEN or type(self.part_number) is not int or self.part_number < 1):
            raise ValueError("Written part number must be positive")
        if self.response_type == ResponseType.MATCH:
            if type(self.item_number) is not int or self.item_number < 1:
                raise ValueError("Matching response requires a sub-item")
        elif self.item_number is not None:
            raise ValueError("Sub-item belongs only to matching responses")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise ValueError("Raw text must be a string")
        for option in self.selected_options:
            identifier(option, "option ID")
        if len(set(self.selected_options)) != len(self.selected_options):
            raise ValueError("Duplicate selections")
        if self.response_type == ResponseType.WRITTEN and self.selected_options:
            raise ValueError("Written responses contain text, not selections")
        if self.status == ReadStatus.READ and not (self.selected_options or self.raw_text):
            raise ValueError("A successful read must contain a response; use blank or uncertain otherwise")
        if self.status == ReadStatus.BLANK and (self.selected_options or self.raw_text):
            raise ValueError("Blank is not an ambiguous response")
        if self.status in (ReadStatus.UNCERTAIN, ReadStatus.ERROR) and not self.review_reasons:
            raise ValueError("Uncertain/error responses need a review reason")
        if self.student_id is None and not self.review_reasons:
            raise ValueError("Missing identity needs review")

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons) or self.status in (ReadStatus.UNCERTAIN, ReadStatus.ERROR)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> ExtractedResponse:
        data = json.loads(value)
        data["layout"] = LayoutRef(**data["layout"])
        data["regions"] = tuple(SourceRegion(**{**r, "bbox": tuple(r["bbox"])}) for r in data["regions"])
        data["response_type"] = ResponseType(data["response_type"])
        data["status"] = ReadStatus(data["status"])
        for key in ("selected_options", "review_reasons"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class RecognitionRequest:
    """Deliberately contains no manifest, reference answers, rubric or marks.

    Callers supply only the answer crop, never a question/answer-key page.
    M2/M5 own crop selection; this DTO cannot inspect the meaning of pixels.
    """
    region: SourceRegion
    image_png: bytes
    language: str


@dataclass(frozen=True)
class Transcription:
    text: str | None
    status: ReadStatus
    review_reasons: tuple[str, ...]
    engine: str
    seconds: float = 0
    model_digest: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class AlignmentResult:
    layout: LayoutRef
    layout_page_number: int
    # 3x3 homography: raw image pixels -> normalized top-left PDF points.
    pixel_to_page: tuple[float, ...]
    error_pt: float


@dataclass(frozen=True)
class ExtractionPage:
    """The source image plus its pixel-to-page mapping, not teacher answer data."""
    source_id: str
    source_page_number: int
    image_png: bytes
    alignment: AlignmentResult
    geometry: Mapping[str, Any]  # use page_geometry(), never the full manifest
    student_id: str | None


@dataclass(frozen=True)
class MarkingResult:
    response_id: str
    response_revision: int
    layout_id: str
    layout_version: int
    score: str | None  # decimal string; None means unresolved, never implicit zero
    status: str
    reason: str
    engine: str


class ManifestResolver(Protocol):
    def resolve(self, layout_id: str, assessment_type: str, assessment_id: str) -> Mapping[str, Any]: ...


class PageAligner(Protocol):
    def align(self, image_png: bytes, page_geometry: Mapping[str, Any], layout: LayoutRef) -> AlignmentResult: ...


class HandwritingRecognizer(Protocol):
    def recognize(self, request: RecognitionRequest) -> Transcription: ...


class ResponseExtractor(Protocol):
    def extract(self, page: ExtractionPage) -> tuple[ExtractedResponse, ...]: ...


class ResponseRepository(Protocol):
    def save(self, response: ExtractedResponse) -> None: ...
    def load(self, response_id: str) -> ExtractedResponse | None: ...


class Marker(Protocol):
    def mark(self, response: ExtractedResponse, answer_rules: Mapping[str, Any]) -> MarkingResult: ...


def page_geometry(manifest: Mapping[str, Any], page_number: int) -> dict[str, Any]:
    """Allowlist an extraction-only view of saved V4/V5 pages.

    Full identity/schema/bounds validation is M2 work. Never pass the original
    manifest to recognizers: it contains teacher-only correct answers.
    """
    page = next(p for p in manifest["pages"] if p["pageNumber"] == page_number)
    def select(obj: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: obj[k] for k in keys if k in obj}
    out = select(page, ("pageNumber", "width", "height"))
    out["markers"] = [select(m, ("id", "cx", "cy", "size")) for m in page["markers"]]
    out["rows"] = []
    for row in page["rows"]:
        entry = select(row, ("questionId", "number", "item", "kind", "x", "y", "selectionMode"))
        entry["bubbles"] = [select(b, ("optionId", "cx", "cy", "diameter")) for b in row["bubbles"]]
        out["rows"].append(entry)
    if "studentNumber" in page:
        grid = page["studentNumber"]
        out["studentNumber"] = select(grid, ("field", "digitCount", "order", "selectionMode", "preserveLeadingZeros"))
        out["studentNumber"]["columns"] = [
            {"position": c["position"], "bubbles": [select(b, ("value", "cx", "cy", "diameter")) for b in c["bubbles"]]}
            for c in grid["columns"]
        ]
    if "writtenRegions" in page:
        out["writtenRegions"] = [select(r, ("regionId","questionId","number","kind","order","x","y","width","height")) for r in page["writtenRegions"]]
    # JSON roundtrip detaches nested mutable values from the teacher manifest.
    return json.loads(json.dumps(out, allow_nan=False))
