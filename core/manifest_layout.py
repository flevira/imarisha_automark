"""Strict V4/V5 QR and geometry validation for the new preparation path."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from core.contracts import LayoutRef


class PreparationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class LegacySheet(PreparationError):
    def __init__(self):
        super().__init__("legacy-sheet", "No manifest reference. Generate answer sheets with a layout QR and import their matching manifest.")


@dataclass(frozen=True)
class SheetIdentity:
    layout_id: str
    version: int
    assessment_type: str
    assessment_id: str
    page_number: int
    student_id: str | None


def require(ok: bool, message: str, code: str = "invalid-manifest") -> None:
    if not ok:
        raise PreparationError(code, message)


def parse_sheet_qr(payload: str) -> SheetIdentity:
    fields = {}
    for part in payload.split(";"):
        if not part.strip():
            continue
        require("=" in part, "Malformed QR field", "invalid-qr")
        key, value = part.split("=", 1)
        key, value = key.strip().lower(), value.strip()
        require(key not in fields, "Duplicate QR field", "invalid-qr")
        fields[key] = value
    if "layoutid" not in fields:
        if fields.get("type", "").upper() in ("EXAM", "TEST"):
            raise LegacySheet()
        raise PreparationError("invalid-qr", "QR is not an assessment reference")
    typ = fields.get("type", "").upper()
    require(typ in ("EXAM", "TEST"), "Unknown assessment type", "invalid-qr")
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", fields["layoutid"]) is not None, "Invalid layout ID", "invalid-qr")
    assessment = fields.get("assessmentid") or fields.get(typ.lower()+"id", "")
    require(re.fullmatch(r"[1-9][0-9]*", assessment) is not None, "Invalid assessment ID", "invalid-qr")
    for key in ("assessmentid", typ.lower()+"id"):
        require(key not in fields or fields[key] == assessment, "Conflicting assessment IDs", "invalid-qr")
    require(("testid" if typ == "EXAM" else "examid") not in fields, "Conflicting assessment type", "invalid-qr")
    require(fields.get("v") in ("4", "5", "6", "7"), "Only manifest sheet versions 4–7 are supported", "unsupported-version")
    require(re.fullmatch(r"[1-9][0-9]*", fields.get("page", "")) is not None, "Invalid page number", "invalid-qr")
    student = fields.get("studentid")
    require(student is None or re.fullmatch(r"[0-9]+", student) is not None, "Invalid student ID", "invalid-qr")
    require((fields["v"] in ("4", "6") and bool(student)) or (fields["v"] in ("5", "7") and student is None),
            "Student identity does not match sheet version", "invalid-qr")
    return SheetIdentity(fields["layoutid"], int(fields["v"]), typ, assessment, int(fields["page"]), student)


def _number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_manifest(manifest: Any) -> dict:
    """Validate the known geometry contract, including all pages before caching.

    Correct-answer fields may be present but are not interpreted by preparation.
    No unknown layout versions or coordinate systems are guessed.
    """
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(type(manifest.get("version")) is int and manifest["version"] in (4, 5, 6, 7), "Unsupported manifest version", "unsupported-version")
    require(isinstance(manifest.get("layoutId"), str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest["layoutId"]) is not None, "Invalid layout ID")
    require(manifest.get("assessmentType") in ("EXAM", "TEST"), "Invalid assessment type")
    for name in ("assessmentId", "assessmentGroupId"):
        require(isinstance(manifest.get(name), str) and re.fullmatch(r"[1-9][0-9]*", manifest[name]) is not None, f"Invalid {name}")
    require(manifest.get("unit") == "pt" and manifest.get("coordinateOrigin") == "top-left", "Unsupported coordinates")
    pages = manifest.get("pages")
    require(isinstance(pages, list) and 1 <= len(pages) <= 200, "Invalid page count")
    row_ids = set()
    region_ids, written_orders, written_numbers = set(), {}, {}
    for number, page in enumerate(pages, 1):
        require(isinstance(page, dict) and type(page.get("pageNumber")) is int and page["pageNumber"] == number, "Pages must be unique and consecutive")
        require(_number(page.get("width"), 595.27, 595.29) and _number(page.get("height"), 841.88, 841.90), "V4/V5 requires A4 page dimensions")
        markers = page.get("markers")
        require(isinstance(markers, list) and len(markers) == 4, "Four markers required")
        expected = {"TL": (30, 30), "TR": (565.28, 30), "BR": (565.28, 811.89), "BL": (30, 811.89)}
        require(all(isinstance(m, dict) for m in markers) and {m.get("id") for m in markers} == set(expected), "Invalid marker identities")
        for marker in markers:
            x, y = expected[marker["id"]]
            require(_number(marker.get("cx"), x-.01, x+.01) and _number(marker.get("cy"), y-.01, y+.01)
                    and marker.get("size") == 16, "Invalid V4/V5 marker geometry")
        rows = page.get("rows")
        require(isinstance(rows, list) and len(rows) <= (54 if manifest["version"] in (5, 7) else 60), "Invalid rows")
        for row in rows:
            require(isinstance(row, dict), "Invalid row")
            require(isinstance(row.get("questionId"), str) and bool(row["questionId"].strip()) and _positive_int(row.get("number")), "Invalid question identity")
            kind = row.get("kind")
            require(kind in ("MC", "MA", "TF", "MATCH"), "Unknown response kind")
            require(row.get("selectionMode") == ("multiple" if kind == "MA" else "single"), "Invalid selection mode")
            require(_number(row.get("x"), 0, page["width"]) and _number(row.get("y"), 0, page["height"]), "Invalid row position")
            require(_positive_int(row.get("item")) if kind == "MATCH" else "item" not in row, "Invalid matching item")
            key = (row["questionId"], row["number"], row.get("item"))
            require(key not in row_ids, "Duplicate response row")
            row_ids.add(key)
            bubbles = row.get("bubbles")
            require(isinstance(bubbles, list) and 1 <= len(bubbles) <= 4, "Invalid bubble list")
            allowed = {"true", "false"} if kind == "TF" else set("ABCD")
            require(all(isinstance(b, dict) and b.get("optionId") in allowed for b in bubbles), "Invalid option ID")
            require(len({b["optionId"] for b in bubbles}) == len(bubbles), "Duplicate options")
            if kind == "TF":
                require({b["optionId"] for b in bubbles} == allowed, "True/false requires both options")
            for b in bubbles:
                _bubble(b, page)
        regions = page.get("writtenRegions", [])
        require(isinstance(regions, list) and len(regions) <= 20, "Invalid written regions")
        require(not regions or manifest["version"] in (6, 7), "Written regions require layout V6/V7")
        occupied = []
        for region in regions:
            require(isinstance(region, dict), "Invalid written region")
            qid = region.get("questionId")
            require(isinstance(qid, str) and bool(qid.strip()) and _positive_int(region.get("number")), "Invalid written question")
            rid = region.get("regionId")
            require(isinstance(rid, str) and bool(rid.strip()) and rid not in region_ids, "Duplicate or missing region ID")
            region_ids.add(rid)
            require(_positive_int(region.get("order")), "Invalid written reading order")
            written_orders.setdefault(qid, []).append(region["order"])
            require(written_numbers.setdefault(qid,region["number"]) == region["number"], "Conflicting written numbering")
            require(region.get("kind") == "WRITTEN", "Invalid written kind")
            x,y,w,h = (region.get(k) for k in ("x","y","width","height"))
            require(_number(x,40,550) and _number(y,300 if manifest["version"]==7 else 180,760)
                    and _number(w,30,515) and _number(h,20,580) and x+w<=555 and y+h<=770, "Written region outside answer area")
            box = (x,y,x+w,y+h)
            for prior in occupied:
                require(box[2]<=prior[0] or box[0]>=prior[2] or box[3]<=prior[1] or box[1]>=prior[3], "Overlapping written regions")
            occupied.append(box)
            for row in rows:
                require(row["questionId"]!=qid, "Question cannot be both bubble and written")
                for b in row["bubbles"]:
                    r=b["diameter"]/2+4
                    require(x+w<=b["cx"]-r or x>=b["cx"]+r or y+h<=b["cy"]-r or y>=b["cy"]+r, "Written region overlaps bubbles")
        if manifest["version"] in (4, 6):
            require("studentNumber" not in page, "Personalized layout cannot contain a shared grid")
        else:
            grid = page.get("studentNumber", {})
            require(isinstance(grid, dict) and grid.get("field") == "studentId" and grid.get("digitCount") == 5
                    and grid.get("order") == "left-to-right" and grid.get("selectionMode") == "single-per-column"
                    and grid.get("preserveLeadingZeros") is True, "Invalid student grid")
            cols = grid.get("columns")
            require(isinstance(cols, list) and len(cols) == 5, "Five student columns required")
            for pos, col in enumerate(cols, 1):
                require(isinstance(col, dict) and col.get("position") == pos, "Invalid digit position")
                bubbles = col.get("bubbles")
                require(isinstance(bubbles, list) and len(bubbles) == 10, "Ten digits required")
                for value, b in enumerate(bubbles):
                    require(isinstance(b, dict) and b.get("value") == str(value), "Invalid digit value")
                    _bubble(b, page)
    for qid, orders in written_orders.items():
        require(sorted(orders)==list(range(1,len(orders)+1)), "Written reading order must be unique and consecutive")
        require(not any(row[0]==qid for row in row_ids), "Question cannot be bubble and written")
    if manifest["version"] in (6,7):
        require(bool(region_ids), "V6/V7 requires written regions")
    return manifest


def _bubble(bubble: dict, page: dict) -> None:
    require(_number(bubble.get("diameter"), 14.17, 14.19), "Invalid V4/V5 bubble size")
    r = bubble["diameter"]/2
    require(_number(bubble.get("cx"), r, page["width"]-r) and _number(bubble.get("cy"), r, page["height"]-r), "Bubble outside page")


def bind_sheet(identity: SheetIdentity, manifest: dict) -> tuple[LayoutRef, dict]:
    validate_manifest(manifest)
    # An authenticated endpoint can resolve historical QR aliases to canonical IDs.
    require(manifest["assessmentType"] == identity.assessment_type, "QR/manifest assessment type mismatch", "identity-mismatch")
    require(manifest["version"] == identity.version, "QR/manifest version mismatch", "identity-mismatch")
    require(identity.page_number <= len(manifest["pages"]), "QR page not in layout", "page-missing")
    ref = LayoutRef(manifest["layoutId"], manifest["version"], manifest["assessmentType"], manifest["assessmentId"], manifest["assessmentGroupId"])
    return ref, manifest["pages"][identity.page_number-1]


def validate_manifest(manifest: Any) -> dict:
    try:
        return _validate_manifest(manifest)
    except PreparationError:
        raise
    except (TypeError, KeyError, ValueError, OverflowError):
        raise PreparationError("invalid-manifest", "Malformed manifest geometry") from None
