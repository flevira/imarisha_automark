"""Manifest-driven OMR. Reads pixels only; no answer keys, OCR or marking."""
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np

from core.contracts import ExtractedResponse, ExtractionPage, ReadStatus, ResponseType, SourceRegion
from core.manifest_layout import PreparationError
from core.page_alignment import normalized_preview


@dataclass(frozen=True)
class BubbleEvidence:
    option_id: str
    darkness: float
    state: str
    region: SourceRegion


@dataclass
class StructuredPage:
    student_id: str | None
    identity_reasons: tuple[str, ...]
    responses: tuple[ExtractedResponse, ...]
    evidence: dict[str, tuple[BubbleEvidence, ...]]
    crops: dict[str, bytes]


class StructuredExtractor:
    """Conservative initial thresholds, to be calibrated on real filled scans.

    Inner disks exclude printed outlines. Mean grayscale darkness preserves faint
    marks instead of thresholding them away. Intermediate marks always need review.
    """
    def extract(self, page: ExtractionPage) -> tuple[ExtractedResponse, ...]:
        return self.extract_page(page).responses

    def extract_page(self, page: ExtractionPage) -> StructuredPage:
        image = cv2.imdecode(np.frombuffer(page.image_png, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise PreparationError("invalid-image", "Cannot decode prepared source image")
        normalized = normalized_preview(image, page.geometry, page.alignment)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        crops, evidence = {}, {}

        def read(bubble, option_key):
            x, y, radius = bubble['cx'], bubble['cy'], bubble['diameter']/2
            # normalized_preview uses two pixels per PDF point.
            left, top = int(np.floor((x-radius)*2)), int(np.floor((y-radius)*2))
            right, bottom = int(np.ceil((x+radius)*2)), int(np.ceil((y+radius)*2))
            if left < 0 or top < 0 or right > gray.shape[1] or bottom > gray.shape[0]:
                raise PreparationError("region-outside-page", "Bubble region exceeds normalized page")
            crop = normalized[top:bottom, left:right]
            encoded = cv2.imencode('.png', crop)[1].tobytes()
            ref = 'sha256:' + hashlib.sha256(encoded).hexdigest()
            crops[ref] = encoded
            region = SourceRegion(page.source_id, page.source_page_number,
                                  (left/2, top/2, (right-left)/2, (bottom-top)/2), ref)
            yy, xx = np.ogrid[top:bottom, left:right]
            inner = (xx-x*2)**2 + (yy-y*2)**2 <= (radius*2*.55)**2
            darkness = float(np.mean(1-gray[top:bottom,left:right][inner]/255))
            state = 'filled' if darkness >= .60 else 'blank' if darkness <= .08 else 'uncertain'
            return BubbleEvidence(bubble[option_key], darkness, state, region)

        def selections(bubbles, single):
            selected = tuple(b.option_id for b in bubbles if b.state == 'filled')
            reasons = []
            if any(b.state == 'uncertain' for b in bubbles):
                reasons.append('ambiguous-mark')
            if single and len(selected) > 1:
                reasons.append('multiple-selections')
            status = ReadStatus.UNCERTAIN if reasons else ReadStatus.READ if selected else ReadStatus.BLANK
            return selected, status, reasons

        student = page.student_id
        identity_reasons = []
        grid = page.geometry.get('studentNumber')
        if grid:
            digits = []
            for column in sorted(grid['columns'], key=lambda c: c['position']):
                key = f"student:{column['position']}"
                bubbles = tuple(read(b,'value') for b in column['bubbles'])
                evidence[key] = bubbles
                selected, status, _ = selections(bubbles, True)
                if status != ReadStatus.READ:
                    identity_reasons.append(f"student-column-{column['position']}-{status.value}")
                else:
                    digits.append(selected[0])
            grid_student = ''.join(digits) if not identity_reasons else None
            if student is not None and grid_student is not None and student != grid_student:
                identity_reasons.append('student-identity-conflict')
            student = grid_student if not identity_reasons else None
        if student is None:
            identity_reasons.append('student-identity-unresolved')
        responses = []
        for row in page.geometry['rows']:
            key = f"question:{row['questionId']}:{row.get('item', 0)}"
            bubbles = tuple(read(b, 'optionId') for b in row['bubbles'])
            evidence[key] = bubbles
            selected, status, reasons = selections(bubbles, row['selectionMode'] == 'single')
            reasons.extend(identity_reasons)
            response_id = hashlib.sha256(f"{page.source_id}|{page.source_page_number}|{page.alignment.layout.layout_id}|{key}".encode()).hexdigest()
            responses.append(ExtractedResponse(response_id, page.alignment.layout, student,
                row['questionId'], row['number'], page.alignment.layout_page_number,
                ResponseType(row['kind']), tuple(b.region for b in bubbles), selected_options=selected,
                item_number=row.get('item'), status=status, review_reasons=tuple(reasons), engine='manifest-omr-v1'))
        return StructuredPage(student, tuple(identity_reasons), tuple(responses), evidence, crops)
