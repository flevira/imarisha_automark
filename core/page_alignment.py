"""Corner-marker alignment with QR orientation and independent edge residuals.

No OCR, answer recognition or nearest-question fallback is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from core.contracts import AlignmentResult, LayoutRef
from core.manifest_layout import LegacySheet, PreparationError, SheetIdentity, parse_sheet_qr, require


@dataclass(frozen=True)
class DecodedSheet:
    identity: SheetIdentity
    qr_quad: np.ndarray  # source image pixels, ordered by the QR decoder


def decode_sheet(image: np.ndarray) -> DecodedSheet:
    require(image is not None and image.size > 0, "Empty page image", "invalid-image")
    scale = min(1., 1800/max(image.shape[:2]))
    view = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    def detect(view, factor, offset=(0, 0)):
        detector = cv2.QRCodeDetector()
        found = []
        try:
            ok, payloads, points, _ = detector.detectAndDecodeMulti(view)
            if ok:
                found = [(p, q/factor+np.array(offset)) for p, q in zip(payloads, points) if p]
            if not found:
                payload, quad, _ = detector.detectAndDecode(view)
                if payload and quad is not None:
                    found = [(payload, quad.reshape(4, 2)/factor+np.array(offset))]
        except cv2.error:
            pass
        return found

    candidates = detect(view, scale)
    if not any("=" in p for p, _ in candidates):
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if view.ndim == 3 else view
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        candidates = detect(binary, scale)
    if not any("=" in p for p, _ in candidates):
        # Bounded corner crops recover small or resampled codes without full-page OCR.
        h, w = image.shape[:2]
        candidates = []
        for x, y in ((w//2, 0), (0, 0), (0, h//2), (w//2, h//2)):
            roi = image[y:y+h//2, x:x+w//2]
            factor = min(3., 1600/max(roi.shape[:2]))
            enlarged = cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            candidates.extend(detect(enlarged, factor, (x, y)))
    if not any("=" in p for p, _ in candidates):
        # Tight corner crops provide a different sampling scale for dense QR codes.
        # Bounded recovery only after the normal page/half-page attempts fail.
        h,w=image.shape[:2]
        for x,y in ((2*w//3,0),(0,0),(0,2*h//3),(2*w//3,2*h//3)):
            roi=image[y:min(h,y+h//3),x:min(w,x+w//3)]
            factor=min(1.5,1600/max(roi.shape[:2]))
            enlarged=cv2.resize(roi,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
            candidates.extend(detect(enlarged,factor,(x,y)))
    valid, errors, legacy = [], [], False
    for payload, quad in candidates:
        # Ignore branding URLs; malformed assessment codes must not silently become legacy.
        if "=" not in payload:
            continue
        try:
            valid.append(DecodedSheet(parse_sheet_qr(payload), quad.astype(np.float32)))
        except LegacySheet:
            legacy = True
        except PreparationError as exc:
            errors.append(exc)
    require(len(valid) <= 1 and not (valid and (legacy or errors)), "Multiple/conflicting assessment QR codes", "ambiguous-qr")
    if valid:
        return valid[0]
    if errors:
        raise errors[0]
    if legacy:
        raise LegacySheet()
    raise PreparationError("qr-unreadable", "No readable assessment QR; page needs review")


def _quads(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (.00012*w*h < area < .004*w*h):
            continue
        quad = cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        quad = quad.reshape(4, 2).astype(np.float32)
        sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
        if min(sides)/max(sides) < .65:
            continue
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        if cv2.mean(binary, mask=mask)[0] < 235:
            continue  # QR finder patterns are hollow, markers are solid.
        cx, cy = quad.mean(axis=0)
        if min(cx, w-cx) > .22*w or min(cy, h-cy) > .22*h:
            continue
        result.append(quad)
    return result


def _ordered_corners(image: np.ndarray) -> list[np.ndarray]:
    h, w = image.shape[:2]
    buckets = [[] for _ in range(4)]
    for quad in _quads(image):
        x, y = quad.mean(axis=0)
        index = (0 if x < w/2 else 1) if y < h/2 else (3 if x < w/2 else 2)
        buckets[index].append(quad)
    require(all(len(b) == 1 for b in buckets), "Missing or ambiguous solid corner markers", "markers-unreliable")
    return [b[0] for b in buckets]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)


class MarkerAligner:
    def align(self, image_png: bytes, page_geometry: dict, layout: LayoutRef) -> AlignmentResult:
        image = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_COLOR)
        require(image is not None, "Cannot decode page image", "invalid-image")
        decoded = decode_sheet(image)
        require(decoded.identity.layout_id == layout.layout_id and decoded.identity.assessment_id == layout.assessment_id,
                "Standalone alignment requires the canonical QR layout reference", "identity-mismatch")
        return self.align_decoded(image, page_geometry, layout, decoded)

    def align_decoded(self, image: np.ndarray, geometry: dict, layout: LayoutRef, decoded: DecodedSheet) -> AlignmentResult:
        require(decoded.identity.version == layout.version and decoded.identity.assessment_type == layout.assessment_type
                and decoded.identity.page_number == geometry["pageNumber"], "QR and alignment target disagree", "identity-mismatch")
        corners = _ordered_corners(image)
        names = ("TL", "TR", "BR", "BL")
        markers = {m["id"]: m for m in geometry["markers"]}
        target = np.array([[markers[n]["cx"], markers[n]["cy"]] for n in names], np.float32)
        candidates = []
        # Four rotations, with orientation resolved by the known V4/V5 QR area.
        for shift in range(4):
            ordered = corners[shift:]+corners[:shift]
            source = np.array([q.mean(axis=0) for q in ordered], np.float32)
            matrix = cv2.getPerspectiveTransform(source, target)
            qr = transform_points(decoded.qr_quad, matrix)
            if not (np.all(qr[:, 0] >= 449) and np.all(qr[:, 0] <= 547) and np.all(qr[:, 1] >= 46) and np.all(qr[:, 1] <= 144)):
                continue
            # Ordered QR corners must be upright, not mirrored or rotated in place.
            if not (qr[1, 0] > qr[0, 0] and qr[3, 1] > qr[0, 1]):
                continue
            residuals = []
            for name, quad in zip(names, ordered):
                m = markers[name]
                warped = transform_points(quad, matrix)
                ideal = np.array([[m["cx"]+dx, m["cy"]+dy] for dx, dy in [(-8,-8), (8,-8), (8,8), (-8,8)]])
                # Independent marker edges: a four-centre fit alone always has zero residual.
                residuals.extend(np.min(np.linalg.norm(warped[:, None, :] - ideal[None, :, :], axis=2), axis=1))
            residuals.append(float(np.linalg.norm(qr.mean(axis=0)-np.array([498., 95.]))))
            error = float(max(residuals))
            diameters = [b["diameter"] for r in geometry["rows"] for b in r["bubbles"]]
            if "studentNumber" in geometry:
                diameters += [b["diameter"] for c in geometry["studentNumber"]["columns"] for b in c["bubbles"]]
            tolerance = min(diameters, default=14.18)/4  # half the smallest radius
            require(np.isfinite(matrix).all(), "Invalid transform", "alignment-unreliable")
            if error < tolerance:
                candidates.append((matrix, error))
        require(len(candidates) == 1, "Page orientation or marker geometry is unreliable", "alignment-unreliable")
        matrix, error = candidates[0]
        return AlignmentResult(layout, geometry["pageNumber"], tuple(float(n) for n in matrix.flat), error)


def normalized_preview(image: np.ndarray, geometry: dict, result: AlignmentResult, pixels_per_point: float = 2) -> np.ndarray:
    matrix = np.diag([pixels_per_point, pixels_per_point, 1.]) @ np.array(result.pixel_to_page).reshape(3, 3)
    return cv2.warpPerspective(image, matrix, (round(geometry["width"]*pixels_per_point), round(geometry["height"]*pixels_per_point)), borderValue=(255,255,255))
