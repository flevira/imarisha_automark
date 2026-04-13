from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract

from core.extractor import has_usable_text_layer


@dataclass(frozen=True)
class OCREnhancedFile:
    path: str
    cleanup_path: str | None
    enhanced: bool


def _pdf_has_usable_text(file_path: str, min_text_pages: int = 1) -> bool:
    try:
        doc = fitz.open(file_path)
    except Exception:
        return False

    try:
        usable_pages = 0
        for page in doc:
            if has_usable_text_layer(page):
                usable_pages += 1
            if usable_pages >= min_text_pages:
                return True
        return False
    finally:
        doc.close()


def _page_to_rgb_bytes(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    zoom = float(dpi) / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if pix.n == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _ocr_image_to_pdf_bytes(img_rgb: np.ndarray) -> bytes:
    return pytesseract.image_to_pdf_or_hocr(img_rgb, extension="pdf")


def _ocr_enhance_pdf(input_pdf_path: str, output_pdf_path: str, *, dpi: int = 300) -> bool:
    src = fitz.open(input_pdf_path)
    out = fitz.open()
    try:
        for page in src:
            rgb = _page_to_rgb_bytes(page, dpi=dpi)
            pdf_bytes = _ocr_image_to_pdf_bytes(rgb)
            ocr_doc = fitz.open("pdf", pdf_bytes)
            out.insert_pdf(ocr_doc)
            ocr_doc.close()
        out.save(output_pdf_path)
        return True
    except Exception:
        return False
    finally:
        out.close()
        src.close()


def prepare_extraction_input(
    file_path: str,
    *,
    enable_ocr_enhancement: bool = True,
    ocr_dpi: int = 300,
) -> OCREnhancedFile:
    lower = file_path.lower()
    is_pdf = lower.endswith(".pdf")

    if not enable_ocr_enhancement:
        return OCREnhancedFile(path=file_path, cleanup_path=None, enhanced=False)

    # For PDFs that already contain a usable text layer, avoid extra OCR latency.
    if is_pdf and _pdf_has_usable_text(file_path):
        return OCREnhancedFile(path=file_path, cleanup_path=None, enhanced=False)

    temp_dir = tempfile.mkdtemp(prefix="ocr_enhanced_")
    out_pdf = str(Path(temp_dir) / f"{Path(file_path).stem}_ocr_enhanced.pdf")

    if is_pdf:
        success = _ocr_enhance_pdf(file_path, out_pdf, dpi=ocr_dpi)
        if success:
            return OCREnhancedFile(path=out_pdf, cleanup_path=temp_dir, enhanced=True)
        return OCREnhancedFile(path=file_path, cleanup_path=temp_dir, enhanced=False)

    # For image inputs, OCR them into searchable PDF first.
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            return OCREnhancedFile(path=file_path, cleanup_path=temp_dir, enhanced=False)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pdf_bytes = _ocr_image_to_pdf_bytes(rgb)
        with open(out_pdf, "wb") as f:
            f.write(pdf_bytes)
        return OCREnhancedFile(path=out_pdf, cleanup_path=temp_dir, enhanced=True)
    except Exception:
        return OCREnhancedFile(path=file_path, cleanup_path=temp_dir, enhanced=False)


def cleanup_prepared_input(cleanup_path: str | None) -> None:
    if not cleanup_path:
        return
    try:
        if os.path.isdir(cleanup_path):
            for child in Path(cleanup_path).iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            Path(cleanup_path).rmdir()
    except Exception:
        # Best-effort cleanup only.
        pass
