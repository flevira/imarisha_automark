# Capability Review: answer extraction and bubbled student ID

Date: 2026-04-14

## Conclusion

This codebase can extract answer bubbles (MC/TF/Matching) and can read a bubbled student ID.

## Logic used to capture different question types

1. **Section type detection**
   - The extractor detects section headers and maps them to internal types:
     - `MC` for Multiple Choice
     - `TF` for True/False
     - `MATCH` for Matching
     - `OPEN_ENDED` for open questions
   - There are two paths:
     - Text-layer PDFs use direct text search in `core/sections.py`.
     - Scanned images use OCR + fuzzy header matching in `core/extractor.py`.
   - The page is split into left/right columns around the page midpoint, and section blocks are built within each column so mixed two-column layouts are handled explicitly.

2. **Row capture per section**
   - For text-layer pages (`core/rows.py`):
     - Extraction starts **after** the detected column heading line (e.g., `NO QID ...`) when present.
     - `MC`, `TF`, `OPEN_ENDED` rows are captured when a 5–6 digit `question_id` is found.
     - `MATCH` rows require both a 5–6 digit `question_id` and an `item_number` in `[1-4]`.
   - For scanned pages (`core/extractor.py`):
     - The scanned section ROI first attempts to locate the table header line and starts row-band detection below that line.
     - Row bands are detected using horizontal projection profiles.
     - `question_id` is OCR'd from the left ROI.
     - `MATCH` additionally OCRs `item_number` from its item ROI.

3. **Bubble answer interpretation by section**
   - In `core/bubbles.py`, option labels are selected by section:
     - `MC` -> `A/B/C/D`
     - `TF` -> `T/F` (mapped to `True/False`)
     - `MATCH` -> `A/B/C/D`
   - For scanned pages, option centers now prefer OCR-detected column labels in the section header row (e.g., columns 3-6), and only fall back to column-layout heuristics if all labels are not reliably found.
   - The confidence model determines marked options.
   - `MC` allows multiple answers; `TF` and `MATCH` require exactly one selected option or row is flagged `needs_review`.

4. **Output shaping**
   - Extracted answers are written as normalized rows via `append_output_rows`.
   - `MATCH` rows also produce `matching_pair` values (e.g., `A=C`) based on `item_number`.

## Student ID extraction logic

- Bubble ID extraction is implemented in `core/student_id.py` and scans a 5-column x 10-row grid in the top-left ID region.
- Handwritten ID OCR is also attempted from the student details ROI.
- `resolve_student_id` chooses the final ID based on mode (`qr`, `text`, `bubble`, `auto`) and conflict rules.

## Evidence from tests

- `tests/test_student_id.py` validates detection of a filled synthetic bubbled ID (`"00530"`) and bubble resolution behavior.
- `tests/test_confidence_model.py` validates blank handling, single-mark handling, and multiple-answer behavior used by the bubble classifier.

## Important caveat

- QR is still preferred for metadata (`sheet_type`, `exam_id` / `test_id`, QR student fields), but answer extraction now continues even when QR is unreadable.
- Without QR, metadata fields stay empty and rows are typically flagged for review depending on student ID mode/resolution confidence.
- QR decoding attempts include both full-page preprocessing variants and multiple top-band/top-right crops to support header-right QR placements.
