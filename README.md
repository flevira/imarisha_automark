# Imarisha Automark

Desktop application for extracting structured data from scanned exam answer sheets using Python and Flet.

## Goal

Build an offline-first desktop app that:

- lets a teacher select PDF or image files
- chooses student ID source mode: QR, TEXT, or AUTO
- extracts:
  - student_id
  - exam_id
  - test_id
  - sheet_type
  - section
  - question_id
  - item_number
  - answer
  - needs_review
- stores extracted rows locally in SQLite
- opens a review screen for correction and approval
- exports approved rows to CSV

## Current extraction prototype

The current working extraction prototype is based on the code in `docs/prototype_extractor.py`.

Main capabilities in the prototype:

- QR extraction
- PDF rendering with PyMuPDF
- student ID resolution
- section detection
- row extraction
- bubble scoring
- MC / TF / Matching support
- CSV output

## Target desktop architecture

```text
app/
  main.py
  routes.py
  views/
    home_view.py
    review_view.py
  controls/
    extraction_table.py

core/
  extractor.py
  qr.py
  student_id.py
  sections.py
  rows.py
  bubbles.py

database/
  db.py
  schema.sql
  repository.py

services/
  extraction_service.py
  export_service.py

docs/
  prototype_extractor.py
  codex_tasks.md

tests/
  test_qr.py
  test_student_id.py
  test_sections.py
  test_rows.py
```
