# Codex Task Plan for Imarisha Automark

## Project objective

Convert the existing extraction prototype into a Python + Flet desktop app.

## Rules

- Build step by step
- Do not replace the prototype with a full rewrite
- Reuse and refactor working logic from `docs/prototype_extractor.py`
- Keep extraction logic separate from UI logic
- Keep code strongly structured and easy to test
- Use SQLite for local persistence
- Keep the app offline-first

## Step 1: Repository scaffolding

Create this structure:

- `app/`
- `app/views/`
- `app/controls/`
- `core/`
- `database/`
- `services/`
- `tests/`

Create placeholder files with imports and minimal structure.

## Step 2: Move prototype logic

Move the extraction logic from `docs/prototype_extractor.py` into:

- `core/qr.py`
- `core/student_id.py`
- `core/sections.py`
- `core/rows.py`
- `core/bubbles.py`
- `core/extractor.py`

Goal:
- `core/extractor.py` should expose one main function:

```python
extract_file(file_path: str, student_id_mode: str) -> list[dict]
```

## Agreed next stage: extraction first, separate marking

Agreed on 2026-10-02. This section records planned work, not completed capabilities. The earlier steps above are retained as task history.

### Extraction and manifest support

Extraction is the primary workflow. Use QR identification to resolve the saved layout, align scanned pages to its coordinates, and capture bubble responses and written answers from their designated regions. Layouts may contain different combinations and positions of question types; do not assume fixed section locations.

Extend the existing layout manifest with written-answer regions, including page, position, size and stable question ID, alongside existing bubble geometry. Supply correct answers, accepted alternatives and scoring criteria through that same manifest for the separate marking workflow.

For digital PDFs, extract answer text directly where a usable text layer exists. For scanned handwriting, use local handwriting recognition, including local AI where needed. Keep transcription independent of correct answers: recognition must record what the student wrote, not infer or correct it using the answer key.

Store extracted responses locally in SQLite with student and question IDs, the manifest/layout version used, source page and region references, access to the source crop, and review flags. Preserve student IDs as strings so leading zeros survive. Uncertain readings must remain distinguishable from blank or incorrect answers and be available for correction and approval.

### Separate marking and limited AI assistance

Marking consumes saved student responses and the manifest's answers and scoring rules. It must be possible to mark or re-mark without rescanning, while retaining the extracted response and recording the resulting score separately.

Use deterministic comparison first, including accepted alternatives and explicit scoring criteria. Do not treat different wording as automatically wrong when the manifest permits it. During marking, use AI only for ambiguity that deterministic comparison cannot resolve; it is not the default grader for every response. Cases that remain unclear require teacher review. Resolve uncertain transcription before awarding a final mark, rather than treating unreadable writing as an incorrect answer.

### Performance target and implementation order

The extraction benchmark target is **100 students × 10 short answers within ten minutes after upload** (1,000 answers). This target is **unverified** and must be measured on the intended hardware using representative handwriting. Include document processing, recognition and local persistence in extraction timing, and report cold-start/model-loading effects. Record marking time and manual review time separately; physical scanning and upload are outside this extraction target. Report accuracy and the proportion requiring review alongside speed.

Implement in this order:

1. Manifest-driven structured extraction: QR/layout resolution, page alignment, bubble responses and student identification.
2. Written-answer regions, text and handwriting capture, local persistence and review.
3. Separate deterministic marking using manifest answers and rules, with re-marking from saved responses.
4. Optional local AI assistance for unresolved marking ambiguity, followed by teacher review where necessary.

### Acceptance checks

- Mixed layouts: correctly associate MC, multiple-answer, true/false, matching and written responses with stable question IDs despite changing section positions or single-type papers.
- Input formats: support text PDFs and scanned documents; align scans before applying manifest coordinates and preserve source evidence for review.
- Student identity: resolve personalized QR identities and shared-sheet bubbled IDs, preserving a five-digit ID such as `00530` through storage and marking.
- Uncertain handwriting: flag ambiguous transcription for review, preserve the source crop, and neither substitute the expected answer nor silently assign an incorrect mark.
- Accepted variants: compare against supplied alternatives and scoring rules; send unresolved semantic ambiguity to optional AI assistance or teacher review.
- Re-marking: produce updated marking results from saved, reviewed responses without repeating scanning or recognition and without overwriting the extracted answers.
- Timing: measure extraction, marking and manual review separately on representative inputs; report hardware, model, cold/warm runs, accuracy and review rate against the unverified ten-minute extraction target.

This update documents the next stage only. Application code, migrations, service restarts and new APIs are outside this documentation change.
