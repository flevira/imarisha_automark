
---

# 2. `docs/codex_tasks.md`

This is the most useful file for Codex.

```markdown
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
