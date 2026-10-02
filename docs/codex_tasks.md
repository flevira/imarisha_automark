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

### Independently testable implementation milestones

**M1–M5 are implemented locally. M6's opt-in trial implementation is available, but real-handwriting qualification and model selection remain pending.** Successful lookup with a real teacher account and a full native desktop user run remain unverified. M7–M9 remain **planned**. Six existing legacy-test failures remain visible rather than declaring the whole application green. Independent means each has a bounded deliverable and can be tested with fixtures or a fake dependency; it does not mean every milestone can be integrated before its prerequisites exist. Complete and validate one milestone at a time. Do not replace the working legacy extractor while introducing the manifest-driven path.

Use small synthetic fixtures for automated tests. Keep real student scans, local databases, model files and virtual environments out of commits. Each milestone closes with its focused test results, documented limitations and a runnable demonstration of its deliverable in this existing document.

#### M1 — Contracts, fixtures and baseline measurements

- **Deliverable:** define the boundary between manifest lookup, page alignment, response extraction, local storage and marking. An extracted response identifies the assessment group, layout/version, student, question and matching sub-item where applicable, plus raw response, source region and review status. Keep scoring results separate.
- **Work:** create fixtures for personalized and shared sheets, mixed and single-type papers, multiple pages, text PDFs and scans. Include expected identities, selected bubbles and handwritten transcriptions. Record the current extractor's time and correctness on the same structured fixtures.
- **Dependencies:** none; use saved or synthetic manifests without a running Imarisha backend. Handwriting fixtures initially exercise contracts only, not a claim of recognition support.
- **Tests:** serialize and reload records without losing IDs, leading zeros, question/sub-item associations or source references. Run the existing extraction tests as a baseline. Verify recognition inputs do not contain correct answers or scoring criteria.
- **Exit:** reproducible fixtures, expected outputs and a baseline report exist; later milestones can consume these contracts without requiring their real dependencies.

#### M1 implementation record — 2026-10-02

Branch: `automark_scans_manifest` (local). No changes to the legacy extraction algorithm, application UI, running services or Imarisha backend were required.

Delivered:

- `core/contracts.py`: versioned extracted-response records, immutable source/layout references, JSON round trips and service protocols for manifest resolution, alignment, extraction, recognition, storage and marking. Source-document page and QR/layout page are separate. Matching sub-items and five-digit student strings survive serialization. Missing identity and uncertain/error readings require review reasons.
- Recognition receives only an answer crop, its source-region reference and language. The geometry projection allowlists fields from enriched manifests and strips teacher answers/rubrics. Crop selection and complete manifest validation belong to M2/M5; contracts cannot detect answer-key content incorrectly included inside an image.
- `tests/fixtures/manifest/`: synthetic V4 personalized/mixed and V5 shared/true-false manifests, separately stored student-response oracles, a draft written-response example and the compact measured baseline. These are test inputs, not backend migrations or production layouts.
- `tools/manifest_fixtures.py`: generates six PDFs (native-text and raster-scan versions of mixed, shared and legacy-control sheets), a hand-drawn synthetic `12` crop and a catalog. The mixed fixture deliberately spans two pages to test page identity; it is not a pagination-capacity test. Written-answer data is contract-only, not demonstrated handwriting recognition.
- `tools/extraction_baseline.py`: times the existing extractor, checks identities and response selections, and records missing/extra/duplicate results, review rates, errors and timeouts. Each fixture runs in a fresh process; `--repeats 2` adds a warm run. An exit code of zero means the measurement completed, not that extraction was accurate.
- `.gitignore` excludes virtual environments, local databases, bytecode and generated artifacts. `requirements-dev.txt` adds pytest to the existing dependencies. Generated PDF fixtures reuse PyMuPDF/OpenCV; no additional PDF runtime is required.

Reproduce from the repository root using the project's Python 3.11 environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_manifest_contracts.py tests/test_manifest_fixtures.py tests/test_extraction_baseline.py
python -m tools.manifest_fixtures --output artifacts/m1-fixtures
python -m tools.extraction_baseline --fixtures artifacts/m1-fixtures --output artifacts/m1-baseline --repeats 1 --timeout 180
python -m pytest -q
```

The extractor benchmark requires Tesseract on PATH. Its explicit offline profile uses Tesseract plus cached English EasyOCR on CPU, with Paddle disabled and model downloads disabled. Missing cached EasyOCR weights are reported in metadata; results from different profiles or installed models must not be compared as equivalent. Fixture generation and contract tests do not need OCR models, backend access or student data. Full per-case logs/results are written under ignored `artifacts/`; the concise baseline is `tests/fixtures/manifest/baseline-macos.json`.

Measured baseline: macOS 15.6 arm64, Python 3.11.15, Tesseract 5.5.2, PyMuPDF 1.28.2, OpenCV 5.0.0.93, EasyOCR 1.7.2 and Torch 2.14.1. Both English EasyOCR weight files were present. One cold-process run per case; OS caches were uncontrolled. Scan fixtures are clean synthetic rasterizations at 150 DPI. CPU model/RAM were not captured, so this is a development baseline, not a hardware capacity claim.

| Fixture | Pages | Extractor seconds | Exact responses / expected | Missing responses |
| --- | ---: | ---: | ---: | ---: |
| personalized-mixed-text | 2 | 21.85 | 0/5 | 5 |
| personalized-mixed-scan | 2 | 29.38 | 0/5 | 5 |
| shared-tf-text | 1 | 4.24 | 0/1 | 1 |
| shared-tf-scan | 1 | 7.73 | 0/1 | 1 |
| legacy-control-text | 1 | 13.01 | 1/1 | 0 |
| legacy-control-scan | 1 | 15.73 | 1/1 | 0 |

Times cover the extraction call including lazy OCR loading; process startup/import time is recorded separately in the JSON report. They exclude marking and SQLite persistence, which are not implemented in this milestone. No ten-minute throughput or handwriting-accuracy claim follows from these small fixtures. Zero rows on modern sheets is the measured gap that M2/M3 must close; review rate is unavailable when there are no detections, not misleadingly reported as zero.

Validation: **39 new tests pass**. They cover serialization and validation, leading zeros, source and sub-item preservation, answer-key isolation, fake service boundaries, native/scanned PDF structure, decoded QR payloads, bubble/grid pixels and benchmark failure accounting. PDF fixture pages were also visually checked with Poppler. The full suite is **94 passed, 6 failed**, with the same six failures observed in the pre-change legacy baseline (55 passed, 6 failed):

- `tests/test_find_option_centers.py`: `test_mc_ignores_labels_beyond_80px`, `test_match_ignores_labels_beyond_150px`, `test_match_does_not_infer_c_when_only_a_and_b_detected`.
- `tests/test_tf_single_answer_enforcement.py`: `test_tf_with_both_options_detected_is_forced_unanswered`, `test_match_with_multiple_options_detected_uses_strongest_and_flags_review`, `test_match_with_weak_winner_is_treated_as_unanswered_without_review`.

These failures remain visible and unmodified. Resolve or explicitly reconcile them in the structured-extraction milestone; do not hide them with skips or claim the existing extraction suite is fully passing. M1 is complete as a contracts/fixtures/baseline deliverable; it does not enable manifest extraction in the application.

#### M2 — QR identification, manifest resolution and page alignment

- **Deliverable:** resolve a QR's layout identifier and page number to an authorized saved manifest, cache it locally, and transform scanned pages into the manifest's coordinate system.
- **Work:** support current personalized V4 and shared V5 sheets; preserve already-supported legacy behavior. QR codes contain identifiers, not the full manifest or answers. Use corner markers and page dimensions rather than section-heading OCR. Load cached manifests offline; report unavailable layouts explicitly.
- **Dependencies:** M1. Test the resolver against a fake endpoint and local cache before checking the existing teacher-authorized backend endpoint.
- **Tests:** skew, rotation, scaling, reordered/missing/duplicate pages, unreadable QR, wrong layout version, missing markers, cache hit and cache miss. Measure transformed fixture coordinates against known locations; require error below half the smallest bubble radius. Reject unreliable alignment rather than guessing question mappings.
- **Exit:** page identity and coordinates are correct on the fixture set, with explicit review/error results for unresolved pages. No answer recognition or marking is needed to demonstrate this milestone.

#### M2 implementation record — 2026-10-02

- **Identity and validation:** `core/manifest_layout.py` validates personalized V4 and shared V5 QR identities and A4 manifests. Student identifiers retain leading zeros. Unsupported versions, malformed geometry and unresolved identities produce explicit errors; recognizable legacy sheets are routed back to the legacy path.
- **Resolution:** `services/manifest_resolver.py` calls the existing teacher-authorized `GET /api/teacher/pdf-exports/layouts/{type}/{id}/{layoutId}` endpoint. It supports canonical layouts returned for historical QR aliases. No new endpoint or backend change is required. Requests use a bearer token, HTTPS (except loopback development), bounded responses and no redirects.
- **Offline cache:** SQLite entries are scoped to backend and staff account, integrity checked and immutable. Tokens are never stored. Offline use trusts previously authorized entries; server-side permission revocation cannot be detected offline. `--refresh` requires online authorization and never falls back after access denial or network failure. Synthetic imports use a separate fixture scope. Account scope must come from the authenticated account when integrated into the application.
- **Alignment:** `core/page_alignment.py` decodes the QR and uses four solid corner markers to map image pixels to PDF points. Bounded QR crop retries handle small codes. Marker edges and QR location provide validation beyond fitting four marker centres. Rotations, skew, scaling and perspective fixtures pass independent bubble-coordinate checks below 3.545 pt (half the smallest bubble radius). Missing or ambiguous markers are rejected.
- **Preparation:** `services/page_preparation.py` returns page identity, geometry, source image, transform, review reasons and timing. Correct answers remain in the cached manifest but are excluded from the geometry passed downstream. Personalized missing/duplicate pages are reported and source order is preserved. Shared sheets report `student-identity-pending`; grouping them by student must wait for M3 grid reading. Document preparation is bounded to 200 pages and 20 million pixels per page.
- **Isolation:** `tools/prepare_manifest_pages.py` is a runnable CLI demonstration. The current UI and legacy extraction workflow are unchanged. Lazy imports in `core/__init__.py` prevent this preparation path from loading the legacy OCR stack. No AI, answer extraction or marking is invoked.

Reproduce the offline demonstration from the repository root:

```sh
.venv/bin/python -m tools.manifest_fixtures --output artifacts/m1-fixtures
.venv/bin/python -m tools.prepare_manifest_pages artifacts/m1-fixtures/personalized-mixed-scan.pdf --fixture-manifests tests/fixtures/manifest --output artifacts/m2-demo --previews
.venv/bin/python -m tools.prepare_manifest_pages artifacts/m1-fixtures/shared-tf-scan.pdf --fixture-manifests tests/fixtures/manifest --output artifacts/m2-shared-demo --previews
.venv/bin/python -m pytest -q tests/test_manifest_preparation.py tests/test_page_alignment.py tests/test_manifest_http.py
```

The demo writes `preparation.json` and optional normalized page previews under ignored `artifacts/`. For an authorized backend check, omit `--fixture-manifests`, provide `--backend` and `--account`, and supply a teacher token through `AUTOMARK_API_TOKEN`, never as a command-line argument. Without the token, previously cached layouts can be used offline. The HTTP integration tests need permission to bind a local test-server socket.

**Validation:** 64 new M2 tests pass (55 preparation/alignment tests and nine HTTP integration tests). Across the suite, 158 tests pass and the same six pre-existing legacy tests listed in M1 fail; none were skipped or modified to hide failures. Tests cover V4/V5, leading-zero identity, transformed geometry, reordered/missing/duplicate pages, invalid QR/version, unreliable markers, cache isolation/corruption/conflicts, authenticated HTTP lookup, offline reuse, access denial, malformed/oversized responses and redirect rejection. Normalized personalized and shared previews were visually checked. A read-only probe of the local Imarisha backend returned the expected HTTP 401 with an invalid token. Successful authorization against a real teacher account is still pending; the successful lookup tests use a local test server.

| Offline cached synthetic scan | Pages | Preparation time | Alignment validation residual |
| --- | ---: | ---: | ---: |
| Personalized mixed layout | 2 | 0.259 s | 0.227 pt per page |
| Shared true/false layout | 1 | 0.140 s | 0.227 pt |

These illustrative single-run times include file hashing, PDF rendering, QR reading, cache resolution, alignment and source-image encoding. They exclude interpreter startup, fixture seeding, output-file writing, answer extraction, marking and teacher review. They are not directly comparable to M1's complete legacy extraction timings and do not verify the ten-minute handwriting target. Support currently assumes the generator's V4/V5 A4 geometry, fixed QR area and planar pages; curved or heavily damaged paper remains unverified. Memory streaming and broader real-scan performance belong to M7. M3 is the next implementation milestone.

#### M3 — Manifest-driven bubble and student-number extraction

- **Deliverable:** extract MC, multiple-answer, true/false and matching responses from aligned regions, with personalized QR identity or the shared sheet's five-column student-number grid.
- **Work:** use each row's selection mode and option IDs. Preserve stable question IDs and displayed numbering; do not infer question type from its location. Keep blank, ambiguous and invalid multiple selections distinct.
- **Dependencies:** M1 and M2 for integration; unit tests receive aligned images and fixture manifests directly.
- **Tests:** every question type alone and in mixed positions; matching sub-items; blank, faint, erased and double marks; `00530`; incomplete ID columns; conflicting identity evidence; page boundaries and ignored options. Verify every clear synthetic mark exactly against expected output and flag ambiguous fixtures.
- **Exit:** structured responses and student identities pass fixture checks without question-ID OCR, an AI model, a marking service or a live backend.

#### M3 implementation record — 2026-10-02

- **Delivered:** `core/structured_extraction.py` implements the response-extractor contract for MC, MA, TF and matching sub-items. It reads only manifest-listed bubbles from normalized images, using question/option IDs and selection mode rather than headings, OCR, row position or correct answers. Native text PDFs and raster scans use the same pixel path.
- **Student identity:** personalized IDs come from the prepared QR identity; shared IDs come from the five digit columns, ordered by position and retained as strings (`00530`). Blank, faint or multiply marked columns require review, as does conflicting QR/grid identity. No partial ID is guessed.
- **Evidence and ambiguity:** each bubble retains mean inner-disk darkness, state, source region and a content-addressed PNG crop. Darkness is a measurement, not a calibrated probability. Printed outlines are excluded. Initial thresholds classify darkness at or below 0.08 as blank, at or above 0.60 as filled, and the interval as uncertain. Any uncertain mark flags the row even beside a solid selection. Multiple solid selections in a single-selection row remain visible as an uncertain response; multiple selections in MA rows are valid. Blank rows remain distinct from uncertain rows. Real scanner/pencil calibration is still required; nearly invisible erasures cannot be guaranteed detectable.
- **Document handling:** `services/structured_extraction.py` skips pages M2 marked for review, continues past invalid source images, and audits shared missing/duplicate pages after student identity becomes available. Duplicate shared responses are retained with review flags, so downstream consumers must honor `needs_review` and document issues. Different students with the same layout are not duplicates.
- **Runnable delivery:** the existing preparation CLI now accepts `--extract`, writes `extraction.json` and referenced bubble crops, and reports preparation and extraction timing separately. Marking and review durations are null because neither ran. Durable jobs, review UI and application integration remain M4; the currently running UI still uses the legacy extractor. This milestone adds no API, migration or marking workflow.

```sh
.venv/bin/python -m pytest -q tests/test_structured_extraction.py
.venv/bin/python -m tools.prepare_manifest_pages artifacts/m1-fixtures/personalized-mixed-scan.pdf --fixture-manifests tests/fixtures/manifest --output artifacts/m3-personalized --extract
.venv/bin/python -m tools.prepare_manifest_pages artifacts/m1-fixtures/shared-tf-scan.pdf --fixture-manifests tests/fixtures/manifest --output artifacts/m3-shared --extract
```

**Acceptance checks:** 45 focused tests cover independent expected answers for text PDFs and scans, all four types alone and mixed, matching sub-items, rotated/skewed/perspective/scaled scans, blank/faint/erased/double marks, leading zeros, incomplete or conflicting identity, shared duplicates and different students, missing pages, source evidence, changed row order/position, ignored unlisted marks and independence from answer keys. The broader run before the final added position test had 193 passing tests and the same six legacy failures; the nine unchanged M2 HTTP tests were excluded from this run (previously passed separately).

**Legacy reconciliation:** the three option-centre failures exercise heading/label-distance inference that M3 bypasses using explicit geometry. The three TF/matching failures exercise legacy answer-selection heuristics; M3 explicitly preserves double selections and ambiguity for review instead of collapsing them into a chosen winner or an unreviewed blank. The old path and its failing tests remain unchanged because M3 has not replaced the UI workflow. This is not a claim that those failures are fixed; legacy maintenance remains necessary until that path is retired.

| Synthetic scan demo | Preparation | Structured extraction | Combined compute time |
| --- | ---: | ---: | ---: |
| Personalized mixed, 2 pages / 5 response rows | 0.251 s | 0.046 s | 0.297 s |
| Shared TF, 1 page / 1 response row + 50 ID bubbles | 0.137 s | 0.040 s | 0.177 s |

These single-run measurements use local fixture manifests and exclude interpreter startup, fixture seeding and output-file writes. Extraction includes normalization, bubble measurements and crop encoding. All fixture answers match the independent student-response oracle, which deliberately differs from the teacher key. These small fixtures do not establish real-scan accuracy, class-scale throughput or the ten-minute handwriting target. No handwriting or marking was performed. Next: M4 durable extraction jobs and teacher review.

#### M4 — Durable extraction jobs and teacher review

- **Deliverable:** replace the placeholder response repository with working SQLite persistence and enable a review workflow showing source crops alongside extracted responses.
- **Work:** retain managed source evidence after the original input is moved. Save original extraction separately from teacher corrections and approval. Track per-page progress, resume interrupted jobs, and prevent duplicate response records on retry. Provide progress and cancellation without freezing the interface.
- **Dependencies:** M1; fake extracted responses allow development independently of M2/M3. Integrate M3 after both are verified.
- **Tests:** restart/resume, repeated input, partial failure, cancellation, source-file relocation, correction/approval and database reopen. Confirm raw responses remain unchanged and uncertain readings are never silently converted to blank or incorrect answers.
- **Exit:** a teacher can complete and recover a structured extraction/review job without any marking implementation. The saved responses and evidence remain available after restart.

#### M4 implementation record — 2026-10-02

- **Durable repository:** `database/structured_repository.py` stores source PDF bytes, page progress, original responses, bubble crops and source-page images in SQLite. `database/repository.py` now exposes this working repository instead of the unused placeholder. The structured database is separate from legacy extraction rows and is created locally on first use. No Imarisha backend schema changes or new APIs are involved. Repeated content within the same backend/account scope returns the same job. Completed pages are immutable on retry; response and evidence writes commit atomically per page.
- **Recoverable runner:** `services/extraction_jobs.py` integrates M2 and M3 page by page. Pending pages resume after interruption; failed pages require explicit retry. Completed pages are not duplicated or overwritten. A process lock prevents concurrent workers/review writes on the same job and is released by the OS after process death. Cancellation stops after the current page finishes; an in-flight manifest request can take up to its timeout. Failure on one page does not discard other pages. The managed source remains usable after the original upload moves.
- **Teacher review:** original readings remain unchanged. Approvals, corrections and exclusions are separate append-only decisions with reviewer, timestamp, note and optimistic revision checking. The latest decision and its history are visible alongside raw readings and crops. Corrections must use listed options and the row's selection mode; shared-sheet IDs require exactly five digits. Ambiguous responses cannot become blank without an explicit teacher decision and explanation. Missing/duplicate/unreadable-page issues need resolution notes. Completion requires all responses reviewed and all issues addressed; approved duplicate student/question responses and inconsistent student IDs on one page are rejected.
- **Desktop entry:** the home screen now offers **Structured extraction & saved review** alongside the legacy path. Select a PDF, configure backend/staff account and an optional transient teacher token, then create/resume a job. Cached layouts work offline. Saved jobs reopen after restart; review is paged to avoid loading every crop into the screen at once. Uncertain student grids show enlarged digit crops. Worker execution uses a background thread, with asynchronous progress updates and cancellation. Tokens are not persisted. Reviewer names are local audit labels, not server-verified signatures; local job storage is not a multi-user access-control system.
- **Scope and storage:** structured jobs currently accept unlocked PDFs, including native text and image-only PDFs, up to 200 pages / 200 MiB and the existing rendering pixel limit. Standalone photo uploads remain on the legacy path. Sources/crops are retained in the local database (POSIX file mode 0600); automatic deletion, storage quotas and large-batch memory tuning remain follow-up work. UI data defaults to the platform application-data directory beside the manifest cache. No app/service was restarted or activated during implementation.

Offline demonstration, using the M1-generated fixtures:

```sh
.venv/bin/python -m tools.extraction_jobs artifacts/m1-fixtures/shared-tf-scan.pdf --fixture-manifests tests/fixtures/manifest --database artifacts/m4-demo/jobs.db
.venv/bin/python -m pytest -q tests/test_extraction_jobs.py tests/test_structured_view.py
```

The CLI prints a saved job ID. Resume it with `--job <id>` instead of a PDF path, retaining the same database and resolver options; add `--retry-failed` for unsuccessful pages. Fixture caches/jobs are isolated from production account scopes. For backend layouts use `--backend`, `--account` and optional `AUTOMARK_API_TOKEN` instead of fixture imports. Teacher corrections and approvals are available through the repository and desktop review flow; no marking is performed.

**Validation:** 21 new M4 tests pass, covering database reopen, simulated interruption/resume, repeated upload, cancellation before and during work, source relocation, retained crops, partial failure/retry, transactional rollback, immutable raw responses, append-only correction history, stale-review conflicts, invalid corrections, explicit ambiguity/blank handling, scope/worker locks, duplicate exclusion, consistent page identity and completion gates. Desktop control tests exercise crop display, approval/completion, Flet control serialization and event-loop responsiveness/cancellation. The broader run reports 215 passing tests and the same six pre-existing legacy failures; the nine unchanged HTTP tests were excluded (previously passed under M2). Native window rendering, file-picker behavior, real teacher authorization and a real filled-paper review session are not yet verified.

### M4 timing clarification: text PDFs versus scanned PDFs

M3's reported 0.18–0.30 seconds used **synthetic scanned PDFs**, containing rasterized page images. M3 also tested native text PDFs and separately transformed scan images; those earlier headline timings did not represent those other formats. Neither milestone has benchmarked real handwriting recognition.

Fresh local M4 jobs were measured separately for each fixture format:

| Fixture format | Pages / response rows | Preparation | Bubble extraction | Total job including source retention and SQLite writes |
| --- | --- | ---: | ---: | ---: |
| Native text PDF, personalized | 2 / 5 | 0.244 s | 0.050 s | 0.308 s |
| Image-only scanned PDF, personalized | 2 / 5 | 0.269 s | 0.048 s | 0.326 s |
| Native text PDF, shared | 1 / 1 + student grid | 0.118 s | 0.027 s | 0.152 s |
| Image-only scanned PDF, shared | 1 / 1 + student grid | 0.132 s | 0.027 s | 0.166 s |

Both formats are rendered into pixels for bubble detection; native text extraction is not used by this OMR path. These are single-run synthetic examples with locally cached manifests. Total job time includes source ingestion and saving evidence/results, but excludes interpreter startup, repository/cache initialization, UI rendering, teacher review and marking. Detailed local measurements are in ignored `artifacts/m4-demo/timings.json`. Review and marking times remain unmeasured, not zero. The 100 students × 10 handwritten answers in ten minutes target remains unverified. Next: M5 written-answer regions and digital-text extraction.

#### M5 — Written-answer layout contract and digital-text extraction

- **Deliverable:** add written-answer regions to the existing manifest and generate corresponding spaces on the question/answer paper. Support direct extraction of text inside those regions where a usable answer text layer exists.
- **Work:** coordinate the Imarisha generator and AutoMark contract: page, region position/size, stable question ID and reading order for answers spanning regions. Keep existing bubble layouts readable. A reference answer alone is not automatically a complete scoring rubric.
- **Dependencies:** M1; M2/M4 for integrated extraction and review. This is the first milestone requiring generator changes in the Imarisha repositories. Use synthetic written-region manifests until that producer is ready.
- **Tests:** generated coordinates match rendered regions; mixed bubble/written papers; multiline and multipage answers; empty answer regions; a PDF with printed question text but image-only student handwriting. Do not mistake question text or a printed answer key for the student's answer. Route image-only answer regions to recognition.
- **Exit:** generator-to-reader fixtures demonstrate correct region mapping and typed-response capture without a handwriting model. Existing structured-sheet tests still pass. Do not apply the 50–60 bubble-responses-per-page target to written answers.

#### M5 implementation record — 2026-10-02

The user explicitly chose **M5 first, with handwritten images reviewed manually**, deferring automatic handwriting transcription to M6. Open-ended questions already existed in Imarisha; this milestone adds their generated answer spaces, saved coordinates, capture and review.

- **Generator:** the existing Imarisha web frontend now appends dedicated written-answer pages to the answer-sheet pack. Each open-ended question gets a numbered 507 × 100 pt capture region, with its printed label and border outside the region. Personalized pages hold four such areas; shared pages hold three beneath the five-digit student grid. Written-only papers do not start with an empty bubble page. These initial spaces target short definitions and explanations; they are not an essay layout or the 50–60 bubble-row density target. Teachers should direct students to these sheets rather than the untracked lines on a standalone question paper.
- **Versioned contract:** sheets containing written areas use **V6 personalized / V7 shared**. Bubble-only sheets remain V4/V5. Each page's `writtenRegions` contains `regionId`, `questionId`, displayed `number`, `kind: WRITTEN`, `order`, `x`, `y`, `width` and `height`. Reading order must be unique and consecutive per question across pages. The reader supports multiple regions for one question; the initial generator emits one region per question. Older readers reject the new versions rather than silently dropping written answers.
- **Backend:** `AnswerSheetLayoutValidator` checks written question IDs/numbers against the accepted paper, geometry, overlap and complete coverage. The current producer uses dedicated writing pages, which the backend enforces. Existing shared-manifest storage and endpoints are reused. Open-ended reference answers and answer details remain teacher-only manifest metadata, with `WRITTEN` response type and `MANUAL_REVIEW` status; they are not treated as a complete scoring rubric. No backend database migration is required.
- **Capture:** `core/written_extraction.py` retains region PNGs and source references. The durable job runner uses visible PDF text only when it is contained within the answer area, with no overlapping image, crossing text boundary or unexplained ink. Multiline and rotated native-PDF text are supported. Hidden PDF text is ignored. Printed prompts outside the box never become a student response. Clean empty regions are blank; nonblank image-only regions have no guessed transcription and are flagged `handwriting-recognition-pending`. This blank/ink heuristic is not handwriting recognition and still needs real-scan calibration.
- **Review and persistence:** written responses carry `part_number` for ordered continuations. The local review table gains a nullable text column through an idempotent local schema upgrade, preserving existing decisions. The UI shows large answer crops, a multiline transcription field and the original extraction alongside append-only teacher corrections. Continuation parts can be approved without being mistaken for duplicate question answers. No correct-answer metadata reaches the capture function; no marking or AI inference runs.
- **Paths and scope:** producer changes are local in `Offline/Imarisha_MacOS_local/frontend` and `backend`; reader changes are on AutoMark's `automark_scans_manifest` branch. The standalone M3 preparation CLI captures written images for review but has no live PDF text context; use the durable job runner for typed-answer extraction. No service restart, deployment, push or commit was performed.

Reproduce the synthetic mixed-paper checks:

```sh
.venv/bin/python -m tools.written_fixtures --output artifacts/m5-written
.venv/bin/python -m tools.extraction_jobs artifacts/m5-written/answers.pdf --fixture-manifests tests/fixtures/manifest --database artifacts/m5-written/jobs.db
.venv/bin/python -m pytest -q tests/test_written_extraction.py
```

**Validation:** 15 M5-focused AutoMark tests cover typed multiline/multipage responses, mixed bubble/written papers, clean blanks, image-only handwriting, shared student IDs, invalid/overlapping regions, hidden text, partial text, unexplained vector ink, PDF rotation, manual transcription, retained history, existing-database upgrade and review-screen controls. Frontend layout/render tests: 23 passed; TypeScript check passed. Backend validator, manifest and layout-service tests: 20 passed. Actual jsPDF-generated V6/V7 files passed AutoMark preparation and capture; a typed answer inserted into the generated V6 region was recovered exactly. Personalized/shared rendered writing pages were visually inspected. The broader AutoMark run before the final two focused tests had 228 passes and the same six legacy failures; unchanged HTTP tests were excluded as in M4.

**User design reference:** `open ended.pdf` was supplied to illustrate answer-space needs, not as a completed student script. Its six pages contain seven multiple-choice questions, three short open-ended questions and the marking scheme; most body content is rasterized. It contains no new assessment-layout answer sheets or student handwriting. A guard check produced no student responses from this package, so its teacher answers were not imported. The file and resulting local test database remain outside committed fixtures. Use its short definition/explanation style to guide the initial writing spaces; do not claim it validates handwritten recognition.

M5 is ready for a real generated-sheet capture/manual-review trial after the local generator/backend changes are activated through the user's normal workflow. Real teacher authorization, physical filled/scanned sheets, native desktop interaction and handwriting accuracy remain unverified. M6 remains a separate, unstarted stage.

#### M6 — Local handwriting recognition

- **Deliverable:** a replaceable local recognizer takes answer-region images and returns transcription, evidence and uncertainty for the extraction/review workflow. It receives no correct-answer key.
- **Work:** begin with words, numbers and one-to-two-sentence answers. Evaluate candidate models on representative target-language handwriting before choosing one. Record model/version and processing time; provision models for offline use. Longer essays, diagrams and mathematical working remain outside this first handwriting milestone.
- **Dependencies:** M1/M5 for the region contract and M4 for review; recognizer tests can run on fixed crops without the full scanning pipeline.
- **Tests:** clear and difficult writing, blank regions, crossings-out, low-quality scans, missing model, inference failure and offline operation. Use labeled real samples with suitable permission for evaluation; synthetic images alone cannot establish handwriting accuracy. Measure character/word error rates and review rate rather than trusting the model's own confidence claim.
- **Exit:** publish the recognition comparison and selected model's measured limitations; approve an accuracy/review-rate gate using the representative evaluation set before enabling automatic acceptance. Until then, handwriting output remains reviewable draft transcription.

#### M6 trial implementation record — 2026-10-02

The user authorized handwriting work for **English and Kiswahili**. This is an opt-in draft-transcription trial, not completion of M6's real-sample accuracy/model-selection exit gate.

- **Installed candidates:** local Ollama reports Gemma 3 `gemma3:4b-it-q4_K_M` with vision capability. The installed AfriqueQwen 3.5 build reports completion/thinking/tools but no vision; it is not used for image transcription. Qwen3 4B is also installed. The displayed Qwen3 8B/14B choices were not installed. No weights were downloaded, and Imarisha's selected tutor model was not changed. Actual capability discovery, not the UI's language/recommendation label, controls suitability for image input.
- **Recognizer boundary:** `core/handwriting.py` implements the existing crop/language-only recognizer protocol. The Ollama adapter sends only a retained answer image and transcription instructions, never question text, manifest answers, scoring rules or student identity. It checks installed GGUF weights and vision capability, restricts requests to a loopback origin, disables proxies/redirects and has no cloud fallback or auto-download. Model digest, engine, language and wall-clock recognition duration are retained. The implementation follows Ollama's [vision](https://github.com/ollama/ollama/blob/main/docs/capabilities/vision.mdx) and [structured-output](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx) APIs.
- **Conservative results:** every nonblank handwriting draft requires teacher review, regardless of model confidence or a plugin's claimed success. Illegible/crossed-out/low-quality flags are retained. Empty output for an ink region remains uncertain, not blank. Missing/nonvision models, unavailable services, malformed output and truncated generation produce explicit errors. A batch caches model/service failures after the first attempt to avoid a timeout for every remaining crop; a fresh resume can retry. The default request timeout is 90 seconds and generation is limited to 512 tokens for short answers.
- **No rescanning:** `services/handwriting_jobs.py` transcribes saved pending handwritten regions after page capture. Append-only `structured_transcriptions` records keep model attempts separate from immutable original responses and teacher decisions. Successful drafts for the same model name/language are reused on ordinary resume; failed attempts can retry. Approved/excluded teacher responses and completed jobs are untouched. Candidate comparisons or model requalification use the benchmark tool rather than silently replacing saved drafts. Cancellation occurs between model calls; the current HTTP call may run until completion/timeout.
- **Desktop/CLI:** the saved-job view adds an opt-in handwriting checkbox, local model field and English/Kiswahili/mixed-language selection. Draft text prepopulates the manual transcription field, with the crop, model and duration visible; approval still requires an explicit teacher decision. CLI flags are `--handwriting-model gemma3:4b-it-q4_K_M --language en-sw`. Preparation, pixel extraction and recognition timings are separate; marking and teacher-review times remain unmeasured. No services were restarted for this milestone.
- **Benchmark harness:** `tools/handwriting_benchmark.py` takes labeled crop files and computes character error rate, word error rate, exact matches, failures, review rate and per-crop timing. Labels are used only after inference and never enter recognition requests. It can compare Ollama with a cached EasyOCR English-character baseline (`download_enabled=False`). That baseline is not a qualified Kiswahili/handwriting recognizer. Reports include hardware/platform and model digests; local reports may contain student transcriptions and belong outside commits.

Two calls used the same synthetic hand-drawn `12` image, once with each language hint:

| Candidate | First call | Second call | Exact output | Review required |
| --- | ---: | ---: | --- | --- |
| Gemma 3 4B Q4_K_M | 10.407 s | 4.289 s | `12` in both | Both |
| Cached EasyOCR English baseline | 5.736 s | 0.074 s | `12` in both | Both |

These are **synthetic digit smoke tests, not English/Kiswahili handwriting accuracy results**. CER/WER were zero only for these two identical tiny samples. First-call timing includes initialization/verification when applicable; Ollama's prior warm state was uncontrolled. The large warm-call speed difference is an evaluation signal, not a model-selection decision. A full three-page M5 synthetic job also produced a separate Gemma draft `12`: preparation 1.188 s, pixel extraction 0.134 s, recognition 4.647 s. These observations do not validate the 1,000-answer/ten-minute target or real-word accuracy.

Reproduce an opt-in saved-job trial:

```sh
.venv/bin/python -m tools.extraction_jobs artifacts/m5-written/answers.pdf --fixture-manifests tests/fixtures/manifest --database artifacts/m6-job/jobs.db --handwriting-model gemma3:4b-it-q4_K_M --language en-sw
.venv/bin/python -m pytest -q tests/test_handwriting.py
.venv/bin/python -m tools.handwriting_benchmark /path/to/labeled-crops/dataset.json --output artifacts/m6-real/gemma.json
.venv/bin/python -m tools.handwriting_benchmark /path/to/labeled-crops/dataset.json --engine easyocr --output artifacts/m6-real/easyocr.json
```

Dataset shape: `{"qualification":"real-labeled","cases":[{"image":"english-01.png","text":"exact student transcription","language":"en","category":"clear"},{"image":"kiswahili-01.png","text":"exact student transcription","language":"sw","category":"difficult"}]}`. Image paths are relative to the dataset file. Labels must transcribe what was written, including mistakes; they are not the teacher's expected correct answers. Include different writers, blank regions, faint writing, crossings-out and actual scan conditions. Supply samples with suitable permission; do not commit real student images or labels.

**Validation:** 25 new focused tests pass, covering crop-only requests, both languages, local-model checks, no remote origins/redirects, blank bypass, ambiguity flags, empty/truncated/malformed output, service timeout/failure, cancellation/resume, saved-crop inference without rescanning, raw-response preservation, teacher-decision protection and metric calculations. The broader run reports 255 passes and the same six pre-existing legacy failures; unchanged HTTP tests were excluded as before. Live local Gemma inference and cached EasyOCR comparison ran successfully on synthetic digits. Real-word/sentence handwriting and actual disconnected-network operation still need representative validation.

**Remaining M6 gate:** obtain labeled real English and Kiswahili handwriting, compare candidate accuracy/review rate and speed on this Mac, choose a model or staged recognition strategy, and agree acceptance thresholds before enabling any automatic acceptance. All model transcriptions remain drafts until that work is complete. Longer essays, equations and diagrams remain outside the initial scope.

#### M7 — Extraction throughput qualification

- **Deliverable:** a repeatable benchmark and a responsive, bounded extraction pipeline; marking remains excluded from the extraction timing.
- **Work:** profile rendering, alignment, recognition and persistence separately before optimizing. Reuse page renders and loaded models, crop only required regions and tune bounded concurrency to the target hardware. Avoid reprocessing approved responses. Record peak memory and concurrent local-AI workload effects.
- **Dependencies:** M3/M4 for structured benchmarks and M6 for the handwritten target. Add timing instrumentation from M1; do not postpone measurement until this milestone.
- **Tests:** 100 students × 10 handwritten short answers, cold and warm starts, offline execution, mixed pages, cancellation/resume and repeated runs. Include preprocessing, model loading when needed, recognition and persistence after upload. Compare correctness and review rate with the same baseline set; do not obtain a faster result by dropping responses or relaxing the accuracy gate.
- **Exit:** report hardware, model, input/page counts, stage timings, total time, memory and accuracy. Claim the ten-minute target only if measured on the agreed representative hardware and inputs; otherwise record the bottleneck and revised options without claiming completion of that target.

#### M8 — Separate deterministic marking

- **Deliverable:** a marking operation consumes saved, reviewed responses plus manifest answers and explicit scoring rules, storing scores separately with the response revision and manifest version used.
- **Work:** handle accepted alternatives, single/multiple selections, true/false, matching and short text/numeric answers where explicit rules suffice. Do not invent partial-credit, unit-conversion or semantic-equivalence policies. Missing policies or unresolved transcription require review. Re-marking creates a new result without rescanning or replacing the original extraction.
- **Dependencies:** M1 contracts and M4 reviewed responses; can be tested with response fixtures independently of handwriting recognition. Keep extraction as the delivery priority.
- **Tests:** accepted/rejected variants, blanks versus unreadable text, exact and multiple selections, matching question totals, numeric/unit rules, absent scoring policy, response correction and repeated marking. Never award the full question allocation independently to each matching sub-item.
- **Exit:** deterministic scores match a teacher-authored expected-results set; ambiguous cases remain unresolved, and re-marking invokes neither scanning nor recognition. Report marking time separately.

#### M9 — Optional AI assistance for marking ambiguity

- **Deliverable:** an optional local-AI step for cases M8 explicitly cannot resolve, with a teacher-review path. Routine deterministic results bypass AI.
- **Work:** provide the reviewed student response, reference answers and rubric to the marking assistant. Store its proposed score and rubric evidence separately; preserve teacher decisions. Treat student text as data, never instructions. Extraction remains available when this feature is disabled or the model is unavailable.
- **Dependencies:** M8; a fake AI provider exercises workflow behavior without model installation.
- **Tests:** valid paraphrases, missing rubric, contradictory or malformed output, out-of-range scores, model timeout, unavailable model and instructions embedded in student answers. Compare real-model decisions with teacher judgments before accepting them automatically.
- **Exit:** AI suggestions cannot overwrite transcription or silently finalize unresolved cases; teacher approval and disagreement are recorded. AI time and teacher review time are reported separately from extraction.

### Milestone delivery boundaries

- M1–M4 deliver a usable structured extraction and review path before handwriting or marking is required.
- M5–M7 extend and qualify written-answer extraction; model choice and the ten-minute target remain evidence-based decisions.
- M8–M9 deliver marking as separate operations that consume saved responses, with AI optional.
- Uploading marked attempts/scores to Imarisha, desktop installers and release/deployment work remain separate follow-up tasks. These milestones do not authorize those changes.

### Acceptance checks

- Mixed layouts: correctly associate MC, multiple-answer, true/false, matching and written responses with stable question IDs despite changing section positions or single-type papers.
- Input formats: support text PDFs and scanned documents; align scans before applying manifest coordinates and preserve source evidence for review.
- Student identity: resolve personalized QR identities and shared-sheet bubbled IDs, preserving a five-digit ID such as `00530` through storage and marking.
- Uncertain handwriting: flag ambiguous transcription for review, preserve the source crop, and neither substitute the expected answer nor silently assign an incorrect mark.
- Accepted variants: compare against supplied alternatives and scoring rules; send unresolved semantic ambiguity to optional AI assistance or teacher review.
- Re-marking: produce updated marking results from saved, reviewed responses without repeating scanning or recognition and without overwriting the extracted answers.
- Timing: measure extraction, marking and manual review separately on representative inputs; report hardware, model, cold/warm runs, accuracy and review rate against the unverified ten-minute extraction target.

This update documents the next stage only. Application code, migrations, service restarts and new APIs are outside this documentation change.
