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

**M1–M5 are implemented locally. M6's opt-in trial implementation is available, but real-handwriting qualification and model selection remain pending. M8's separate deterministic marking workflow is implemented locally.** The user requested proceeding to marking independently of M7 throughput qualification, which remains pending. Successful lookup with a real teacher account and a full native desktop user run remain unverified. M9 has a local assistance trial; real-answer accuracy and review-rate qualification remain pending. Six existing legacy-test failures remain visible rather than declaring the whole application green. Independent means each has a bounded deliverable and can be tested with fixtures or a fake dependency; it does not mean every milestone can be integrated before its prerequisites exist. Do not replace the working legacy extractor while introducing the manifest-driven path.

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

##### M8 implementation record (2026-10-02)

- **Boundary:** `core/marking.py` scores whole questions from saved approved responses and the existing layout's answer fields. `services/marking_jobs.py` resolves that layout in the original account scope, verifies its identity/geometry, and groups matching items and written continuations. It never reads a PDF, crop, handwriting draft or recognizer. Extraction review must be complete first.
- **Scoring policy:** existing backend manifests say `ASSESSMENT_POLICY_REQUIRED`. This placeholder does not authorize a scoring rule. Desktop marking defaults to manifest rules; the teacher can explicitly choose all-or-nothing per structured question, including the entire matching question. Multiple-answer selections must equal the complete correct set. Matching equivalent option IDs are accepted, but the question's maximum is awarded once. No partial-credit policy is inferred. Missing/invalid answers, conflicting marks, unsupported policies and absent/excluded question parts remain unresolved, never implicit zeros.
- **Written answers:** the teacher may explicitly enable case-insensitive, whitespace-normalized comparison to reference answers/accepted alternatives. Unlisted wording remains unresolved unless an explicit rule declares the list exhaustive (`rejectUnlisted: true`). NFC Unicode normalization is used; spelling, punctuation and meaning are not guessed. Numeric comparison requires explicit absolute tolerance and a literal unit. No unit conversion or semantic/AI grading occurs. Blank approved text scores zero only after a valid key and rule have been established; unapproved handwriting cannot be marked. Continuation parts are combined in manifest order and receive one question allocation.
- **Persistence:** the local jobs database adds only `structured_marking_runs`, through idempotent initialization. Each append-only run stores the original response snapshot, review sequences, raw revisions, exact manifest documents and SHA-256 digests, layout versions, teacher/rules, question results and student totals. Re-marking adds a run without modifying extraction or earlier scores. Corrections invalidate the displayed previous result until review is completed and marking rerun. Students with unresolved questions have no final total; the resolved subtotal is explicitly labeled. Excluded-only students have no score. Acknowledged document issues remain in the run audit.
- **Desktop:** complete review, select an explicit policy if needed, then use **Mark / re-mark reviewed answers**. The UI shows question scores/review reasons, totals, history IDs and marking duration. Written normalization is opt-in. Detailed per-question numeric or text rules can be supplied in a manifest's `questions[].scoringRules`, or with the CLI's teacher policy file. This stage does not add a manual-score editor, upload results to Imarisha, modify backend records, restart services or activate the desktop app.
- **CLI:** `python -m tools.marking_jobs --job JOB --reviewer 'Teacher' --backend http://127.0.0.1:18080 --account STAFF --structured-policy all-or-nothing`. Use the same database/account as extraction. Previously authorized cached layouts work offline; `AUTOMARK_API_TOKEN` is needed only for uncached lookup. `--fixture-manifests tests/fixtures/manifest` isolates synthetic runs. Omit `--structured-policy` to require manifest rules. `--policy rules.json` supplies explicit teacher rules and is mutually exclusive with the preset.
- **Rule contract:** teacher policy JSON is `{"version":1,"structured":"all-or-nothing","written":{"QUESTION_ID":{"version":1,"mode":"numeric","tolerance":"0.1","unit":"cm"}}}`. Text rules use `{"version":1,"mode":"text-exact","caseSensitive":false,"collapseWhitespace":true}`; `writtenDefault` optionally applies a text rule to written questions without a per-question override. Per-question manifest rules take precedence over teacher presets. Numeric reference answers must include the specified unit; an unrecognized numeric format/unit requires review.
- **Validation:** 22 new marking tests cover explicit keys and expected scores, set comparison, matching alternatives/totals, absent policies, invalid keys, blanks versus uncertainty, Unicode/text alternatives, numeric tolerance/units, continuation ordering, approved handwritten transcription, excluded parts, scope/identity rejection, immutable raw extraction, response corrections, persisted run history and the desktop action. Integration tests make extraction/recognition/source/crop access fail if invoked during marking. These are synthetic developer-authored expectations; real teacher qualification and native desktop end-to-end use remain pending.
- **Timing:** `marking_seconds` includes loading saved responses, manifest resolution/verification, comparison and constructing the audit snapshot. It excludes the final SQLite run insert, UI display and human review; `review_seconds` remains null. Extraction and recognition times are not added to it. This is not the M7 throughput benchmark and makes no ten-minute handwriting claim.

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

### Temporary local development manifest lookup (2026-10-02)

The user requested token-free testing. In the desktop workflow, an empty token with a loopback backend URL now fetches `/api/development/automark/layouts/{type}/{id}/{layoutId}`. Staff account ID remains the local cache/job namespace, so retain the same value when retrying saved jobs. Remote cached layouts still work offline; remote uncached lookup requires a token. Providing a token uses the existing teacher endpoint.

The backend controller exists only under the `development` profile and checks loopback binding, a loopback caller, no proxy forwarding headers and the exact `imarisha_development` datasource before reading a manifest. It provides no write endpoint and does not weaken the authenticated teacher route. This temporary path can read development answer keys without staff authentication; it must not be used as a production sign-in mechanism. Backend activation and native desktop retry still require runtime validation. Failed-page metadata now retains the preparation error message, and the review view directs users to retry instead of acknowledging an unextracted page as resolved.

Testing follow-up: removed the token and staff-account inputs from the structured desktop view. It now uses a fixed `local-development-testing` cache namespace with the loopback-only development endpoint. Existing account-scoped jobs are retained; users must create a new job from the same PDF to use this testing namespace. Reviewer name remains an audit label. Backend URL remains because the QR still contains a manifest reference, not its contents. The user's intended full-manifest-in-QR design has not yet been implemented; this temporary testing change does not make initial lookup independent of the backend.

### Sample-based bulk approval (2026-10-02)

At the user's request, response review comments are now optional, including individual corrections and exclusions. Reviewer identity and append-only history remain. The desktop **Approve all saved responses** action approves previously unreviewed saved values across every page, including available handwriting drafts; it preserves prior teacher decisions. Unsaved on-screen edits must be saved individually first. Invalid identity/selections and written regions without transcription remain unresolved, rather than being converted to blanks. The action reports approved, retained and skipped counts/pages. Completing review still checks document issues and duplicate/inconsistent identities before enabling separate marking. Document-issue resolution notes are unchanged. This supersedes earlier notes requiring a comment for every uncertain response.

### Manifest scoring and local AI assistance (2026-10-02)

The user authorized using existing manifest answers directly, partial marking and accepting equivalent work, with local AI resolving marking ambiguity. This supersedes M8's earlier requirement to select a rule whenever the backend carries `ASSESSMENT_POLICY_REQUIRED`.

- The absent/legacy placeholder policy now falls back to the agreed scoring rules. Explicit question/manifest rules and teacher overrides remain supported. Missing keys, invalid identities, excluded/missing parts and unreadable text are not resolved by inventing answers.
- MC/TF receive full marks for an accepted answer, otherwise zero. Matching uses `question marks × correct items / total items`; the question allocation is never repeated per item. Multiple-answer credit is `question marks × max(0, correct selections − wrong selections) / correct options`. This prevents selecting all options from automatically earning full credit. Partial scores round half-up to two decimals and never exceed the question allocation. The UI describes these rules and retains an all-or-nothing override.
- Written responses first compare approved text against references and accepted alternatives using case-insensitive NFC/whitespace normalization. Blank approved responses score zero. Other wording is eligible for local AI; extraction, transcription and structured marking bypass this AI operation.
- `core/ai_marking.py` provides text-only Ollama assistance using installed local GGUF weights, loopback-only transport, no proxy/redirect/cloud fallback or model downloads, bounded input/output and timeouts. The marking model is independent of the handwriting model. The desktop defaults to the installed Qwen3 4B text candidate; the AI checkbox can disable assistance. CLI opt-in is `--ai-model MODEL`.
- Equivalent meaning can receive full credit. Partial written credit requires supplied `answerDetail` marking guidance. Model uncertainty, missing guidance for a partly correct answer, incomplete/invalid output, impossible marks and missing quoted evidence remain review cases. There is no claim that schema validation proves the model's judgment correct. The selected local model still needs teacher qualification, including English/Kiswahili variants and adversarial student text.
- Results keep deterministic/AI provenance, proposal, quoted student evidence, explanation, model/digest, prompt version and timing in the existing append-only marking-run audit. The UI displays AI explanations. Same-question, same-rule, identical approved text reuses the AI result within a run. Total marking time includes AI; `ai_seconds` reports it separately. Raw extraction, prior approvals and earlier marking runs remain intact.
- Validation: 99 relevant tests passed across marking, AI contracts, desktop review, saved jobs, written capture and handwriting. Tests cover fractional matching marks, wrong-choice deductions, placeholder fallback, guidance requirements, out-of-range/model output rejection, local-only weights, fail-fast unavailability, structured bypass and persisted AI scoring versus re-marking with AI disabled. Live synthetic checks exposed Gemma's overly literal interpretation and Qwen explanation truncation; output bounds were tightened. These checks do not establish real-exam accuracy or that AI will resolve most review cases. No real student answers were marked during implementation, no services were restarted and no remote changes were made.

Final synthetic smoke check after bounding explanation length: installed `qwen3:4b-instruct-2507-q4_K_M` accepted an equivalent plant-food/sunlight paraphrase for 2/2 in 2.575 seconds, with quoted evidence and rationale. This is a single synthetic example, not a handwriting, class-scale or real grading accuracy benchmark. Final focused marking/AI/UI checks were rerun after the prompt/output change.

### Marking output report (2026-10-02)

Added an explicit **Export results report and audit** action to saved marking runs.
It writes a printable HTML report and the complete saved run as JSON to a unique
folder, without rescanning, re-marking, uploading, or overwriting earlier exports.
The report includes student IDs, resolved subtotals, question marks, reviewed
responses, unresolved reasons, AI explanations, and marking/AI durations. Pending
scores stay pending rather than becoming zero; exports of changed response
versions are prominently labelled outdated. Excluded responses are disclosed.
The JSON is a local audit export, not an Imarisha import API contract.

Validation: export preservation/escaping test and existing structured view tests:
7 passed. Generated the report for saved run 3. Written marking limitations remain;
this output feature does not resolve or alter those scores.

### Written marking validation and context (2026-10-02)

- Evidence comparison normalizes Unicode NFC and whitespace only. A line break
  no longer rejects a genuine quotation; different words still fail validation.
- Backend-generated manifests now retain `questionText` from the accepted paper
  snapshot and pass through explicit `scoringCriteria` when available. Existing
  manifests and historical runs are not rewritten. No service was restarted.
- The local review screen has a collapsed **Written question wording and
  partial-credit criteria** editor. Supply the original wording and one criterion
  per line as `marks | expected idea`. These apply across students for the layout,
  are preserved in the run's policy snapshot, and do not change the backend.
- Criteria use `{id, description, marks}`; positive allocations must sum to the
  question maximum. AI must return each criterion once, evidence for each met
  criterion, and a total equal to the deterministic sum of awarded points.
  General answer explanations are background, not extra required criteria.
- Unresolved written answers receive a separate token-overlap F1 similarity
  suggestion: `maximumMarks * 2 * sharedTokenCount / combinedTokenCount`, using
  the best reference/accepted alternative. This is an uncalibrated lexical
  estimate, not a statistical probability of correctness. It is labelled in
  the UI/report and excluded from final totals; negation, synonyms and translated
  equivalents make automatic use unsafe without validation.
- `python -m tools.evaluate_written_marking examples.json output.json` tests local
  AI without altering saved jobs. Each example contains `question`, `studentAnswer`
  and optionally teacher-established `expectedScore`. Unlabelled cases measure
  resolution/latency only, not accuracy. Output retains model decisions and timing.
- Live trial: five saved answers from student 00535, using the revised Qwen 4B
  prompt with paraphrase examples, took 13.40 seconds; 0/5 resolved. These older
  manifests lack question wording/criteria. No teacher scores have been supplied;
  real grading accuracy remains unverified. Prompt changes alone have not solved
  the model's excessive deferral. No saved scores were modified by these trials.

### Offline manifest handoff (2026-10-02)

Supersedes the backend-fetch and token-free development workflow described above.
Imarisha Print & PDF now provides `automark-manifest.json` as a separate download
and includes it in `print-pack.zip`. It wraps the saved, answer-enriched layout in
`{format: "imarisha-automark-manifest", packageVersion: 1, manifest: ...}`.
One file serves every student using that layout. Keep it with the teacher because
it includes answers. Student QR contents remain layout/page/student references.

AutoMark's **Upload manifest** action imports this file into its local cache;
manual paths and CLI `--manifest` are also supported. Backend URL/token/account
inputs and the HTTP manifest client have been removed. The temporary unauthenticated
development endpoint was removed. The authenticated backend layout storage remains
part of Imarisha's print generation, not an AutoMark runtime dependency.
Imports validate geometry and package version, enforce a 1 MB limit, and reject
changes to an already imported immutable identity. A missing layout asks for an
import, never a backend connection. Existing review jobs retain their stored scope
and decisions while reading manifests from the new local import cache.

Validation: 137 targeted AutoMark tests, 38 frontend PDF tests, and backend layout,
assignment, and topic-storage migration tests passed. This includes a file-import
through PDF-extraction test with no backend. Service activation is separate;
no restart or deployment was performed. AI grading accuracy limitations above remain.

### OCR-first written extraction and image inputs (2026-10-03)

Written regions now use this order: visible PDF text when usable, otherwise local
Tesseract OCR on the saved crop; optional local AI only for missing/uncertain OCR.
OCR runs by default with no AI checkbox required, including when resuming an older
capture-only job. Approved responses and successful saved results are retained.
OCR records include text, engine, language, elapsed seconds, confidence and review
flags. Confidence is a routing heuristic, not a calibrated correctness score:
mean token confidence >=80 and no token below45 qualifies as confident only when
all requested language packs are present. Empty/error/low-confidence outputs remain
reviewable; failed AI retains usable earlier OCR text. No reference answers enter
transcription requests. Review approval is still explicit.

The structured source reader and desktop picker now accept PDF, PNG, JPEG, TIFF
(including multiple frames), BMP and WebP. Images follow the same QR/manifest,
alignment, bubble and written-region pipeline. Limits: 200 pages/frames, 200 MiB
source file, 40 megapixels per input image. Original input bytes are retained.

Local test on SIM_3_Completed_Student_Answer_Sheet.pdf extracted all five written
regions with ordinary OCR and no AI: approximately2.1 seconds overall,0.8 seconds
OCR. Tesseract reported92–96% mean token confidence; this is not a handwritten
accuracy benchmark. This Mac currently has English OCR data but no Kiswahili pack;
requested mixed-language OCR explicitly flags that absence. Tesseract executable
and corresponding language packs are local runtime requirements (no auto-download).

Validation:88 targeted extraction, handwriting, review and marking tests passed;
4 OCR/image tests passed, including confident-OCR bypass of AI, resume reuse,
PNG extraction and multi-frame TIFF input. No service restart, remote push or
production-job reprocessing was performed for this change.

### Written marking review and side-by-side output (2026-10-03)

- Results and HTML exports show the manifest reference answer beside the reviewed extracted answer. Older saved runs retrieve reference answers from their saved manifest snapshot.
- The default written comparison tolerates explicit multi-line numbered lists and terminal full stops, alongside case and whitespace normalization. It preserves meaningful numbers and negation; equivalent wording beyond these formatting changes still needs an accepted alternative, optional AI, or teacher decision.
- Unresolved results offer a teacher mark and optional note. When explicit scoring criteria exist, teachers can select individual marking points and sum their weights before saving. Missing criteria are not invented from lexical similarity.
- Saving a teacher mark creates a new immutable marking-run version and recalculates totals. Invalid/out-of-range marks, stale results and incomplete extraction reviews are rejected. Original runs remain available as history.
- Teacher decisions belong to that saved run. Starting a fresh re-mark computes new results; it does not automatically carry teacher overrides forward. Export the latest run after reviewing marks.
- Regression checks cover list formatting, preserved negation/decimals, mark bounds and version conflicts, total recalculation, and adjacent escaped reference/extracted report cells. Statistical similarity remains advisory and does not enter totals automatically.

### Combined extraction and marking report (2026-10-03)

- One `automark-results-<job-key>.html` per job, exported to Downloads by default using **Export extraction and marking report**. Available before marking and before review is complete.
- Export again after marking or teacher corrections to atomically replace the same HTML file. No extra JSON file is created by this action; saved marking versions remain in the internal database.
- Includes extracted/reviewed responses, reference answers (option letters for MC/MA), review status, marks, explanations and student totals. Matching items and written continuations are grouped under their question so its score is shown once.
- Unmarked results say “Not marked yet”; unresolved marks stay pending. Changed responses make previous scores outdated and hide their totals until re-marking.
- Verified extraction-only UI export, same-file replacement after marking, and outdated score handling after a response correction. Existing older export folders are not deleted.

### Requested concept count for written partial credit (2026-10-03)

- Explicit enumerative question wording such as “State two negative behaviours” now produces two distinct-concept credit slots when no explicit rubric exists. At five marks, one slot earns 2.5 and both earn five. Reference examples are an acceptable pool, not a requirement to repeat every example.
- Existing scoring criteria take precedence. Missing counts, multiple count requests and “at least/at most” wording do not generate guidance. Initial parsing supports common English enumerative commands and a limited set of Kiswahili forms; unsupported wording retains the existing review workflow.
- Local AI receives the slots and must identify distinct valid ideas with student evidence. Duplicate evidence is rejected; synonymous repetitions must not earn separate credit. Semantic distinctness remains a model judgment and needs accuracy testing with teacher-labelled examples.
- Saved runs record derived criteria/count/source and AI prompt version v3. The UI displays the requested count. Count derivation does not alter extraction, manifest files or existing marks; re-marking applies it.
- Mocked model regression checks cover two valid concepts/full marks, one/half marks, duplicate evidence, explicit rubric precedence and ambiguous count fallback. Actual model accuracy has not been measured for this change.

- Evidence validation correction: a combined AI summary quote need not be contiguous when each credited criterion has its own verified student-text quote. This resolves numbered-list separators rejecting otherwise valid evidence (observed Q16, run 13). Invented criterion quotes still fail; zero-credit decisions still require valid overall evidence. Regression coverage includes all three cases; validation version recorded as written-marking-v4. Existing saved scores are unchanged until re-marking.

### Multiple PDF/image inputs (2026-10-04)

- Structured extraction accepts multiple PDFs and supported images in one selection (macOS native picker and standard Flet picker). Manual entry accepts one path per line.
- A batch processes files sequentially, keeping a separate saved job/report per input. Duplicate paths are removed; existing content resumes through the repository's existing source identity logic.
- A failed file is reported while subsequent files continue. Cancellation stops after the current page/recognition request and skips remaining inputs. Existing single-job Resume/Retry remains available.
- Regression checks cover mixed PDF/PNG queueing, missing input continuation, duplicate selection, cancellation and native picker path handling. This does not combine separate image files into one multipage student script.

### Repeated student/exam attempts (2026-10-04)

- Different uploaded files/jobs may represent different attempts for the same student and assessment. There is no cross-job student/exam uniqueness restriction.
- The job ID is exposed as Attempt ID in the review UI, combined HTML and newly saved marking runs. Each attempt keeps independent responses, reviews, scores and report file.
- Regression check marks two distinct uploads with the same student and exam successfully. Byte-identical uploads still resume their existing job. Repeated copies of the same layout page inside one job remain ambiguous duplicates; this change does not infer attempt boundaries within a combined PDF.

### Batch review, marking and combined report (2026-10-04)

- Saved scripts now have selection checkboxes and Select all scripts. Newly uploaded jobs are selected automatically for the current view; reopening the app requires explicit selection.
- Approve selected scripts uses saved extraction/transcription values and retains previous decisions. Missing/invalid responses stay pending. Complete selected reviews validates every job separately.
- Mark selected scripts processes each reviewed attempt independently, honours the current marking policy/local AI choice and question guidance, and reports blocked jobs without aborting the remainder.
- Marking and Export combined report write one stable HTML file in Downloads (or Batch results folder) for the selected job set, with navigation and a separate section per attempt. Re-exporting the same selection replaces that report. Pending or stale marks remain labelled.
- UI regression covers blocked unreviewed jobs, multi-script approval/completion/marking, report inclusion and same-file re-export. No existing user approvals or scores are changed merely by installing this change.

### Answer-detail context and incorrect written answers (2026-10-04)

- Confirmed the SIM 3 manifest already supplies answerDetail for all written questions and the local AI receives it as explanatoryNotes. Prompt v5 explicitly uses those notes to clarify intended concepts without adding requirements.
- Clearly unrelated answers, contradictions and misconceptions should receive zero with quoted student evidence, rather than unresolved. Semantic equivalents remain acceptable; shared words alone are insufficient. Genuine ambiguity remains for teacher review.
- Marking snapshots now retain question wording and answer details; combined/single job reports show both alongside reference and extracted responses.
- Regression verifies context delivery and validated zero for a clearly incorrect definition. This tests integration/validation, not actual-model grading accuracy; re-marking is needed to apply updated guidance to saved responses.

### Preserve credit for incomplete but valid concept answers (2026-10-04)

- Observed Q20 model failure: explanation acknowledged one correct tool, but both concept slots were false. Prompt v6 and derived slot descriptions now explicitly assess each slot independently; absence of later concepts must not cancel earlier credit.
- For derived count-based criteria, application code calculates score/decision from validated credited slots, overriding an inconsistent model aggregate. The original proposal remains in the audit; UI/report show the calculated concept credit.
- Live local Qwen3 4B check: “A bucket of dry sand.” for the two-tool, five-mark question scored 2.50/5 with one credited concept (about 7.1 seconds). This is one verified example, not a broad accuracy benchmark. Saved production marks were not changed by the test.

### Named concepts, source-segment evidence and written benchmark (2026-10-04)

- Replaced anonymous concept slots and AI-proposed numeric scores with `written-marking-v7`: named requirements, discrete semantic judgments, application-calculated scores and explanations. Model output no longer contains a free-form score or overall explanation that can contradict its concept decisions.
- Explicit weighted `scoringCriteria` take precedence; optional `acceptedConcepts` (stable IDs, descriptions, accepted alternatives) plus `requiredConceptCount` implement any-N distinct credit. For legacy counted list questions, a conservative split of a single reference list supplies named ideas; ambiguous lists require teacher guidance. No required count is inferred from reference length alone. Credit is capped and rounding happens after proportional calculation.
- Question wording and reference/accepted answers remain grading inputs. `answerDetail` is retained as background in audit/report, excluded from the scoring request to prevent incidental consequences/examples becoming requirements. Teachers can promote actual requirements into explicit criteria.
- Evidence uses IDs pointing to saved transcription segments. Unknown IDs, missing evidence, duplicate concept IDs, extra score fields and incomplete model output are rejected. Inferred semantic matches receive a separate focused verification; disputed matches stay uncertain rather than earning unsupported credit.
- Definitions and other uncounted questions support weighted teacher criteria. Without such criteria, complete semantic equivalents may earn full credit, clearly incorrect responses zero, and partly correct responses require a rubric rather than being collapsed to zero.
- The existing guidance editor exposes weighted criteria, any-N accepted concepts/variants and count; edits are shared across selected jobs with the same layout. Effective requirements and AI evidence/calculations are saved in marking runs. Production extraction/reviews/scores are not changed by implementation or benchmark runs.
- Prepared 50 private benchmark cases and proposed teacher-review marks in `artifacts/written-marking-review/` (ignored local artifacts). These are NOT teacher-approved. Q17's primary-contribution wording versus multiple reference activities, and Q18's bare bribery example, need explicit teacher judgement.
- Evaluator separates teacher-labelled false-zero, over-credit, partial-credit error and unresolved metrics from unapproved drafts. No teacher labels means accuracy counts are unavailable, not zero errors. Use `python -m tools.evaluate_written_marking cases.json output.json --draft-rubrics` only for evaluating proposed rubrics; it does not approve or apply them.

Validation for v7:
- 73 targeted automated checks passed; 19 UI/report/benchmark checks passed again after the guidance-preview metadata fix.
- Live 50-case local Qwen3 4B trial with proposed rubrics and match verification: 46 resolved, 4 pending, 199.5 seconds; 44/50 agreed with unapproved proposed marks. Remaining differences include Q17 rubric overlap and treatment of a bare bribery definition. No teacher-labelled accuracy claim is made.
- Six representative cases using original/default manifest rules (no proposed rubrics): valid Q17/Q18 paraphrases earned 5; one sand bucket earned 2.50; broom/bell earned 0; two cases remained for review. Total about 19 seconds.
- Review artifact: `artifacts/written-marking-review/teacher-review.html`; raw private cases and local-run audits are beside it. Provisional expected scores remain separate from `expectedScore` and `teacherApproved` is false. Restart and re-mark are required to change user-visible production results; neither was performed automatically.

### Remove legacy desktop extraction (2026-10-04)

- Desktop launch now opens manifest-based extraction/review directly. Removed the legacy home/review screens, unused table control and Back to legacy extraction navigation.
- Startup no longer initializes the old extraction database. Existing legacy data is retained and remains covered by explicit Reset all AutoMark data; removing the UI does not delete user records.
- Sheets without a manifest reference now explain that matching layout-QR sheets and manifests are needed, instead of pointing to a removed screen. Older extraction service/CLI utilities remain for compatibility and regression tests, not as desktop entry points.


### Use manifest answer details for equivalent wording (2026-10-04)

- `written-marking-v8` passes existing manifest `answerDetail` to both initial semantic grading and focused match verification. Previously v7 retained it only in audit/report. The verification pass also receives question wording so concise responses can rely on the setting/purpose already stated in the question.
- Details clarify equivalent wording of existing reference concepts; they do not create extra requirements, accepted answers, weights or concepts. Explicit accepted alternatives and criteria retain precedence. No generated synonym guesses are written into imported immutable manifests. Existing concept-level `acceptedAlternatives` continue to reach both passes.
- Interpretation context is recorded in the marking audit. The importer, original manifest files, OCR/transcription, saved approvals and production marks are unchanged. This changes interpretation of existing manifest data, not the frontend manifest generator or schema. Missing details remain supported.
- 54 targeted automated tests passed, including manifest detail propagation through marking, both AI requests, fixed concept counts/partial scores, absent details and ambiguous-rubric blocking. `git diff --check` passed.
- Six-case live local Qwen3 trial with original manifest rules: script 02 Q16 now 5/5; script 06 Q20 remains 2.5/5; script 08 Q20 remains 0/5. Script 02 Q20 remains pending, script 03 Q19 remains under-credited at 2.5/5, and script 05 Q19 incorrectly receives 0 despite one valid theft concept. Context delivery alone does not resolve semantic reliability; explicit concept alternatives and teacher-approved benchmark labels remain necessary. These are observations against provisional expectations, not teacher-validated accuracy.
- Private trial audit: `artifacts/written-marking-review/local-v8-answer-detail.json`. No production re-mark, restart, remote push or deployment was performed.


### Planned: teacher-grounded open-ended marking

Status: future implementation plan, agreed 2026-10-04. The six improvements below are planned work, not a record of completed features.

Teacher-provided correct answers and their full marks are already approved reference examples. They can seed the example bank without another approval step. The principal missing supervision is partial-credit rules and scored partial answers. Teacher approval of a reference answer does not automatically approve AI-generated paraphrases or scores assigned to student responses; transcription approval is also separate from score approval.

Using examples in prompts or retrieval does not train model weights. Fine-tuning remains a later possibility, outside this planned first approach.

#### Six planned improvements

1. **Explicit scoring rubrics — planned.** Use question wording, teacher answers, full marks and answer details to draft named concepts, required counts, accepted alternatives and partial-credit weights. Retain the teacher's original answer. Review newly inferred rules before activation; explanatory details must not silently introduce requirements.

2. **Teacher-grounded example bank — planned.** Seed full-credit examples from teacher answers and maximum marks. Add teacher-scored partial, incorrect and borderline responses as they become available. Record provenance and approval status. Keep AI-generated examples provisional and separate from evaluation answers.

3. **Concept-level semantic marking — planned.** Apply approved deterministic matches first, retrieve relevant scored examples next, and use local AI for unresolved concepts. Calculate marks in application code. Enforce distinct-concept credit, score caps and contradiction rules; similarity alone must not determine marks.

4. **Grouped teacher review — planned.** Group identical reviewed responses to the same question and rubric version. Suggest similar-answer groups for confirmation. Allow a teacher decision to apply to confirmed group members while preserving individual overrides and audit history.

5. **Uncertainty and technical recovery — planned.** Separate extraction uncertainty, semantic ambiguity, missing scoring rules and invalid model output. Use bounded retries for technical failures. Show disputed concepts and affected mark ranges; introduce confidence percentages only after calibration against teacher-scored cases. Mark ranges describe the current concept decisions, not a guarantee of correctness.

6. **Benchmarking and controlled adoption — planned.** Establish approved expected marks for the existing 50-answer benchmark, concentrating teacher input on partial-credit and ambiguous cases. Maintain separate development examples and held-out tests; expand to unseen questions and English/Kiswahili responses. Compare accuracy, review workload and timing before adopting changes.

#### Acceptance criteria for future implementation

- Teacher reference answers receive their declared full marks.
- For an approved equal-weight two-concept rubric worth five marks, one valid concept earns 2.5; repetition adds no credit.
- Test equivalent wording, negation, irrelevant additions, clear errors and explanatory details that are not scoring requirements.
- Cover the observed Q16–Q20 failures, particularly valid theft concepts and concise definitions.
- Report exact agreement, average score error, false zeros, excessive credit and unresolved rate; measure extraction, marking and human-review time separately.
- Prevent evaluation-response and score-label leakage into prompts, retrieval or training. The question's teacher reference answer remains a legitimate marking input; copying that answer is a functional check, not evidence of generalisation to unseen student answers.
- Evaluate held-out answers separately from development examples, including language-specific results. Fewer pending answers alone does not establish improved accuracy.

#### Research and guidance informing the plan

- [ETS: Best Practices for Constructed-Response Scoring](https://www.ets.org/pdfs/about/cr_best_practices.pdf) — established guidance on scoring rubrics, expert-scored examples, quality control and evaluation.
- [ETS: Evaluations of Automated Scoring Systems in Practice](https://www.ets.org/research/policy_research_reports/publications/report/2020/kbxs.html) — evaluation against quality human ratings and the intended assessment use.
- [SemEval-2013: Student Response Analysis and Recognizing Textual Entailment](https://aclanthology.org/S13-2045/) — semantic assessment of short responses rather than treating word overlap as correctness.
- [Language Models are Few-Shot Graders (2025)](https://arxiv.org/abs/2502.13337) — research supporting instructor-graded examples and retrieval-based example selection; benefit must be validated with our local models and target languages.
- [Gradescope: AI-assisted grading and answer groups](https://guides.gradescope.com/hc/en-us/articles/24838908062093-AI-assisted-grading-and-answer-groups) — an established teacher-controlled grouping workflow; automated grouping support varies by question type.
- [Confidence Estimation in Automatic Short Answer Grading with LLMs (2026)](https://arxiv.org/abs/2605.00200) — research on combining model signals with graded-response data; model-stated confidence alone is insufficient, and local calibration is still required.

#### Documentation boundaries

This update records future work only. It does not change application code, manifest files, database records or model weights, and does not restart services, commit, push or deploy. Existing implementation history above is preserved; this section clarifies the agreed direction without claiming that earlier experiments established grading accuracy.

### Teacher-grounded marking implementation (2026-10-04)

- Implemented the six-part workflow as local capabilities; teacher approval of partial-credit rules and benchmark labels remains a human step. Teacher reference answers and manifest-approved alternatives supply full-credit examples directly, without duplicating them in every student record or asking for another approval. No model weights are trained.
- Added draft-only rubric preparation from question wording, teacher reference answers, full marks and answer details. The UI displays proposed concepts, synonyms and allocations for explicit approval before staging them for the next marking run. Original manifests remain immutable. Weighted criteria now support accepted alternatives, alongside existing any-N concept rules. Legacy manifest interpretation remains compatible.
- Added a local append-only `teacher_examples` table, created idempotently when the repository opens. Saving a teacher mark atomically saves its approved written-answer example and immutable marking-run version. Example scope fingerprints question identity, wording, reference answers, rubric, maximum and scoring rule; changed requirements do not reuse old examples. Latest explicit corrections supersede earlier versions for retrieval while history remains stored. Individual job deletion retains this reusable bank; Reset all AutoMark data removes it.
- Marking now applies deterministic reference comparison, approved exact example matches and unambiguous explicit concept alternatives before local AI. Concept credit is distinct and capped; unknown clauses and negations do not receive substring-based credit. Relevant approved examples are retrieved using lightweight lexical ranking solely to select prompt examples, never to calculate marks. Reference examples always anchor retrieval; provisional examples are excluded.
- `written-marking-v9` supplies approved examples and rules to semantic grading. Answer details are reserved for rubric authoring and audit/report context, addressing the v8 background-prose regression. Invalid/incomplete model output receives at most one retry with attempt metadata; genuine uncertainty does not trigger repeated guessing. An uncertain holistic response may omit evidence, but a scored decision must cite valid student segments.
- Added Review answer groups for checked scripts. Identical answers share a group only within the same question/rule fingerprint. Similarity-based suggestions stay separate and unchecked until the teacher confirms selected members. Teacher scores can correct already-scored answers, not only pending ones. Individual overrides, source text, leading-zero identities and immutable result history are preserved; stale or changed results block application. A group failure reports per-member outcomes rather than claiming all succeeded.
- UI and HTML reports distinguish extraction, meaning, rubric and technical review reasons. Concept uncertainty includes disputed ideas and a provisional lower/upper mark range, not a confidence percentage. Teacher marking interactions record elapsed review-interface time separately from marking; group time is apportioned across members. This is elapsed interaction time, not a calibrated measure of attention. Unmeasured OCR/extraction-review time remains unavailable.
- Benchmark tooling now evaluates deterministic grounding before AI and reports exact agreement, scored-answer accuracy, mean absolute mark error, false zeros, over-credit, review rate and language-specific counts. It blocks identical held-out answers supplied as non-reference teaching examples. Development/test split discipline and independent teacher labels are still required; the teacher answer remains legitimate input, not evidence of generalisation.

Validation and limitations:

- 95 targeted tests passed across marking, rubrics, AI, desktop controls, reports, manifest import and saved-data cleanup. New regressions cover approved references, partial credit, repetition, negation, changed scopes, example persistence/reset, stale groups, preserved overrides, draft approval, technical retries, uncertainty ranges and evaluation leakage. No tests start or deploy application services.
- Six-case local Qwen3 trial: the clear corruption definition, funding/educators and theft/hatred examples earned full marks; one sand bucket earned 2.50; broom/bell earned zero; theft plus helping old people remained pending. About 32.3 seconds; expected outcomes remain provisional.
- Isolated 50-answer trial with original manifest rules and no newly approved partial-score examples: 41 scored, 9 pending, 203.1 seconds. Accuracy metrics remain null because there are no approved student-score labels. These benchmark inputs are not an exact replay of every production OCR text/line break, so this is not a controlled before/after accuracy claim. English/Kiswahili qualification remains outstanding.
- Private benchmark output is under `artifacts/written-marking-review/local-v9-full-regression.json`. Remaining cases include ambiguous definitions, irrelevant NGO answers and the valid theft concept alongside an irrelevant answer. The Q17 partial-credit allocations also still require teacher agreement.
- Existing production jobs, scores and manifests were not changed by tests. No restart, commit, push, deployment, fine-tuning or additional model download was performed. Reopening the app enables the new controls; re-marking is an explicit user action. To establish partial-credit guidance, use Draft rubric from teacher answer, review/approve it, then mark; use Save teacher mark and approved example or Review answer groups to supply actual teacher scores.


### Clear unrelated answers receive zero (2026-10-04)

Updated local marking prompt to `written-marking-v10`: understandable but unrelated answers are incorrect (holistic) or not-demonstrated (named concepts), rather than semantically uncertain. Genuine ambiguity remains reviewable; a valid concept retains credit even alongside unrelated content. No blanket conversion of pending results to zero.

The teacher-approved NGO example (roads and selling schoolbooks instead of anti-corruption education) is a zero-credit prompt anchor and opt-in live regression, not a held-out evaluation item. Local Ollama returned 0/5 (`ai-incorrect`) in 4.84 seconds. This verifies the reported case, not general grading accuracy. Existing saved runs are unchanged; re-marking applies the updated prompt. No model weights were trained.


### Independent credit in mixed answers (2026-10-04)

`written-marking-v11` adds concept-independence guidance to named-concept classification and semantic verification. The teacher-approved example “Stealing and helping old people” demonstrates theft only: one of two required concepts, earning 2.5/5. A positive activity alongside a valid negative behavior does not negate the valid behavior. Negation of theft still does not earn theft credit, and repetition adds no credit. Holistic classification retains its existing unrelated-answer guidance. These prompt anchors are development regressions, not held-out accuracy evidence. Saved runs are unchanged until re-marked.


### List conjunctions as candidate evidence (2026-10-04)

`written-marking-v12` segments “and”-joined candidate items for any-N list rubrics. Both AI passes receive the original student answer for context; segment count never determines credited concept count. Shared negation and recognized compound reference phrases remain intact. Holistic answers retain sentence segmentation. Regression coverage includes the mixed theft/positive-behavior answer, negation, repeated concepts and compound phrases. No saved extraction or marking records are rewritten.


### Additional answer list formats (2026-10-04)

`written-marking-v13` extends any-N candidate segmentation to commas, semicolons, Roman-numeral labels, numeric/alphabetic labels and bullets, including inline lists. Decimal and digit-grouping commas remain intact. Shared negation and known compound reference phrases remain in context; holistic prose keeps sentence segmentation. Full student text remains available to both marking passes. Candidate count never grants concept credit automatically. Local regression tests cover these formats; saved results require re-marking to use the change.


### Single-pass semantic marking (2026-10-04)

At the user's request, `written-marking-v14` removes the second AI concept-verification request. The first pass still judges each concept against the rubric and cites evidence; application code validates the output and calculates marks. Genuine uncertainty remains pending for individual teacher review. Teacher group review and overrides remain available. Bounded retries for malformed output remain, so technical recovery can still make a second request. Existing audit history and saved scores are preserved. Automated regressions check one successful grading request, partial-credit arithmetic and unresolved-case handling. Live-model validation was not run because the permission request was declined; accuracy and speed improvements are not yet measured on a new run.

Follow-up validation (2026-10-04): the user authorised the previously declined local Ollama checks. All five opt-in live regressions passed with single-pass marking v14 (pytest elapsed 29.61 seconds, including test setup). Cases covered the unrelated NGO answer (0), theft plus helping older people (2.5), helping older people alone (0), negated theft (0), and repeated theft (2.5), each out of five. Saved production results were not changed. This small regression run is not a throughput or general accuracy benchmark.


### Matching letter pairs in generated papers and reports (2026-10-04)

The web frontend now labels matching left-hand items A–D on question papers and answer sheets, consistent with existing letter-based answer keys. Numeric manifest item positions and bubble geometry remain unchanged. AutoMark extraction and marking HTML comparisons format matching pairs as A=B, C=D, retaining original positions when placeholders are omitted. Older manifests remain readable; existing exported HTML must be exported again to show the new comparison format. No backend migration, restart or remote push is included.

### Phrase recognition, selective OCR fallback and marking recovery (2026-10-05)

Implemented locally:

- Semantic marking v15 compares complete action phrases, their objects and polarity. “Looking after school equipment” can demonstrate “taking care of school property”; destroying it cannot. Concise valid concepts retain independent credit. Readable unrelated terms receive zero rather than semantic uncertainty. Application code still calculates credit, caps scores and preserves genuine uncertainty for teacher review. There is no second AI verification pass.
- Count recognition now covers “What/which are the two…” and “Identify the two…”. Reference-list splitting keeps balanced parenthetical glosses attached to their concept. The election-rights example has two concepts worth 4.5 each out of nine. Ambiguous counts/reference lists still require a rubric.
- A missing secondary OCR language pack remains a visible configuration warning but no longer makes confident OCR uncertain by itself. Missing the explicitly selected language, low confidence and empty text remain reading problems. Historical warning-only OCR results with sufficient recorded confidence also avoid unnecessary AI fallback. This does not establish that OCR confidence is calibrated handwriting accuracy.
- Each completed question is saved in a local marking checkpoint. An interrupted run resumes unchanged results, keyed by reviewed responses, manifest, policy, teacher examples/overrides and installed model digest/prompt version. Successful completion atomically saves the final run and clears its checkpoints; deletion includes checkpoints. Single-job and batch interfaces display question counts, student/question identity and restored progress. Saved historical marking runs remain unchanged until re-marked.

Validation: 11 live local Qwen v15 regression cases passed (57.92 seconds including setup), covering unrelated answers, independent theft credit, negation, repetition, school-property phrase equivalence and unrelated terminology with different capitalization. These prompt-development cases are not held-out evaluation or a general accuracy benchmark. Automated tests also cover 4.5/9 partial credit, language warning versus reading uncertainty, interrupted-run recovery and changed-policy invalidation. No production results were rewritten, services restarted or changes pushed.

### Ordinary OCR improvements, priority 2 (2026-10-06; subsequently reverted)

Implemented locally as `tesseract-written-v2`:

- Conservative ink-band detection chooses single-line PSM 7 or block PSM 6. Source pixels remain intact. NFC Unicode and whitespace cleanup never changes spelling, negations, numbers or answer meaning.
- Uncertain first readings receive at most one alternate pass: PSM 13 for a single line, or PSM 11 with contrast normalization for a block. Both passes share a 20-second OCR budget. Confident first readings skip retries. Missing secondary language packs alone do not trigger retries.
- Matching normalized readings may use the higher-confidence attempt. Differing nonempty readings retain the first text and flag disagreement; confidence alone cannot replace words. Retry failure retains the first reading. Original text, both attempts, modes and confidence are saved in optional transcription JSON details without a database migration.
- Confident v1 and v2 OCR bypass optional Gemma transcription. Uncertain results can still use Gemma only when handwriting assistance is enabled. Approved responses and existing successful saved transcriptions are preserved; this change does not automatically reprocess existing jobs.
- The handwriting benchmark accepts `tesseract`, `tesseract-single` (v2 without retry), and `paddleocr`, alongside the existing engines. PaddleOCR requires explicit local PP-OCRv5 English recognition weights and separately installed dependencies. It uses line crops and remains benchmark-only/unqualified; it is not active in extraction and Kiswahili is not qualified. The current environment lacks PaddleOCR, so no specialist accuracy or throughput claim is made.

Validation covers cleanup, single-line mode, bounded retries, disagreement, failure recovery, persistence, AI gating and optional specialist boundaries. Evaluate CER/WER, exact transcriptions, scoring-impact errors, fallback rate and timing against independent transcription labels before claiming improved handwriting accuracy. The existing benchmark reports per-language counts/errors; the new single-pass option is not the historical v1 baseline. Scan resolution changes (priority 1), service restarts and production-result changes are outside this update.

PaddleOCR API reference: [Text recognition module](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html). OCR preprocessing reference: [Tesseract quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html).

Local validation: OCR/handwriting/written-extraction suite passed 52 tests; specialist-adapter/report suite passed 4; final OCR/manifest-contract suite passed 38 (overlapping OCR tests). A four-crop saved-handwriting smoke comparison found no selected-text improvement: two confident readings remained unchanged and two uncertain alternate readings disagreed, correctly retaining review. This is not evidence of reduced AI usage. The PaddleOCR adapter was tested with a fake local predictor, not an installed model. No services were restarted and no commits or pushes were made.

### Preserve 300-DPI written-answer detail (2026-10-06)

Written regions now render directly from bounded clips of the original PDF at 300 DPI, then map through the existing page homography. Image uploads retain their original embedded raster through the same path. Page identification, QR decoding and bubble extraction continue to use 150-DPI previews; a full 300-DPI page is not allocated. Crop dimensions and clipped source renders are bounded to 20 million pixels each. Pixel-center mapping preserves crop alignment on rotated pages. Visible PDF text and residual-ink checks use the same new crop scale. Captured image/blank responses identify `written-region-capture-v2`; page metadata records `written_crop_dpi`.

This preserves detail present in a 300-DPI source; it cannot restore missing detail in a low-resolution or blurred original. New extractions use the change. Resume does not replace already saved crops or approved responses; testing needs a fresh extraction job. No running application was restarted or production results rewritten.

Validation includes native fine-detail image content versus the previous low-resolution enlargement, all four PDF rotations, original-resolution PNG uploads, oversized crop rejection, persisted 300-DPI crop dimensions and metadata, visible/hidden PDF text, blank regions and existing structured extraction behavior.

### Simplify OCR after the 300-DPI trial (2026-10-06)

At the user's request, retain only the three extraction improvements: 300-DPI source crops, single-line/block mode selection (PSM 7/6), and NFC/whitespace cleanup. `tesseract-written-v3` now performs exactly one OCR call per answer. Removed the alternate PSM 13/11 retry, retry contrast normalization and disagreement selection logic. Removed the inactive PaddleOCR candidate and its benchmark CLI option. The benchmark's `tesseract` option measures the current single-pass recognizer.

The existing 20-second Tesseract timeout, uncertainty thresholds, optional AI fallback, audit metadata and saved-history compatibility remain. Original text and selected mode are still recorded; no second attempt is generated. AI gating understands v1, v2 and v3 results: confident readings skip AI and uncertain readings reach it only when enabled. No existing job, transcription, crop or score is rewritten; previously saved v2 retry history remains historical evidence. No restart or push is included.

Regression coverage checks one call for low-confidence and empty readings, both selected modes, unchanged cleanup, 300-DPI preservation, confident-result AI bypass and uncertain-result AI fallback.

### Restore block-only written OCR (2026-10-06)

At the user's request, `tesseract-written-v4` removes automatic line-band detection and PSM 7/6 selection. Every written answer now receives exactly one Tesseract PSM 6 call, including single-line answers. Retain 300-DPI source crops and NFC/whitespace cleanup. Confidence thresholds, timeout, optional AI fallback and historical saved results are unchanged. AI gating recognises v4 alongside older versions. This does not establish that fallback will return to 50%; comparisons must use the same handwriting set. New extraction after reopening the app is needed; saved successful OCR is not silently replaced.

Validation covers PSM 6 for single/multiple lines, no retries for uncertain or empty readings, cleanup, 300-DPI crop preservation and confident/uncertain AI gating. No service restart, production-data rewrite, commit or push is included.


### Revert both morning extraction priorities (2026-10-06)

At the user's request, all application changes from the Priority 1/2 experiment are reverted. Restore 150-DPI page preparation and approximately 144-DPI normalized written crops; restore the original `tesseract-written-v1` one-pass PSM 6 recognizer. Remove the added Unicode/whitespace cleanup, line detection, alternate retries, contrast adjustment, transcription details field, crop-DPI metadata and new benchmark options. Specialist benchmark files were already removed. The preceding experiment entries are retained as history, not active behavior.

Preserve the pre-experiment language-configuration warning fix, semantic marking v15, marking checkpoints and unrelated UI/review work. Saved jobs, experimental transcriptions/crops, reports and source PDFs are not rewritten or deleted. Reopening the app and using a fresh extraction is required to test restored behavior; no automatic restart or remote action is performed. A rollback does not guarantee the earlier 50% fallback rate on different handwriting.

### Bounded parallel AI handwriting trial (2026-10-06)

Implemented an optional **AI transcription pages at once** selector: one (default, sequential) or two (parallel trial). This changes only optional AI handwriting fallback after ordinary OCR. Within each student/layout group, submit at most one answer crop from each of two source pages; finish that group before starting the next student. Files remain sequential. This is concurrent independent crop transcription, not a multi-image prompt or full-page transcription. A student with uncertain answers on only one page remains sequential. No question, reference answer or marking context enters transcription requests.

The coordinator alone reads crops and writes results, preserving response IDs even when requests finish out of order. Each finished draft is saved immediately. Cancellation stops new submissions and waits for bounded in-flight requests to finish and save; resume skips successful saved drafts and reviewed responses. Failed AI calls retain any OCR text, remain retryable and reduce subsequent concurrency to one for the rest of that job. Shared model verification is protected against duplicate concurrent initialization. Existing timeouts and crop limits remain in force.

Progress reports completed AI answers, active pages and elapsed wall time. The completion message retains the total AI wall time across sequential files for comparison. Individual transcription durations may overlap and must not be summed as elapsed time in parallel mode; the returned scheduler summary separates wall time from summed request time. This update does not add persistent run-level timing storage, so record the completion message during trials.

The two-page limit is a conservative trial for the previously identified 16-GB Mac, not a measured capacity or speed guarantee. Ollama may serialize requests: its documented default `OLLAMA_NUM_PARALLEL` is one, and parallel context memory grows with concurrency. No Ollama configuration, service restart, model installation or production-result rewrite is included. Reference: [Ollama concurrency and memory guidance](https://docs.ollama.com/faq).

Validation: controlled fake inference verifies overlap on separate pages, student isolation, a two-request bound, correct response mapping, cancellation draining/resume, OCR retention on failure, serial fallback, confidence/approval skipping and one-time model verification. Existing extraction/OCR/handwriting/UI regressions passed (72 tests); the final UI test also checks explicit default/selection forwarding and retained elapsed-time feedback. These tests establish scheduling behavior, not real-model speed or transcription accuracy. Before adopting two-page mode as the default, compare identical unreviewed crops using one versus two pages, record elapsed time, failures, memory pressure and transcription differences, and retain sequential mode if throughput does not improve. Marking and the reverted OCR configuration are unchanged.

### Semantic marking v16: equivalent wording and clear decisions (2026-10-06)

Target the latest run's rejected reporting alternatives, pending concise definitions/equal-vote descriptions, unrelated terms and contradictory voting claims. Reflow single wrapped lines in evidence segmentation so an action and its object stay together; explicit list markers, negation and paragraph boundaries remain meaningful. The original student answer and extraction records remain unchanged. No spelling replacement or reference-guided transcription is introduced.

Simplify the marking prompt and explain concrete actions that instantiate a general reference, either branch of an OR alternative, concise equivalent definitions, and the distinction between absent evidence and genuinely ambiguous meaning. Retain modifiers and contradictions instead of salvaging isolated matching words. Ask for a bounded semantic comparison before the categorical judgment in the same model request; save that comparison in the audit. Application code still calculates marks and validates concept/evidence IDs. Ambiguous meanings, missing rubrics and invalid output remain reviewable. No additional verification call is introduced; the extra comparison can increase output time per request.

Use `written-marking-v16` in audit/checkpoint identity so v15 semantic checkpoints cannot silently supply a new v16 marking run. Existing approved responses, historical scores and exported reports are not rewritten. Re-marking after reopening the updated application is required to obtain new results. No restart, commit, push or deployment is part of this change.

The selected problem cases are development cases from this run, not held-out evidence. Their simulation scores are provisional benchmark labels, not newly teacher-approved examples. A shorter prompt alone regressed previously correct zero scores to pending and was not retained; the comparison-before-decision format is being validated with both failures and previously correct controls.

Final validation: 75 regression tests passed; 11 opt-in live tests were skipped by the normal suite. A separate real local Qwen3 4B run evaluated 20 selected saved answers without writing to production records: all 12 previously correct controls retained their expected marks, and seven of eight selected failure/pending cases now matched the simulation labels (19/20 total). This resolved the selected equal-vote and patience paraphrases, two unrelated-term cases, the prefect-reporting answer, and both repeated-voting/removing-candidates cases. Student 13042 Q15 remains wrongly zero against the simulation label; its damaged transcription includes “Tea her polit ely,” so robustness to that input remains unresolved. This is targeted development validation, not a new whole-batch accuracy estimate or independent teacher validation. The added semantic comparison costs generation time; measure full-run marking time before accepting the speed trade-off.

### V17 ablation: remove generated semantic comparison (2026-10-06)

At the user's request, remove the generated comparison field and its instruction from marking output. Retain V16's other semantic guidance, wrapped-line evidence handling, OR-alternative handling, partial-credit calculations, validation and genuine-ambiguity review. This is a comparison-removal experiment, not a return to V15. Application-generated score explanations and existing V16 historical comparisons remain available; no saved record is rewritten.

New inference/checkpoint identity is `written-marking-v17`, preventing reuse of earlier semantic checkpoints in a new marking run. Reopening the application and re-marking saved reviewed responses is sufficient; extraction does not need to repeat. No service restart or remote operation is included.

Local regression validation: 73 passed, 11 opt-in model tests skipped. Additional paired local-model validation uses the same 20 development answers with V16 and V17, alternating evaluation order and retaining the same model, reviewed text, rubric and generation settings. Outcomes are compared with provisional simulation labels; these selected cases do not establish held-out accuracy. Reports are temporary and production scores remain unchanged.

Paired live result (20 selected development answers, same installed Qwen3 4B): V16 took 102.44 seconds, matched 19 labels and left none pending; V17 took 47.21 seconds, matched 18 labels and left none pending (approximately 54% less request time). All 12 previously correct controls stayed correct. Removing comparison regressed student 13038 Q16 from correct zero to 4.5/9 for “vote many times and remove other candidates.” Student 13042 Q15 remained incorrectly zero under both versions. The first V16 request included possible model warm-up; this small single-pass trial is not a whole-batch speed or accuracy guarantee. V17 remains the requested experimental configuration for the user's full rerun; do not describe it as uniformly better. No saved grades were changed by the test.

### Implemented locally: TrOCR written-answer extraction (2026-10-07)

- Development branch: `automark_TrOCR`. Replaces the ordinary written-answer
  Tesseract adapter with `microsoft/trocr-base-handwritten`. OpenCV continues to
  decode and prepare answer crops; PDF rendering, QR/layout identification,
  bubbles, text-PDF extraction, review and marking retain their existing paths.
  The separate student-ID routines still use their existing OCR engines.
- The [TrOCR model card](https://huggingface.co/microsoft/trocr-base-handwritten)
  specifies single text-line images. OpenCV removes long horizontal rules from
  the segmentation mask, groups nearby ink bands, and produces padded lines in
  top-to-bottom order. TrOCR reads each line and newline-joins the output.
  Tall/ambiguous bands, empty readings and generation truncation remain uncertain.
- Model runs locally through Transformers/PyTorch, using MPS when available,
  otherwise CUDA or CPU. Weights are cached across student jobs and model
  inference is serialized. No teacher answers or marking context enter OCR.
- Install requirements and run `.venv/bin/python -m tools.prepare_trocr` once
  to cache the model. Runtime uses `local_files_only=True`; missing dependencies
  or weights produce `ocr-unavailable` and a setup message, not a silent download.
  `AUTOMARK_TROCR_MODEL` can point to a pre-provisioned local model directory.
- Confidence is a provisional token-score heuristic, not calibrated accuracy:
  geometric mean below 0.90 or any token below 0.50 triggers review/optional AI.
  These thresholds are distinct from Tesseract's. The IAM English checkpoint
  has not been validated for Kiswahili: `sw` and `en-sw` are flagged
  `ocr-language-unvalidated` and remain eligible for optional AI assistance.
  Consequently the default mixed-language setting may refer every written
  answer to AI; use English for an English-only comparison.
- Existing optional Gemma fallback receives uncertain/error crops; confident
  English TrOCR readings skip it. Approvals and existing AI drafts are preserved.
  Resuming unreviewed old Tesseract crops reruns ordinary OCR with TrOCR; completed
  jobs remain unchanged. Fresh jobs are recommended for paired comparisons.
- Validation covers ordered multiline reading, rules, ambiguous line bands,
  blank images, missing weights, confidence/truncation, language limitations,
  AI eligibility and resume/review behavior. Real handwriting accuracy and
  referral-rate calibration remain benchmark work, not established improvements.
- Local validation: 76 targeted tests passed across OCR, parallel fallback,
  written extraction, review UI and marking; `git diff --check` passed.
  Installed Transformers 4.57.6 and cached model revision
  `eaacaf452b06415df8f10bb6fad3a4c11e609406` for local testing.
  Read-only smoke tests on four retained answer crops produced text, including
  a two-line Q17 answer reconstructed in reading order. Q13's `feeling` reading
  was flagged low-confidence; Q16/Q20 and multiline Q17 passed provisional
  checks. This is a functionality check, not a comparative accuracy benchmark.
  No saved jobs/results were modified and no running app was restarted.

### Offline selective TrOCR retry benchmark (2026-10-07)

- Added `python -m tools.trocr_retry_benchmark --database <saved-jobs.db>
  --dataset <simulation-directory> --output <separate-report.json>`.
  It opens SQLite read-only, freezes a read snapshot, checks source PDF hashes,
  and selects English TrOCR results flagged only `ocr-low-confidence`.
- Reproduces greedy decoding on unchanged OpenCV line crops and retries only
  low-confidence, nonempty, complete lines once with three beams. Limits remain
  128 new tokens and 20 seconds per line. Selected-token probabilities use
  `compute_transition_scores` with beam ancestry and normalized logits; EOS and
  padding are excluded. Every line must pass the existing 0.90/0.50 heuristic
  thresholds before simulating bypass of Gemma.
- The separate report preserves baseline/retry/Gemma text and per-line timing,
  model-load time, mean/median/p95 durations, strict and punctuation-insensitive
  error metrics, corrections/regressions and every accepted transcription that
  differs from its simulation label. Expected text is loaded only after inference.
  Estimated net savings subtract all retry time from saved Gemma time avoided.
- Unit checks cover candidate exclusions, order, retry bounds, incomplete/empty
  output, failures and selected-beam score calculation. The tool is isolated
  from live extraction: no activation, application-data writes, marking changes,
  service restart, commit or push.
- Initial execution stopped before inference: the production database now holds
  120xx jobs, replacing the planned 130xx run's 101 candidates. Earlier comparison
  reports survive but their crop pixels are absent. Dataset choice is pending;
  no accuracy or speed benefit has yet been established for retries.

#### Retry benchmark result: current 120xx run (2026-10-07)

- User selected the current saved run: 27 English low-confidence-only answers.
  Saved PDFs matched the original 120xx simulation by SHA-256. All 27 fresh
  greedy baselines reproduced saved text exactly. The benchmark ran on CPU;
  timings must not be generalized to MPS or other device configurations.
- Three-beam retries bypassed **0/27** Gemma calls under unchanged thresholds.
  No technical failures occurred. Five answers passed the mean-token threshold
  and eleven passed the minimum-token threshold, but none passed both for every
  line. Thus there are no newly auto-accepted mismatches to inspect.
- Ignoring case, punctuation and whitespace differences, exact transcription
  increased from 16/27 to 18/27 (two corrected, zero exact-match regressions).
  Character error rate improved 2.02% to 1.35%; word error rate 10.05% to 7.41%.
  Saved Gemma readings reached 25/27 exact, 0.58% CER and 1.06% WER.
  Raw punctuation-sensitive metrics are retained in the separate JSON report.
- Changes inspected: 12040 Q17 corrected `PROFECT` to `prefect`; 12030 Q15
  corrected `PLAYTES`/`means` to `plates`/`meals`; 12030 Q13 improved
  `itiscalledaffering` to `itiscalled a feeling`, but spacing remained wrong.
- Model load: 4.49 s. Fresh greedy: 42.12 s total, 1.56 s mean, 1.65 s median,
  2.81 s p95 per answer. Retry: 52.68 s total, 1.95 s mean, 2.04 s median,
  3.56 s p95. Corresponding saved Gemma calls totalled 120.28 s.
  Estimated net saving is **negative 52.68 s**: retries add time and avoid no
  fallback calls. Baseline replay and loading are excluded from that incremental
  estimate. Simulated final results are identical to the existing Gemma results.
- Decision: **do not activate this retry strategy**. A wider decoding search
  modestly improves text, but fails the required speed benefit. Confidence
  calibration can be investigated separately; thresholds were not weakened to
  force a favorable result. This small development subset is not held-out proof.
- Separate local report: `/tmp/automark-trocr-retry-12000.json`.
  Application data and live extraction behavior remain unchanged.

#### Offline PaddleOCR-before-Gemma benchmark (2026-10-07)

- Tested the current 120xx run's 266 English answers with uncertain TrOCR
  readings and saved Gemma readings, including segmentation warnings. Source
  PDFs matched the simulation by SHA-256. Read the database in a read-only
  snapshot and used saved crops; expected transcriptions were loaded only after
  inference. No live extraction, marking, approval or database changes.
- Added `tools/paddle_fallback_benchmark.py` and two passing tests for result
  order, missing/mismatched results and lexical evaluation. Dependencies and
  models are isolated under `/tmp`, outside the application's environment.
  PaddleOCR 3.7.0 / PaddlePaddle 3.3.1 used `PP-OCRv5_mobile_det` and
  `en_PP-OCRv5_mobile_rec`, CPU with four threads, no document unwarping or
  orientation model, and no recognition-score filtering before evaluation.
- Reproduction after preparing those local dependencies and weights:

  ```sh
  PADDLE_PDX_CACHE_HOME=/tmp/automark-paddlex-cache \
    /tmp/automark-paddle-benchmark-env/bin/python -m tools.paddle_fallback_benchmark \
    --models /tmp/automark-paddle-models \
    --database "$HOME/Library/Application Support/Imarisha AutoMark/structured-jobs.db" \
    --dataset "$HOME/Downloads/Imarisha_Phase2_50_Student_Scripts" \
    --output /tmp/automark-paddle-fallback-12000.json
  ```

- Same-subset transcription results (case, punctuation and repeated whitespace
  ignored; word boundaries, numbers and negation retained):

  | Reading | Exact / 266 | Character error rate | Word error rate |
  | --- | ---: | ---: | ---: |
  | Saved TrOCR | 236 | 0.49% | 1.89% |
  | PaddleOCR | 97 | 35.93% | 48.07% |
  | Saved Gemma | 244 | 0.45% | 1.35% |

- PaddleOCR inference total: 94.57 s, mean 0.356 s, median 0.325 s,
  p95 0.826 s per answer, including detection and recognition. Model
  construction/load took 1.73 s separately (Python imports excluded).
  No technical exceptions; 21 answers produced no nonempty reading.
- Exploratory minimum per-detection confidence policies, with remaining answers
  retaining saved Gemma results:

  | Policy | Gemma calls avoided | Accepted mismatches | Combined exact / 266 | Estimated net saving |
  | --- | ---: | ---: | ---: | ---: |
  | Score ≥0.90 | 157 | 63 | 198 | 577.56 s |
  | Score ≥0.95 | 142 | 57 | 204 | 508.87 s |
  | Score ≥0.98 | 72 | 24 | 227 | 202.90 s |
  | Score ≥0.95 and normalized agreement with TrOCR | 78 | 0 | 244 | 239.96 s |

- Inspected all 63 mismatches accepted by the weakest threshold (which includes
  the higher-threshold subsets). Errors include word joining/splitting, omitted
  list numbers, spelling mutations and severe text loss. For example, 12008 Q15
  returned only `F` for a full sentence with score 0.926; Q16 returned only `T`
  with score 0.942. Confidence measures recognized detections, not whether all
  handwriting was detected. The tool's `complete` field means nonempty returned
  detections only, not verified image coverage.
- The agreement policy accepted 78 answers across 34 students: 66 with
  segmentation-only warnings, nine low-confidence-only and three with both.
  Twelve accepted answers contained at most two words; 66 were longer.
  All 78 matched simulation transcriptions under the stated normalization.
  The combined CER/WER remained identical to saved Gemma, with 22 mismatches
  remaining among the 188 answers still sent to Gemma.
- Estimated savings subtract **all 266 Paddle calls** from corresponding saved
  Gemma durations avoided: 334.53 − 94.57 = 239.96 s. That is 29.3% fewer
  Gemma calls and 21.2% less summed fallback processing time relative to
  1131.04 s of saved Gemma durations. This is not measured end-to-end wall-time
  improvement: concurrency, model contention and startup can change it.
- Decision: reject confidence-only substitution. Agreement is promising for a
  separate unseen-handwriting validation, not activated by this test. Both OCR
  engines can agree on the same mistake; these development simulation labels
  are not independent teacher verification. This evaluates these specific
  mobile models, not every PaddleOCR configuration or handwriting language.
- Report: `/tmp/automark-paddle-fallback-12000.json`. Official configuration:
  [PaddleOCR pipeline documentation](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
  and [recognition models](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html).
  Live TrOCR → optional Gemma behavior is unchanged; no restart, commit or push.

#### Implemented: PaddleOCR agreement before optional Gemma (2026-10-07)

- User authorized integrating the tested agreement strategy. New written-answer
  flow: TrOCR → selective PaddleOCR corroboration → optional Gemma for remaining
  uncertainty. Bubble extraction and marking are unchanged.
- Run Paddle only for English, nonempty uncertain TrOCR readings flagged solely
  for low confidence and/or uncertain line segmentation. Confident readings,
  blank regions, failures, incomplete decoding and unvalidated languages do not
  gain automatic acceptance from Paddle. Neither recognizer sees answer keys.
- Accept only when every returned Paddle detection has finite confidence ≥0.95
  and the entire normalized reading agrees with TrOCR. Normalize Unicode NFC,
  case, punctuation and whitespace as in the benchmark; preserve word boundaries,
  numbers and negation. High confidence without agreement never bypasses Gemma.
- Preserve the original TrOCR text and transcription history. Store a separate
  `paddle-trocr-agreement-v1` record with stage duration and minimum detection
  score. Accepted corroborations are `read`; rejected/failed corroborations keep
  TrOCR text and remain uncertain. Recognition scores are uncalibrated heuristics.
  Teacher review remains a separate step; agreement does not approve a response.
- Missing dependencies/weights and inference failures retain the normal optional
  Gemma path. Missing setup is reported in review; one unavailable-model load is
  attempted per job. Resume reuses completed corroboration, retries unavailable
  or failed corroboration, and can corroborate older unreviewed TrOCR readings
  without repeating TrOCR. Existing Gemma drafts, approvals and completed jobs
  are preserved. Cancellation prevents starting the next stage.
- Setup is explicit: `python -m pip install -r requirements-paddle.txt`, then
  `python -m tools.prepare_paddle`. Development dependencies and previously
  downloaded weights were prepared locally. Default weights directory is
  `~/.cache/imarisha-automark/paddleocr`; override with
  `AUTOMARK_PADDLE_MODEL_DIR`. Model loading is local-only, lazy and cached;
  Paddle inference is serialized and uses four CPU threads. No application
  restart, production-data update, commit or push was performed.
- Validation covers whole-answer agreement, missing text, confidence boundaries,
  NaN/malformed output, unsupported languages, truncation exclusion, fallback
  eligibility, unavailable-model recovery, history retention, resume reuse,
  cancellation and completed-job protection. Extraction, handwriting and view
  regression tests passed; `pip check` found no broken dependencies.
- Limitation: two OCR systems can agree on the same error. The development
  benchmark supports this gate, but unseen handwriting and Kiswahili still need
  separate validation. Confidence does not establish image coverage.
- Replay through the actual installed implementation reproduced **78 accepted,
  188 uncertain** on all 266 saved crops, with zero normalized mismatches among
  accepted readings. The production database was opened read-only for this
  validation; no saved response was replaced.
