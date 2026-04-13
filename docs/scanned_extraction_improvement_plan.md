# Improving scanned-image extraction quality

This project already has a strong scanned-image pipeline (deskew, perspective alignment, denoise/contrast normalization, adaptive thresholding, OCR fallback, and row de-duplication).

## Current scanned pipeline strengths

From `core/extractor.py` and `core/ocr_engine.py`, scanned mode currently includes:

- automatic fallback from text-layer mode to scanned mode when skew is high,
- rotation normalization and page-to-template alignment,
- quality scoring (brightness/contrast/blur/skew),
- fuzzy section heading detection,
- multi-variant OCR for question IDs and item numbers,
- post-detection de-duplication of row candidates.

## Why performance can still be poor on scans

The hard cases are usually from one (or more) of these:

1. **Template drift**: global perspective correction is not enough when page curvature/warping is local.
2. **Row segmentation sensitivity**: projection-threshold segmentation can under/over-split rows for noisy pages.
3. **Fixed ROI assumptions**: fixed `qid_x1` / `item_x0` ratios are brittle across scanner crops and print scaling.
4. **OCR-only ID extraction**: low-res or compressed scans can make numeric OCR unstable.
5. **Weak operational feedback loop**: quality metrics are computed, but not yet used to dynamically tune extraction.

## High-impact improvements (priority order)

### 1) Make extraction quality-aware (highest ROI)

Use `scan_quality_score` and sub-metrics to branch extraction behavior:

- For **poor** scans, run a heavier pipeline:
  - render with higher zoom,
  - widen row-band merge tolerance,
  - run OCR on more threshold variants.
- For **good** scans, keep the current faster path.

Expected impact: fewer misses in poor scans with controlled latency increase only where needed.

### 2) Anchor the layout with fiducials/registration marks

Instead of relying only on largest quadrilateral + relative column ratios, detect known anchors (corners, header lines, label tokens) and compute an affine/homography transform from template coordinates.

Expected impact: more stable section and column geometry across different scanners/crops.

### 3) Replace single-pass row segmentation with hybrid row finding

Combine horizontal projection with connected-component analysis:

- build candidate row bands from projection,
- validate/split/merge using blob density and expected row-height priors per section type,
- reject obviously implausible bands before OCR.

Expected impact: reduced duplicate rows and fewer missing question IDs.

### 4) Use deterministic digit classifiers for numeric fields

Question ID and item-number fields are constrained (digits only, known lengths). Add a lightweight digit classifier (or template matching) as a second opinion when OCR confidence is low.

Expected impact: better stability in compressed/blurred scans where OCR confuses similar digits.

### 5) Add adaptive ROI calibration per page

Estimate true column boundaries from detected vertical separators/text clusters, then derive `qid`/`item` ROIs from those boundaries (instead of static width ratios).

Expected impact: fewer failures on pages with variable margins and print scale drift.

## Engineering roadmap

### Phase A: observability and fast wins

- Log/retain per-page counters: rows detected, rows without qid, OCR confidence stats.
- Add debug artifacts (optional): aligned page, row-band overlay, ROI crops.
- Enable quality-aware branching using existing quality metrics.

### Phase B: geometry robustness

- Introduce template anchor detection and homography refinement.
- Calibrate ROIs from detected anchors/columns.

### Phase C: recognition robustness

- Add constrained digit classifier fallback for qid/item fields.
- Ensemble score between OCR and classifier for final token selection.

### Phase D: model tuning and regression protection

- Build a labeled benchmark set of poor scans.
- Track exact metrics in CI:
  - row recall/precision,
  - question-id exact match,
  - end-to-end row accuracy,
  - review-rate.

## Suggested acceptance targets

- +20-30% relative improvement in question-id exact match on poor scans.
- -30% relative reduction in `needs_review` for scanned-image mode.
- No measurable regression on text-layer mode.

## Practical parameter knobs to tune first

In current code, tune these first before large refactors:

- rendering zoom (`RENDER_ZOOM`) for scan-heavy workflows,
- section OCR confidence threshold in scanned heading detection,
- `min_tesseract_confidence` for qid/item extraction,
- row segmentation thresholds (`threshold` percentile, `min_row_h`, `max_row_h`, `merge_gap`).

These parameters already exist and can provide immediate gains with controlled experiments.

## Operator guidance: searchable PDF vs pure image scan

Yes — in most cases, **searchable PDF with a usable text layer performs better** than purely image-only scans for this extractor.

Why:

- text-layer mode can read structured text directly (less OCR noise),
- scanned-image mode depends more heavily on image quality, geometric alignment, and OCR confidence,
- low-resolution JPEG-like scans lose digit detail that question-ID extraction needs.

Recommended capture/export settings for best results:

1. Prefer **PDF output with OCR text layer** enabled (searchable PDF), not image-only exports.
2. Scan at **300 DPI minimum** (400 DPI if handwriting/light marks are common).
3. Use **grayscale or color** scan modes; avoid aggressive black-and-white compression presets.
4. Disable heavy auto-compression and “small file size” presets.
5. Keep pages flat, uncropped, and aligned (avoid clipped margins and perspective warp).

When to force scanned-image mode anyway:

- the PDF text layer exists but is badly mis-registered (common in skewed or mobile-captured pages),
- extracted tokens are clearly inconsistent with the visible marks,
- quality metrics indicate poor geometry/blur and you need the deskew/alignment path.

## Adding an OCR-enhanced PDF step in the workflow

Yes — it is feasible to add an **OCR-enhanced PDF component** (pre-OCR pass that embeds a searchable text layer) before extraction.

Recommended integration pattern:

1. Input PDF/image arrives.
2. Run quick preflight checks (dpi, skew, blur, existing text-layer usability).
3. If text layer is missing/weak, run OCR enhancement to generate a searchable PDF sidecar.
4. Run extractor on enhanced PDF in text-layer mode first; fallback to scanned-image mode if mis-registration is detected.
5. Store original + enhanced artifacts for audit/review.

### Will it delay scanning?

Usually **yes, somewhat** — but the delay can be controlled and often pays off by reducing downstream `needs_review`.

- Added latency depends on page count, DPI, OCR engine, and hardware.
- For small batches, the overhead is often acceptable.
- For large batches, use asynchronous/background OCR enhancement and parallel workers.

Practical ways to keep latency low:

- only OCR-enhance pages that fail text-layer usability checks,
- cache enhanced outputs by file hash so retries are instant,
- cap OCR resolution to the minimum quality target (for example 300 DPI),
- process pages in parallel with a bounded worker pool,
- expose a “fast mode” (no enhancement) and an “accuracy mode” (enhancement enabled).

Operational recommendation:

- default to conditional OCR enhancement (triggered by weak text layer / poor scan metrics),
- measure end-to-end throughput and review-rate together; accept moderate preprocessing delay if manual review drops significantly.
