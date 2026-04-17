# Quantitative thresholds currently configured vs typical industry values

This document captures hard-coded numeric thresholds and setup parameters used in the current automarking pipeline, then compares each to commonly used ranges in OCR/OMR production systems.

> Note: “Industry values” below are practical ranges commonly used in document AI/OCR/OMR deployments. They are reference bands, not strict standards.

## 1) Bubble confidence model (OMR classification)

| Parameter | Current value | Typical industry range | Assessment |
|---|---:|---:|---|
| `bias` | -1.05 | -2.0 to 0.0 | In-range; moderately conservative baseline. |
| `w_score` | 0.022 | 0.005 to 0.05 | In-range; light raw-intensity influence. |
| `w_z` | 1.45 | 0.8 to 2.5 | In-range; strong relative-separation weighting. |
| `w_rel_to_max` | 1.25 | 0.5 to 1.8 | In-range. |
| `w_rel_to_mean` | 0.85 | 0.3 to 1.5 | In-range. |
| `positive_threshold` | 0.60 | 0.55 to 0.75 | In-range; slightly quality-first. |
| `strong_threshold` | 0.78 | 0.75 to 0.90 | In-range; matches common auto-accept gates. |
| `uncertainty_band` | 0.06 | 0.05 to 0.15 | In-range; moderate ambiguity capture. |

## 2) OCR confidence gates and token constraints

| Use case | Current setup | Typical industry range | Assessment |
|---|---|---|---|
| Generic primary OCR gate | Legacy param `min_tesseract_confidence` = 55.0 (applies to Paddle+Tesseract pool) | 50–70 | In-range; balanced precision/recall. |
| Question ID (5–6 digits) | min confidence = 58.0, token length 5–6 | 55–75, fixed length preferred | In-range; reasonable for numeric IDs. |
| Matching item number | min confidence = 50.0, token length = 1, pattern `[1-4]` | 45–65 for single-char fields | In-range; recall-friendly for tiny symbols. |
| Handwritten student ID | min confidence = 45.0, low-conf flag at `<55` | 35–60 (handwritten OCR), review near 50–70 | In-range; appropriately permissive for handwriting. |
| Multi-variant consistency boost | +6 per repeated hit, capped at +18 | +3 to +20 style bonus patterns | In-range; avoids over-amplification. |
| Engine bias in final ranking | Tesseract +6 (or +10 for numeric), EasyOCR +2, Paddle +0 | 0 to +12 fusion bias | In-range; calibrated to avoid under-ranking numeric Tesseract hits. |

## 3) Page/scan preprocessing and geometry thresholds

| Stage | Current setup | Typical industry range | Assessment |
|---|---|---|---|
| Render zoom | 3.5x | 2.0x–4.0x | In-range; quality-first rendering. |
| Usable text-layer trigger | ≥20 alphanumeric chars | 10–40 chars | In-range; avoids weak OCR-layer trust. |
| Foreground ratio for skew detection | at least 0.5% | 0.2%–2% | In-range. |
| Minimum foreground pixels for skew | at least 300 points | 150–1000 | In-range. |
| Ignore tiny skew | abs(angle) < 0.35° => no rotate | 0.2°–1.0° | In-range, slightly aggressive correction. |
| Skew correction clamp | [-60°, +60°] | ±30° to ±75° | In-range; robust to severe scans. |
| Force scanned mode despite text layer | if estimated skew ≥1.0° | 0.7°–2.0° | In-range; pragmatic reliability switch. |
| Perspective quad minimum area | 35% of image | 20%–60% | In-range; moderate strictness. |
| Canny edges | low/high = 70/180 | 50–100 / 150–250 | In-range. |

## 4) Section/row segmentation thresholds (scanned forms)

| Parameter | Current value | Typical industry range | Assessment |
|---|---|---|---|
| Heading OCR token filter | keep conf ≥20 | 15–40 | In-range; recall-first for structure cues. |
| Fuzzy heading edit distance | max distance = 2 | 1–3 | In-range. |
| Section top offset from heading | +20 px | +5 to +30 px | In-range. |
| Section split margin before next heading | -8 px | -3 to -20 px | In-range. |
| Minimum section height accepted | >10 px | 8–30 px | In-range. |
| Projection smoothing window | odd window from 5 to 21 | 3–31 | In-range. |
| Activity threshold | max(8, 45th percentile) | percentile-based 35th–60th, floor 3–15 | In-range. |
| Minimum active band height | 6 px | 3–12 px | In-range. |
| Row min height | max(10, 1.5% section height) | 1%–3% | In-range. |
| Row max height | max(min+8, 18% section height) | 10%–25% | In-range. |
| Merge gap between adjacent bands | max(3, min_row_h/2) | 2 px to ~row_h/2 | In-range. |

## 5) Image enhancement and binarization parameters

| Stage | Current setup | Typical industry range | Assessment |
|---|---|---|---|
| Blue-ink blend | gray*0.85 + blue_boost*0.15 | 0.05–0.30 ink-channel weighting | In-range. |
| NLM denoise | h=5, template=7, search=21 | h=3–10, template=7, search=21 | In-range. |
| Illumination blur sigma | 21 | 10–35 | In-range. |
| CLAHE | clipLimit=2.2, grid=(8,8) | clip 1.5–4.0, grid 4–16 | In-range. |
| Adaptive threshold | blockSize=31, C=7 (binary inv) | block 11–51, C=2–12 | In-range. |
| Morph cleanup kernel | 2x2 open+close, 1 iter | 2x2 to 3x3, 1–2 iter | In-range. |
| QR CLAHE | clipLimit=3.0 | 2.0–4.0 | In-range. |
| QR adaptive threshold | blockSize=11, C=2 | block 9–31, C=1–10 | In-range. |
| QR upscale attempts | 2.5x full gray, 3.0x crop | 1.5x–4.0x | In-range. |

## 6) Spatial ROI/layout assumptions

| Component | Current setup | Typical industry range | Assessment |
|---|---|---|---|
| Handwritten ID ROI | x=[2%,48%], y=[16%,30%] of page | 10%–50% width, 10%–35% height depending form | In-range (template-specific). |
| QR crop ROI | x=[72%,98%], y=[10%,38%] of page | Corner ROI 15%–35% area bands | In-range (top-right assumption). |
| Option center horizontal search | shifts ±12 px, step 4 | ±5 to ±20 px | In-range. |
| Top zone for option labels | section y0 + 80 px | +40 to +120 px | In-range. |
| Text-line grouping tolerance | 4.0 px | 2–8 px | In-range. |

## Overall interpretation

- Most configured thresholds are inside normal OCR/OMR production bands.
- The system is slightly tuned toward quality (review-safe) rather than maximum throughput (examples: `positive_threshold=0.60`, `strong_threshold=0.78`, scanned-mode fallback at 1.0° skew).
- No parameter appears dramatically out-of-family for modern form-processing pipelines.
