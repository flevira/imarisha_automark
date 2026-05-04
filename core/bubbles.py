from __future__ import annotations

from typing import Any

import cv2
import fitz
import numpy as np

from core.confidence_model import BubbleConfidenceModel

MATCH_LEFT_LABELS = {1: "A", 2: "B", 3: "C", 4: "D"}
BUBBLE_MODEL = BubbleConfidenceModel()


def page_to_image_scale(page: fitz.Page, img_bgr: np.ndarray) -> tuple[float, float]:
    sx = img_bgr.shape[1] / page.rect.width
    sy = img_bgr.shape[0] / page.rect.height
    return sx, sy


def sample_bubble_score(gray: np.ndarray, cx: int, cy: int, radius: int) -> float:
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(gray.shape[1], cx + radius)
    y1 = min(gray.shape[0], cy + radius)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    rr = min(roi.shape[0], roi.shape[1]) / 2.2
    cy0 = roi.shape[0] / 2.0
    cx0 = roi.shape[1] / 2.0
    mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= rr**2

    pixels = roi[mask]
    if pixels.size == 0:
        pixels = roi.flatten()

    return 255.0 - float(np.mean(pixels))


def _bubble_composite_metrics(
    img_bgr: np.ndarray,
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> dict[str, float]:
    """Compute multi-signal confidence metrics for one bubble center."""
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(gray.shape[1], cx + radius)
    y1 = min(gray.shape[0], cy + radius)
    if x1 <= x0 or y1 <= y0:
        return {"fill": 0.0, "stroke": 0.0, "blue": 0.0}

    roi_gray = gray[y0:y1, x0:x1]
    roi_bgr = img_bgr[y0:y1, x0:x1]
    if roi_gray.size == 0:
        return {"fill": 0.0, "stroke": 0.0, "blue": 0.0}

    h, w = roi_gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx0 = w / 2.0
    cy0 = h / 2.0
    r_outer = min(h, w) / 2.2
    r_center = max(2.0, r_outer * 0.55)
    outer_mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r_outer**2
    center_mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r_center**2

    if not np.any(outer_mask):
        return {"fill": 0.0, "stroke": 0.0, "blue": 0.0}

    bubble_pixels = roi_gray[outer_mask]
    # Fill ratio in circle (adaptive local Otsu inside bubble ROI)
    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = float(np.mean(inv[outer_mask] > 0))

    # Stroke density near center to emphasize actual pen marks
    gradx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gradx, grady)
    center_grad = grad_mag[center_mask] if np.any(center_mask) else grad_mag[outer_mask]
    stroke_density = float(np.mean(center_grad > np.percentile(grad_mag, 75)))

    # Blue-ink emphasis: higher when B channel exceeds R/G
    b = roi_bgr[:, :, 0].astype(np.float32)
    g = roi_bgr[:, :, 1].astype(np.float32)
    r = roi_bgr[:, :, 2].astype(np.float32)
    blue_excess = np.clip(b - ((r + g) * 0.5), 0, 255)
    blue_conf = float(np.mean(blue_excess[outer_mask]) / 255.0)

    darkness = float((255.0 - np.mean(bubble_pixels)) / 255.0)
    return {
        "fill": fill_ratio,
        "stroke": stroke_density,
        "blue": blue_conf,
        "darkness": darkness,
    }


def sample_bubble_score_with_center_refinement(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    *,
    search_px: int,
    step_px: int = 2,
) -> tuple[float, int, int]:
    """Sample bubble darkness around (cx, cy), allowing small center drift.

    Scanned sheets frequently have tiny row/header OCR offsets. A fixed center can
    miss the filled region and make real answers look blank. We sample a compact
    neighborhood and keep the darkest response.
    """
    best_score = -1.0
    best_cx = cx
    best_cy = cy

    x_candidates = list(range(cx - search_px, cx + search_px + 1, step_px))
    y_candidates = list(range(cy - search_px, cy + search_px + 1, step_px))

    for yc in y_candidates:
        if yc < 0 or yc >= gray.shape[0]:
            continue
        for xc in x_candidates:
            if xc < 0 or xc >= gray.shape[1]:
                continue
            score = sample_bubble_score(gray, xc, yc, radius)
            if score > best_score:
                best_score = score
                best_cx = xc
                best_cy = yc

    if best_score < 0:
        return 0.0, cx, cy
    return best_score, best_cx, best_cy


def _is_intentional_multi_mark(
    ordered_labels: list[str],
    scores: dict[str, float],
    probabilities: dict[str, float],
    filled_labels: list[str],
    *,
    similarity_threshold: float = 0.10,
    dominance_threshold: float = 0.30,
) -> bool:
    """Determine whether multiple filled bubbles represent a genuine multi-mark.

    Returns True only when all filled answers have similar darkness scores
    (within ``similarity_threshold`` of the maximum score), meaning the student
    deliberately filled more than one bubble.

    Returns False when one answer is clearly dominant — its score exceeds every
    other filled answer by more than ``dominance_threshold`` of the max score —
    indicating the secondary marks are noise or accidental smudges.

    Args:
        ordered_labels: All option labels for this row (e.g. [\"A\",\"B\",\"C\",\"D\"]).
        scores: Raw darkness scores keyed by label.
        probabilities: Model confidence probabilities keyed by label.
        filled_labels: Labels the model classified as filled.
        similarity_threshold: Max fractional score gap allowed for marks to be
            considered \"similar\" (default 10 %).
        dominance_threshold: Minimum fractional score gap required for one mark
            to be considered \"clearly dominant\" (default 30 %).
    """

    if len(filled_labels) <= 1:
        return False  # Not a multi-mark situation

    filled_scores = [scores.get(label, 0.0) for label in filled_labels]
    if not filled_scores:
        return False

    max_score = max(filled_scores)
    if max_score <= 0:
        return False

    # Check whether any secondary answer is clearly weaker than the strongest.
    for label in filled_labels:
        s = scores.get(label, 0.0)
        gap_fraction = (max_score - s) / max_score
        if gap_fraction > dominance_threshold:
            # One answer is clearly dominant — this is NOT an intentional multi-mark.
            return False

    # All filled answers are within similarity_threshold of the max — intentional.
    for label in filled_labels:
        s = scores.get(label, 0.0)
        gap_fraction = (max_score - s) / max_score
        if gap_fraction > similarity_threshold:
            return False

    return True


def extract_row_answers(
    page: fitz.Page,
    img_bgr: np.ndarray,
    section: dict[str, Any],
    row: dict[str, Any],
    option_centers: dict[str, float],
) -> tuple[list[str], bool, dict[str, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx, sy = page_to_image_scale(page, img_bgr)

    if section["section_type"] == "MC":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    elif section["section_type"] == "TF":
        ordered_labels = ["T", "F"]
        output_map = {"T": "True", "F": "False"}
    elif section["section_type"] == "MATCH":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    else:
        return [], True, {}

    row_h_page = max(10.0, row["row_bbox"][3] - row["row_bbox"][1])
    radius = max(10, int((row_h_page * sy) * 0.38))
    cy = int(row["y_center"] * sy)

    scores: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    center_search_px = max(8, int(radius * 0.85))
    for label in ordered_labels:
        if label not in option_centers:
            scores[label] = 0.0
            continue
        cx = int(option_centers[label] * sx)
        score, best_cx, best_cy = sample_bubble_score_with_center_refinement(
            gray,
            cx,
            cy,
            radius,
            search_px=center_search_px,
            step_px=2,
        )
        m = _bubble_composite_metrics(img_bgr, gray, best_cx, best_cy, radius)
        # Composite score reduces strict dependence on fill darkness alone.
        composite = (
            0.45 * m.get("darkness", 0.0)
            + 0.30 * m.get("fill", 0.0)
            + 0.15 * m.get("stroke", 0.0)
            + 0.10 * m.get("blue", 0.0)
        )
        metrics[label] = m
        scores[label] = (score * 0.5) + (composite * 100.0)

    if not scores:
        return [], True, scores

    # Strict blank detection: treat the row as unanswered when all bubble scores
    # are uniformly low (student left the question blank).  Two independent
    # criteria are checked on the raw (pre-calibration) scores:
    #   1. Every score is below an absolute low-mark threshold (30).
    #   2. The spread across all options is tiny (< 5 pts) AND the maximum is
    #      below a weak-signal ceiling (50) — a flat profile with no standout.
    # Either condition alone is sufficient to declare the row blank.
    _raw_score_vals = [scores[lbl] for lbl in ordered_labels if lbl in scores]
    if _raw_score_vals:
        _raw_max = max(_raw_score_vals)
        _raw_spread = _raw_max - min(_raw_score_vals)
        _all_below_floor = _raw_max < 30.0
        _flat_and_weak = _raw_spread < 5.0 and _raw_max < 50.0
        if _all_below_floor or _flat_and_weak:
            return [], False, {f"score_{k}": v for k, v in scores.items()}

    # Row-local calibration pass: use likely empty bubbles as local baseline.
    ordered_score_vals = np.array([scores[label] for label in ordered_labels if label in scores], dtype=np.float32)
    baseline = float(np.percentile(ordered_score_vals, 35)) if ordered_score_vals.size else 0.0
    calibrated_scores = {k: max(0.0, v - baseline) for k, v in scores.items()}

    # Adaptive thresholding per section block type (only when model config is available).
    section_positive_offsets = {"MC": 0.00, "TF": -0.04, "MATCH": -0.02}
    base_positive = float(getattr(getattr(BUBBLE_MODEL, "config", object()), "positive_threshold", 0.55))
    adaptive_positive = max(0.45, min(0.8, base_positive + section_positive_offsets.get(section["section_type"], 0.0)))

    # keep original classifier behavior for ambiguity handling with calibrated scores
    filled_labels, model_review2, probabilities = BUBBLE_MODEL.classify(
        ordered_labels=ordered_labels,
        scores=calibrated_scores,
        allow_multiple_answers=(section["section_type"] == "MC"),
    )
    model_review = bool(model_review2)

    # Optional adaptive threshold override after model classification.
    if probabilities:
        filled_labels = [label for label in ordered_labels if probabilities.get(label, 0.0) >= adaptive_positive]

    # MC single-answer enforcement: MC questions must always have exactly one
    # answer. If multiple options passed the adaptive threshold, keep only the
    # highest-probability one and flag for review when it falls in the
    # ambiguous confidence range (0.45–0.65).
    if section["section_type"] == "MC" and len(filled_labels) > 1:
        best_label = max(filled_labels, key=lambda lbl: probabilities.get(lbl, 0.0))
        filled_labels = [best_label]
        best_prob = probabilities.get(best_label, 0.0)
        model_review = 0.45 <= best_prob <= 0.65


    needs_review = model_review

    if section["section_type"] == "TF" and len(filled_labels) != 1:
        # TF questions: handle both-marked or neither-marked cases intelligently.
        score_t = calibrated_scores.get("T", 0.0)
        score_f = calibrated_scores.get("F", 0.0)

        # If both are truly blank (both < 30 calibrated), return unanswered.
        if score_t < 30.0 and score_f < 30.0:
            return [], False, {f"score_{k}": v for k, v in scores.items()}

        # If both are marked, pick the stronger one.
        if len(filled_labels) == 2:  # Both T and F detected
            if score_t > score_f:
                filled_labels = ["T"]
                # Flag for review if they're too close (ambiguous, within 20%).
                if (score_t - score_f) < (score_t * 0.20):
                    needs_review = True
            else:
                filled_labels = ["F"]
                # Flag for review if they're too close (ambiguous, within 20%).
                if (score_f - score_t) < (score_f * 0.20):
                    needs_review = True
        # If neither marked, return unanswered.
        elif len(filled_labels) == 0:
            return [], False, {f"score_{k}": v for k, v in scores.items()}

    positive_threshold = float(getattr(BUBBLE_MODEL.config, "positive_threshold", 0.55))
    strong_threshold = float(getattr(BUBBLE_MODEL.config, "strong_threshold", 0.78))
    uncertainty_band = float(getattr(BUBBLE_MODEL.config, "uncertainty_band", 0.08))

    if section["section_type"] == "MC" and not filled_labels and probabilities:
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_label, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            top_label in output_map
            and top_prob >= (positive_threshold - uncertainty_band)
            and (top_prob - second_prob) >= (uncertainty_band * 0.5)
        ):
            answers = [output_map[top_label]]
            debug_scores = {f"score_{key}": value for key, value in scores.items()}
            debug_scores.update({f"prob_{key}": value for key, value in probabilities.items()})
            return answers, True, debug_scores

    if section["section_type"] == "MATCH" and len(filled_labels) != 1:
        if probabilities:
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            top_label, top_prob = ranked[0]
            second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
            score_values = sorted(scores.values(), reverse=True)
            score_gap = score_values[0] - (score_values[1] if len(score_values) > 1 else 0.0)

            has_clear_winner = (
                top_label in output_map
                and top_prob >= max(positive_threshold, strong_threshold - uncertainty_band)
                and (
                    (top_prob - second_prob) >= (uncertainty_band * 0.5)
                    or score_gap >= 2.0
                )
            )
            if has_clear_winner:
                answers = [output_map[top_label]]
                debug_scores = {f"score_{key}": value for key, value in scores.items()}
                debug_scores.update({f"prob_{key}": value for key, value in probabilities.items()})
                return answers, True, debug_scores
        return [], False, {}

    answers = [output_map[label] for label in filled_labels if label in output_map]
    if not answers:
        return [], needs_review, {}

    debug_scores = {f"score_{key}": value for key, value in calibrated_scores.items()}
    debug_scores.update({f"prob_{key}": value for key, value in probabilities.items()})
    debug_scores.update({f"baseline_{section['section_type']}": baseline, "adaptive_positive_threshold": adaptive_positive})
    return answers, needs_review, debug_scores


def build_matching_pair(item_number: int | None, answer: str | None) -> str | None:
    if item_number is None or answer is None:
        return None
    left = MATCH_LEFT_LABELS.get(item_number)
    if not left:
        return None
    return f"{left}={answer}"


def append_output_rows(
    output_rows: list[dict[str, Any]],
    base: dict[str, Any],
    row: dict[str, Any],
    answers: list[str],
    needs_review: bool,
    scores: dict[str, float],
) -> None:
    if row["section_type"] == "OPEN_ENDED":
        output_rows.append(
            {
                **base,
                "section": "OPEN_ENDED",
                "question_id": row["question_id"],
                "item_number": None,
                "answer": None,
                "answered": "No",
                "matching_pair": None,
                "manual_marking": True,
                "needs_review": True,
                "score_debug": None,
            }
        )
        return

    if not answers:
        output_rows.append(
            {
                **base,
                "section": row["section_type"],
                "question_id": row["question_id"],
                "item_number": row["item_number"],
                "answer": None,
                "answered": "No",
                "matching_pair": None,
                "manual_marking": False,
                "needs_review": bool(needs_review),
                "score_debug": None,
            }
        )
        return

    for ans in answers:
        output_rows.append(
            {
                **base,
                "section": row["section_type"],
                "question_id": row["question_id"],
                "item_number": row["item_number"],
                "answer": ans,
                "answered": "Yes",
                "matching_pair": build_matching_pair(row["item_number"], ans)
                if row["section_type"] == "MATCH"
                else None,
                "manual_marking": False,
                "needs_review": bool(needs_review),
                "score_debug": str(scores),
            }
        )
