from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BubbleModelConfig:
    bias: float
    w_score: float
    w_z: float
    w_rel_to_max: float
    w_rel_to_mean: float
    positive_threshold: float
    strong_threshold: float
    uncertainty_band: float
    blank_score_spread_abs_max: float
    blank_score_spread_rel_max: float


DEFAULT_MODEL = BubbleModelConfig(
    bias=-1.05,
    w_score=0.022,
    w_z=1.45,
    w_rel_to_max=1.25,
    w_rel_to_mean=0.85,
    positive_threshold=0.60,
    strong_threshold=0.78,
    uncertainty_band=0.06,
    blank_score_spread_abs_max=3.0,
    blank_score_spread_rel_max=0.08,
)

MODEL_CONFIG_PATH = Path("core/model_params/bubble_confidence.json")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_model_config() -> BubbleModelConfig:
    if not MODEL_CONFIG_PATH.exists():
        return DEFAULT_MODEL

    try:
        raw = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_MODEL

    merged: dict[str, Any] = {
        "bias": DEFAULT_MODEL.bias,
        "w_score": DEFAULT_MODEL.w_score,
        "w_z": DEFAULT_MODEL.w_z,
        "w_rel_to_max": DEFAULT_MODEL.w_rel_to_max,
        "w_rel_to_mean": DEFAULT_MODEL.w_rel_to_mean,
        "positive_threshold": DEFAULT_MODEL.positive_threshold,
        "strong_threshold": DEFAULT_MODEL.strong_threshold,
        "uncertainty_band": DEFAULT_MODEL.uncertainty_band,
        "blank_score_spread_abs_max": DEFAULT_MODEL.blank_score_spread_abs_max,
        "blank_score_spread_rel_max": DEFAULT_MODEL.blank_score_spread_rel_max,
    }
    for key in merged:
        if key in raw:
            try:
                merged[key] = float(raw[key])
            except Exception:
                pass
    return BubbleModelConfig(**merged)


class BubbleConfidenceModel:
    def __init__(self) -> None:
        self.config = _load_model_config()

    def label_probabilities(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        labels = list(scores.keys())
        values = np.array([float(scores[label]) for label in labels], dtype=np.float32)
        mean = float(np.mean(values))
        std = float(np.std(values))
        std = max(std, 1e-6)
        vmax = float(np.max(values))
        vmax = max(vmax, 1e-6)

        probs: dict[str, float] = {}
        for label in labels:
            score = float(scores[label])
            z = (score - mean) / std
            rel_to_max = score / vmax
            rel_to_mean = score / max(abs(mean), 1e-6)

            logit = (
                self.config.bias
                + (self.config.w_score * score)
                + (self.config.w_z * z)
                + (self.config.w_rel_to_max * rel_to_max)
                + (self.config.w_rel_to_mean * rel_to_mean)
            )
            probs[label] = _sigmoid(logit)
        return probs

    def classify(
        self,
        ordered_labels: list[str],
        scores: dict[str, float],
        *,
        allow_multiple_answers: bool = False,
    ) -> tuple[list[str], bool, dict[str, float]]:
        probs = self.label_probabilities(scores)
        if not probs:
            return [], True, {}

        threshold = self.config.positive_threshold
        positives = [label for label in ordered_labels if probs.get(label, 0.0) >= threshold]
        positives = [label for label in positives if label in probs]

        top_label = max(probs, key=probs.get)
        top_prob = probs[top_label]
        sorted_probs = sorted(probs.values(), reverse=True)
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        ordered_scores = [float(scores[label]) for label in ordered_labels if label in scores]
        if ordered_scores:
            score_max = max(ordered_scores)
            score_min = min(ordered_scores)
            score_spread = score_max - score_min
            rel_spread = score_spread / max(score_max, 1e-6)
            is_flat_score_profile = (
                score_spread <= self.config.blank_score_spread_abs_max
                and rel_spread <= self.config.blank_score_spread_rel_max
            )
            if (
                is_flat_score_profile
                and abs(top_prob - second_prob) <= self.config.uncertainty_band
            ):
                return [], False, probs

        needs_review = False
        if not positives:
            # Treat rows as intentionally unanswered when every option remains
            # clearly below the positive threshold and none stands out.
            blank_cutoff = threshold - self.config.uncertainty_band
            is_likely_blank = (
                top_prob < blank_cutoff
                and abs(top_prob - second_prob) <= self.config.uncertainty_band
            )
            if is_likely_blank:
                return [], False, probs
            needs_review = True

        has_uncertain_top_gap = abs(top_prob - second_prob) <= self.config.uncertainty_band
        if has_uncertain_top_gap and not (allow_multiple_answers and len(positives) > 1):
            needs_review = True

        if (
            not allow_multiple_answers
            and len(positives) > 1
            and (top_prob - second_prob) <= (self.config.uncertainty_band * 0.7)
        ):
            needs_review = True

        return positives, needs_review, probs
