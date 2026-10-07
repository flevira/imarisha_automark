"""Offline, line-based TrOCR for written regions; never receives answer keys.

OpenCV segments the existing crops. Weights must be downloaded explicitly with
``python -m tools.prepare_trocr``; extraction never downloads models.
"""
import os
import threading
import time
from functools import lru_cache

import cv2
import numpy as np
from core.contracts import Transcription, ReadStatus

MODEL_ID = 'microsoft/trocr-base-handwritten'
ENGINE = 'trocr-written-v1'
MAX_LINES = 40


def split_lines(image):
    """Return top-to-bottom line images and conservative segmentation flags.

    Remove long rules from the segmentation mask, retain punctuation, and merge
    nearby horizontal ink bands (dots/accents). Ambiguous tall bands are flagged
    rather than silently considered reliable single lines.
    """
    _, ink = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, w = ink.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 2), 1)))
    ink = cv2.subtract(ink, rules)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    # Ignore isolated scan speckles for line finding (keep original crop pixels).
    tiny = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < 3)
    ink[np.isin(labels, tiny)] = 0
    components = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 3]
    if not components:
        return [], ()
    heights = [s[cv2.CC_STAT_HEIGHT] for s in components if s[cv2.CC_STAT_HEIGHT] >= 4]
    char_height = float(np.median(heights)) if heights else 4.
    occupied = np.flatnonzero(np.count_nonzero(ink, axis=1))
    bands = []
    for y in occupied:
        if bands and y - bands[-1][1] <= max(2, int(char_height * .3)):
            bands[-1][1] = int(y)
        else:
            bands.append([int(y), int(y)])
    flags = []
    lines = []
    clean = image.copy()
    clean[rules != 0] = 255
    for top, bottom in bands:
        xs = np.flatnonzero(np.count_nonzero(ink[top:bottom + 1], axis=0))
        if not len(xs):
            continue
        if bottom - top + 1 > char_height * 2.2:
            flags.append('ocr-line-segmentation-uncertain')
        # Do not clip neighboring lines or resize the whole multiline answer.
        crop = clean[top:bottom + 1, xs[0]:xs[-1] + 1]
        lines.append(cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255))
    if len(lines) > MAX_LINES:
        raise ValueError('ocr-too-many-lines')
    return lines, tuple(dict.fromkeys(flags))


class TrOCRBackend:
    def __init__(self, source):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained(source, local_files_only=True, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=True,
                                                               use_safetensors=True)
        self.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    def read_line(self, image):
        from PIL import Image
        with self.lock, self.torch.inference_mode():
            pixels = self.processor(images=Image.fromarray(image).convert('RGB'),
                                    return_tensors='pt').pixel_values.to(self.device)
            output = self.model.generate(pixels, max_new_tokens=128, do_sample=False,
                                         num_beams=1, max_time=20., return_dict_in_generate=True, output_scores=True)
            ids = output.sequences[0, 1:]
            eos = self.model.generation_config.eos_token_id
            if isinstance(eos, int):
                eos = [eos]
            probabilities = []
            finished = False
            for token, logits in zip(ids, output.scores):
                if int(token) in (eos or []):
                    finished = True
                    break
                probabilities.append(float(logits[0].softmax(-1)[token].item()))
            text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
            # A heuristic, NOT a calibrated probability of transcription accuracy.
            confidence = float(np.exp(np.mean(np.log(np.maximum(probabilities, 1e-12))))) if probabilities else 0.
            return text, confidence, min(probabilities, default=0.), finished


@lru_cache(maxsize=1)
def _backend(source):
    # Reuse model weights across all students; serialize inference on this model.
    return TrOCRBackend(source)


class LocalTextOCR:
    def __init__(self, backend=None):
        self.backend = backend

    def recognize(self, request):
        start = time.perf_counter()
        texts = []
        try:
            image = cv2.imdecode(np.frombuffer(request.image_png, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('ocr-invalid-image')
            if image.size > 20_000_000:
                raise ValueError('ocr-region-too-large')
            lines, flags = split_lines(image)
            reasons = list(flags)
            # IAM English weights do not establish Kiswahili accuracy, even for
            # apparently confident Latin-script output.
            if request.language != 'en':
                reasons.append('ocr-language-unvalidated')
            if not lines:
                return Transcription(None, ReadStatus.UNCERTAIN, ('ocr-no-text',), ENGINE,
                                     time.perf_counter() - start, confidence=0.)
            backend = self.backend or _backend(os.environ.get('AUTOMARK_TROCR_MODEL', MODEL_ID))
            confidence = []
            for line in lines:
                text, average, lowest, finished = backend.read_line(line)
                texts.append(text)
                confidence.append(average)
                if not text:
                    reasons.append('ocr-no-text')
                if average < .90 or lowest < .50:
                    reasons.append('ocr-low-confidence')
                if not finished:
                    reasons.append('ocr-incomplete-recognition')
            return Transcription('\n'.join(texts) or None,
                ReadStatus.UNCERTAIN if reasons else ReadStatus.READ,
                tuple(dict.fromkeys(reasons)), ENGINE, time.perf_counter() - start,
                confidence=min(confidence) * 100.)
        except Exception as exc:
            reason = 'ocr-unavailable' if isinstance(exc, (ImportError, OSError)) else 'ocr-failed'
            if isinstance(exc, ValueError) and str(exc).startswith('ocr-'):
                reason = str(exc)
            return Transcription('\n'.join(texts) or None, ReadStatus.ERROR, (reason,), ENGINE,
                                 time.perf_counter() - start)
