"""Conservative offline corroboration of uncertain English TrOCR readings.

Never sees question text or answer keys. Confidence describes detected text,
not coverage; acceptance also requires whole-answer agreement with TrOCR.
"""
import math
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
from functools import lru_cache

import cv2
import numpy as np
from core.contracts import Transcription, ReadStatus

ENGINE = 'paddle-trocr-agreement-v1'
MODELS = ('PP-OCRv5_mobile_det', 'en_PP-OCRv5_mobile_rec')
THRESHOLD = .95
ELIGIBLE_REASONS = {'ocr-low-confidence', 'ocr-line-segmentation-uncertain'}


def model_directory():
    return Path(os.environ.get('AUTOMARK_PADDLE_MODEL_DIR',
        str(Path.home() / '.cache' / 'imarisha-automark' / 'paddleocr')))


def normalized_words(text):
    # Do not repair spelling or join words. Keep negation and numbers.
    return re.findall(r'\w+', unicodedata.normalize('NFC', text or '').casefold())


def eligible(draft, language):
    reasons = set(draft.review_reasons)
    return (language == 'en' and draft.engine == 'trocr-written-v1'
            and draft.status == ReadStatus.UNCERTAIN
            and bool(normalized_words(draft.text)) and bool(reasons)
            and reasons <= ELIGIBLE_REASONS)


class PaddleBackend:
    def __init__(self, directory):
        directory = Path(directory)
        for model in MODELS:
            for name in ('inference.json', 'inference.pdiparams', 'inference.yml'):
                if not (directory / model / name).is_file():
                    raise OSError('PaddleOCR weights are not prepared')
        # Explicit local weights only; runtime must not contact model registries.
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(directory / 'runtime-cache'))
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(text_detection_model_name=MODELS[0],
            text_recognition_model_name=MODELS[1],
            text_detection_model_dir=str(directory / MODELS[0]),
            text_recognition_model_dir=str(directory / MODELS[1]),
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, device='cpu', cpu_threads=4,
            text_rec_score_thresh=0.0)
        self.lock = threading.Lock()

    def read(self, image):
        with self.lock:
            results = list(self.model.predict(image))
        if len(results) != 1:
            raise ValueError('Expected one result per answer crop')
        data = results[0].json['res']
        return data['rec_texts'], data['rec_scores']


@lru_cache(maxsize=1)
def _backend(directory):
    return PaddleBackend(directory)


class PaddleAgreementOCR:
    def __init__(self, backend=None):
        self.backend = backend
        self.unavailable = False

    def corroborate(self, request, original):
        if not eligible(original, request.language):
            return None
        start = time.perf_counter()
        confidence = None
        try:
            if self.unavailable:
                raise OSError('PaddleOCR unavailable in this job')
            image = cv2.imdecode(np.frombuffer(request.image_png, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] * image.shape[1] > 20_000_000:
                raise ValueError('Invalid or oversized crop')
            backend = self.backend or _backend(str(model_directory()))
            texts, scores = backend.read(image)
            if len(texts) != len(scores):
                raise ValueError('Mismatched recognition results')
            scores = [float(s) for s in scores]
            if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                raise ValueError('Invalid confidence')
            confidence = min(scores, default=0) * 100
            if not texts or any(not t.strip() for t in texts):
                reason = 'paddle-no-text'
            elif min(scores) < THRESHOLD:
                reason = 'paddle-low-confidence'
            elif normalized_words('\n'.join(texts)) != normalized_words(original.text):
                reason = 'paddle-trocr-disagreement'
            else:
                # Preserve original spelling/punctuation and TrOCR history;
                # Paddle only corroborates it, never silently rewrites it.
                return Transcription(original.text, ReadStatus.READ, (), ENGINE,
                    time.perf_counter()-start, model_digest='+'.join(MODELS), confidence=confidence)
        except (ImportError, OSError):
            self.unavailable = True
            reason = 'paddle-unavailable'
        except Exception:
            reason = 'paddle-failed'
        return Transcription(original.text, ReadStatus.UNCERTAIN,
            tuple(dict.fromkeys(original.review_reasons + (reason,))), ENGINE,
            time.perf_counter()-start, model_digest='+'.join(MODELS), confidence=confidence)
