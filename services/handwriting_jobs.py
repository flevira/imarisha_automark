"""Recognition on retained crops, independent of rescanning and marking."""
from dataclasses import replace
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

from core.contracts import RecognitionRequest, ResponseType, Transcription, ReadStatus
from core.paddle_text_ocr import ENGINE as PADDLE_ENGINE, PaddleAgreementOCR


def _eligible(row, recognizer, language):
    raw = row['response']
    if row['review'] or raw.response_type != ResponseType.WRITTEN or 'handwriting-recognition-pending' not in raw.review_reasons:
        return False
    old = row.get('transcription')
    if old and old.get('engine')=='image-inspection-repair-v1':return False
    if old and old['engine'] == 'tesseract-written-v1':
        warning_only = (old.get('text') and old.get('language') == language and language != 'sw'
            and set(old.get('review_reasons', [])) == {'ocr-language-pack-missing'}
            and (old.get('confidence') or 0) >= 80)
        if old['status'] == 'read' or warning_only:
            return False
    if old and old['engine'] in ('trocr-written-v1', PADDLE_ENGINE) and old.get('language') == language and old['status'] == 'read':
        return False
    engine = 'ollama:' + getattr(recognizer, 'model', '')
    return not (old and old['status'] != 'error' and old['engine'] == engine
        and old['language'] == language and 'ai-failed-ocr-text-retained' not in old.get('review_reasons', []))


def _transcribe(recognizer, request, old):
    start = time.perf_counter()
    try:
        draft = recognizer.recognize(request)
    except Exception:
        draft = Transcription(None, ReadStatus.ERROR, ('recognizer-failed',), 'recognizer-error')
    failed = draft.status == ReadStatus.ERROR
    if failed and old and old.get('text'):
        draft = replace(draft, text=old['text'], status=ReadStatus.UNCERTAIN,
            review_reasons=draft.review_reasons + ('ai-failed-ocr-text-retained',))
    draft = replace(draft, status=ReadStatus.ERROR if draft.status == ReadStatus.ERROR else ReadStatus.UNCERTAIN,
        review_reasons=tuple(dict.fromkeys(draft.review_reasons + ('handwriting-draft-needs-review',))),
        seconds=time.perf_counter()-start)
    return draft, failed


def recognize_job(repository, job, recognizer, language='en-sw', cancel=None, workers=1, progress=None):
    """Bounded page lanes within one student; only the coordinator touches SQLite.

    At most one request per page and two requests total. Cancellation drains and
    saves already-started requests without submitting more. No answer keys enter
    requests. Elapsed wall time is distinct from overlapping request durations.
    """
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('Choose one or two AI transcription pages')
    cancel = cancel or threading.Event()
    started = time.perf_counter()
    state = dict(total=0, completed=0, errors=0, workers=workers, active_pages=[],
                 student=None, wall_seconds=0., request_seconds=0., reduced_to_serial=False)
    def emit():
        state['wall_seconds'] = time.perf_counter()-started
        if progress:
            progress(dict(state))
    with repository.lock(job):
        info = repository.job(job)
        if info['status'] == 'complete':
            return state
        if any(p['status'] == 'pending' for p in info['pages']):
            raise ValueError('Finish page capture before handwriting recognition')
        groups = {}
        for row in repository.responses(job):
            if not _eligible(row, recognizer, language):
                continue
            raw = row['response']
            # Unknown identities do not share lanes across different pages.
            key = (raw.layout.layout_id, raw.student_id or ('unknown', row['page']))
            groups.setdefault(key, {}).setdefault(row['page'], deque()).append(row)
            state['total'] += 1
        repository.status(job, 'recognizing')
        limit = workers
        try:
            emit()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='automark-handwriting') as pool:
                for (_, student), pages in groups.items():
                    if cancel.is_set():
                        break
                    state['student'] = student if isinstance(student, str) else None
                    pending = {}
                    while pending or any(pages.values()):
                        active = {page for page, _ in pending.values()}
                        for page, rows in pages.items():
                            if cancel.is_set() or len(pending) >= limit:
                                break
                            if not rows or page in active:
                                continue
                            row = rows.popleft()
                            raw = row['response']; region = raw.regions[0]
                            request = RecognitionRequest(region, repository.crop(job, region.crop_ref), language)
                            # Check again after crop loading; do not prequeue a batch.
                            if cancel.is_set():
                                break
                            future = pool.submit(_transcribe, recognizer, request, row.get('transcription'))
                            pending[future] = (page, row)
                            active.add(page)
                        state['active_pages'] = sorted(page for page, _ in pending.values())
                        emit()
                        if not pending:
                            break
                        ready, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in ready:
                            _, row = pending.pop(future)
                            draft, failed = future.result()
                            repository.save_transcription(job, row['response'].response_id, draft, language)
                            state['completed'] += 1
                            state['errors'] += int(failed)
                            state['request_seconds'] += draft.seconds
                            # Lower pressure for the remainder of this run after a failure.
                            if failed and limit > 1:
                                limit = 1
                                state['reduced_to_serial'] = True
                        state['active_pages'] = sorted(page for page, _ in pending.values())
                        emit()
            repository.status(job, 'recognition-cancelled' if cancel.is_set() else 'review')
            emit()
            return dict(state)
        except BaseException:
            repository.status(job, 'interrupted')
            raise


def ocr_job(repository,job,language='en',cancel=None):
    """Resume OCR on saved crops, without rescanning or changing approvals."""
    from core.local_text_ocr import LocalTextOCR
    cancel=cancel or threading.Event();recognizer=LocalTextOCR();paddle=PaddleAgreementOCR()
    with repository.lock(job):
        if repository.job(job)['status']=='complete':return
        repository.status(job,'ocr')
        try:
            for row in repository.responses(job):
                if cancel.is_set():break
                raw=row['response'];old=row.get('transcription')
                if old and old.get('engine')=='image-inspection-repair-v1':continue
                if row['review'] or raw.response_type!=ResponseType.WRITTEN or raw.raw_text or raw.status==ReadStatus.BLANK:continue
                reuse = old and old['status'] != 'error' and old['language'] == language
                if reuse and old['engine'].startswith('ollama:'):continue
                if reuse and old['engine'] == PADDLE_ENGINE:
                    if not {'paddle-unavailable','paddle-failed'} & set(old['review_reasons']):continue
                    # Retry only the unavailable/failed corroboration stage.
                    draft=Transcription(old.get('text'),ReadStatus.UNCERTAIN,
                        tuple(r for r in old['review_reasons'] if not r.startswith('paddle-')),
                        'trocr-written-v1')
                elif reuse and old['engine']=='trocr-written-v1':
                    draft=Transcription(old.get('text'),ReadStatus(old['status']),
                        tuple(old['review_reasons']),old['engine'])
                    if draft.status==ReadStatus.READ:continue
                else:
                    draft=None
                region=raw.regions[0]
                request=RecognitionRequest(region,repository.crop(job,region.crop_ref),language)
                if draft is None:
                    draft=recognizer.recognize(request)
                    repository.save_transcription(job,raw.response_id,draft,language)
                if cancel.is_set():break
                corroborated=paddle.corroborate(request,draft)
                if corroborated is not None:
                    repository.save_transcription(job,raw.response_id,corroborated,language)
            repository.status(job,'recognition-cancelled' if cancel.is_set() else 'review')
        except BaseException:
            repository.status(job,'interrupted');raise
