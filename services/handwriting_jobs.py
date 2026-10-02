"""Recognition on retained crops, independent of rescanning and marking."""
from dataclasses import replace
import threading
import time

from core.contracts import RecognitionRequest, ResponseType, Transcription, ReadStatus


def recognize_job(repository,job,recognizer,language='en-sw',cancel=None):
    cancel=cancel or threading.Event()
    with repository.lock(job):
        info=repository.job(job)
        if info['status']=='complete':return
        if any(p['status']=='pending' for p in info['pages']):
            raise ValueError('Finish page capture before handwriting recognition')
        repository.status(job,'recognizing')
        try:
            for row in repository.responses(job):
                if cancel.is_set():break
                raw=row['response']
                if row['review'] or raw.response_type!=ResponseType.WRITTEN or 'handwriting-recognition-pending' not in raw.review_reasons:
                    continue
                old=row.get('transcription')
                # A saved successful draft is not rerun automatically. Explicitly
                # compare another candidate using the benchmark tool; retain history.
                engine='ollama:'+getattr(recognizer,'model','')
                if old and old['status']!='error' and old['engine']==engine and old['language']==language:
                    continue
                region=raw.regions[0]
                request=RecognitionRequest(region,repository.crop(job,region.crop_ref),language)
                start=time.perf_counter()
                try:draft=recognizer.recognize(request)
                except Exception:draft=Transcription(None,ReadStatus.ERROR,('recognizer-failed',),'recognizer-error')
                draft=replace(draft,status=ReadStatus.ERROR if draft.status==ReadStatus.ERROR else ReadStatus.UNCERTAIN,
                    review_reasons=tuple(dict.fromkeys(draft.review_reasons+('handwriting-draft-needs-review',))),
                    seconds=time.perf_counter()-start)
                repository.save_transcription(job,raw.response_id,draft,language)
            repository.status(job,'recognition-cancelled' if cancel.is_set() else 'review')
        except BaseException:
            repository.status(job,'interrupted');raise
