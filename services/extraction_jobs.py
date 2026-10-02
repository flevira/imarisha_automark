"""Resumable page-at-a-time M2/M3 runner; no marking or legacy OCR."""
import json
import threading
import time

import cv2
import fitz
import numpy as np

from core.structured_extraction import StructuredExtractor
from core.written_extraction import capture_written
from services.handwriting_jobs import recognize_job
from services.page_preparation import prepare_image


def audit_job(repository, job):
    issues, groups = [], {}
    for page in repository.job(job)['pages']:
        if page['status'] == 'pending':
            continue
        if page['status'] != 'extracted':
            issues.append(dict(code='page-not-extracted',page=page['number'],reason=page['reason']))
            continue
        meta = json.loads(page['metadata'])
        if not meta['student']:
            issues.append(dict(code='student-identity-unresolved',page=page['number']))
            continue
        groups.setdefault((meta['layout'],meta['student']),[]).append((page['number'],meta))
    for (layout,student), entries in groups.items():
        numbers={}
        for source,meta in entries:
            numbers.setdefault(meta['layout_page'],[]).append(source)
        missing=sorted(set(range(1,entries[0][1]['page_count']+1))-numbers.keys())
        if missing:
            issues.append(dict(code='missing-pages',layout=layout,student=student,pages=missing))
        for number,sources in numbers.items():
            if len(sources)>1:
                issues.append(dict(code='duplicate-page',layout=layout,student=student,layout_page=number,source_pages=sources))
    repository.set_issues(job,issues)


def run_job(repository, job, resolver, cancel=None, progress=None, retry_failed=False, recognizer=None, language="en"):
    cancel = cancel or threading.Event()
    with repository.lock(job):
        info = repository.job(job)
        if info['scope'] != resolver.scope:
            raise ValueError('Select the same backend/account used to create this job')
        if info['status']=='complete':
            return
        repository.status(job,'running')
        try:
            with fitz.open(stream=repository.source(job),filetype='pdf') as document:
                for saved in info['pages']:
                    if cancel.is_set():
                        break
                    if saved['status']=='extracted' or (saved['status']!='pending' and not retry_failed):
                        continue
                    number=saved['number']
                    start=time.perf_counter()
                    image_png=None
                    try:
                        pdf_page=document[number-1]
                        if pdf_page.rect.width*pdf_page.rect.height*(150/72)**2>20_000_000:
                            raise ValueError('Page exceeds preparation pixel limit')
                        pix=pdf_page.get_pixmap(dpi=150,alpha=False,colorspace=fitz.csRGB)
                        image=cv2.cvtColor(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3),cv2.COLOR_RGB2BGR)
                        image_png=cv2.imencode('.png',image)[1].tobytes()
                        prepared=prepare_image(image,'sha256:'+info['digest'],number,resolver)
                        prep_seconds=time.perf_counter()-start
                        if prepared.status!='prepared':
                            repository.save_page(job,number,'failed',prepared.reason,image_png,{},prep_seconds,0)
                        else:
                            page=prepared.page
                            extraction_start=time.perf_counter()
                            result=capture_written(page,StructuredExtractor().extract_page(page),pdf_page)
                            extraction_seconds=time.perf_counter()-extraction_start
                            meta=dict(layout=page.alignment.layout.layout_id,student=result.student_id,
                                      layout_page=page.alignment.layout_page_number,page_count=page.geometry['layoutPageCount'],
                                      identity_reasons=result.identity_reasons,
                                      identity_digit_count=5 if 'studentNumber' in page.geometry else None,
                                      bubble_evidence={key:[dict(option=b.option_id,darkness=b.darkness,state=b.state,crop=b.region.crop_ref) for b in bubbles]
                                                       for key,bubbles in result.evidence.items()})
                            repository.save_page(job,number,'extracted',None,image_png,meta,prep_seconds,extraction_seconds,result,page.geometry)
                    except Exception as exc:
                        # A failed page remains retryable; do not discard completed pages.
                        repository.save_page(job,number,'failed',f'{type(exc).__name__}: {exc}',image_png,{},time.perf_counter()-start,0)
                    if progress:
                        progress(number,info['total'])
            audit_job(repository,job)
            repository.status(job,'cancelled' if cancel.is_set() else 'review')
        except BaseException:
            repository.status(job,'interrupted')
            raise

    if recognizer is not None and not cancel.is_set():
        recognize_job(repository,job,recognizer,language,cancel)
