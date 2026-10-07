from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from core.contracts import Transcription, ReadStatus
from core.paddle_text_ocr import PaddleAgreementOCR, ENGINE, eligible, normalized_words


def request(language='en'):
    image=np.full((50,250,3),255,np.uint8)
    return SimpleNamespace(image_png=cv2.imencode('.png',image)[1].tobytes(),language=language)


def original(text='Do not steal.\nProtect school property.', reasons=('ocr-line-segmentation-uncertain',)):
    return Transcription(text,ReadStatus.UNCERTAIN,reasons,'trocr-written-v1')


def test_accepts_whole_answer_agreement_and_preserves_original():
    backend=SimpleNamespace(read=lambda image:(['DO NOT STEAL','Protect school property'],[.95,.99]))
    result=PaddleAgreementOCR(backend).corroborate(request(),original())
    assert result.status==ReadStatus.READ and result.text==original().text
    assert result.engine==ENGINE and result.confidence==95 and result.seconds>=0


@pytest.mark.parametrize('text,scores,reason',[
    (['Do steal. Protect school property.'],[.99],'paddle-trocr-disagreement'),
    (['Do not steal.'],[.999],'paddle-trocr-disagreement'),
    (['Do not steal. Protect schoolproperty.'],[.99],'paddle-trocr-disagreement'),
    (['Do not steal. Protect school property.'],[.949],'paddle-low-confidence'),
    ([],[],'paddle-no-text'),
    ([''],[.99],'paddle-no-text'),
    (['text'],[float('nan')],'paddle-failed'),
    (['text'],[],'paddle-failed'),
])
def test_rejection_keeps_original_and_remains_ai_eligible(text,scores,reason):
    from services.handwriting_jobs import _eligible
    from core.contracts import ResponseType
    result=PaddleAgreementOCR(SimpleNamespace(read=lambda image:(text,scores))).corroborate(request(),original())
    assert result.status==ReadStatus.UNCERTAIN and result.text==original().text
    assert reason in result.review_reasons
    row=dict(response=SimpleNamespace(response_type=ResponseType.WRITTEN,review_reasons=['handwriting-recognition-pending']),
        review=None,transcription=dict(engine=ENGINE,language='en',status=result.status.value))
    assert _eligible(row,SimpleNamespace(model='gemma'),'en')
    row['transcription']['status']='read'
    assert not _eligible(row,SimpleNamespace(model='gemma'),'en')


def test_language_failure_truncation_and_confident_exclusions():
    assert not eligible(original(),'en-sw')
    for reason in ('ocr-no-text','ocr-incomplete-recognition','ocr-language-unvalidated'):
        assert not eligible(original(reasons=(reason,)),'en')
    assert not eligible(Transcription('text',ReadStatus.READ,(),'trocr-written-v1'),'en')
    assert normalized_words('2 tools') != normalized_words('3 tools')


def test_unavailable_only_loads_once_per_job(monkeypatch):
    import core.paddle_text_ocr as module
    calls=[]
    def missing(*args):
        calls.append(1);raise ImportError('optional dependency missing')
    monkeypatch.setattr(module,'_backend',missing)
    recognizer=PaddleAgreementOCR()
    for _ in range(2):
        result=recognizer.corroborate(request(),original())
        assert 'paddle-unavailable' in result.review_reasons
    assert len(calls)==1


def test_job_resume_and_ai_skip_preserve_history(setup, monkeypatch):
    from test_written_extraction import read
    from core.local_text_ocr import LocalTextOCR
    from services.handwriting_jobs import ocr_job,recognize_job
    import core.paddle_text_ocr as module
    calls=[]
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:original('12',('ocr-low-confidence',)))
    def recognize(image):
        calls.append(1);return ['12'],[.99]
    monkeypatch.setattr(module,'_backend',lambda *args:SimpleNamespace(read=recognize))
    _,_,_,repo=setup;job,_=read(setup,'answers.pdf')
    rows=repo.responses(job)
    row=next(r for r in rows if r['transcription'])
    assert row['transcription']['engine']==ENGINE
    count=len(calls)
    ocr_job(repo,job,'en')
    assert len(calls)==count
    class Never:
        def recognize(self,request):raise AssertionError('Gemma should be skipped')
    assert recognize_job(repo,job,Never(),'en')['total']==0
    with repo.connect() as db:
        engines=[__import__('json').loads(r[0])['engine'] for r in db.execute(
            'select result from structured_transcriptions where job=? and response=? order by sequence',(job,row['id']))]
    assert engines==['trocr-written-v1',ENGINE]


from test_written_extraction import setup


def test_resume_adds_paddle_without_rerunning_trocr_and_retries_unavailable(setup,monkeypatch):
    from test_written_extraction import read
    from core.local_text_ocr import LocalTextOCR
    from services.handwriting_jobs import ocr_job
    import core.paddle_text_ocr as module
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:original('12',('ocr-low-confidence',)))
    # Simulate a saved pre-Paddle extraction by initially bypassing this stage.
    real=PaddleAgreementOCR.corroborate
    monkeypatch.setattr(PaddleAgreementOCR,'corroborate',lambda *args:None)
    _,_,_,repo=setup;job,_=read(setup,'answers.pdf')
    monkeypatch.setattr(PaddleAgreementOCR,'corroborate',real)
    def never(*args):raise AssertionError('Saved TrOCR must be reused')
    monkeypatch.setattr(LocalTextOCR,'recognize',never)
    def missing(*args):raise OSError('weights missing')
    monkeypatch.setattr(module,'_backend',missing)
    ocr_job(repo,job,'en')
    rows=[r for r in repo.responses(job) if r['transcription']]
    assert rows and all('paddle-unavailable' in r['transcription']['review_reasons'] for r in rows)
    monkeypatch.setattr(module,'_backend',lambda *args:SimpleNamespace(read=lambda image:(['12'],[.99])))
    ocr_job(repo,job,'en')
    assert all(r['transcription']['status']=='read' for r in repo.responses(job) if r['transcription'])


def test_cancelled_and_completed_jobs_do_not_start_paddle(setup,monkeypatch):
    import threading
    from test_written_extraction import read
    from core.local_text_ocr import LocalTextOCR
    from services.handwriting_jobs import ocr_job
    _,_,_,repo=setup
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:original('12'))
    monkeypatch.setattr(PaddleAgreementOCR,'corroborate',lambda *args:None)
    job,_=read(setup,'answers.pdf')
    cancel=threading.Event();cancel.set()
    def never(*args):raise AssertionError('Cancelled jobs must not start Paddle')
    monkeypatch.setattr(PaddleAgreementOCR,'corroborate',never)
    ocr_job(repo,job,'en',cancel)
    assert repo.job(job)['status']=='recognition-cancelled'
    repo.status(job,'complete')
    ocr_job(repo,job,'en')
    assert repo.job(job)['status']=='complete'
