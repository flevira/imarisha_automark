import json
from dataclasses import fields
import threading

import cv2
import numpy as np
import pytest

from core.contracts import RecognitionRequest,SourceRegion,Transcription,ReadStatus
from core.handwriting import OllamaHandwritingRecognizer,RecognitionFailure
from tools.handwriting_benchmark import evaluate
from services.handwriting_jobs import recognize_job
from test_written_extraction import setup,read


def request(blank=False):
    image=np.full((50,120),255,np.uint8)
    if not blank:cv2.line(image,(15,10),(15,40),0,3)
    return RecognitionRequest(SourceRegion('test',1,(0,0,120,50),'test'),cv2.imencode('.png',image)[1].tobytes(),'en-sw')


def fake_client(monkeypatch,**changes):
    client=OllamaHandwritingRecognizer();payloads=[]
    def post(path,data=None):
        payloads.append((path,data))
        if path=='/api/tags':return {'models':[{'name':client.model,'digest':'a'*64}]}
        if path=='/api/show':return {'capabilities':['vision'],'details':{'format':'gguf'},**changes.get('show',{})}
        return {'done':True,'done_reason':'stop','message':{'content':json.dumps({'text':'Mti ni mmea.','flags':[]})},**changes.get('reply',{})}
    monkeypatch.setattr(client,'_request',post)
    return client,payloads


def test_crop_only_recognition_always_requires_review(monkeypatch):
    client,payloads=fake_client(monkeypatch)
    result=client.recognize(request())
    assert result.text=='Mti ni mmea.' and result.status==ReadStatus.UNCERTAIN
    assert result.model_digest=='a'*64 and result.seconds>=0
    assert result.review_reasons==('handwriting-draft-needs-review',)
    assert {f.name for f in fields(RecognitionRequest)}=={'region','image_png','language'}
    body=payloads[-1][1]
    assert body['messages'][1]['images'] and 'question' not in body['messages'][1]['content']
    assert not any(k in json.dumps(body) for k in ['correctAnswers','referenceAnswers','assessmentId'])


@pytest.mark.parametrize('endpoint',['https://example.com','http://192.168.1.2:11434','http://localhost:11434/path','http://user:secret@localhost:11434'])
def test_remote_or_credentialed_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):OllamaHandwritingRecognizer(endpoint=endpoint)


def test_blank_image_skips_model(monkeypatch):
    client,payloads=fake_client(monkeypatch)
    result=client.recognize(request(True))
    assert result.status==ReadStatus.BLANK and not payloads


@pytest.mark.parametrize('show,reason',[({'capabilities':['completion']},'model-has-no-vision'),({'remote_model':'cloud'},'model-is-not-local-weights')])
def test_nonvision_and_remote_models_fail_safely(monkeypatch,show,reason):
    client,_=fake_client(monkeypatch,show=show)
    result=client.recognize(request())
    assert result.status==ReadStatus.ERROR and reason in result.review_reasons


@pytest.mark.parametrize('reply',[{'done':False},{'done_reason':'length'},{'message':{'content':'not JSON'}},{'message':{'content':'{"text": null, "flags": []}'} }])
def test_invalid_or_truncated_output_is_not_accepted(monkeypatch,reply):
    client,_=fake_client(monkeypatch,reply=reply)
    result=client.recognize(request());assert result.status==ReadStatus.ERROR and result.text is None


def test_missing_model_and_service_failure(monkeypatch):
    client=OllamaHandwritingRecognizer()
    monkeypatch.setattr(client,'_request',lambda *a,**k:{'models':[]})
    assert 'local-model-not-installed' in client.recognize(request()).review_reasons
    def unavailable(*a,**k):raise RecognitionFailure('local-recognition-unavailable')
    client=OllamaHandwritingRecognizer()
    monkeypatch.setattr(client,'_request',unavailable)
    assert client.recognize(request()).status==ReadStatus.ERROR


class FakeRecognizer:
    model='synthetic'
    def __init__(self):self.requests=[]
    def recognize(self,req):
        self.requests.append(req)
        return Transcription('12',ReadStatus.READ,(),'ollama:synthetic',.1,'a'*64)


def test_saved_crop_recognition_keeps_raw_and_teacher_review_separate(setup):
    _,_,_,repo=setup;job,rows=read(setup,'answers.pdf');raws=[r['raw'] for r in rows]
    engine=FakeRecognizer();recognize_job(repo,job,engine)
    assert len(engine.requests)==1
    rows=repo.responses(job);assert [r['raw'] for r in rows]==raws
    row=next(r for r in rows if r['response'].question_id=='4')
    assert row['transcription']['text']=='12' and row['transcription']['status']=='uncertain'
    assert row['review'] is None
    recognize_job(repo,job,engine);assert len(engine.requests)==1
    repo.review(job,row['id'],action='approve',student='00530',selected=[],reviewer='Teacher',note='Checked handwriting',raw_text='12')
    engine.model='other';recognize_job(repo,job,engine);assert len(engine.requests)==1


def test_recognition_cancellation_before_first_crop(setup):
    _,_,_,repo=setup;job,_=read(setup,'answers.pdf')
    cancel=threading.Event();cancel.set();engine=FakeRecognizer()
    recognize_job(repo,job,engine,cancel=cancel)
    assert not engine.requests and repo.job(job)['status']=='recognition-cancelled'
    recognize_job(repo,job,engine)
    assert len(engine.requests)==1


def test_model_failure_does_not_remove_evidence_and_can_retry(setup):
    _,_,_,repo=setup;job,_=read(setup,'answers.pdf')
    class Broken:
        def recognize(self,req):raise RuntimeError('failure')
    recognize_job(repo,job,Broken())
    row=next(r for r in repo.responses(job) if r['response'].question_id=='4')
    assert row['transcription']['status']=='error'
    assert repo.crop(job,row['response'].regions[0].crop_ref)
    recognize_job(repo,job,FakeRecognizer())
    assert next(r for r in repo.responses(job) if r['response'].question_id=='4')['transcription']['text']=='12'


def test_benchmark_error_rates_and_labels_never_sent_to_model():
    engine=FakeRecognizer()
    result=evaluate([dict(image_png=request().image_png,text='13',language='en')],engine)
    assert result['character_error_rate']==.5 and result['word_error_rate']==1
    assert not hasattr(engine.requests[0],'text')


@pytest.mark.parametrize('flag',['illegible','crossed-out','low-quality'])
def test_difficult_writing_flags_are_preserved(monkeypatch,flag):
    client,_=fake_client(monkeypatch,reply={'message':{'content':json.dumps({'text':'[illegible]','flags':[flag]})}})
    result=client.recognize(request())
    assert 'handwriting-'+flag in result.review_reasons
    assert result.status==ReadStatus.UNCERTAIN


def test_model_empty_text_on_ink_is_uncertain_not_blank(monkeypatch):
    client,_=fake_client(monkeypatch,reply={'message':{'content':'{"text":"","flags":[]}'}})
    result=client.recognize(request())
    assert result.status==ReadStatus.UNCERTAIN and 'recognition-empty' in result.review_reasons


def test_http_failure_is_bounded_and_not_retried(monkeypatch):
    client=OllamaHandwritingRecognizer(timeout=1)
    calls=[]
    def timeout(*args,**kwargs):
        calls.append(kwargs);raise TimeoutError()
    monkeypatch.setattr(client.opener,'open',timeout)
    result=client.recognize(request())
    assert result.status==ReadStatus.ERROR and len(calls)==1
    assert calls[0]['timeout']==1
    assert client.recognize(request()).status==ReadStatus.ERROR
    assert len(calls)==1


def test_redirect_handler_refuses_forwarding():
    from core.handwriting import NoRedirect
    assert NoRedirect().redirect_request(None,None,302,'Moved',{},'https://external.invalid') is None


def test_english_and_kiswahili_language_hints_are_sent(monkeypatch):
    from dataclasses import replace
    for language in ['en','sw','en-sw']:
        client,payloads=fake_client(monkeypatch)
        assert client.recognize(replace(request(),language=language)).status==ReadStatus.UNCERTAIN
        assert payloads[-1][1]['messages'][1]['content'].endswith(language)


def test_job_integration_persists_a_separate_draft(setup):
    from services.extraction_jobs import run_job
    path,_,resolver,repo=setup
    job=repo.create(path/'papers/answers.pdf',resolver.scope)
    engine=FakeRecognizer();run_job(repo,job,resolver,recognizer=engine,language='en-sw')
    row=next(r for r in repo.responses(job) if r['response'].question_id=='4')
    assert row['response'].raw_text is None and row['transcription']['text']=='12'
    assert repo.job(job)['status']=='review'


@pytest.fixture(autouse=True)
def uncertain_ocr_for_ai_fallback_tests(monkeypatch):
    from core.local_text_ocr import LocalTextOCR
    from core.contracts import Transcription, ReadStatus
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *a:Transcription(None,ReadStatus.UNCERTAIN,('ocr-no-text',),'tesseract-written-v1'))
