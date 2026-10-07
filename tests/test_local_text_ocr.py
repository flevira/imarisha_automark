import cv2
import numpy as np
from types import SimpleNamespace
from core.local_text_ocr import LocalTextOCR


def test_trocr_lines_confidence_and_language():
    from core.local_text_ocr import split_lines
    image=np.full((120,400),255,np.uint8)
    cv2.putText(image,'First line',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.7,0,2)
    cv2.putText(image,'Second line',(15,95),cv2.FONT_HERSHEY_SIMPLEX,.7,0,2)
    lines,flags=split_lines(image)
    assert len(lines)==2 and not flags
    png=cv2.imencode('.png',image)[1].tobytes()
    class Backend:
        def __init__(self):self.n=0
        def read_line(self,image):
            self.n+=1
            return ('First line' if self.n%2 else 'Second line',.98,.95,True)
    ocr=LocalTextOCR(Backend())
    result=ocr.recognize(SimpleNamespace(image_png=png,language='en'))
    assert result.text=='First line\nSecond line' and result.status.value=='read'
    assert result.engine=='trocr-written-v1'
    result=ocr.recognize(SimpleNamespace(image_png=png,language='en-sw'))
    assert result.status.value=='uncertain' and 'ocr-language-unvalidated' in result.review_reasons


def test_trocr_blank_does_not_hallucinate_or_load_model(monkeypatch):
    import core.local_text_ocr as module
    def never(*a):raise AssertionError('Do not load model for blank image')
    monkeypatch.setattr(module,'_backend',never)
    png=cv2.imencode('.png',np.full((50,200),255,np.uint8))[1].tobytes()
    result=LocalTextOCR().recognize(SimpleNamespace(image_png=png,language='en'))
    assert result.text is None and result.status.value=='uncertain'


def test_trocr_uncertain_and_unavailable(monkeypatch):
    import core.local_text_ocr as module
    image=np.full((70,250),255,np.uint8)
    cv2.putText(image,'Answer',(10,40),cv2.FONT_HERSHEY_SIMPLEX,.8,0,2)
    req=SimpleNamespace(image_png=cv2.imencode('.png',image)[1].tobytes(),language='en')
    backend=SimpleNamespace(read_line=lambda _:('Answer',.7,.3,False))
    r=LocalTextOCR(backend).recognize(req)
    assert r.status.value=='uncertain'
    assert {'ocr-low-confidence','ocr-incomplete-recognition'}<=set(r.review_reasons)
    def unavailable(*a):raise OSError('Not cached')
    monkeypatch.setattr(module,'_backend',unavailable)
    r=LocalTextOCR().recognize(req)
    assert r.status.value=='error' and r.review_reasons==('ocr-unavailable',)


def test_image_input_and_multiframe_tiff(tmp_path):
    from PIL import Image
    from database.structured_repository import JobRepository
    from core.source_document import open_source
    path=tmp_path/'pages.tiff'
    first=Image.new('RGB',(500,700),'white');second=Image.new('RGB',(500,700),'black')
    first.save(path,save_all=True,append_images=[second])
    repo=JobRepository(tmp_path/'jobs.db');job=repo.create(path,'local')
    assert repo.job(job)['total']==2
    with open_source(repo.source(job)) as doc:assert len(doc)==2


def test_confident_ocr_skips_ai_and_resume_reuses_result(setup,monkeypatch):
    from test_written_extraction import read
    from core.contracts import Transcription,ReadStatus
    from services.handwriting_jobs import recognize_job,ocr_job
    calls=[]
    def recognize(*args):
        calls.append(1)
        return Transcription('12',ReadStatus.READ,(),'trocr-written-v1',confidence=95)
    monkeypatch.setattr(LocalTextOCR,'recognize',recognize)
    _,_,_,repo=setup;job,rows=read(setup,'answers.pdf')
    count=len(calls)
    class Never:
        def recognize(self,request):raise AssertionError('AI must not run for confident OCR')
    recognize_job(repo,job,Never(),language='en')
    ocr_job(repo,job,language='en')
    assert len(calls)==count
    assert next(r for r in repo.responses(job) if r['response'].question_id=='4')['transcription']['engine']=='trocr-written-v1'


def test_png_uses_same_manifest_extraction(setup):
    import fitz
    from services.extraction_jobs import run_job
    path,_,resolver,repo=setup
    with fitz.open(path/'papers'/'answers.pdf') as document:
        image=path/'sheet.png';document[0].get_pixmap(dpi=150).save(image)
    job=repo.create(image,resolver.scope);run_job(repo,job,resolver)
    assert repo.job(job)['pages'][0]['status']=='extracted'
    assert repo.responses(job)


from test_written_extraction import setup

def test_historical_configuration_warning_alone_does_not_trigger_ai(setup,monkeypatch):
    from test_written_extraction import read
    from core.contracts import Transcription,ReadStatus
    from services.handwriting_jobs import recognize_job
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:Transcription(
        '12',ReadStatus.UNCERTAIN,('ocr-language-pack-missing',),'tesseract-written-v1',confidence=95))
    _,_,_,repo=setup
    job,_=read(setup,'answers.pdf')
    # The extraction fixture uses English; make the stored language agree.
    class Never:
        def recognize(self,request):raise AssertionError('Configuration alone must not invoke AI')
    recognize_job(repo,job,Never(),language='en')


def test_trocr_fallback_gate():
    from services.handwriting_jobs import _eligible
    from core.contracts import ResponseType
    raw=SimpleNamespace(response_type=ResponseType.WRITTEN,review_reasons=('handwriting-recognition-pending',))
    row=dict(response=raw,review=None,transcription=dict(engine='trocr-written-v1',language='en',status='read'))
    model=SimpleNamespace(model='gemma3')
    assert not _eligible(row,model,'en')
    for status in ('uncertain','error'):
        row['transcription']['status']=status
        assert _eligible(row,model,'en')
    row['review']={'action':'approve'}
    assert not _eligible(row,model,'en')


def test_line_rules_removed_and_tall_band_flagged():
    from core.local_text_ocr import split_lines
    image=np.full((120,400),255,np.uint8)
    cv2.putText(image,'First',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,0,1)
    cv2.putText(image,'Second',(10,85),cv2.FONT_HERSHEY_SIMPLEX,.6,0,1)
    cv2.line(image,(0,50),(399,50),0,1)
    lines,_=split_lines(image)
    assert len(lines)==2
    # A connected ink stroke spanning two lines must not silently pass.
    cv2.line(image,(12,18),(12,85),0,1)
    _,flags=split_lines(image)
    assert 'ocr-line-segmentation-uncertain' in flags
