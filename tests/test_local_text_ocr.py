import cv2
import numpy as np
from types import SimpleNamespace
from core.local_text_ocr import LocalTextOCR


def test_ocr_text_confidence_and_missing_language(monkeypatch):
    import core.local_text_ocr as module
    monkeypatch.setattr(module.pytesseract,'get_languages',lambda **kw:['eng'])
    data=dict(text=['Water','helps'],conf=['95','90'],block_num=[1,1],par_num=[1,1],line_num=[1,1])
    monkeypatch.setattr(module.pytesseract,'image_to_data',lambda *a,**kw:data)
    png=cv2.imencode('.png',np.full((50,200),255,np.uint8))[1].tobytes()
    result=LocalTextOCR().recognize(SimpleNamespace(image_png=png,language='en'))
    assert result.text=='Water helps' and result.status.value=='read' and result.confidence==92.5
    result=LocalTextOCR().recognize(SimpleNamespace(image_png=png,language='en-sw'))
    assert 'ocr-language-pack-missing' in result.review_reasons
    data['conf']=['10','20']
    assert 'ocr-low-confidence' in LocalTextOCR().recognize(SimpleNamespace(image_png=png,language='en')).review_reasons


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
        return Transcription('12',ReadStatus.READ,(),'tesseract-written-v1',confidence=95)
    monkeypatch.setattr(LocalTextOCR,'recognize',recognize)
    _,_,_,repo=setup;job,rows=read(setup,'answers.pdf')
    count=len(calls)
    class Never:
        def recognize(self,request):raise AssertionError('AI must not run for confident OCR')
    recognize_job(repo,job,Never(),language='en')
    ocr_job(repo,job,language='en')
    assert len(calls)==count
    assert next(r for r in repo.responses(job) if r['response'].question_id=='4')['transcription']['engine']=='tesseract-written-v1'


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
