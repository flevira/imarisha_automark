from test_written_extraction import setup,read
import pytest


def test_repair_preserves_history_requires_review_and_accepts_new_approval(setup):
    _,_,_,repo=setup
    job,rows=read(setup,'answers.pdf')
    row=next(r for r in rows if r['selection_mode']=='text')
    raw=row['response']
    repo.review(job,row['id'],action='approve',student=raw.student_id,selected=[],reviewer='Teacher',raw_text='wrong')
    previous=next(r for r in repo.responses(job) if r['id']==row['id'])['review']
    repo.save_transcription_repair(job,row['id'],'Correct reading','Checked against saved crop')
    repaired=next(r for r in repo.responses(job) if r['id']==row['id'])
    assert repaired['review'] is None and repaired['transcription']['text']=='Correct reading'
    with repo.connect() as db:
        assert db.execute('SELECT raw_text FROM structured_reviews WHERE sequence=?',(previous['sequence'],)).fetchone()[0]=='wrong'
    with pytest.raises(ValueError):repo.complete(job)
    repo.review(job,row['id'],action='approve',student=raw.student_id,selected=[],reviewer='Teacher',raw_text='Correct reading')
    assert next(r for r in repo.responses(job) if r['id']==row['id'])['review']['raw_text']=='Correct reading'


def test_image_checked_repair_is_not_replaced_by_ai(setup):
    from services.handwriting_jobs import _eligible
    from types import SimpleNamespace
    _,_,_,repo=setup
    job,rows=read(setup,'answers.pdf')
    row=next(r for r in rows if r['response'].question_id=='4')
    repo.save_transcription_repair(job,row['id'],'12','Image inspected')
    repaired=next(r for r in repo.responses(job) if r['id']==row['id'])
    assert not _eligible(repaired,SimpleNamespace(model='test'),'en')
