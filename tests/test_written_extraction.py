import copy
import json

import pytest

from core.manifest_layout import validate_manifest, PreparationError
from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from services.manifest_resolver import CachedManifestResolver
from tools.written_fixtures import generate


@pytest.fixture
def setup(tmp_path):
    manifest=generate(tmp_path/'papers',continuation=True)
    resolver=CachedManifestResolver(tmp_path/'cache.db','synthetic-fixtures-only');resolver.seed_fixture(manifest)
    repo=JobRepository(tmp_path/'jobs.db')
    return tmp_path,manifest,resolver,repo


def read(setup,name):
    path,manifest,resolver,repo=setup
    job=repo.create(path/'papers'/name,resolver.scope);run_job(repo,job,resolver)
    assert all(p['status']=='extracted' for p in repo.job(job)['pages']),repo.job(job)
    return job,repo.responses(job)


def test_typed_multiline_and_multipage_answers_with_image_only_handwriting(setup):
    job,rows=read(setup,'answers.pdf')
    written=[r['response'] for r in rows if r['response'].response_type.value=='WRITTEN']
    assert len(written)==5
    parts=[r for r in written if r.question_id=='2']
    assert [r.part_number for r in parts]==[1,2]
    assert parts[0].raw_text=='First line\nSecond line'
    assert parts[1].raw_text=='Continued response'
    handwritten=next(r for r in written if r.question_id=='4')
    assert handwritten.raw_text is None
    assert 'handwriting-recognition-pending' in handwritten.review_reasons
    blank=next(r for r in written if r.question_id=='3')
    assert blank.status.value=='blank'
    assert all('Question' not in (r.raw_text or '') for r in written)


def test_scanned_answer_regions_do_not_reuse_printed_question_text(setup):
    _,rows=read(setup,'scan.pdf')
    assert all(r['response'].raw_text is None for r in rows)
    assert next(r['response'] for r in rows if r['response'].question_id=='2').status.value=='uncertain'


def test_empty_generated_boxes_are_blank_not_the_printed_labels(setup):
    _,rows=read(setup,'blank.pdf')
    assert all(r['response'].status.value=='blank' for r in rows if r['response'].response_type.value=='WRITTEN')


def test_manual_transcription_saved_separately_and_continuations_can_complete(setup):
    path,manifest,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        r=row['response'];written=r.response_type.value=='WRITTEN'
        repo.review(job,row['id'],action='approve',student='00530',selected=list(r.selected_options),reviewer='Teacher',
            note='Manually transcribed visible handwriting' if r.needs_review else '',raw_text=('12' if r.question_id=='4' else r.raw_text or '') if written else None)
    repo.complete(job)
    saved=JobRepository(repo.path).responses(job)
    row=next(r for r in saved if r['response'].question_id=='4')
    assert row['response'].raw_text is None and row['review']['raw_text']=='12'


@pytest.mark.parametrize('defect',['overlap','outside','order','duplicate','legacy'])
def test_invalid_written_manifest_rejected(setup,defect):
    _,manifest,_,_=setup
    m=copy.deepcopy(manifest);regions=m['pages'][1]['writtenRegions']
    if defect=='overlap':regions[1]['y']=regions[0]['y']
    if defect=='outside':regions[0]['height']=900
    if defect=='order':regions[0]['order']=8
    if defect=='duplicate':regions[1]['regionId']=regions[0]['regionId']
    if defect=='legacy':m['version']=4
    with pytest.raises(PreparationError):validate_manifest(m)


def test_shared_written_sheet_retains_student_bubbles(tmp_path):
    manifest=generate(tmp_path/'papers',shared=True)
    cache=CachedManifestResolver(tmp_path/'cache.db','synthetic-fixtures-only');cache.seed_fixture(manifest)
    repo=JobRepository(tmp_path/'jobs.db');job=repo.create(tmp_path/'papers/answers.pdf',cache.scope);run_job(repo,job,cache)
    assert len(repo.responses(job))==6
    assert all(r['response'].student_id=='00530' for r in repo.responses(job))


def test_hidden_pdf_text_is_not_student_answer(setup):
    import fitz
    path,manifest,resolver,repo=setup
    source=path/'papers/blank.pdf';target=path/'papers/hidden.pdf'
    with fitz.open(source) as doc:
        r=manifest['pages'][1]['writtenRegions'][0]
        doc[1].insert_text((r['x']+10,r['y']+20),'HIDDEN ANSWER KEY',render_mode=3)
        doc.save(target)
    _,rows=read(setup,'hidden.pdf')
    assert all(r['response'].raw_text is None for r in rows)


def test_partial_text_and_unexplained_vector_ink_require_review(setup):
    import fitz
    path,manifest,resolver,repo=setup
    with fitz.open(path/'papers/answers.pdf') as doc:
        region=manifest['pages'][1]['writtenRegions'][0]
        doc[1].draw_line((region['x']+200,region['y']+60),(region['x']+300,region['y']+75),width=5)
        other=manifest['pages'][1]['writtenRegions'][1]
        doc[1].insert_text((other['x']-15,other['y']+20),'Crosses boundary',fontsize=12)
        doc.save(path/'papers/partial.pdf')
    _,rows=read(setup,'partial.pdf')
    q2=next(r['response'] for r in rows if r['response'].question_id=='2')
    q3=next(r['response'] for r in rows if r['response'].question_id=='3')
    assert q2.raw_text is None and q2.needs_review
    assert 'text-crosses-answer-boundary' in q3.review_reasons


def test_rotated_native_pdf_retains_typed_region_mapping(setup):
    import fitz
    path,_,_,_=setup
    with fitz.open(path/'papers/answers.pdf') as doc:
        for page in doc:page.set_rotation(90)
        doc.save(path/'papers/rotated.pdf')
    _,rows=read(setup,'rotated.pdf')
    q2=next(r['response'] for r in rows if r['response'].question_id=='2')
    assert q2.raw_text=='First line\nSecond line'


def test_existing_review_database_gains_text_column_without_losing_history(tmp_path):
    import sqlite3
    path=tmp_path/'old.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE structured_reviews(sequence INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, response TEXT, action TEXT, student TEXT, selected TEXT, reviewer TEXT, note TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)')
        db.execute("INSERT INTO structured_reviews(job,response,action,student,selected,reviewer,note) VALUES ('j','r','approve','00530','[\"A\"]','Teacher','Original review')")
    repo=JobRepository(path)
    history=repo.review_history('j','r')
    assert len(history)==1 and history[0]['note']=='Original review'
    assert history[0]['raw_text'] is None


def test_review_screen_allows_manual_written_transcription(setup):
    import flet as ft
    from types import SimpleNamespace
    from app.views.structured_view import build_structured_view
    from test_structured_view import Page,find,descendants
    _,_,_,repo=setup
    job,_=read(setup,'answers.pdf')
    page=Page();build_structured_view(page,repo)
    find(page,'text','answers.pdf — review (3 pages)').on_click(None)
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'label','Review page').on_change(SimpleNamespace(control=SimpleNamespace(value='2')))
    card=next(c for c in descendants(page) if isinstance(c,ft.Container) and isinstance(c.content,ft.Column)
              and c.content.controls and getattr(c.content.controls[0],'value',None)=='Page 2 · Question 4 (part 1)')
    find(card,'label','Student answer transcription (empty means blank)').value='12'
    find(card,'label','Correction / review note').value='Read the retained handwriting crop manually'
    find(card,'text','Approve current values').on_click(None)
    row=next(r for r in repo.responses(job) if r['response'].question_id=='4')
    assert row['review']['raw_text']=='12' and row['response'].raw_text is None
    for control in page.controls:assert control._build_add_commands()


def test_bulk_approval_does_not_turn_untranscribed_handwriting_into_blank(setup):
    _,_,_,repo=setup
    job,rows=read(setup,'answers.pdf')
    result=repo.approve_all(job,'Teacher')
    assert any(r['question']==4 and 'transcription' in r['reason'] for r in result['skipped'])
    row=next(r for r in repo.responses(job) if r['response'].question_id=='4')
    assert row['review'] is None
    # Individual transcription approval no longer requires a comment.
    repo.review(job,row['id'],action='approve',student='00530',selected=[],reviewer='Teacher',raw_text='12')
    assert next(r for r in repo.responses(job) if r['id']==row['id'])['review']['raw_text']=='12'
