import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from core.marking import mark_question
from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from services.marking_jobs import mark_job, digest, response_snapshot
from test_page_alignment import fixture_dir, resolver


def question(kind='MC', maximum=4):
    q=dict(questionId='1',displayNumber=1,maxMarks=maximum,responseType=kind,status='READY',referenceAnswers=['Water'])
    row=dict(kind=kind,bubbles=[dict(optionId=c) for c in ['A','B','C','D']],
             correctAnswers=['A'],answerStatus='READY',questionMarks=maximum)
    return q,row


def answer(selected=(),text='',approved=True):
    return dict(selected=list(selected),text=text,approved=approved)


RULE=dict(version=1,mode='all-or-nothing')
TEXT=dict(version=1,mode='text-exact',caseSensitive=False,collapseWhitespace=True)


@pytest.mark.parametrize('kind,key,selected,expected',[
    ('MC',['A'],['A'],'4'),('MC',['A'],['B'],'0'),('MC',['A'],[],'0'),
    ('MA',['A','C'],['C','A'],'4'),('MA',['A','C'],['A'],'0'),
    ('MA',['A','C'],['A','B','C'],'0'),('TF',['A'],['A'],'4')])
def test_structured_answer_oracle(kind,key,selected,expected):
    q,row=question(kind);row['correctAnswers']=key
    result=mark_question(q,[row],[answer(selected)],RULE)
    assert result['score']==expected and result['status']=='scored'


def test_matching_is_one_allocation_and_accepts_equivalent_labels():
    q,row=question('MATCH');second=copy.deepcopy(row);second['correctAnswers']=['B'];second['acceptableAnswers']=['B','C']
    assert mark_question(q,[row,second],[answer(['A']),answer(['C'])],RULE)['score']=='4'
    assert mark_question(q,[row,second],[answer(['A']),answer(['D'])],RULE)['score']=='0'
    assert mark_question(q,[row,second],[answer(['A']),None],RULE)['score'] is None


@pytest.mark.parametrize('change',[
    {'maxMarks':None},{'maxMarks':-1},{'maxMarks':float('nan')},{'status':'REVIEW_REQUIRED'}])
def test_invalid_keys_do_not_become_zero(change):
    q,row=question();q.update(change)
    assert mark_question(q,[row],[answer(['B'])],RULE)['score'] is None


def test_missing_policy_and_uncertain_response_are_unresolved():
    q,row=question()
    assert mark_question(q,[row],[answer(['A'])],None)['reason']=='scoring-policy-required'
    assert mark_question(q,[row],[answer(['A'],approved=False)],RULE)['reason']=='unresolved-transcription'
    row['correctAnswers']=[]
    assert mark_question(q,[row],[answer()],RULE)['score'] is None


def test_written_variants_unicode_and_unlisted_wording():
    q,row=question('WRITTEN');q.update(status='MANUAL_REVIEW',referenceAnswers=['Maji'],acceptedAlternatives=['Water','café'])
    for text in ['  MAJI  ','water','cafe\u0301']:
        assert mark_question(q,[row],[answer(text=text)],TEXT)['score']=='4'
    result=mark_question(q,[row],[answer(text='H2O')],TEXT)
    assert result['score'] is None and result['reason']=='unlisted-wording-needs-review'
    assert mark_question(q,[row],[answer(text='H2O')],dict(TEXT,rejectUnlisted=True))['score']=='0'
    assert mark_question(q,[row],[answer()],TEXT)['score']=='0'
    assert mark_question(q,[row],[answer(approved=False)],TEXT)['score'] is None


def test_written_continuations_are_joined_and_scored_once():
    q,row=question('WRITTEN');q['referenceAnswers']=['first second']
    assert mark_question(q,[row,row],[answer(text='first'),answer(text='second')],TEXT)['score']=='4'


def test_numeric_tolerance_and_units_are_explicit():
    q,row=question('WRITTEN');q['referenceAnswers']=['10 cm']
    rule=dict(version=1,mode='numeric',tolerance='0.1',unit='cm')
    assert mark_question(q,[row],[answer(text='10.1 cm')],rule)['score']=='4'
    assert mark_question(q,[row],[answer(text='10.2 cm')],rule)['score']=='0'
    for text in ['0.1 m','10','ten cm']:
        assert mark_question(q,[row],[answer(text=text)],rule)['score'] is None
    assert mark_question(q,[row],[answer(text='10 cm')],dict(rule,tolerance=None))['score'] is None


def completed(tmp_path,fixture_dir,resolver):
    repository=JobRepository(tmp_path/'jobs.db')
    job=repository.create(fixture_dir/'personalized-mixed-text.pdf',resolver.scope)
    run_job(repository,job,resolver)
    for row in repository.responses(job):
        raw=row['response']
        repository.review(job,row['id'],action='approve',student=raw.student_id,
                          selected=list(raw.selected_options),reviewer='Teacher',note='Checked image')
    for issue in repository.issues(job):repository.resolve_issue(job,issue['key'],'Teacher','Checked fixture')
    repository.complete(job)
    return repository,job


def test_mark_remark_persists_revisions_without_scanning(tmp_path,fixture_dir,resolver,monkeypatch):
    repo,job=completed(tmp_path,fixture_dir,resolver)
    original=[r['raw'] for r in repo.responses(job)]
    def forbidden(*a,**kw):raise AssertionError('Marking must not scan or recognize')
    monkeypatch.setattr(repo,'source',forbidden)
    monkeypatch.setattr(repo,'crop',forbidden)
    monkeypatch.setattr('services.extraction_jobs.run_job',forbidden)
    monkeypatch.setattr('services.handwriting_jobs.recognize_job',forbidden)
    policy=dict(version=1,structured='all-or-nothing')
    first=mark_job(repo,job,resolver,'Teacher',policy)
    assert len(first['results'])==4
    assert {r['student_id'] for r in first['results']}=={'00530'}
    assert sum(Decimal(r['max_marks']) for r in first['results'])==8
    assert first['marking_seconds']>=0 and first['review_seconds'] is None
    row=repo.responses(job)[0]
    repo.review(job,row['id'],action='approve',student='00530',selected=['A'],reviewer='Teacher',
                note='Corrected transcription',expected_sequence=row['review']['sequence'])
    with pytest.raises(ValueError,match='Complete extraction review'):mark_job(repo,job,resolver,'Teacher',policy)
    repo.complete(job)
    second=mark_job(repo,job,resolver,'Teacher',policy)
    assert first['response_digest']!=second['response_digest']
    assert second['results'][0]['score']=='2'
    assert first['results'][0]['responses'][0]['review_sequence']!=second['results'][0]['responses'][0]['review_sequence']
    reopened=JobRepository(repo.path)
    assert len(reopened.marking_runs(job))==2
    assert reopened.marking_runs(job)[1]['response_digest']==first['response_digest']
    assert [r['raw'] for r in repo.responses(job)]==original


def test_manifest_placeholder_uses_agreed_rules_and_wrong_scope_is_rejected(tmp_path,fixture_dir,resolver):
    repo,job=completed(tmp_path,fixture_dir,resolver)
    result=mark_job(repo,job,resolver,'Teacher')
    assert result['totals'][0]['score'] is not None
    assert result['totals'][0]['review_count']==0
    assert all(r['rule']['mode']=='partial' for r in result['results'])
    with pytest.raises(ValueError,match='same backend'):
        mark_job(repo,job,SimpleNamespace(scope='other'),'Teacher')


def test_excluded_matching_item_keeps_question_unresolved(tmp_path,fixture_dir,resolver):
    repo,job=completed(tmp_path,fixture_dir,resolver)
    row=repo.responses(job)[-1]
    repo.review(job,row['id'],action='exclude',student='00530',selected=[],reviewer='Teacher',note='Unrecoverable',
                expected_sequence=row['review']['sequence'])
    repo.complete(job)
    result=mark_job(repo,job,resolver,'Teacher',dict(version=1,structured='all-or-nothing'))
    assert result['results'][-1]['reason']=='missing-or-excluded-response'
    assert result['totals'][0]['score'] is None


def test_ui_marking_action_and_outdated_results(tmp_path,fixture_dir,resolver,monkeypatch):
    from test_structured_view import Page,find
    import app.views.structured_view as view
    import asyncio
    repo,job=completed(tmp_path,fixture_dir,resolver)
    monkeypatch.setattr(view,'ImportedManifestResolver',lambda *a,**kw:resolver)
    page=Page();view.build_structured_view(page,repo)
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'label','Structured marking policy').value='all-or-nothing'
    find(page,'text','personalized-mixed-text.pdf — complete (2 pages)').on_click(None)
    async def run():
        find(page,'text','Mark / re-mark reviewed answers').on_click(None)
        await page.tasks[-1]
    asyncio.run(run())
    assert len(repo.marking_runs(job))==1
    for control in page.controls:assert control._build_add_commands()


def test_written_saved_teacher_text_and_continuations(tmp_path):
    from tools.written_fixtures import generate
    from services.manifest_resolver import CachedManifestResolver
    manifest=generate(tmp_path/'papers',continuation=True)
    manifest['answerSchemaVersion']=1
    manifest['questions']=[dict(questionId=str(i),displayNumber=i,maxMarks=2,
        responseType='MC' if i==1 else 'WRITTEN',status='READY' if i==1 else 'MANUAL_REVIEW',
        referenceAnswers=['First line Second line Continued response' if i==2 else '12']) for i in range(1,6)]
    manifest['pages'][0]['rows'][0].update(correctAnswers=['A'],questionMarks=2,answerStatus='READY')
    cache=CachedManifestResolver(tmp_path/'cache.db','synthetic-fixtures-only');cache.seed_fixture(manifest)
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(tmp_path/'papers/answers.pdf',cache.scope);run_job(repo,job,cache)
    for row in repo.responses(job):
        r=row['response'];written=r.response_type.value=='WRITTEN'
        repo.review(job,row['id'],action='approve',student='00530',selected=list(r.selected_options),reviewer='Teacher',
            note='Checked image',raw_text=('12' if r.question_id=='4' else r.raw_text or '') if written else None)
    repo.complete(job)
    result=mark_job(repo,job,cache,'Teacher',dict(version=1,structured='all-or-nothing',writtenDefault=TEXT))
    by_id={r['question_id']:r for r in result['results']}
    assert by_id['2']['score']=='2' and len(by_id['2']['responses'])==2
    assert by_id['4']['score']=='2'
    assert by_id['3']['score']=='0'
    assert next(r['response'] for r in repo.responses(job) if r['response'].question_id=='4').raw_text is None


def test_manifest_identity_failure_leaves_no_partial_run(tmp_path,fixture_dir,resolver):
    repo,job=completed(tmp_path,fixture_dir,resolver)
    def wrong(*args):
        m=copy.deepcopy(resolver.resolve(*args));m['assessmentGroupId']='999';return m
    with pytest.raises(ValueError,match='does not match'):
        mark_job(repo,job,SimpleNamespace(scope=resolver.scope,resolve=wrong),'Teacher')
    assert repo.marking_runs(job)==[]


def test_list_formatting_accepts_reference_but_preserves_meaning():
    q,row=question('WRITTEN')
    q['referenceAnswers']=['Buckets filled with dry sand and portable dry chemical fire extinguishers.']
    rule=dict(TEXT,normalizeListFormatting=True)
    text='(i) Buckets filled with dry sand.\n(ii) Portable dry chemical fire extinguishers.'
    assert mark_question(q,[row],[answer(text=text)],rule)['score']=='4'
    assert mark_question(q,[row],[answer(text=text.replace('filled','not filled'))],rule)['score'] is None
    from core.marking import text_value
    assert text_value('1.25 and 2.5',rule)=='1.25 and 2.5'


def test_teacher_marks_are_versioned_validated_and_update_totals(tmp_path,fixture_dir,resolver):
    from services.marking_jobs import apply_teacher_mark
    repo,job=completed(tmp_path,fixture_dir,resolver)
    run=mark_job(repo,job,resolver,'Teacher')
    r=run['results'][0]
    args=(repo,job,run['sequence'],r['student_id'],r['layout_id'],r['question_id'])
    for invalid in ['-1','999','NaN','0.001']:
        with pytest.raises(ValueError):apply_teacher_mark(*args,invalid,'Teacher')
    seq=apply_teacher_mark(*args,'1.5','Teacher','One marking point earned')
    latest,old=repo.marking_runs(job)
    assert latest['sequence']==seq and latest['parent_run']==run['sequence']
    assert latest['results'][0]['score']=='1.5'
    assert latest['results'][0]['teacher_decision']['reviewer']=='Teacher'
    assert old['results']==run['results']
    assert Decimal(latest['totals'][0]['score'])==sum(Decimal(x['score']) for x in latest['results'])
    with pytest.raises(ValueError,match='Results changed'):apply_teacher_mark(*args,'1','Teacher')
    row=repo.responses(job)[0]
    repo.review(job,row['id'],action='approve',student='00530',selected=['A'],reviewer='Teacher',expected_sequence=row['review']['sequence'])
    with pytest.raises(ValueError):
        apply_teacher_mark(repo,job,seq,r['student_id'],r['layout_id'],r['question_id'],'1','Teacher')


def test_same_student_exam_in_separate_jobs_can_both_be_marked(tmp_path,fixture_dir,resolver):
    import fitz
    repo,first=completed(tmp_path,fixture_dir,resolver)
    # A distinct upload with identical page content represents another attempt.
    second_pdf=tmp_path/'second-attempt.pdf'
    with fitz.open(fixture_dir/'personalized-mixed-text.pdf') as doc:
        doc.set_metadata({'title':'Second attempt'})
        doc.save(second_pdf)
    second=repo.create(second_pdf,resolver.scope)
    assert second!=first
    run_job(repo,second,resolver)
    for row in repo.responses(second):
        repo.review(second,row['id'],action='approve',student='00530',
                    selected=list(row['response'].selected_options),reviewer='Teacher')
    repo.complete(second)
    a=mark_job(repo,first,resolver,'Teacher')
    b=mark_job(repo,second,resolver,'Teacher')
    assert a['attempt_id']==first and b['attempt_id']==second
    assert a['totals'][0]['student_id']==b['totals'][0]['student_id']=='00530'
    assert len(repo.marking_runs(first))==len(repo.marking_runs(second))==1
