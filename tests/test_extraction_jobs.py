import json
import threading
from pathlib import Path

import pytest

from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from test_page_alignment import fixture_dir, resolver


@pytest.fixture
def repository(tmp_path):
    return JobRepository(tmp_path/'jobs.db')


def create(repository,fixture_dir,resolver,name='personalized-mixed-text'):
    return repository.create(fixture_dir/f'{name}.pdf',resolver.scope)


def approve(repository,job,row,**overrides):
    args=dict(action='approve',student=row['response'].student_id,selected=list(row['response'].selected_options),
              reviewer='Synthetic teacher',note='',expected_sequence=row['review']['sequence'] if row['review'] else None)
    args.update(overrides)
    repository.review(job,row['id'],**args)


def test_restart_resume_and_duplicate_upload_keep_original_records(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver)
    def interrupt(n,total):
        raise SystemExit('Simulated process interruption after first page commit')
    with pytest.raises(SystemExit):
        run_job(repository,job,resolver,progress=interrupt)
    assert repository.job(job)['status']=='interrupted'
    raw=[r['raw'] for r in repository.responses(job)]
    assert len(raw)==3
    reopened=JobRepository(repository.path)
    assert create(reopened,fixture_dir,resolver)==job
    run_job(reopened,job,resolver)
    assert len(reopened.responses(job))==5
    assert [r['raw'] for r in reopened.responses(job)][:3]==raw
    run_job(reopened,job,resolver)
    assert len(reopened.responses(job))==5


def test_cancellation_commits_current_page_and_resumes(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver)
    cancel=threading.Event()
    run_job(repository,job,resolver,cancel,lambda n,total:cancel.set())
    assert repository.job(job)['status']=='cancelled'
    assert [p['status'] for p in repository.job(job)['pages']]==['extracted','pending']
    with pytest.raises(ValueError):repository.complete(job)
    run_job(repository,job,resolver)
    assert len(repository.responses(job))==5


def test_cancel_before_start(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver)
    cancel=threading.Event();cancel.set()
    run_job(repository,job,resolver,cancel)
    assert not repository.responses(job)
    assert repository.job(job)['status']=='cancelled'


def test_source_moved_database_reopened_and_crops_remain(repository,fixture_dir,resolver,tmp_path):
    source=tmp_path/'uploaded.pdf'; source.write_bytes((fixture_dir/'shared-tf-scan.pdf').read_bytes())
    job=repository.create(source,resolver.scope)
    source.rename(tmp_path/'moved.pdf')
    run_job(JobRepository(repository.path),job,resolver)
    reopened=JobRepository(repository.path)
    row=reopened.responses(job)[0]
    assert row['response'].student_id=='00530'
    assert reopened.crop(job,row['response'].regions[0].crop_ref).startswith(b'\x89PNG')
    assert reopened.page_image(job,1).startswith(b'\x89PNG')
    assert reopened.source(job).startswith(b'%PDF')


def test_partial_failure_and_explicit_retry(repository,fixture_dir,resolver,monkeypatch):
    import services.extraction_jobs as service
    real=service.prepare_image
    def broken(image,source,number,resolver):
        if number==1:raise ValueError('Synthetic page failure')
        return real(image,source,number,resolver)
    monkeypatch.setattr(service,'prepare_image',broken)
    job=create(repository,fixture_dir,resolver)
    run_job(repository,job,resolver)
    assert [p['status'] for p in repository.job(job)['pages']]==['failed','extracted']
    assert len(repository.responses(job))==2
    monkeypatch.setattr(service,'prepare_image',real)
    run_job(repository,job,resolver,retry_failed=True)
    assert len(repository.responses(job))==5
    assert not repository.issues(job)


def test_correction_is_append_only_and_approval_survives_reopen(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver,'shared-tf-scan')
    run_job(repository,job,resolver)
    row=repository.responses(job)[0];original=row['raw']
    approve(repository,job,row,selected=['true'],note='Checked bubble crop; corrected selection')
    revised=repository.responses(job)[0]
    assert revised['raw']==original
    assert json.loads(revised['review']['selected'])==['true']
    with pytest.raises(ValueError,match='changed'):approve(repository,job,row)
    approve(repository,job,revised,selected=[],note='Teacher confirmed blank')
    repository.complete(job)
    reopened=JobRepository(repository.path)
    assert reopened.job(job)['status']=='complete'
    assert reopened.responses(job)[0]['raw']==original
    with reopened.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM structured_reviews').fetchone()[0]==2


@pytest.mark.parametrize('change', [dict(selected=['Z']),dict(selected=['true','false']),dict(student='530'),dict(reviewer=''),dict(student='00531')])
def test_invalid_or_unexplained_correction_is_rejected(repository,fixture_dir,resolver,change):
    job=create(repository,fixture_dir,resolver,'shared-tf-text');run_job(repository,job,resolver)
    with pytest.raises(ValueError):approve(repository,job,repository.responses(job)[0],**change)
    assert repository.responses(job)[0]['review'] is None


def test_completion_requires_every_response_review(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver);run_job(repository,job,resolver)
    with pytest.raises(ValueError):repository.complete(job)
    for row in repository.responses(job):approve(repository,job,row)
    repository.complete(job)
    assert repository.job(job)['status']=='complete'


def test_worker_lock_and_account_scope(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver)
    with repository.lock(job):
        with pytest.raises(ValueError,match='running'):run_job(repository,job,resolver)
    resolver.scope='another-account'
    with pytest.raises(ValueError,match='same backend'):run_job(repository,job,resolver)


def test_failed_page_needs_explicit_document_resolution(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver,'legacy-control-scan');run_job(repository,job,resolver)
    with pytest.raises(ValueError):repository.complete(job)
    issue=repository.issues(job)[0]
    repository.resolve_issue(job,issue['key'],'Teacher','Excluded legacy sheet from this structured job; process in legacy workflow')
    repository.complete(job)
    assert not repository.responses(job)


def test_page_commit_rolls_back_crops_and_responses_together(repository,fixture_dir,resolver,monkeypatch):
    job=create(repository,fixture_dir,resolver)
    with repository.connect() as db:
        db.execute("CREATE TRIGGER simulate_disk_failure BEFORE INSERT ON structured_crops BEGIN SELECT RAISE(ABORT,'synthetic disk error'); END")
    run_job(repository,job,resolver)
    assert not repository.responses(job)
    assert all(p['status']=='failed' for p in repository.job(job)['pages'])
    with repository.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM structured_crops').fetchone()[0]==0
        db.execute('DROP TRIGGER simulate_disk_failure')
    run_job(repository,job,resolver,retry_failed=True)
    assert len(repository.responses(job))==5


def test_duplicate_pages_cannot_be_completed_with_duplicate_approvals(repository,fixture_dir,resolver,tmp_path):
    import fitz
    path=tmp_path/'duplicates.pdf'
    with fitz.open(fixture_dir/'shared-tf-text.pdf') as source,fitz.open() as output:
        output.insert_pdf(source);output.insert_pdf(source);output.save(path)
    job=repository.create(path,resolver.scope);run_job(repository,job,resolver)
    issue=repository.issues(job)[0]
    for row in repository.responses(job):approve(repository,job,row)
    repository.resolve_issue(job,issue['key'],'Teacher','Duplicate confirmed; exclude the second copy')
    with pytest.raises(ValueError,match='duplicate'):repository.complete(job)
    row=repository.responses(job)[1]
    approve(repository,job,row,action='exclude',note='Duplicate scan of page one')
    repository.complete(job)
    assert len(repository.responses(job))==2


def test_uncertain_original_cannot_silently_become_blank(repository,fixture_dir,resolver,monkeypatch):
    from dataclasses import replace
    from core.contracts import ReadStatus
    import services.extraction_jobs as service
    extract=service.StructuredExtractor.extract_page
    def uncertain(self,page):
        result=extract(self,page)
        result.responses=tuple(replace(r,status=ReadStatus.UNCERTAIN,selected_options=('true','false'),review_reasons=('multiple-selections',)) for r in result.responses)
        return result
    monkeypatch.setattr(service.StructuredExtractor,'extract_page',uncertain)
    job=create(repository,fixture_dir,resolver,'shared-tf-text');run_job(repository,job,resolver)
    row=repository.responses(job)[0]
    with pytest.raises(ValueError):approve(repository,job,row,selected=[])
    approve(repository,job,row,selected=[],note='Teacher explicitly confirmed no valid response')
    saved=repository.responses(job)[0]
    assert saved['response'].status==ReadStatus.UNCERTAIN
    assert saved['response'].selected_options==('true','false')
    assert json.loads(saved['review']['selected'])==[]


def test_approved_page_identity_must_be_consistent(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver);run_job(repository,job,resolver)
    for i,row in enumerate(repository.responses(job)):
        approve(repository,job,row,student='00531' if i==0 else '00530',note='Synthetic identity correction' if i==0 else '')
    with pytest.raises(ValueError,match='same student'):repository.complete(job)


def test_two_reviewers_cannot_overwrite_each_other(repository,fixture_dir,resolver):
    job=create(repository,fixture_dir,resolver);run_job(repository,job,resolver)
    row=repository.responses(job)[0]
    approve(repository,job,row)
    with pytest.raises(ValueError,match='changed'):
        approve(JobRepository(repository.path),job,row,reviewer='Second reviewer')
