import pytest
from database.structured_repository import JobRepository
from test_page_alignment import fixture_dir,resolver
from services.extraction_jobs import run_job


def test_delete_job_preserves_other_jobs_shared_source_and_manifest(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db');pdf=fixture_dir/'shared-tf-text.pdf'
    first=repo.create(pdf,resolver.scope);second=repo.create(pdf,'different-scope')
    run_job(repo,first,resolver)
    row=repo.responses(first)[0]
    repo.review(first,row['id'],action='approve',student='00530',selected=['true'],reviewer='Teacher')
    assert repo.delete_saved_data(first)['deleted_jobs']==1
    assert repo.source(second) and not repo.responses(first)
    with repo.connect() as db:
        for table in ('structured_pages','structured_responses','structured_crops','structured_reviews'):
            assert db.execute(f'SELECT count(*) FROM {table} WHERE job=?',(first,)).fetchone()[0]==0
    assert repo.delete_saved_data()['deleted_jobs']==1
    assert repo.jobs()==[] and pdf.exists() and resolver.path.exists()
    with repo.connect() as db:
        assert db.execute('SELECT count(*) FROM structured_sources').fetchone()[0]==0


def test_cleanup_refuses_locked_job(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    with repo.lock(job):
        with pytest.raises(ValueError,match='already running'):repo.delete_saved_data()
    assert len(repo.jobs())==1


def test_full_reset_clears_manifest_and_legacy_but_preserves_files(tmp_path,fixture_dir,resolver):
    import sqlite3
    repo=JobRepository(tmp_path/'jobs.db')
    repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    legacy=tmp_path/'legacy.db'
    with sqlite3.connect(legacy) as db:
        db.execute('CREATE TABLE extractions(id INTEGER PRIMARY KEY, answer TEXT)')
        db.execute("INSERT INTO extractions VALUES(1,'answer')")
    report=tmp_path/'results.html';report.write_text('exported report')
    result=repo.delete_saved_data(manifest_path=resolver.path,legacy_path=legacy)
    assert result['deleted_jobs']==1 and repo.jobs()==[]
    for path,table in [(resolver.path,'layout_cache'),(legacy,'extractions')]:
        with sqlite3.connect(path) as db:assert db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]==0
    assert report.read_text()=='exported report'
    assert (fixture_dir/'shared-tf-text.pdf').exists()
