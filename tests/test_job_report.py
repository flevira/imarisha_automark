from test_marking import completed
from test_page_alignment import fixture_dir, resolver
from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from services.marking_jobs import mark_job
from services.job_report import export_job_report


def test_same_report_before_after_marking_and_after_correction(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'personalized-mixed-text.pdf',resolver.scope)
    run_job(repo,job,resolver)
    path=export_job_report(repo,job,tmp_path/'exports',resolver)
    text=path.read_text()
    assert 'Not marked yet' in text and 'Correct / reference answer' in text
    for row in repo.responses(job):
        raw=row['response']
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),reviewer='Teacher')
    repo.complete(job)
    mark_job(repo,job,resolver,'Teacher')
    assert export_job_report(repo,job,tmp_path/'exports',resolver)==path
    text=path.read_text()
    assert 'Latest saved marking results' in text and '00530' in text
    row=repo.responses(job)[0]
    repo.review(job,row['id'],action='approve',student='00530',selected=['A'],reviewer='Teacher',expected_sequence=row['review']['sequence'])
    export_job_report(repo,job,tmp_path/'exports',resolver)
    assert 'Outdated — re-mark' in path.read_text()
    assert len(list((tmp_path/'exports').iterdir()))==1


def test_ui_exports_before_marking(tmp_path,fixture_dir,resolver,monkeypatch):
    import app.views.structured_view as view
    from test_structured_view import Page,find
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    run_job(repo,job,resolver)
    monkeypatch.setattr(view,'ImportedManifestResolver',lambda *a,**kw:resolver)
    page=Page();view.build_structured_view(page,repo)
    find(page,'text','shared-tf-text.pdf — review (1 pages)').on_click(None)
    find(page,'label','Results export folder').value=str(tmp_path/'exports')
    find(page,'text','Export extraction and marking report').on_click(None)
    reports=list((tmp_path/'exports').glob('*.html'))
    assert len(reports)==1 and 'Not marked yet' in reports[0].read_text()
