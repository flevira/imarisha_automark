import asyncio
from types import SimpleNamespace

from app.views.structured_view import build_structured_view
from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from test_page_alignment import fixture_dir,resolver


class Page:
    def __init__(self):
        self.controls=[];self.overlay=[];self.tasks=[];self.updates=0
    def add(self,*controls):self.controls.extend(controls)
    def update(self):self.updates+=1
    def run_task(self,handler,*args):
        task=asyncio.create_task(handler(*args));self.tasks.append(task);return task


def descendants(root):
    yield root
    for control in getattr(root,'controls',[]) or []:
        yield from descendants(control)
    content=getattr(root,'content',None)
    if content is not None:yield from descendants(content)


def find(page,attribute,value):
    return next(c for c in descendants(page) if getattr(c,attribute,None)==value)


def test_review_screen_shows_crops_and_saves_approval(tmp_path,fixture_dir,resolver):
    repository=JobRepository(tmp_path/'jobs.db')
    job=repository.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    run_job(repository,job,resolver)
    page=Page();build_structured_view(page,repository)
    find(page,'text','shared-tf-text.pdf — review (1 pages)').on_click(None)
    # Validate the actual Flet control serialization without starting a service.
    for control in page.controls:
        assert control._build_add_commands()
    assert any(getattr(c,'src_base64',None) for c in descendants(page))
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'text','Approve current values').on_click(None)
    assert repository.responses(job)[0]['review']['action']=='approve'
    find(page,'text','Complete review').on_click(None)
    assert repository.job(job)['status']=='complete'


def test_ui_worker_yields_to_event_loop_and_cancels(tmp_path,fixture_dir,resolver,monkeypatch):
    import threading
    import app.views.structured_view as view
    repository=JobRepository(tmp_path/'jobs.db')
    page=Page();gate=threading.Event();entered=threading.Event()
    monkeypatch.setattr(view,'ImportedManifestResolver',lambda *a,**kw:resolver)
    actual=view.run_job
    def delayed(*args):
        entered.set();gate.wait(5);return actual(*args)
    monkeypatch.setattr(view,'run_job',delayed)
    build_structured_view(page,repository)
    assert not any('token' in str(getattr(c,'label','')).lower() or getattr(c,'label',None)=='Staff account ID' for c in descendants(page))
    find(page,'label','Answer-sheet PDF path').value=str(fixture_dir/'personalized-mixed-text.pdf')
    async def scenario():
        find(page,'text','Create / resume extraction job').on_click(None)
        for _ in range(100):
            if entered.is_set():break
            await asyncio.sleep(.01)
        assert entered.is_set(), 'Background worker did not start'
        assert not page.tasks[0].done()
        find(page,'text','Cancel after current page').on_click(None)
        gate.set()
        await asyncio.wait_for(page.tasks[0],5)
    asyncio.run(scenario())
    assert repository.jobs()[0]['status']=='cancelled'
    assert not repository.responses(repository.jobs()[0]['id'])


def test_macos_pdf_button_selects_and_preserves_path_on_cancel(tmp_path,monkeypatch):
    import app.views.structured_view as view
    monkeypatch.setattr(view.sys,'platform','darwin')
    selected=iter(['/tmp/Answer sheets.pdf',None])
    monkeypatch.setattr(view,'choose_macos_pdf',lambda:next(selected))
    page=Page();build_structured_view(page,JobRepository(tmp_path/'jobs.db'))
    async def scenario():
        for _ in range(2):
            find(page,'text','Choose PDF').on_click(None)
            await page.tasks[-1]
            assert find(page,'label','Answer-sheet PDF path').value=='/tmp/Answer sheets.pdf'
            assert not find(page,'text','Choose PDF').disabled
    asyncio.run(scenario())


def test_complete_review_explains_blocker_beside_buttons(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    run_job(repo,job,resolver)
    page=Page();build_structured_view(page,repo)
    find(page,'text','shared-tf-text.pdf — review (1 pages)').on_click(None)
    find(page,'text','Complete review').on_click(None)
    assert repo.job(job)['status']=='review'
    assert find(page,'text','Mark / re-mark reviewed answers').disabled
    feedback=[c for c in descendants(page) if str(getattr(c,'value','')).startswith('Review not complete:')]
    assert len(feedback)==2  # Global message and feedback next to the job actions.
    assert '1 responses' in feedback[0].value and 'pages 1' in feedback[0].value
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'text','Approve current values').on_click(None)
    find(page,'text','Complete review').on_click(None)
    assert not find(page,'text','Mark / re-mark reviewed answers').disabled


def test_approval_with_missing_reviewer_shows_inline_error_and_then_saves(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope);run_job(repo,job,resolver)
    page=Page();build_structured_view(page,repo)
    find(page,'text','shared-tf-text.pdf — review (1 pages)').on_click(None)
    find(page,'text','Approve current values').on_click(None)
    assert repo.responses(job)[0]['review'] is None
    feedback=[c for c in descendants(page) if str(getattr(c,'value','')).startswith('Approval not saved:')]
    assert len(feedback)==3  # Question-local, job-local, and global feedback.
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'text','Approve current values').on_click(None)
    assert repo.responses(job)[0]['review']['reviewer']=='Teacher'
    find(page,'value','Latest decision: approve by Teacher')


def test_bulk_approval_covers_job_and_preserves_existing_decisions(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'personalized-mixed-text.pdf',resolver.scope);run_job(repo,job,resolver)
    first=repo.responses(job)[0]
    repo.review(job,first['id'],action='approve',student=first['response'].student_id,selected=['A'],reviewer='Teacher')
    original=repo.responses(job)[0]['review']
    page=Page();build_structured_view(page,repo)
    find(page,'text','personalized-mixed-text.pdf — review (2 pages)').on_click(None)
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'text','Approve all saved responses').on_click(None)
    assert all(r['review'] for r in repo.responses(job))
    assert repo.responses(job)[0]['review']==original
    assert repo.approve_all(job,'Teacher')['approved']==0
    find(page,'text','Complete review').on_click(None)
    assert repo.job(job)['status']=='complete'


def test_saved_review_clears_note_and_collapses_history(tmp_path,fixture_dir,resolver):
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'shared-tf-text.pdf',resolver.scope)
    run_job(repo,job,resolver)
    page=Page();build_structured_view(page,repo)
    find(page,'text','shared-tf-text.pdf — review (1 pages)').on_click(None)
    find(page,'label','Reviewer name').value='Teacher'
    find(page,'label','Correction / review note').value='GOOD'
    find(page,'text','Approve current values').on_click(None)
    assert find(page,'label','Correction / review note').value==''
    assert repo.responses(job)[0]['review']['note']=='GOOD'
    find(page,'value','Current review: approved.')
    tiles=[c for c in descendants(page) if type(c).__name__=='ExpansionTile']
    assert tiles and all(not tile.initially_expanded for tile in tiles)
    for control in page.controls:
        assert control._build_add_commands()
