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
    monkeypatch.setattr(view,'CachedManifestResolver',lambda *a,**kw:resolver)
    actual=view.run_job
    def delayed(*args):
        entered.set();gate.wait(5);return actual(*args)
    monkeypatch.setattr(view,'run_job',delayed)
    build_structured_view(page,repository)
    find(page,'label','Staff account ID').value='synthetic-teacher'
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
