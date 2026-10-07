"""Scheduling tests use controlled fake inference, never a running model."""
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from core.contracts import SourceRegion, Transcription, ReadStatus, ResponseType
from services.handwriting_jobs import recognize_job


class Repository:
    def __init__(self, entries):
        self.owner = threading.get_ident()
        self.state = 'review'
        self.saved = []
        self.rows = []
        for student, page, name in entries:
            raw = SimpleNamespace(response_id=name, student_id=student,
                layout=SimpleNamespace(layout_id='layout'), response_type=ResponseType.WRITTEN,
                review_reasons=('handwriting-recognition-pending',),
                regions=(SourceRegion('source', page, (0, 0, 10, 10), name),))
            self.rows.append(dict(response=raw, page=page, review=None, transcription=None))
    def check(self):
        assert threading.get_ident() == self.owner, 'Workers must not access repository'
    def lock(self, job):
        self.check(); return nullcontext()
    def job(self, job):
        self.check(); return dict(status=self.state, pages=[dict(status='extracted')])
    def responses(self, job):
        self.check(); return self.rows
    def status(self, job, status):
        self.check(); self.state = status
    def crop(self, job, ref):
        self.check(); return ref.encode()
    def save_transcription(self, job, response_id, draft, language):
        self.check(); self.saved.append((response_id, draft))
        row = next(r for r in self.rows if r['response'].response_id == response_id)
        row['transcription'] = dict(asdict(draft), language=language)


class Recognizer:
    model = 'test'
    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []
    def recognize(self, request):
        name = request.image_png.decode()
        self.calls.append(name)
        if self.callback:
            self.callback(name)
        return Transcription(name, ReadStatus.READ, (), 'ollama:test')


def test_two_pages_overlap_within_student_and_keep_response_identity():
    repo = Repository([('001', 1, 'a'), ('001', 1, 'b'), ('001', 2, 'c'), ('002', 3, 'd')])
    barrier = threading.Barrier(2)
    c_finished = threading.Event()
    def infer(name):
        if name in ('a', 'c'):
            barrier.wait(timeout=3)
            if name == 'a':
                assert c_finished.wait(3)
            else:
                c_finished.set()
        if name == 'd':
            assert {key for key, _ in repo.saved} == {'a', 'b', 'c'}
    engine = Recognizer(infer)
    states = []
    result = recognize_job(repo, 'job', engine, workers=2, progress=states.append)
    assert result['completed'] == result['total'] == 4
    assert any(len(s['active_pages']) == 2 for s in states)
    assert all(len(s['active_pages']) <= 2 for s in states)
    assert all(key == draft.text for key, draft in repo.saved)
    assert all(draft.status == ReadStatus.UNCERTAIN for _, draft in repo.saved)
    assert repo.state == 'review'
    recognize_job(repo, 'job', engine, workers=2)
    assert len(engine.calls) == 4  # Saved successes are not repeated.


def test_cancel_drains_two_inflight_requests_then_resume():
    repo = Repository([('001', 1, 'a'), ('001', 2, 'b'), ('001', 3, 'c')])
    barrier = threading.Barrier(2)
    cancel = threading.Event()
    def infer(name):
        barrier.wait(timeout=3)
        cancel.set()
    engine = Recognizer(infer)
    result = recognize_job(repo, 'job', engine, cancel=cancel, workers=2)
    assert result['completed'] == 2 and len(repo.saved) == 2
    assert repo.state == 'recognition-cancelled'
    engine.callback = None
    recognize_job(repo, 'job', engine, workers=2)
    assert sorted(engine.calls) == ['a', 'b', 'c']
    assert repo.state == 'review'


def test_failure_retains_ocr_reduces_pressure_and_can_retry():
    repo = Repository([('001', 1, 'a'), ('001', 2, 'b'), ('001', 3, 'c')])
    repo.rows[0]['transcription'] = dict(text='old OCR', status='uncertain',
        engine='tesseract-written-v1', language='en-sw', review_reasons=['ocr-low-confidence'])
    def infer(name):
        if name == 'a':
            raise RuntimeError('timeout')
    engine = Recognizer(infer)
    states = []
    result = recognize_job(repo, 'job', engine, workers=2, progress=states.append)
    assert result['errors'] == 1 and result['reduced_to_serial']
    assert repo.rows[0]['transcription']['text'] == 'old OCR'
    assert 'ai-failed-ocr-text-retained' in repo.rows[0]['transcription']['review_reasons']
    engine.callback = None
    recognize_job(repo, 'job', engine, workers=2)
    assert engine.calls.count('a') == 2
    assert engine.calls.count('b') == engine.calls.count('c') == 1


def test_default_serial_and_approved_or_confident_ocr_skipped():
    repo = Repository([('001', 1, 'a'), ('001', 2, 'b'), ('001', 3, 'c'), ('001', 4, 'd')])
    repo.rows[0]['review'] = {'action': 'approve'}
    repo.rows[1]['transcription'] = dict(text='good', status='read', engine='tesseract-written-v1')
    states = []
    result = recognize_job(repo, 'job', Recognizer(), progress=states.append)
    assert result['completed'] == 2 and result['workers'] == 1
    assert all(len(s['active_pages']) <= 1 for s in states)


@pytest.mark.parametrize('workers', [0, 3, True, '2'])
def test_invalid_concurrency_rejected(workers):
    with pytest.raises(ValueError):
        recognize_job(Repository([]), 'job', Recognizer(), workers=workers)


def test_shared_ollama_client_verifies_once(monkeypatch):
    from test_handwriting import fake_client, request
    client, calls = fake_client(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.recognize(request()), range(2)))
    assert all(r.text == 'Mti ni mmea.' for r in results)
    assert [path for path, _ in calls].count('/api/show') == 1
    assert [path for path, _ in calls].count('/api/chat') == 2
