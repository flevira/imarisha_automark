import pytest
from core.minimal_matching import candidate,pair,distance
from database.structured_repository import JobRepository
from test_page_alignment import fixture_dir,resolver


def q(ref):return dict(referenceAnswers=[ref],maxMarks=5)


def test_cleanup_and_multiple_edits():
    assert candidate(q('School property'),' SCHOOL\n property ')['reason']=='accepted-answer'
    r=candidate(q('Taking care of school property'),'Taking kare of school proparty')
    assert r['accepted'] and r['matches'][0]['distance']==2
    assert r['result']['score']=='5'
    assert distance('kitten','sitting')==3
    assert candidate(q('Emotions'),'Emotioms')['accepted'] is True


def test_threshold():
    assert pair('abcdefghij','abcdxyghij')['similarity']==.8
    assert pair('abcdefghi','abcdxyghi') is None
    assert candidate(q('feeling'),'feelings')['accepted']
    assert not candidate(q('feeling'),'feelings .')['accepted']


@pytest.mark.parametrize('ref,text',[
 ('The right to vote','The right to vote many times'),
 ('The right to be elected','The right to be deleted'),
 ('Use 10 litres of water','Use 20 litres of water'),
 ('Always tell the truth','Never tell the truth'),
 ('Taking care of school property','Not taking care of school property'),
 ('Taking care of school property','Taking care of school property and dancing'),
 ('The cat is sleeping','The bat is sleeping'),
 ('Use sand and extinguishers','Use sand'),
 ('Taking care of school property.','Taking care of school property'),
] )
def test_guards(ref,text):
    assert not candidate(q(ref),text)['accepted']


def test_ambiguous():
    question=q('School proparty');question['acceptedAlternatives']=['School properti']
    assert candidate(question,'School property')['reason']=='ambiguous-complete-answer'


def test_cleanup_retains_extraction_and_refuses_busy(tmp_path):
    repo=JobRepository(tmp_path/'x.db')
    with repo.connect() as db:
        db.execute("INSERT INTO structured_jobs(id,digest,scope,name,status,total) VALUES('j','d','s','n','complete',0)")
        db.execute("INSERT INTO structured_marking_runs(job,result) VALUES('j','{}')")
        db.execute("INSERT INTO marking_checkpoints VALUES('j','i','{}')")
        db.execute("INSERT INTO teacher_examples(scope,text,score,reviewer,provenance) VALUES('s','t','5','r','teacher-score')")
        db.execute("INSERT INTO question_glossaries(scope,material,context,status,model,prompt,reviewer,error,seconds) VALUES('s','{}','{}','draft','m','p','','',0)")
    with repo.lock('j'):
        with pytest.raises(ValueError,match='running'):repo.clear_marking_data()
    assert len(repo.marking_runs('j'))==1
    counts=repo.clear_marking_data()
    assert all(v==1 for v in counts.values())
    assert repo.job('j')['status']=='complete' and repo.marking_runs('j')==[]


def test_cleanup_atomic_rollback(tmp_path):
    repo=JobRepository(tmp_path/'x.db')
    with repo.connect() as db:
        db.execute("INSERT INTO teacher_examples(scope,text,score,reviewer,provenance) VALUES('s','t','5','r','teacher-score')")
        db.execute("CREATE TRIGGER refuse_delete BEFORE DELETE ON teacher_examples BEGIN SELECT RAISE(ABORT,'blocked'); END")
        db.execute("INSERT INTO question_glossaries(scope,material,context,status,model,prompt,reviewer,error,seconds) VALUES('s','{}','{}','draft','m','p','','',0)")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):repo.clear_marking_data()
    with repo.connect() as db:assert db.execute('SELECT COUNT(*) FROM question_glossaries').fetchone()[0]==1


def test_marking_cleanup_allows_remark(tmp_path,fixture_dir,resolver,monkeypatch):
    from test_marking import completed
    from services.marking_jobs import mark_job
    repo,job=completed(tmp_path,fixture_dir,resolver)
    before=[(r['id'],r['review']) for r in repo.responses(job)]
    first=mark_job(repo,job,resolver,'Teacher')
    repo.clear_marking_data()
    assert repo.marking_runs(job)==[]
    def forbidden(*args,**kwargs):raise AssertionError('Removed source must not be loaded')
    monkeypatch.setattr(repo,'glossary',forbidden)
    monkeypatch.setattr(repo,'examples',forbidden)
    second=mark_job(repo,job,resolver,'Teacher')
    assert first['totals']==second['totals']
    assert before==[(r['id'],r['review']) for r in repo.responses(job)]


@pytest.mark.parametrize('reference,student',[
    ('The force increases.','The force increases .'),
    ('A valid explanation!','A valid explanation !'),
    ('Is it reversible?','Is it reversible ?'),
])
def test_terminal_layout_spacing(reference,student):
    result=candidate(q(reference),student)
    assert result['accepted'] and result['reason']=='accepted-answer'
    assert result['matches'][0]['student']==student


@pytest.mark.parametrize('reference,student',[
    ("sister's",'sisters'), ('3.5','35'), ('x + y','x - y'),
    ('not reversible.','reversible .'), ('10 kg','100 kg'),
    ('AB','A B'), ('hers','hers .'), ('x!','x'),
])
def test_layout_cleanup_preserves_meaning(reference,student):
    assert not candidate(q(reference),student)['accepted']
