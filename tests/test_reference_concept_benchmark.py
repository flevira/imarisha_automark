import pytest
from tools.benchmark_reference_concepts import candidate,compare,html_report
from core.teacher_grounding import explicit_concept_match
from tools.benchmark_nonai_marking import DEFAULT_RULE


def question(reference='dry sand; extinguisher',count=2,marks=5):
    return dict(questionId='1',displayNumber=1,responseType='WRITTEN',status='READY',
                questionText=f'Name {count} tools.',referenceAnswers=[reference],maxMarks=marks)


@pytest.mark.parametrize('text,score',[('dry sand','2.50'),('dry sand; extinguisher','5.00'),
    ('dry sand and dry sand','2.50'),('i) dry sand\nii) extinguisher','5.00')])
def test_credit(text,score):
    r=candidate(question(),text,DEFAULT_RULE)
    assert r['accepted'] and r['result']['score']==score
    assert explicit_concept_match(question(),text,DEFAULT_RULE)['score']==score


def test_three_concepts_extra_alternatives_and_cap():
    q=question('sand; water; extinguisher; blanket',3,10)
    assert candidate(q,'sand',DEFAULT_RULE)['result']['score']=='3.33'
    assert candidate(q,'sand; water',DEFAULT_RULE)['result']['score']=='6.67'
    assert candidate(q,'sand; water; extinguisher; blanket',DEFAULT_RULE)['result']['score']=='10.00'


@pytest.mark.parametrize('text',['not dry sand','dry sand and dancing','dry sand but not extinguisher',
                               'extinguisher?','sand','dry sand or extinguisher'])
def test_unknown_and_contradictory_text(text):
    assert not candidate(question(),text,DEFAULT_RULE)['accepted']


@pytest.mark.parametrize('reference',['sand or water; extinguisher','tools such as sand; extinguisher',
    'bread and butter, milk','sand (dry); extinguisher','sand (dry; extinguisher',
    'sand; water e.g. river water'])
def test_ambiguous_reference(reference):
    assert not candidate(question(reference),'sand',DEFAULT_RULE)['accepted']


def test_no_count_invalid_count_and_case():
    q=question();q['questionText']='Explain the tools.'
    assert not candidate(q,'dry sand',DEFAULT_RULE)['accepted']
    q=question();q['requiredConceptCount']=0
    assert not candidate(q,'dry sand',DEFAULT_RULE)['accepted']
    assert not candidate(question(),'DRY SAND',dict(DEFAULT_RULE,caseSensitive=True))['accepted']


def test_report_labels_only_evaluate_and_html_escapes():
    case=dict(id='1:1',question=question(),rule=DEFAULT_RULE,studentAnswer='dry sand',
              expectedScore=2.5,saved_ai=dict(seconds=4),saved_score='2.5')
    a=compare([case]);b=compare([dict(case,expectedScore=0)])
    assert a['rows'][0]['candidate']==b['rows'][0]['candidate']
    assert a['summary']['newly_resolved']==a['summary']['potential_ai_requests_avoided']==0
    assert a['summary']['excessive_credit']==0 and b['summary']['excessive_credit']==1
    assert a['summary']['saved_ai_seconds']==0
    a['rows'][0]['studentAnswer']='<script>alert(1)</script>'
    assert '<script>' not in html_report(a)


def test_loader_uses_latest_run_read_only(tmp_path):
    import json,sqlite3
    from tools.benchmark_reference_concepts import load_cases
    p=tmp_path/'jobs.db'
    result=dict(scoring_question=question(),student_id='1',display_number=1,rule=DEFAULT_RULE,
                responses=[dict(text='dry sand')],score='2.5')
    with sqlite3.connect(p) as db:
        db.execute('CREATE TABLE structured_marking_runs(sequence INTEGER,job TEXT,result TEXT)')
        for n in (1,2):
            db.execute('INSERT INTO structured_marking_runs VALUES(?,?,?)',(n,'job',json.dumps(dict(results=[result]))))
    before=p.read_bytes()
    cases=load_cases(p,{('1','1'):2.5})
    assert len(cases)==1 and cases[0]['studentAnswer']=='dry sand'
    assert p.read_bytes()==before
