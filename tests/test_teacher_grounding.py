import copy
import json
import pytest
from core.teacher_grounding import (approved_match,explicit_concept_match,reference_examples,
    question_scope,retrieve_examples,uncertainty)
from database.structured_repository import JobRepository
from core.ai_marking import LocalMarkingAssistant


def question():
    return dict(questionId='20',questionText='Identify two tools.',responseType='WRITTEN',maxMarks=5,
        referenceAnswers=['Buckets filled with dry sand and portable dry chemical fire extinguishers'],
        requiredConceptCount=2,acceptedConcepts=[
            dict(id='sand',description='Buckets filled with dry sand',acceptedAlternatives=['A bucket of dry sand','sand bucket']),
            dict(id='ext',description='portable dry chemical fire extinguishers',acceptedAlternatives=['dry chemical extinguisher'])])


def test_teacher_reference_is_full_credit_without_second_approval():
    q=question();examples=reference_examples(q)
    assert examples[0]['approved'] and examples[0]['provenance']=='teacher-reference'
    assert approved_match(q['referenceAnswers'][0],examples)['score']=='5'


def test_explicit_partial_credit_duplicates_unknowns_and_negation():
    q=question()
    assert explicit_concept_match(q,'A bucket of dry sand')['score']=='2.50'
    assert explicit_concept_match(q,'sand bucket and sand bucket')['score']=='2.50'
    assert explicit_concept_match(q,'sand bucket and dry chemical extinguisher')['score']=='5.00'
    assert explicit_concept_match(q,'not a bucket of dry sand') is None
    assert explicit_concept_match(q,'A bucket of dry sand and a broom') is None
    assert explicit_concept_match(dict(q,acceptedConcepts=None),'sand bucket') is None


def test_scope_changes_with_rules_and_criteria_and_teacher_examples_persist(tmp_path):
    q=question();scope=question_scope(q)
    assert scope!=question_scope(dict(q,maxMarks=6))
    assert scope!=question_scope(q,dict(mode='different'))
    repo=JobRepository(tmp_path/'jobs.db')
    repo.save_example(scope,'A bucket',2.5,'Teacher')
    assert JobRepository(repo.path).examples(scope)[0]['score']=='2.5'
    repo.save_example(scope,'A bucket',0,'Teacher','job')
    assert len(repo.examples(scope))==1 and repo.examples(scope)[0]['score']=='0'
    repo.delete_saved_data()
    assert repo.examples(scope)==[]


def test_examples_are_approved_only_and_conflicts_block():
    q=question();e=dict(text='One tool',score='2.5',approved=True,provenance='teacher-score')
    assert approved_match('one tool',[e])['score']=='2.5'
    assert approved_match('one tool',[dict(e,approved=False)]) is None
    assert approved_match('one tool',[e,dict(e,score='5')])['status']=='review'
    assert retrieve_examples(q,'One tool',[e,dict(e,approved=False,text='bad')])[1]==e


def test_uncertainty_range_and_categories():
    from core.written_rubric import prepare_rubric,grade_judgments
    r=grade_judgments(prepare_rubric(question()),[
        dict(conceptId='sand',status='matched',segmentIds=['s1']),
        dict(conceptId='ext',status='uncertain',segmentIds=[])],[dict(id='s1',text='sand bucket')],5)
    assert r['score_range']==dict(lower='2.50',upper='5.00')
    assert uncertainty(dict(r,ai=r))['category']=='meaning'
    assert uncertainty(dict(status='review',reason='ai-invalid-evidence-segment'))['category']=='technical'
    assert uncertainty(dict(status='review',reason='named-concept-rubric-required'))['category']=='rubric'


def test_technical_retry_once_and_semantic_uncertainty_no_retry(monkeypatch):
    ai=LocalMarkingAssistant();calls=[]
    def assess(q,t):
        calls.append(1)
        return dict(status='review',score=None,reason='ai-invalid-output',seconds=1)
    monkeypatch.setattr(ai,'_assess_once',assess)
    r=ai.assess(question(),'x');assert len(calls)==2 and r['seconds']==2
    monkeypatch.setattr(ai,'_assess_once',lambda q,t:dict(status='review',score=None,reason='ai-unresolved',seconds=1))
    assert len(ai.assess(question(),'x')['attempts'])==1


def test_rubric_draft_uses_details_without_student_or_mutation(monkeypatch):
    from services.rubric_drafts import draft_rubric
    from core.handwriting import OllamaHandwritingRecognizer
    q=question();q['answerDetail']='Sand smothers a small fire.';original=copy.deepcopy(q)
    def request(self,path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        sent=json.loads(payload['messages'][1]['content'])
        assert sent['answerDetail']==q['answerDetail'] and 'studentAnswer' not in sent
        return dict(done=True,message=dict(content=json.dumps(dict(mode='any-n',concepts=[
            dict(description=c['description'],marks='0',acceptedAlternatives=c['acceptedAlternatives']) for c in q['acceptedConcepts']]))))
    monkeypatch.setattr(OllamaHandwritingRecognizer,'_request',request)
    draft=draft_rubric(q,'local')
    assert draft['approved'] is False and q==original
    assert draft['context']['requiredConceptCount']==2


def test_evaluation_has_metrics_and_rejects_example_leakage():
    from tools.evaluate_written_marking import evaluate
    class Never:
        def assess(self,*a):pytest.fail('Reference must be deterministic')
    q=question()
    case=dict(question=q,studentAnswer=q['referenceAnswers'][0],expectedScore='5',teacherApproved=True,language='en')
    r=evaluate([case],Never())
    assert r['exact_agreement_all_labelled']==1 and r['mean_absolute_error_scored']==0
    assert r['by_language']['en']['teacher_labelled']==1
    q['approvedExamples']=[dict(text=case['studentAnswer'],score='5',approved=True,provenance='teacher-score')]
    with pytest.raises(ValueError,match='leaked'):evaluate([case],Never())

from test_written_extraction import setup,read


def test_saved_teacher_score_reused_without_ai_and_groups_preserve_override(setup,monkeypatch):
    from types import SimpleNamespace
    from services.marking_jobs import mark_job,apply_teacher_mark
    from services.teacher_groups import review_groups,mark_group
    _,manifest,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        raw=row['response']
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),reviewer='Teacher',
            raw_text=('one tool' if raw.question_id=='4' else raw.raw_text or '') if raw.response_type.value=='WRITTEN' else None)
    repo.complete(job)
    m=copy.deepcopy(manifest);m['answerSchemaVersion']=1
    m['questions']=[dict(questionId=str(i),displayNumber=i,maxMarks=2,responseType='MC' if i==1 else 'WRITTEN',
        questionText=f'Question {i}',status='READY' if i==1 else 'MANUAL_REVIEW',referenceAnswers=['two tools']) for i in range(1,6)]
    m['pages'][0]['rows'][0].update(correctAnswers=['A'],questionMarks=2,answerStatus='READY')
    res=SimpleNamespace(scope=resolver.scope,resolve=lambda *a:m)
    run=mark_job(repo,job,res,'Teacher')
    g=next(g for g in review_groups(repo,[job]) if g['number']==4)
    from test_structured_view import Page,find
    from app.views.structured_view import build_structured_view
    page=Page();page.open=lambda dialog:page.controls.append(dialog)
    page.close=lambda dialog:page.controls.remove(dialog)
    build_structured_view(page,repo)
    find(page,'text','Select all scripts').on_click(None)
    find(page,'text','Review answer groups').on_click(None)
    assert any(getattr(c,'title',None) and getattr(c.title,'value',None)=='Review answer groups' for c in page.controls)
    for control in page.controls:assert control._build_add_commands()
    page.close(page.controls[-1])
    find(page,'text',f"{repo.job(job)['name']} — complete ({repo.job(job)['total']} pages)").on_click(None)
    import services.rubric_drafts as drafts
    def fake_draft(q,model):
        return dict(context=dict(questionText=q['questionText'],scoringCriteria=[dict(id='a',description='Essential idea',marks=2)],
            acceptedConcepts=None,requiredConceptCount=None),referenceAnswers=q['referenceAnswers'],answerDetail='Background only')
    monkeypatch.setattr(drafts,'draft_rubric',fake_draft)
    import asyncio
    async def open_draft():
        find(page,'text','Draft Q4 rubric from teacher answer').on_click(None)
        await page.tasks[-1]
    asyncio.run(open_draft())
    dialog=page.controls[-1]
    assert dialog.title.value=='Review proposed rubric'
    before=len(repo.marking_runs(job))
    dialog.actions[1].on_click(None)  # Unconfirmed draft does nothing.
    assert len(repo.marking_runs(job))==before
    find(page,'label','I reviewed the concepts, alternatives and partial-credit allocation').value=True
    dialog.actions[1].on_click(None)
    assert len(repo.marking_runs(job))==before  # Approval stages, never silently re-marks.
    result=mark_group(repo,g['members'],'1','Teacher','One of two concepts',review_seconds=12)
    assert repo.marking_runs(job)[0]['review_seconds']==12

    assert result[0]['status']=='saved'
    assert all(g['number']!=4 for g in review_groups(repo,[job]))
    latest=repo.marking_runs(job)[0]
    answer=next(r for r in latest['results'] if r['display_number']==4)
    assert repo.examples(answer['example_scope'])[0]['score']=='1'
    rerun=mark_job(repo,job,res,'Teacher')
    answer=next(r for r in rerun['results'] if r['display_number']==4)
    assert answer['score']=='1' and answer['teacher_decision']
    with pytest.raises(ValueError,match='Results changed'):mark_group(repo,g['members'],'0','Teacher')
    # A fresh run with identical reviewed text can use the approved example even
    # without an individual override attached to that run.
    old=repo.marking_runs(job)[0]
    for r in old['results']:r.pop('teacher_decision',None)
    old.pop('sequence');old.pop('created');repo.save_marking_run(job,old)
    rerun=mark_job(repo,job,res,'Teacher')
    answer=next(r for r in rerun['results'] if r['display_number']==4)
    assert answer['score']=='1' and answer['reason']=='approved-example-match'


def test_weighted_alternatives_and_finite_uncertainty():
    q=dict(questionId='18',questionText='Define corruption.',maxMarks=5,referenceAnswers=['Abuse of office for private gain'],
        scoringCriteria=[dict(id='abuse',description='Abuse of entrusted power',marks=3,acceptedAlternatives=['misusing public office']),
                         dict(id='gain',description='For private benefit',marks=2)])
    assert explicit_concept_match(q,'misusing public office')['score']=='3'
    from core.written_rubric import grade_judgments,prepare_rubric
    r=grade_judgments(prepare_rubric(q),[dict(conceptId='abuse',status='matched',segmentIds=['s1']),
        dict(conceptId='gain',status='uncertain',segmentIds=[])],[dict(id='s1',text='misusing public office')],5)
    assert r['score_range']==dict(lower='3',upper='5.00')


def test_group_suggestions_never_merge_unconfirmed_or_different_rubrics():
    from services.teacher_groups import review_groups
    from services.marking_jobs import digest
    def result(text,scope='same',override=False):
        return dict(scoring_question=dict(responseType='WRITTEN'),responses=[dict(text=text)],example_scope=scope,
            question_text='Name a resource.',display_number=1,max_marks='5',reference_answers=['water'],
            student_id='00530',layout_id='layout',question_id='1',score=None,
            **(dict(teacher_decision={"reviewer":"Teacher"}) if override else {}))
    values={'a':result('water'), 'b':result('water'), 'c':result('not water'),
            'd':result('water','different-rubric'), 'e':result('water',override=True)}
    class Repo:
        def marking_runs(self,job):return [dict(sequence=1,response_digest=digest([]),results=[values[job]])]
        def job(self,job):return dict(status='complete',name=job)
        def responses(self,job):return []
    groups=review_groups(Repo(),values)
    same=next(g for g in groups if g['scope']=='same' and g['text']=='water')
    assert len(same['members'])==2
    assert all(m['student']=='00530' for m in same['members'])
    assert len(groups)==3
    assert all(m['job']!='e' for g in groups for m in g['members'])


def test_holistic_uncertain_empty_evidence_is_review_not_invalid(monkeypatch):
    ai=LocalMarkingAssistant();ai.client._verified=True;ai.client.model_digest='a'*64
    calls=[]
    def request(*args):
        calls.append(1)
        return dict(done=True,message=dict(content=json.dumps(dict(relation='uncertain',segmentIds=[]))))
    monkeypatch.setattr(ai.client,'_request',request)
    result=ai.assess(dict(questionText='Define a word.',referenceAnswers=['definition'],maxMarks=5),'unclear')
    assert result['reason']=='ai-unresolved' and len(calls)==1


def test_invalid_json_retries_and_recovers(monkeypatch):
    ai=LocalMarkingAssistant();ai.client._verified=True;ai.client.model_digest='a'*64
    replies=iter(['not JSON',json.dumps(dict(relation='equivalent',segmentIds=['s1']))])
    monkeypatch.setattr(ai.client,'_request',lambda *a:dict(done=True,message=dict(content=next(replies))))
    result=ai.assess(dict(referenceAnswers=['water'],maxMarks=5),'maji')
    assert result['score']=='5' and len(result['attempts'])==2
    assert result['attempts'][0]['reason']=='ai-invalid-json'
