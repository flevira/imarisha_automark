import json
import pytest
from core.ai_marking import LocalMarkingAssistant
from core.marking import mark_question
from test_marking import question,answer,completed
from test_page_alignment import fixture_dir,resolver
from services.marking_jobs import mark_job
from test_written_extraction import setup,read


def test_matching_partial_and_ma_wrong_selection_deduction():
    q,row=question('MATCH',5)
    result=mark_question(q,[row,row,row],[answer(['A']),answer(['B']),answer(['A'])],dict(version=1,mode='partial'))
    assert result['score']=='3.33'
    q,row=question('MA',4);row['correctAnswers']=['A','C']
    for selected,expected in [(['A'],'2.00'),(['A','C'],'4'),(['A','B','C'],'2.00'),(['A','B','C','D'],'0.00'),([], '0.00')]:
        assert mark_question(q,[row],[answer(selected)],dict(version=1,mode='partial'))['score']==expected

def test_structured_marking_bypasses_ai(tmp_path,fixture_dir,resolver):
    repo,job=completed(tmp_path,fixture_dir,resolver)
    class Never:
        def assess(self,*args):raise AssertionError('Structured answers must bypass AI')
    result=mark_job(repo,job,resolver,'Teacher',assistant=Never())
    assert result['ai_seconds']==0
    assert all('ai' not in r for r in result['results'])

def test_written_ai_scores_are_separate_audited_and_repeatable(setup):
    from types import SimpleNamespace
    import copy
    _,manifest,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        raw=row['response']
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),reviewer='Teacher',
                    raw_text=('twelve' if raw.question_id=='4' else raw.raw_text or '') if raw.response_type.value=='WRITTEN' else None)
    repo.complete(job)
    m=copy.deepcopy(manifest);m['answerSchemaVersion']=1
    m['questions']=[dict(questionId=str(i),displayNumber=i,maxMarks=2,responseType='MC' if i==1 else 'WRITTEN',
                        status='READY' if i==1 else 'MANUAL_REVIEW',answerDetail='Twelve is the number after eleven.',referenceAnswers=[
                        'First line Second line Continued response' if i==2 else '12']) for i in range(1,6)]
    m['pages'][0]['rows'][0].update(correctAnswers=['A'],questionMarks=2,answerStatus='READY')
    manifest_resolver=SimpleNamespace(scope=resolver.scope,resolve=lambda *a:m)
    class AI:
        calls=[]
        def assess(self,q,text):
            self.calls.append((q['questionId'],text))
            assert q['answerDetail']=='Twelve is the number after eleven.'
            return dict(status='scored',score='2',reason='ai-equivalent',model='fake',model_digest='a'*64,seconds=.1,
                        proposal=dict(evidence='twelve',explanation='Equivalent number'))
    ai=AI()
    result=mark_job(repo,job,manifest_resolver,'Teacher',assistant=ai)
    assert ai.calls==[('4','twelve')]
    r=next(r for r in result['results'] if r['question_id']=='4')
    assert r['score']=='2' and r['engine']=='local-ai-v1'
    assert repo.marking_runs(job)[0]['results']==result['results']
    assert result['ai_seconds']==.1
    # Turning assistance off leaves only the equivalent wording unresolved.
    second=mark_job(repo,job,manifest_resolver,'Teacher')
    assert next(r for r in second['results'] if r['question_id']=='4')['score'] is None
    assert len(repo.marking_runs(job))==2
    context={'questionText':'Write twelve as a word.', 'scoringCriteria':[
        dict(id='number',description='The word twelve',marks=2)]}
    policy=dict(version=1,writtenContext={m['layoutId']:{'4':context}})
    class ContextAI:
        def assess(self,q,text):
            assert q['questionText']==context['questionText']
            assert q['scoringCriteria']==context['scoringCriteria']
            return dict(status='review',score=None,reason='ai-unresolved',seconds=0)
    third=mark_job(repo,job,manifest_resolver,'Teacher',policy=policy,assistant=ContextAI())
    assert third['policy']==policy
    pending=next(r for r in third['results'] if r['question_id']=='4')
    assert pending['score'] is None and pending['similarity_estimate']['requires_teacher_review']
    bad=dict(version=1,writtenContext={'wrong-layout':{'4':context}})
    with pytest.raises(ValueError,match='Unknown written-context layout'):
        mark_job(repo,job,manifest_resolver,'Teacher',policy=bad)


def provider(monkeypatch,reply,check=None):
    ai=LocalMarkingAssistant();ai.client._verified=True;ai.client.model_digest='a'*64
    def request(path,payload):
        assert path=='/api/chat'
        sent=json.loads(payload['messages'][1]['content'])
        if 'verificationPairs' in sent:
            return dict(done=True,message=dict(content=json.dumps(dict(checks=[dict(conceptId=p['conceptId'],relation='same-meaning') for p in sent['verificationPairs']]))))
        if check:check(sent)
        value=reply(json.loads(payload['messages'][1]['content'])) if callable(reply) else reply
        return dict(done=True,message=dict(content=json.dumps(value)))
    monkeypatch.setattr(ai.client,'_request',request)
    return ai


@pytest.mark.parametrize('relation,score,status',[
    ('equivalent','5','scored'),('incorrect','0','scored'),('partly-correct',None,'review'),('uncertain',None,'review')])
def test_holistic_scores_are_application_owned(monkeypatch,relation,score,status):
    q=dict(questionText='Define corruption.',referenceAnswers=['Misuse of power for private gain'],
           answerDetail='It damages public trust and slows development.',maxMarks=5)
    def check(payload):
        assert 'answerDetail' not in payload
        assert payload['gradedExamples'][0]['score']=='5'
        assert payload['rubric']['mode']=='holistic'
        assert payload['rubric']['concepts']==[]
    result=provider(monkeypatch,dict(relation=relation,segmentIds=['s1']),check).assess(q,'Using public power for personal benefit')
    assert result['status']==status and result['score']==score
    assert result['background_context']==q['answerDetail']


@pytest.mark.parametrize('status,expected',[('matched','2.50'),('not-demonstrated','0.00'),('uncertain',None)])
def test_named_concepts_bound_partial_credit(monkeypatch,status,expected):
    q=dict(questionText='Identify two tools.',referenceAnswers=['Dry sand buckets and fire extinguishers'],maxMarks=5)
    def reply(data):
        cs=data['rubric']['concepts']
        assert cs[0]['description']=='Dry sand buckets'
        return dict(judgments=[dict(conceptId=cs[0]['id'],status=status,segmentIds=['s1']),
                              dict(conceptId=cs[1]['id'],status='not-demonstrated',segmentIds=[])])
    r=provider(monkeypatch,reply).assess(q,'A bucket of dry sand.')
    assert r['score']==expected
    assert 'score' not in r['proposal'] and 'explanation' not in r['proposal']
    if expected:assert expected in r['explanation']


def test_no_invented_concepts_or_segments_or_aggregate_scores(monkeypatch):
    q=dict(questionText='Name two tools.',referenceAnswers=['sand and extinguishers'],maxMarks=5)
    def valid(data):return dict(judgments=[dict(conceptId=c['id'],status='matched',segmentIds=['s1']) for c in data['rubric']['concepts']])
    for mutation in ['concept','segment','duplicate','score']:
        def bad(data):
            value=valid(data)
            if mutation=='concept':value['judgments'][0]['conceptId']='broom'
            if mutation=='segment':value['judgments'][0]['segmentIds']=['invented']
            if mutation=='duplicate':value['judgments'][1]['conceptId']=value['judgments'][0]['conceptId']
            if mutation=='score':value['score']='5'
            return value
        assert provider(monkeypatch,bad).assess(q,'Sand and extinguishers')['status']=='review'


def test_segment_ids_handle_numbering_and_linebreaks(monkeypatch):
    q=dict(questionText='State two problems.',referenceAnswers=['Insufficient funds and too few staff'],maxMarks=5)
    def reply(data):
        return dict(judgments=[dict(conceptId=c['id'],status='matched',segmentIds=[data['studentSegments'][i]['id']]) for i,c in enumerate(data['rubric']['concepts'])])
    r=provider(monkeypatch,reply).assess(q,'(i) Not enough money.\n(ii) Too few trained\npeople.')
    assert r['score']=='5.00'
    assert r['evidence']


def test_explicit_definition_rubric_allows_partial_marks(monkeypatch):
    q=dict(questionText='Define corruption.',referenceAnswers=['Abuse of power for personal benefit'],maxMarks=5,
        scoringCriteria=[dict(id='abuse',description='Dishonest misuse of entrusted authority or resources',marks=3),
                         dict(id='gain',description='For private benefit',marks=2)])
    r=provider(monkeypatch,dict(judgments=[dict(conceptId='abuse',status='matched',segmentIds=['s1']),
        dict(conceptId='gain',status='not-demonstrated',segmentIds=[])])).assess(q,'Misuse of public power')
    assert r['score']=='3' and r['reason']=='ai-partial'


def test_explicit_any_n_pool_with_variants_and_extra_examples(monkeypatch):
    q=dict(questionText='State two behaviours.',referenceAnswers=['Theft, hatred, irresponsibility'],maxMarks=5,
        requiredConceptCount=2,acceptedConcepts=[dict(id='theft',description='Theft',acceptedAlternatives=['stealing']),
        dict(id='hate',description='Hatred'),dict(id='irresponsibility',description='Irresponsibility')])
    def reply(data):return dict(judgments=[dict(conceptId=c['id'],status='matched' if i<2 else 'not-demonstrated',segmentIds=['s1'] if i<2 else []) for i,c in enumerate(data['rubric']['concepts'])])
    r=provider(monkeypatch,reply).assess(q,'Stealing and hating others')
    assert r['score']=='5.00' and r['calculation']['credited_concepts']==2


def test_ambiguous_reference_count_requires_rubric_without_calling_ai(monkeypatch):
    ai=LocalMarkingAssistant()
    monkeypatch.setattr(ai.client,'_request',lambda *a:pytest.fail('Must not call AI'))
    r=ai.assess(dict(questionText='Name two resources.',referenceAnswers=['Funding, including transport and salaries'],maxMarks=5),'Funding')
    assert r['reason']=='named-concept-rubric-required'


def test_model_availability_and_incomplete_responses(monkeypatch):
    ai=LocalMarkingAssistant()
    monkeypatch.setattr(ai.client,'_request',lambda *a:{'models':[]})
    q=dict(referenceAnswers=['water'],maxMarks=5)
    assert ai.assess(q,'maji')['reason']=='local-model-not-installed'
    ai.client._verified=True;ai.client.model_digest='a'*64;ai.failure=None
    monkeypatch.setattr(ai.client,'_request',lambda *a:dict(done=False,message=dict(content='{}')))
    assert ai.assess(q,'maji')['reason']=='ai-incomplete-output'


def test_uncertain_first_pass_still_requires_teacher_review(monkeypatch):
    q=dict(questionText='Identify two tools.',referenceAnswers=['Dry sand buckets and fire extinguishers'],maxMarks=5)
    def reply(data):
        return dict(judgments=[dict(conceptId=c['id'],status='uncertain' if i==0 else 'not-demonstrated',
                                   segmentIds=['s1'] if i==0 else []) for i,c in enumerate(data['rubric']['concepts'])])
    result=provider(monkeypatch,reply).assess(q,'An unclear tool.')
    assert result['status']=='review' and result['score'] is None
    assert result['calculation']['credited_concepts']==0


def test_single_pass_keeps_context_and_alternatives_without_mutating_manifest(monkeypatch):
    import copy
    ai=LocalMarkingAssistant();ai.client._verified=True;ai.client.model_digest='a'*64
    q=dict(questionText='State two challenges.',referenceAnswers=['Funding and trained personnel'],maxMarks=5,
        answerDetail='Personnel means trained educators who teach the public. Campaigns also improve trust.',
        requiredConceptCount=2,acceptedConcepts=[
            dict(id='funds',description='Funding',acceptedAlternatives=['campaign money']),
            dict(id='staff',description='Trained personnel',acceptedAlternatives=['trained educators'])])
    original=copy.deepcopy(q);calls=[]
    def request(path,payload):
        data=json.loads(payload['messages'][1]['content']);calls.append(data)
        assert 'verificationPairs' not in data
        assert 'answerDetail' not in data
        assert data['gradedExamples']
        assert data['rubric']['requiredCount']==2
        assert data['rubric']['concepts'][1]['acceptedAlternatives']==['trained educators']
        assert data['studentAnswer']=='Too few people trained to teach the public.'
        value=dict(judgments=[dict(conceptId='funds',status='not-demonstrated',segmentIds=[]),
            dict(conceptId='staff',status='matched',segmentIds=['s1'])])
        return dict(done=True,message=dict(content=json.dumps(value)))
    monkeypatch.setattr(ai.client,'_request',request)
    r=ai.assess(q,'Too few people trained to teach the public.')
    assert r['score']=='2.50' and len(calls)==1 and q==original
    assert r['background_context']==q['answerDetail']
    assert 'answerDetail' not in r['interpretation_context']
    assert r['prompt_version']=='written-marking-v14'


def test_detail_does_not_create_requirements_or_bypass_no_ai_rubric_guard(monkeypatch):
    ai=LocalMarkingAssistant()
    monkeypatch.setattr(ai.client,'_request',lambda *a:pytest.fail('Ambiguous rubric must not call AI'))
    q=dict(questionText='Name two resources.',referenceAnswers=['Funding, including salaries and transport'],
        answerDetail='Vehicles and teachers are useful.',maxMarks=5)
    assert ai.assess(q,'Vehicles')['reason']=='named-concept-rubric-required'


def test_absent_answer_detail_remains_supported(monkeypatch):
    def check(data):
        assert 'answerDetail' not in data
    r=provider(monkeypatch,dict(relation='equivalent',segmentIds=['s1']),check).assess(
        dict(referenceAnswers=['water'],maxMarks=1),'maji')
    assert r['score']=='1' and 'answerDetail' not in r['interpretation_context']


@pytest.mark.skipif(__import__('os').environ.get('AUTOMARK_LIVE_MODEL_TESTS')!='1',
                    reason='Opt-in local Ollama semantic regression')
def test_live_teacher_approved_unrelated_ngo_answer_receives_zero():
    # Teacher explicitly supplied this zero-credit label; no generated gold score.
    q=dict(questionId='ngo-zero-regression',maxMarks=5,
           questionText='What is the primary contribution of Non-Governmental Organisations (NGOs) to the anti-corruption campaign?',
           referenceAnswers=['They provide free civic education, raise awareness on anti-bribery laws, and teach citizens methods to protect public resources.'])
    result=LocalMarkingAssistant().assess(q,'NGOs build roads and sell school books.')
    assert result['status']=='scored', result
    assert result['score']=='0', result
    assert result['reason']=='ai-incorrect'


@pytest.mark.skipif(__import__('os').environ.get('AUTOMARK_LIVE_MODEL_TESTS')!='1',
                    reason='Opt-in local Ollama semantic regression')
@pytest.mark.parametrize('text,expected',[
    ('Stealing and helping old people.','2.50'),
    ('Helping old people.','0.00'),
    ('Not stealing and helping old people.','0.00'),
    ('Stealing and theft.','2.50')])
def test_live_teacher_negative_behaviors_partial_credit(text,expected):
    q=dict(questionId='village-behaviors-regression',maxMarks=5,
           questionText='State two negative behaviors that destroy social relationships in the village.',
           referenceAnswers=['Acts of theft, public hatred, irresponsibility, and refusing to reconcile after disagreements.'])
    result=LocalMarkingAssistant().assess(q,text)
    assert result['status']=='scored', result
    assert result['score']==expected, result
