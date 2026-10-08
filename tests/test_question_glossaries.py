import copy
import pytest
from database.structured_repository import JobRepository
from services.question_glossaries import source,scope,prepare,apply,validate
from core.fuzzy_marking import candidate,pair
from tools.benchmark_nonai_marking import DEFAULT_RULE


def data():
    return {'assessmentId':'1','layoutId':'a','version':1},dict(questionId='q',questionText='Define a feeling.',referenceAnswers=['Feeling'],maxMarks=5,responseType='WRITTEN')


def test_approval_restart_scope_override_rejection(tmp_path,monkeypatch):
    import services.question_glossaries as service
    repo=JobRepository(tmp_path/'x.db');m,q=data();calls=[]
    def generate(material,model):calls.append(1);return {'acceptedAlternatives':['Emotional state']},.2
    monkeypatch.setattr(service,'generate',generate)
    e=prepare(repo,m,q,'test');prepare(repo,m,q,'test');assert len(calls)==1
    assert apply(repo,m,q)[1]==e['id']
    assert apply(repo,m,q)[0]['glossaryStatus']=='draft'
    approved=repo.save_glossary(e['scope'],e['material'],e['context'],'approved','test','v1',reviewer='Teacher')
    repo=JobRepository(repo.path)
    effective,version=apply(repo,m,q);assert version==approved['id'] and effective['acceptedAlternatives']==['Emotional state']
    assert apply(repo,m,dict(q,answerDetail='new'))[1] is None
    assert apply(repo,m,q,{'acceptedAlternatives':['Override']})[0]['acceptedAlternatives']==['Override']
    repo.save_glossary(e['scope'],e['material'],e['context'],'rejected','test','v1')
    assert apply(repo,m,q)[1] is None


def test_failed_draft_cached_and_no_approval_without_name(tmp_path,monkeypatch):
    import services.question_glossaries as service
    repo=JobRepository(tmp_path/'x.db');m,q=data();calls=[]
    def fail(*args):calls.append(1);raise ValueError('unavailable')
    monkeypatch.setattr(service,'generate',fail)
    assert prepare(repo,m,q,'x')['status']=='failed'
    prepare(repo,m,q,'x');assert len(calls)==1
    with pytest.raises(ValueError):repo.save_glossary(scope(source(m,q)),source(m,q),{},'approved','x','v1')


def test_manifest_criteria_preserved():
    m,q=data();q.update(scoringCriteria=[dict(id='a',description='Feeling',marks=5)])
    material=source(m,q)
    with pytest.raises(ValueError):validate(material,{'scoringCriteria':[dict(id='a',description='New',marks=5)]})
    assert validate(material,{'acceptedAlternatives':['Emotional state']})


def test_live_fuzzy_guards_and_credit():
    m,q=data();q['referenceAnswers']=['school rules']
    assert candidate(q,'schoolrules',DEFAULT_RULE)['result']['score']=='5'
    for a,b in [('therapist','the rapist'),('nowhere','now here'),('vote many times','vote'),('elected','deleted'),('10 cm','11 cm')]:
        assert pair(a,b,DEFAULT_RULE,.97) is None
    q.update(questionText='Name three items.',referenceAnswers=['one; two; three'],requiredConceptCount=3,maxMarks=10,
        acceptedConcepts=[dict(id=str(i),description=t) for i,t in enumerate(['school rules','school desks','school books'])])
    assert candidate(q,'schoolrules',DEFAULT_RULE)['result']['score']=='3.33'
    assert candidate(q,'schoolrules and schooldesks',DEFAULT_RULE)['result']['score']=='6.67'
    assert not candidate(q,'schoolrules and stealing',DEFAULT_RULE)['accepted']


def test_reset_removes_glossaries(tmp_path):
    repo=JobRepository(tmp_path/'x.db');m,q=data();material=source(m,q);key=scope(material)
    repo.save_glossary(key,material,{},'approved','test','v1',reviewer='Teacher')
    repo.delete_saved_data()
    assert repo.glossary(key) is None


def test_generation_preserves_concepts_and_holistic_mode(monkeypatch):
    import json
    from types import SimpleNamespace
    import core.ai_marking as ai
    from services.question_glossaries import generate
    requests=[]
    def request(path,payload):
        requests.append((path,payload))
        if path=='/api/show':return {'details':{'format':'gguf'}}
        return dict(done=True,message=dict(content=json.dumps(dict(answer=['Emotional state']))))
    monkeypatch.setattr(ai,'LocalMarkingAssistant',lambda model:SimpleNamespace(client=SimpleNamespace(_request=request)))
    m,q=data();result,seconds=generate(source(m,q),'local')
    assert result=={'acceptedAlternatives':['Emotional state']}
    payload=json.loads(requests[-1][1]['messages'][-1]['content'])
    assert 'studentAnswer' not in str(payload) and payload['rubric']['mode']=='holistic'


def test_inferred_count_cannot_change():
    m,q=data();q.update(questionText='Name two tools.',referenceAnswers=['sand; extinguisher'])
    with pytest.raises(ValueError):validate(source(m,q),dict(acceptedConcepts=[dict(id='a',description='sand'),dict(id='b',description='extinguisher')],requiredConceptCount=1))


from test_written_extraction import setup,read


def test_live_job_ignores_stored_glossary(setup,monkeypatch):
    from types import SimpleNamespace
    from core.contracts import Transcription,ReadStatus
    from core.local_text_ocr import LocalTextOCR
    from services.marking_jobs import mark_job
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *a:Transcription(None,ReadStatus.UNCERTAIN,('ocr-no-text',),'trocr-written-v1'))
    _,manifest,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        raw=row['response'];text='Schoolrules'
        if raw.question_id=='2':text='School' if raw.part_number==1 else 'rules'
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),reviewer='Teacher',raw_text=text if raw.response_type.value=='WRITTEN' else None)
    repo.complete(job)
    m=copy.deepcopy(manifest);m['answerSchemaVersion']=1
    m['questions']=[dict(questionId=str(i),displayNumber=i,maxMarks=5,responseType='MC' if i==1 else 'WRITTEN',questionText='Name the rules.',status='READY',referenceAnswers=['Rules']) for i in range(1,6)]
    m['pages'][0]['rows'][0].update(correctAnswers=['A'],questionMarks=5,answerStatus='READY')
    for q in m['questions'][1:]:
        q.update(subject='Civics',topic='Democracy')
        material=source(m,q);repo.save_glossary(scope(material),material,{'acceptedAlternatives':['School rules']},'approved','test','v1',reviewer='Teacher')
    resolver=SimpleNamespace(scope=resolver.scope,resolve=lambda *args:m)
    run=mark_job(repo,job,resolver,'Teacher')
    written=[r for r in run['results'] if r['display_number']>1]
    assert all(r['score'] is None and not r.get('glossary_version') for r in written)
    assert all(r['scoring_question']['subject']=='Civics' and r['scoring_question']['topic']=='Democracy' for r in written)
    saved=repo.marking_runs(job)[0]
    assert next(r for r in saved['results'] if r['display_number']>1)['scoring_question']['topic']=='Democracy'
    assert not any(r.get('fuzzy_evidence') for r in written)
    assert all(r['fuzzy_policy']['threshold']==.80 for r in written)
    off=mark_job(repo,job,resolver,'Teacher',{'version':1,'fuzzyMatching':False})
    assert next(r for r in off['results'] if r['question_id']=='3')['score'] is None
    # Delete one job without destroying the shared glossary.
    key=scope(source(m,m['questions'][1]));repo.delete_saved_data(job)
    assert repo.glossary(key)['status']=='approved'


def test_ui_has_no_glossary_controls(setup,monkeypatch):
    import app.views.structured_view as view
    from test_structured_view import Page,descendants
    _,_,resolver,repo=setup
    monkeypatch.setattr(view,'ImportedManifestResolver',lambda *a,**kw:resolver)
    page=Page()
    view.build_structured_view(page,repo)
    labels={getattr(c,'text',None) for c in descendants(page)}
    assert not labels.intersection({'Prepare question glossaries','Approve glossary','Regenerate glossary'})


def test_live_fuzzy_80_percent_threshold():
    m,q=data();q['referenceAnswers']=['happy']
    result=candidate(q,'happi',DEFAULT_RULE)
    assert result['accepted']
    assert result['matches'][0]['similarity']==.8
    assert not candidate(q,'happi',DEFAULT_RULE,threshold=.81)['accepted']
    assert not candidate(q,'haxxi',DEFAULT_RULE)['accepted']


def test_generated_counted_glossary_does_not_award_full_credit_for_one_concept(tmp_path):
    from core.teacher_grounding import explicit_concept_match
    from core.marking import mark_question
    repo=JobRepository(tmp_path/'x.db');m,q=data()
    q.update(questionText='Name two trustworthy acts.',maxMarks=10,referenceAnswers=['telling the truth; caring for school property'])
    context=dict(acceptedAlternatives=['being honest'],requiredConceptCount=2,acceptedConcepts=[
        dict(id='a',description='telling the truth',acceptedAlternatives=['being honest']),
        dict(id='b',description='caring for school property',acceptedAlternatives=['looking after school equipment'])])
    material=source(m,q);entry=repo.save_glossary(scope(material),material,context,'draft','test','v1')
    effective,version=apply(repo,m,q)
    assert version==entry['id'] and effective['acceptedAlternatives']==[]
    effective['status']='READY';effective['displayNumber']=20
    exact=mark_question(effective,[{'kind':'WRITTEN'}],[dict(approved=True,text='being honest',selected=[])],DEFAULT_RULE)
    assert exact['score'] is None
    assert explicit_concept_match(effective,'being honest')['score']=='5.00'
    assert explicit_concept_match(effective,'being honest and looking after school equipment')['score']=='10.00'


def test_generated_phrases_preserve_ids_and_resolve_partial_credit(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    import core.ai_marking as ai
    from core.written_rubric import prepare_rubric
    from core.teacher_grounding import explicit_concept_match
    from services.question_glossaries import generate
    m,q=data()
    q.update(questionText='State two trustworthy acts.',maxMarks=10,
        referenceAnswers=['Abiding by school rules; being truthful'])
    requests=[]
    def request(path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        requests.append(payload)
        return dict(done=True,message=dict(content=json.dumps(dict(
            c1=dict(conciseForm='obey school rules',alternatives=['obeying school rules']),c2=dict(conciseForm='tell the truth',alternatives=['telling the truth'])))))
    monkeypatch.setattr(ai,'LocalMarkingAssistant',lambda model:SimpleNamespace(client=SimpleNamespace(_request=request)))
    context,_=generate(source(m,q),'local')
    assert [c['id'] for c in context['acceptedConcepts']]==[c['id'] for c in prepare_rubric(q)['concepts']]
    assert requests[0]['format']['required']==['c1','c2']
    assert 'acceptedAlternatives' not in requests[0]['format']['properties']
    repo=JobRepository(tmp_path/'x.db');material=source(m,q)
    repo.save_glossary(scope(material),material,context,'draft','local','v2')
    effective,_=apply(repo,m,q)
    assert explicit_concept_match(effective,'obey school rules and tell the truth')['score']=='10.00'
    assert explicit_concept_match(effective,'tell the truth')['score']=='5.00'
    assert explicit_concept_match(effective,'tell the truth and tell the truth')['score']=='5.00'
    for text in ['do not tell the truth','obey school rules and steal books','vote many times']:
        assert explicit_concept_match(effective,text) is None
    assert context['acceptedAlternatives']==[]


def test_generation_repairs_empty_or_copy_only_output_once(monkeypatch):
    import json
    from types import SimpleNamespace
    import core.ai_marking as ai
    from services.question_glossaries import generate
    m,q=data();calls=[]
    outputs=[{'answer':['Feeling']},{'answer':['An emotional state']}]
    def request(path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        calls.append(payload)
        return dict(done=True,message=dict(content=json.dumps(outputs.pop(0))))
    monkeypatch.setattr(ai,'LocalMarkingAssistant',lambda model:SimpleNamespace(client=SimpleNamespace(_request=request)))
    assert generate(source(m,q),'local')[0]['acceptedAlternatives']==['An emotional state']
    assert len(calls)==2
    outputs[:]=[{'answer':[]},{'answer':[]}];calls.clear()
    with pytest.raises(ValueError,match='after one repair'):generate(source(m,q),'local')
    assert len(calls)==2


def test_generated_key_and_alias_validation():
    from services.question_glossaries import _generation_context
    from core.written_rubric import prepare_rubric
    m,q=data();q.update(questionText='Name two acts.',referenceAnswers=['telling the truth; caring for school property'])
    material=source(m,q);rubric=prepare_rubric(q)
    for value in [{'c1':['being honest']}, {'c1':['being honest'],'c2':['being honest']},
                  {'c1':['telling the truth'],'c2':['looking after school equipment']},
                  {'c1':['being honest'],'c2':['looking after school equipment'],'c3':['new requirement']}]:
        with pytest.raises(ValueError):_generation_context(material,rubric,value)
    value=_generation_context(material,rubric,{'c1':['being honest'],'c2':['looking after school equipment']})
    assert value['requiredConceptCount']==2


def test_preparation_refreshes_old_generated_versions_only(tmp_path,monkeypatch):
    import services.question_glossaries as service
    repo=JobRepository(tmp_path/'x.db');m,q=data();material=source(m,q);key=scope(material);calls=[]
    def generate(*args):calls.append(1);return {'acceptedAlternatives':['Emotional state']},.1
    monkeypatch.setattr(service,'generate',generate)
    for status in ('draft','failed'):
        repo.save_glossary(key,material,{},status,'local','question-glossary-v1')
        result=prepare(repo,m,q,'local')
        assert result['prompt']==service.VERSION and result['status']=='draft'
        n=len(calls);assert prepare(repo,m,q,'local')['id']==result['id'];assert len(calls)==n
    for status in ('approved','rejected'):
        entry=repo.save_glossary(key,material,{},status,'local','question-glossary-v1',reviewer='Teacher')
        n=len(calls);assert prepare(repo,m,q,'local')['id']==entry['id'];assert len(calls)==n


def test_generated_alternatives_keep_weights_and_thirds():
    from services.question_glossaries import _generation_context
    from core.written_rubric import prepare_rubric
    from core.teacher_grounding import explicit_concept_match
    m,q=data();q.update(questionText='Name three acts.',maxMarks=10,
        referenceAnswers=['being truthful; caring for school property; abiding by rules'])
    values={'c1':['tell the truth'],'c2':['look after school equipment'],'c3':['obey rules']}
    context=_generation_context(source(m,q),prepare_rubric(q),values)
    assert explicit_concept_match(dict(q,**context),'tell the truth')['score']=='3.33'
    q['scoringCriteria']=[dict(id='fixed-a',description='being truthful',marks=4),
        dict(id='fixed-b',description='caring for school property',marks=6)]
    context=_generation_context(source(m,q),prepare_rubric(q),{k:v for k,v in values.items() if k!='c3'})
    assert [(c['id'],c['marks']) for c in context['scoringCriteria']]==[('fixed-a',4),('fixed-b',6)]
    assert explicit_concept_match(dict(q,**context),'tell the truth')['score']=='4'


@pytest.mark.parametrize('response',[
    {'done':True,'done_reason':'length','message':{'content':'{"answer":["Emotional state"]}'}},
    {'done':False,'message':{'content':'{"answer":["Emotional state"]}'}},
    {'done':True,'message':{'content':'invalid json'}}])
def test_generation_failure_is_bounded(monkeypatch,response):
    from types import SimpleNamespace
    import core.ai_marking as ai
    from services.question_glossaries import generate
    calls=[]
    def request(path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        calls.append(1);return response
    monkeypatch.setattr(ai,'LocalMarkingAssistant',lambda model:SimpleNamespace(client=SimpleNamespace(_request=request)))
    m,q=data()
    with pytest.raises(ValueError,match='after one repair'):generate(source(m,q),'local')
    assert len(calls)==2


def test_independent_or_options_keep_shared_objects_and_qualifiers():
    from services.question_glossaries import _independent_options
    text="Wait calmly for the issues to be resolved or report any concerns through the school's leadership channels"
    assert len(_independent_options(text))==2
    for text in ['use red or blue buckets','not wait calmly or report concerns to staff',
                 'wait calmly if asked or report concerns to staff','vote (elect or choose)',
                 'wash hands before meals or after using the toilet']:
        assert _independent_options(text)==[text]


def test_generation_keeps_or_options_as_one_scoring_concept(monkeypatch):
    import json
    from types import SimpleNamespace
    import core.ai_marking as ai
    from services.question_glossaries import generate
    from core.teacher_grounding import explicit_concept_match
    m,q=data();q.update(questionText='State one constructive action.',maxMarks=9,
        referenceAnswers=['Wait calmly for the issues to be resolved or report any concerns through the school leadership channels'])
    def request(path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        assert payload['format']['required']==['c1_option1','c1_option2']
        result=dict(c1_option1=dict(conciseForm='wait calmly',alternatives=['Wait calmly until the food problem is solved']),
                    c1_option2=dict(conciseForm='report concerns to school leaders',alternatives=['Tell a teacher about the delay']))
        return dict(done=True,message=dict(content=json.dumps(result)))
    monkeypatch.setattr(ai,'LocalMarkingAssistant',lambda model:SimpleNamespace(client=SimpleNamespace(_request=request)))
    context,_=generate(source(m,q),'local');effective=dict(q,**context)
    assert len(context['acceptedConcepts'])==1
    for text in ['Wait calmly until the food problem is solved','Tell a teacher about the delay']:
        assert explicit_concept_match(effective,text)['score']=='9.00'
    assert explicit_concept_match(effective,'Wait calmly and destroy plates') is None


@pytest.mark.parametrize('field',['subject','topic'])
def test_question_subject_topic_changes_invalidate_glossary(tmp_path,field):
    repo=JobRepository(tmp_path/'metadata.db');m,q=data();q.update(subject='Civics',topic='Democracy')
    material=source(m,q)
    repo.save_glossary(scope(material),material,{'acceptedAlternatives':['Emotional state']},'approved','test','v3',reviewer='Teacher')
    assert apply(repo,m,q)[1] is not None
    changed=dict(q,**{field:'Changed'})
    effective,version=apply(repo,m,changed)
    assert version is None and {k:v for k,v in effective.items() if k!='scoringAuthority'}==changed
    assert effective['scoringAuthority']['glossary'] is None


def test_glossary_authority_provenance(tmp_path):
    from core.teacher_grounding import reference_examples
    repo=JobRepository(tmp_path/'authority.db');m,q=data()
    material=source(m,q)
    repo.save_glossary(scope(material),material,{'acceptedAlternatives':['Emotional state']},'draft','fake','v4')
    effective,_=apply(repo,m,q)
    assert [e['text'] for e in reference_examples(effective)]==['Feeling']
    assert effective['scoringAuthority']['glossary']['authority']=='suggestion'
    repo.save_glossary(scope(material),material,{'acceptedAlternatives':['Emotional state']},'approved','fake','v4',reviewer='Teacher')
    effective,_=apply(repo,m,q)
    assert reference_examples(effective)[1]['provenance']=='teacher-approved-glossary'
    effective,_=apply(repo,m,q,{'acceptedAlternatives':['Teacher correction']})
    assert reference_examples(effective)[1]['provenance']=='teacher-alternative'
    assert effective['scoringAuthority']['teacherCorrections']['acceptedAlternatives']==['Teacher correction']


def test_glossary_generation_tie_subject_topic(monkeypatch):
    import json
    from core.handwriting import OllamaHandwritingRecognizer
    from services.question_glossaries import generate
    m,q=data();q.update(subject='Language',topic='Vocabulary')
    def request(self,path,payload):
        if path=='/api/show':return {'details':{'format':'gguf'}}
        sent=json.loads(payload['messages'][1]['content'])
        assert sent['educationalContext']['subject']=='Language'
        assert sent['educationalContext']['topic']=='Vocabulary'
        assert sent['educationalContext']['country']=='Tanzania'
        assert 'TIE' in payload['messages'][0]['content']
        assert 'studentAnswer' not in str(sent)
        return dict(done=True,message=dict(content=json.dumps({'answer':['Emotional state']})))
    monkeypatch.setattr(OllamaHandwritingRecognizer,'_request',request)
    assert generate(source(m,q),'fake')[0]['acceptedAlternatives']==['Emotional state']
