import copy
import pytest
from core.marking import mark_question
from core.teacher_grounding import explicit_concept_match
from core.written_matching import typo_suggestions,complete_criteria

RULE=dict(version=1,mode='text-exact',caseSensitive=False,collapseWhitespace=True,
          normalizePresentation=True,allowAnswerFraming=True)


def question():
    return dict(questionId='1',displayNumber=1,responseType='WRITTEN',status='READY',maxMarks=5,
                questionText='Name the emotional state.',referenceAnswers=['Feeling'])


def mark(text,q=None,rule=None):
    return mark_question(q or question(),[dict(kind='WRITTEN')],
        [dict(approved=True,text=text,selected=[])],rule or RULE)


def concepts():
    q=question();q.update(questionText='Name two tools.',requiredConceptCount=2,
        referenceAnswers=['Buckets of dry sand and portable dry chemical fire extinguishers'],
        acceptedConcepts=[dict(id='sand',description='Buckets of dry sand',acceptedAlternatives=['A bucket of dry sand']),
                          dict(id='ext',description='portable dry chemical fire extinguishers',acceptedAlternatives=['dry chemical extinguisher'])])
    return q


@pytest.mark.parametrize('text',['  FEELING\n','“Feeling.”','The answer is feeling.','Answer: Feeling','It is feeling.'])
def test_normalized_whole_answer_and_neutral_frame(text):
    assert mark(text)['score']=='5'


@pytest.mark.parametrize('text',['Not feeling','It is not feeling','Feeling but leadership','Leadership',
    'Feelings','Feeling and leadership','Feeling?','Feeling!','No feeling'])
def test_unknown_contradictory_or_extra_claims_still_need_ai(text):
    assert mark(text)['reason']=='unlisted-wording-needs-review'


def test_alternatives_and_explicit_exhaustive_terms():
    q=question();q['acceptedAlternatives']=['A feeling']
    assert mark('The answer is a feeling.',q)['score']=='5'
    assert mark('Leadership',q)['score'] is None
    assert mark('Leadership',q,dict(RULE,rejectUnlisted=True))['score']=='0'
    assert mark('Feelig',q,dict(RULE,rejectUnlisted=True))['score']=='0'
    assert mark('Feelig',q)['score'] is None
    assert typo_suggestions('Feelig',q['referenceAnswers'])[0]['candidate']=='feeling'


def test_literal_policy_is_respected():
    assert mark('feeling',rule=dict(RULE,caseSensitive=True))['score'] is None
    assert mark('It is Feeling',rule=dict(RULE,allowAnswerFraming=False))['score'] is None
    for flag in ('rejectUnlisted','normalizePresentation','allowAnswerFraming'):
        assert mark('Feeling',rule=dict(RULE,**{flag:'true'}))['score'] is None
    q=question();q['referenceAnswers']=['5!']
    assert mark('5',q)['score'] is None
    q['referenceAnswers']=['-5']
    assert mark('5',q)['score'] is None
    q['referenceAnswers']=['1.5']
    assert mark('15',q)['score'] is None


@pytest.mark.parametrize('text',[
    'A bucket of dry sand and dry chemical extinguisher',
    'A bucket of dry sand, dry chemical extinguisher',
    'i) A bucket of dry sand ii) dry chemical extinguisher',
    '(i) A bucket of dry sand; (ii) dry chemical extinguisher',
    '1. A bucket of dry sand\n2. dry chemical extinguisher',
    'A bucket of dry sand\ndry chemical extinguisher',
    'The answer is A bucket of\ndry sand and dry chemical extinguisher',
])
def test_complete_concepts_across_lists_and_wrapping(text):
    result=explicit_concept_match(concepts(),text)
    assert result['score']=='5.00'


def test_fractional_credit_deduplicated_capped_and_weighted():
    q=concepts()
    assert explicit_concept_match(q,'A bucket of dry sand')['score']=='2.50'
    assert explicit_concept_match(q,'A bucket of dry sand and A bucket of dry sand')['score']=='2.50'
    q['maxMarks']=10;q['requiredConceptCount']=3
    q['acceptedConcepts'].append(dict(id='water',description='water bucket'))
    assert explicit_concept_match(q,'A bucket of dry sand')['score']=='3.33'
    assert explicit_concept_match(q,'A bucket of dry sand and water bucket')['score']=='6.67'
    q.pop('acceptedConcepts');q['scoringCriteria']=[dict(id='a',description='sand',marks='3'),dict(id='b',description='extinguisher',marks='7')]
    assert explicit_concept_match(q,'sand')['score']=='3'


@pytest.mark.parametrize('text',['not A bucket of dry sand','A bucket of dry sand but water',
    'A bucket of dry sand and a broom','A bucket of dry sand is dangerous','A bucket of dry sand without extinguisher'])
def test_no_keyword_scoring_or_unknown_clause_discard(text):
    assert explicit_concept_match(concepts(),text) is None


def test_ambiguous_alias_and_overlapping_phrase_protection():
    q=concepts();q['acceptedConcepts'][0]['acceptedAlternatives'].append('tool')
    q['acceptedConcepts'][1]['acceptedAlternatives'].append('tool')
    assert explicit_concept_match(q,'tool') is None
    assert complete_criteria('public property',[dict(id='a',description='public'),dict(id='b',description='public property')])=={'b'}
    assert complete_criteria('working hard and honestly',[dict(id='a',description='working hard and honestly')])=={'a'}
    assert complete_criteria('not stealing',[dict(id='a',description='not stealing')])=={'a'}


def test_inferred_concepts_score_but_conflicting_exhaustive_policy_does_not():
    q=concepts();q.pop('acceptedConcepts')
    assert explicit_concept_match(q,'Buckets of dry sand')['score']=='2.50'
    assert mark('A bucket of dry sand',concepts(),dict(RULE,rejectUnlisted=True))['score'] is None


def test_numeric_and_typo_limits():
    q=question();q['referenceAnswers']=['10 cm']
    rule=dict(version=1,mode='numeric',tolerance='.1',unit='cm')
    assert mark('10.1 cm',q,rule)['score']=='5'
    assert mark('10.2 cm',q,rule)['score']=='0'
    assert mark('10 m',q,rule)['score'] is None
    assert not typo_suggestions('10',['11'])
    assert not typo_suggestions('not stealing',['stealing'])
    assert not typo_suggestions('cat',['car'])
    # A real English word is only a suggestion, never automatically corrected.
    assert typo_suggestions('coast',['toast'])[0]['advisory_only'] is True

from test_written_extraction import setup,read


def test_minimal_job_falls_back_for_framing_and_concepts(setup,monkeypatch):
    from types import SimpleNamespace
    from core.contracts import Transcription,ReadStatus
    from core.local_text_ocr import LocalTextOCR
    from services.marking_jobs import mark_job
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:Transcription(None,ReadStatus.UNCERTAIN,('ocr-no-text',),'trocr-written-v1'))
    _,manifest,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        raw=row['response']
        value={'2':'The answer is Feeling.','3':'Leadership','4':'A bucket of dry sand','5':'Feelig'}.get(raw.question_id,'')
        if raw.question_id=='2':value='The answer is' if raw.part_number==1 else 'Feeling.'
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),reviewer='Teacher',
                    raw_text=value if raw.response_type.value=='WRITTEN' else None)
    repo.complete(job)
    m=copy.deepcopy(manifest);m['answerSchemaVersion']=1
    m['questions']=[dict(questionId=str(i),displayNumber=i,maxMarks=5,responseType='MC' if i==1 else 'WRITTEN',
        questionText='Name a feeling.',status='READY',referenceAnswers=['Feeling']) for i in range(1,6)]
    m['questions'][2]['scoringRules']=dict(RULE,rejectUnlisted=True)
    m['questions'][3].update(concepts(),questionId='4',displayNumber=4)
    m['pages'][0]['rows'][0].update(correctAnswers=['A'],questionMarks=5,answerStatus='READY')
    res=SimpleNamespace(scope=resolver.scope,resolve=lambda *args:m)
    calls=[]
    class Assistant:
        def assess(self,q,text):
            calls.append((q['questionId'],text))
            return dict(status='review',score=None,reason='ai-unresolved',seconds=0)
    run=mark_job(repo,job,res,'Teacher',policy={'version':1,'fuzzyMatching':False},assistant=Assistant())
    results={r['question_id']:r for r in run['results']}
    assert all(results[k]['score'] is None and results[k].get('ai') for k in ('2','3','4','5'))
    assert {q for q,t in calls}=={'2','3','4','5'}
    count=len(calls)
    mark_job(repo,job,res,'Teacher',policy={'version':1,'fuzzyMatching':False},assistant=Assistant())
    assert len(calls)==count+4
    m['questions'][1]['acceptedAlternatives']=[next(t for q,t in calls if q=='2')]
    rerun=mark_job(repo,job,res,'Teacher',assistant=Assistant())
    assert next(r for r in rerun['results'] if r['question_id']=='2')['score']=='5'


def test_saved_case_sensitive_policy_cannot_be_bypassed_by_examples():
    from core.teacher_grounding import approved_match
    examples=[dict(text='Na',approved=True,score='5')]
    assert approved_match('NA',examples,dict(RULE,caseSensitive=True)) is None


def test_view_passes_teacher_alternatives_and_exhaustive_rule(setup,monkeypatch):
    import asyncio
    import app.views.structured_view as view
    from test_structured_view import Page,find
    from core.contracts import Transcription,ReadStatus
    from core.local_text_ocr import LocalTextOCR
    monkeypatch.setattr(LocalTextOCR,'recognize',lambda *args:Transcription(None,ReadStatus.UNCERTAIN,('ocr-no-text',),'trocr-written-v1'))
    _,_,resolver,repo=setup
    job,rows=read(setup,'answers.pdf')
    for row in rows:
        raw=row['response']
        repo.review(job,row['id'],action='approve',student='00530',selected=list(raw.selected_options),
                    reviewer='Teacher',raw_text='Feeling' if raw.response_type.value=='WRITTEN' else None)
    repo.complete(job)
    monkeypatch.setattr(view,'ImportedManifestResolver',lambda *a,**kw:resolver)
    captured=[]
    def marking(*args):
        captured.append(args[4]);return dict(marking_seconds=0,sequence=1)
    monkeypatch.setattr(view,'mark_job',marking)
    page=Page();view.build_structured_view(page,repo)
    saved=repo.job(job)
    find(page,'text',f"answers.pdf — {saved['status']} ({len(saved['pages'])} pages)").on_click(None)
    find(page,'label','Question 2: original wording').value='Name the emotional state.'
    find(page,'label','Q2: approved full-credit alternatives').value='A feeling\nEmotion'
    find(page,'label','Q2: only listed complete answers are valid; others receive zero').value=True
    find(page,'label','Use local AI for equivalent written answers and rubric-based partial credit').value=False
    async def run():
        find(page,'text','Mark / re-mark reviewed answers').on_click(None)
        await page.tasks[-1]
    asyncio.run(run())
    context=next(iter(captured[0]['writtenContext'].values()))['2']
    assert context['acceptedAlternatives']==['A feeling','Emotion']
    assert context['scoringRules']['rejectUnlisted'] is True


def test_derived_reference_concepts_use_existing_calculator_and_keep_fallback():
    q=question();q.update(questionText='Name three resources.',maxMarks=10,referenceAnswers=['school books; school desks; school equipment; school buildings'])
    r=explicit_concept_match(q,'school books')
    assert r['score']=='3.33' and r['concept_source']=='reference-list'
    assert r['reason']=='derived-reference-concept-match'
    assert explicit_concept_match(q,'school books and school desks')['score']=='6.67'
    assert explicit_concept_match(q,'school books and school books')['score']=='3.33'
    assert explicit_concept_match(q,'school books and dancing') is None
    assert explicit_concept_match(q,'not school books') is None
    assert explicit_concept_match(dict(q,questionText='Explain the resources.'),'school books') is None
    from core.fuzzy_marking import candidate
    fuzzy=candidate(q,'schoolbooks',RULE)
    assert fuzzy['accepted'] and fuzzy['result']['score']=='3.33'
    assert fuzzy['concept_source']=='reference-list'
