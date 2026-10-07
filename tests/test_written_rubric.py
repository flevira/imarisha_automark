import pytest
from core.written_rubric import prepare_rubric,requested_count,grade_judgments,answer_segments


def test_reference_pool_names_real_ideas_and_preserves_or():
    q=dict(questionText='State two behaviours.',referenceAnswers=['Theft, hatred, irresponsibility, and refusing to reconcile'],maxMarks=5)
    r=prepare_rubric(q)
    assert r['requiredCount']==2 and len(r['concepts'])==4
    assert [c['description'] for c in r['concepts']]==['Theft','hatred','irresponsibility','refusing to reconcile']
    assert prepare_rubric(dict(questionText='Define corruption.',referenceAnswers=['Abuse of power or office for personal gain'],maxMarks=5))['mode']=='holistic'


@pytest.mark.parametrize('text',['Give at least two examples','State two causes and list three effects','Discuss two events in 1990','Explain each of two problems'])
def test_ambiguous_counts_not_assumed(text):
    assert requested_count(dict(questionText=text)) is None


def test_weighted_rubric_wins_over_count():
    q=dict(questionText='Give two causes.',maxMarks=5,scoringCriteria=[dict(id='a',description='Main cause',marks=4),dict(id='b',description='Other cause',marks=1)])
    assert prepare_rubric(q)['mode']=='weighted'


def test_duplicate_concepts_invalid():
    with pytest.raises(ValueError,match='duplicate-accepted-concept'):
        prepare_rubric(dict(maxMarks=5,requiredConceptCount=2,acceptedConcepts=[dict(id='a',description='Theft'),dict(id='b',description='theft')]))


def test_score_is_capped_and_rounds_after_summing():
    rubric=dict(mode='any-n',requiredCount=3,concepts=[dict(id=str(i),description=str(i)) for i in range(4)])
    segments=answer_segments('Ideas here.')
    for matches,expected in [(1,'1.67'),(2,'3.33'),(3,'5.00'),(4,'5.00')]:
        judgments=[dict(conceptId=str(i),status='matched' if i<matches else 'not-demonstrated',segmentIds=['s1'] if i<matches else []) for i in range(4)]
        assert grade_judgments(rubric,judgments,segments,5)['score']==expected


def test_uncertain_concept_keeps_total_pending_unless_cap_already_met():
    rubric=dict(mode='any-n',requiredCount=1,concepts=[dict(id='a',description='A'),dict(id='b',description='B')])
    js=[dict(conceptId='a',status='matched',segmentIds=['s1']),dict(conceptId='b',status='uncertain',segmentIds=[])]
    assert grade_judgments(rubric,js,answer_segments('A'),5)['score']=='5.00'
    rubric['requiredCount']=2
    assert grade_judgments(rubric,js,answer_segments('A'),5)['score'] is None


def test_and_separates_list_evidence_without_dropping_shared_context():
    rubric=dict(mode='any-n',concepts=[dict(id='theft',description='Acts of theft')])
    assert [s['text'] for s in answer_segments('Stealing and helping old people.',rubric)]==['Stealing','helping old people.']
    assert len(answer_segments('Not stealing and helping old people.',rubric))==1
    assert len(answer_segments('Stealing and helping old people.'))==1
    compound=dict(mode='any-n',concepts=[dict(id='care',description='care and support')])
    assert len(answer_segments('care and support',compound))==1


@pytest.mark.parametrize('text',[
    'Stealing, hatred, irresponsibility',
    '(i) Stealing (ii) hatred (iii) irresponsibility',
    'i. Stealing ii. hatred iii. irresponsibility',
    'i, Stealing ii, hatred iii, irresponsibility',
    '1. Stealing 2. hatred 3. irresponsibility',
    'a) Stealing b) hatred c) irresponsibility',
    '• Stealing • hatred • irresponsibility',
    '- Stealing\n- hatred\n- irresponsibility',
    'Stealing;hatred;irresponsibility',
    'Stealing, hatred and irresponsibility',
    'Stealing, hatred, and irresponsibility',
])
def test_list_formats(text):
    rubric=dict(mode='any-n',concepts=[])
    assert [s['text'] for s in answer_segments(text,rubric)]==['Stealing','hatred','irresponsibility']


def test_list_numbers_negation_and_holistic_prose_preserved():
    rubric=dict(mode='any-n',concepts=[])
    assert [s['text'] for s in answer_segments('1.5 litres, 1,000 litres',rubric)]==['1.5 litres','1,000 litres']
    assert len(answer_segments('Not theft, hatred or irresponsibility',rubric))==1
    assert len(answer_segments('A concise definition, with context.'))==1

def test_school_election_interrogative_count_and_parenthetical_glosses():
    q=dict(questionText='What are the two primary rights exercised by pupils during a standard school election?',
           referenceAnswers=['The right to elect (vote) and the right to be elected (contest for a leadership position).'],maxMarks=9)
    rubric=prepare_rubric(q)
    assert rubric['requiredCount']==2
    assert len(rubric['concepts'])==2
    judgments=[dict(conceptId=c['id'],status='matched' if i==0 else 'not-demonstrated',
                    segmentIds=['s1'] if i==0 else []) for i,c in enumerate(rubric['concepts'])]
    result=grade_judgments(rubric,judgments,[dict(id='s1',text='The right to vote')],9)
    assert result['score']=='4.50'


def test_wrapped_prose_keeps_action_and_object_together():
    rubric=dict(mode='any-n',concepts=[])
    assert answer_segments('Ask the prefect to tell the school\nleaders.',rubric)==[
        dict(id='s1',text='Ask the prefect to tell the school leaders.')]
    assert [s['text'] for s in answer_segments('The right to vote and\nremove other candidates.',rubric)]==[
        'The right to vote','remove other candidates.']
    assert answer_segments('I do not give up.\nI stay calm.')[1]['text']=='I stay calm.'


def test_wrapped_negation_is_not_detached_from_candidate():
    rubric=dict(mode='any-n',concepts=[])
    segments=answer_segments('Not\nstealing and helping old people.',rubric)
    assert [s['text'] for s in segments]==['Not stealing and helping old people.']
    # Explicit numbered items still separate after reflow.
    assert [s['text'] for s in answer_segments('i) Stealing\nii) hatred',rubric)]==['Stealing','hatred']
