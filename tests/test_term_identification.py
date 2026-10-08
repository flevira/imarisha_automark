"""Subject-neutral term regression set; model decisions require opt-in inference."""
import json
import os
import pytest
from core.ai_marking import LocalMarkingAssistant

CASES = [
    ('Biology', 'Name the process by which plants make food using light.', 'Photosynthesis',
     'The process is photosynthesis.', 'equivalent'),
    ('Mathematics', 'Name a polygon with three sides.', 'Triangle',
     'These shapes are triangles.', 'equivalent'),
    ('Civics', 'What is the name for the biological physical responses that result from strong feelings like anger or fear?', 'Emotions',
     'These reactions are emotions.', 'equivalent'),
    ('Civics', 'What is the name for the biological physical responses that result from strong feelings like anger or fear?', 'Emotions',
     'The response is an emotion.', 'equivalent'),
    ('Biology', 'Name the process by which plants make food using light.', 'Photosynthesis',
     'It is not photosynthesis; it is respiration.', 'incorrect'),
    ('Mathematics', 'Name a polygon with three sides.', 'Triangle',
     'The answer is a square, not a triangle.', 'incorrect'),
    ('Civics', 'What is the name for the biological physical responses that result from strong feelings like anger or fear?', 'Emotions',
     'Leadership.', 'incorrect'),
]


def question(subject, wording, reference):
    return dict(subject=subject, questionText=wording, referenceAnswers=[reference], maxMarks=2)


@pytest.mark.parametrize('subject,wording,reference,text,relation', CASES)
def test_term_protocol(monkeypatch, subject, wording, reference, text, relation):
    # This tests transport/scoring only, not the mocked semantic judgment.
    ai=LocalMarkingAssistant();ai.client._verified=True;ai.client.model_digest='a'*64
    calls=[]
    def request(path,payload):
        calls.append(payload)
        system=payload['messages'][0]['content']
        assert 'identify what the student asserts is the answer' in system
        assert 'Mentioning the term is insufficient' in system
        assert 'offers an incompatible alternative' in system
        assert reference not in system
        data=json.loads(payload['messages'][1]['content'])
        assert data['studentAnswer']==text
        assert data['referenceAnswers']==[reference]
        return dict(done=True,message=dict(content=json.dumps(dict(relation=relation,segmentIds=['s1']))))
    monkeypatch.setattr(ai.client,'_request',request)
    result=ai.assess(question(subject,wording,reference),text)
    assert len(calls)==1
    assert result['score']==('2' if relation=='equivalent' else '0')
    assert result['prompt_version']=='written-marking-v25'


@pytest.mark.skipif(os.environ.get('AUTOMARK_LIVE_MODEL_TESTS')!='1',reason='Opt-in local semantic evaluation')
@pytest.mark.parametrize('subject,wording,reference,text,relation', CASES)
def test_live_term_identification(subject,wording,reference,text,relation):
    result=LocalMarkingAssistant().assess(question(subject,wording,reference),text)
    assert result['status']=='scored',result
    assert float(result['score'])==(2 if relation=='equivalent' else 0),result
