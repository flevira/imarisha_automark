import pytest
from tools.benchmark_nonai_marking import evaluate
from test_nonai_written import question, concepts, RULE


def test_offline_gate_scores_known_answers_and_defers_unknowns():
    cases=[dict(question=question(),studentAnswer='The answer is feeling.',expectedScore=5),
           dict(question=concepts(),studentAnswer='A bucket of dry sand',expectedScore=2.5),
           dict(question=question(),studentAnswer='Leadership',expectedScore=0),
           dict(question=question(),studentAnswer='Not feeling',expectedScore=0),
           dict(question=question(),studentAnswer='Feelig',expectedScore=5),
           dict(question=question(),studentAnswer='Leadership',expectedScore=0,rule=dict(RULE,rejectUnlisted=True))]
    report=evaluate(cases)
    assert report['nonai_scored']==3 and report['needs_ai']==3
    assert report['mismatches']==report['false_zeros']==report['excessive_credit']==0
    assert report['rows'][4]['typo_suggestions'][0]['candidate']=='feeling'
    assert report['rows'][4]['studentAnswer']=='Feelig'


def test_evaluation_labels_do_not_change_inference_and_leakage_is_rejected():
    q=question()
    a=evaluate([dict(question=q,studentAnswer='Feeling',expectedScore=0)])
    b=evaluate([dict(question=q,studentAnswer='Feeling',expectedScore=5)])
    assert a['rows'][0]['result']==b['rows'][0]['result']
    q['approvedExamples']=[dict(text='Leadership',score=0,approved=True,provenance='teacher-score')]
    with pytest.raises(ValueError,match='leaked'):
        evaluate([dict(question=q,studentAnswer='Leadership',split='test')])
