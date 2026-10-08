import pytest
from tools.fuzzy_marking_trial import pair,candidate,distance
from tools.benchmark_nonai_marking import DEFAULT_RULE as RULE
from test_nonai_written import question,concepts


def test_distance():
    for a,b,n in [('abc','abc',0),('abc','ac',1),('ac','abc',1),('abc','adc',1),('ab','ba',2)]:
        assert distance(a,b)==n


def test_spacing_and_thresholds():
    assert pair('schoolrules','school rules',RULE)['method']=='spacing'
    a='Report concerns through the school leadership channels'
    b='Report concerns through the school leadershiip channels'
    assert pair(a,b,RULE,.95)
    assert pair(a,b,RULE,.97)
    assert pair(a,b,RULE,.99) is None
    assert pair(a,b,RULE) is None


@pytest.mark.parametrize('a,b', [('not feeling','feeling'),('vote many times','vote'),('10 metres','11 metres'),
    ('elected','deleted'),('cooks','cocks'),('no schoolrules','school rules'),('feeling and theft','feeling'),
    ('5 cm','5 mm'),('vote!','vote?')])
def test_reject_meaning_changes(a,b):
    assert pair(a,b,RULE,.95) is None


def test_full_answer_and_concept_scores():
    q=question();q['referenceAnswers']=['School rules']
    assert candidate(q,'Schoolrules',RULE)['result']['score']=='5'
    c=concepts()
    assert candidate(c,'A bucket of drysand',RULE)['result']['score']=='2.50'
    assert not candidate(c,'A bucket of drysand and dancing',RULE)['accepted']
    assert candidate(c,'A bucket of drysand and A bucket of drysand',RULE)['result']['score']=='2.50'


def test_ambiguous_and_no_mutation():
    import copy
    q=question();q['referenceAnswers']=['schoolrules','school rules']
    assert not candidate(q,'school rules',RULE)['accepted']
    c=concepts();before=copy.deepcopy(c);candidate(c,'A bucket of drysand',RULE)
    assert c==before
