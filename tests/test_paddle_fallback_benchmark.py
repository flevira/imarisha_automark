from types import SimpleNamespace
from tools.paddle_fallback_benchmark import decode,evaluate
import pytest


def test_preserves_order_and_low_confidence_lines():
    r=decode(SimpleNamespace(json={'res':dict(rec_texts=['First','Second'],rec_scores=[.99,.2],rec_boxes=[[0,0,10,10],[0,20,10,30]])}))
    assert r['text']=='First\nSecond' and r['scores']==[.99,.2] and r['complete']
    assert not decode(SimpleNamespace(json={'res':dict(rec_texts=[],rec_scores=[],rec_boxes=[])}))['complete']
    with pytest.raises(ValueError):decode(SimpleNamespace(json={'res':dict(rec_texts=['x'],rec_scores=[],rec_boxes=[])}))


def test_metrics_preserve_negation_and_word_boundaries():
    rows=[dict(expected='Do not steal.',paddle='Do steal'),dict(expected='school rules',paddle='schoolrules')]
    assert evaluate(rows,'paddle')['exact']==0
