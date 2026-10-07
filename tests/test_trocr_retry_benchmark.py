import math
from types import SimpleNamespace
import cv2
import numpy as np
from tools.trocr_retry_benchmark import selected_confidence, trial, candidate


def test_selected_sequence_scores_exclude_eos_and_padding():
    avg,low,done=selected_confidence([7,8,2,0],[math.log(.9),math.log(.8),-9,-99],2)
    assert abs(avg-math.sqrt(.72))<1e-8 and abs(low-.8)<1e-8 and done
    assert not selected_confidence([7],[math.log(.9)],2)[2]


def test_retry_order_and_limit(monkeypatch):
    import tools.trocr_retry_benchmark as m
    monkeypatch.setattr(m,'split_lines',lambda _:([1,2],()))
    calls=[]
    def read(line):return (str(line),.95 if line==1 else .6,.7,True)
    def retry(line):calls.append(line);return ('second',.99,.9,True)
    png=cv2.imencode('.png',np.full((20,20),255,np.uint8))[1].tobytes()
    r=trial(png,SimpleNamespace(read_line=read,retry_line=retry))
    assert calls==[2] and r['retry']=='1\nsecond' and r['accepted']
    for value in [('second',.99,.9,False),('',.99,.9,True)]:
        r=trial(png,SimpleNamespace(read_line=read,retry_line=lambda _:value))
        assert not r['accepted']
    def failure(_):raise RuntimeError()
    assert not trial(png,SimpleNamespace(read_line=read,retry_line=failure))['accepted']
    monkeypatch.setattr(m,'split_lines',lambda _:([1,2],('ocr-line-segmentation-uncertain',)))
    calls.clear();r=trial(png,SimpleNamespace(read_line=read,retry_line=retry))
    assert not r['accepted'] and not calls


def test_candidate_exclusions():
    base=dict(engine='trocr-written-v1',status='uncertain',review_reasons=['ocr-low-confidence'])
    assert candidate(base)
    assert not candidate(dict(base,status='read'))
    assert not candidate(dict(base,review_reasons=['ocr-low-confidence','ocr-line-segmentation-uncertain']))


def test_beam_ancestry_is_passed_to_transition_score_calculation():
    import threading
    import torch
    from tools.trocr_retry_benchmark import RetryBackend
    output=SimpleNamespace(sequences=torch.tensor([[2,10,11,2]]),
        scores=(torch.zeros((3,20)),)*3,beam_indices=torch.tensor([[0,2,1,-1]]))
    def generate(pixels,**kwargs):
        assert kwargs['num_beams']==3 and kwargs['num_return_sequences']==1
        assert kwargs['max_new_tokens']==128 and kwargs['max_time']==20.
        assert kwargs['do_sample'] is False
        return output
    def transitions(sequences,scores,beam_indices,normalize_logits):
        assert beam_indices is output.beam_indices and normalize_logits
        return torch.tensor([[math.log(.99),math.log(.95),math.log(.7)]])
    class Processor:
        def __call__(self,**kwargs):return SimpleNamespace(pixel_values=torch.zeros(1))
        def batch_decode(self,*a,**kw):return ['valid text']
    backend=RetryBackend.__new__(RetryBackend)
    backend.torch=torch;backend.device='cpu';backend.lock=threading.Lock();backend.processor=Processor()
    backend.model=SimpleNamespace(generate=generate,compute_transition_scores=transitions,
                                   generation_config=SimpleNamespace(eos_token_id=2))
    text,avg,low,done=backend.retry_line(np.full((20,40),255,np.uint8))
    assert text=='valid text' and done and abs(avg-math.sqrt(.99*.95))<1e-6 and abs(low-.95)<1e-6
