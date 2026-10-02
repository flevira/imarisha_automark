"""Evaluate local crop transcription against user-supplied labels, not answer keys.

Dataset JSON: {"cases":[{"image":"crop.png","text":"verbatim handwriting",
"language":"en","category":"clear"}],"qualification":"real-labeled"}
Paths are relative to the dataset. Reports contain draft text; keep them local.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import unicodedata

from core.contracts import RecognitionRequest, SourceRegion
from core.handwriting import OllamaHandwritingRecognizer, EasyOCRBaseline


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


def evaluate(cases,recognizer):
    records=[];chars=words=char_errors=word_errors=0
    for index,case in enumerate(cases):
        start=time.perf_counter()
        # Label is deliberately not included in RecognitionRequest.
        draft=recognizer.recognize(RecognitionRequest(SourceRegion('benchmark',index+1,(0,0,1,1),'benchmark-crop'),case['image_png'],case['language']))
        expected=unicodedata.normalize('NFC',case['text'])
        actual=unicodedata.normalize('NFC',draft.text or '')
        ce=distance(expected,actual);we=distance(expected.split(),actual.split())
        chars+=len(expected);words+=len(expected.split());char_errors+=ce;word_errors+=we
        records.append(dict(index=index,category=case.get('category','unspecified'),language=case['language'],
                            expected=expected,draft=asdict(draft),wall_seconds=time.perf_counter()-start,
                            character_errors=ce,word_errors=we,exact=expected==actual))
    return dict(cases=records,character_error_rate=char_errors/chars if chars else None,
        word_error_rate=word_errors/words if words else None,character_errors=char_errors,word_errors=word_errors,
        exact_matches=sum(r['exact'] for r in records),count=len(records),
        review_rate=sum(bool(r['draft']['review_reasons']) for r in records)/len(records) if records else None,
        failures=sum(r['draft']['status']=='error' for r in records),
        total_seconds=sum(r['wall_seconds'] for r in records),first_call_seconds=records[0]['wall_seconds'] if records else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['ollama','easyocr'],default='ollama')
    parser.add_argument('dataset',type=Path);parser.add_argument('--model',default='gemma3:4b-it-q4_K_M')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    dataset=json.loads(args.dataset.read_text())
    cases=[dict(c, image_png=(args.dataset.parent/c['image']).read_bytes()) for c in dataset['cases']]
    result=evaluate(cases,OllamaHandwritingRecognizer(args.model) if args.engine=='ollama' else EasyOCRBaseline())
    result.update(qualification=dataset.get('qualification','unverified'),hardware=platform.platform(),machine=platform.machine(),
                  timing_note='Includes model verification/loading in first call when applicable; server warm state uncontrolled. No marking or manual review.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
