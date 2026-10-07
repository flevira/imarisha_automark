"""Offline PaddleOCR-before-Gemma experiment; no live workflow integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import re
import unicodedata
import statistics

import cv2
import numpy as np


def words(text):
    return re.findall(r'\w+',unicodedata.normalize('NFC',text or '').casefold())


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


def evaluate(rows,key):
    exact=ce=we=nc=nw=0
    for r in rows:
        a,b=words(r['expected']),words(r[key]);x,y=' '.join(a),' '.join(b)
        exact+=a==b;ce+=distance(x,y);we+=distance(a,b);nc+=len(x);nw+=len(a)
    return dict(n=len(rows),exact=exact,CER=ce/nc if nc else None,WER=we/nw if nw else None)


def decode(result):
    """Use Paddle's documented reading order; never hide low-confidence lines."""
    data=result.json['res']
    texts=data['rec_texts'];scores=[float(s) for s in data['rec_scores']]
    if len(texts)!=len(scores):raise ValueError('Mismatched recognition results')
    # `complete` only means all returned detections contain text. Detection may
    # still miss whole words/lines; confidence alone cannot establish coverage.
    return dict(text='\n'.join(texts),scores=scores,boxes=data['rec_boxes'],
                complete=bool(texts) and all(t.strip() for t in texts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--database',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    from paddleocr import PaddleOCR
    t=time.perf_counter()
    ocr=PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',
                  text_recognition_model_name='en_PP-OCRv5_mobile_rec',
                  text_detection_model_dir=str(a.models/'PP-OCRv5_mobile_det'),
                  text_recognition_model_dir=str(a.models/'en_PP-OCRv5_mobile_rec'),
                  use_doc_orientation_classify=False,use_doc_unwarping=False,
                  use_textline_orientation=False,device='cpu',cpu_threads=4,
                  text_rec_score_thresh=0.0)
    load=time.perf_counter()-t
    if a.prepare:
        print('Models prepared; no student data read.');return
    if not all((a.database,a.dataset,a.output)):p.error('Supply database, dataset and output')
    db=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    jobs={r['id']:dict(r) for r in db.execute('select * from structured_jobs')}
    history={}
    for r in db.execute('select * from structured_transcriptions order by sequence'):
        history.setdefault((r['job'],r['response']),[]).append(dict(r,result=json.loads(r['result'])))
    cases=[];verified=set()
    for r in db.execute('select * from structured_responses'):
        h=history.get((r['job'],r['id']),[])
        tr=[t for t in h if t['result']['engine']=='trocr-written-v1']
        ai=[t for t in h if t['result']['engine'].startswith('ollama:')]
        if not tr or not ai or tr[-1]['result']['status']=='read':continue
        job=jobs[r['job']]
        if r['job'] not in verified:
            source=a.dataset/'student_scripts'/job['name']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=job['digest']:raise ValueError('Dataset mismatch')
            verified.add(r['job'])
        raw=json.loads(r['raw'])
        if len(raw['regions'])!=1:raise ValueError('Expected one crop per response')
        png=db.execute('select data from structured_crops where job=? and ref=?',
            (r['job'],raw['regions'][0]['crop_ref'])).fetchone()[0]
        cases.append(dict(student=raw['student_id'],question=str(raw['display_number']),
            original=tr[-1]['result'].get('text') or '',gemma=ai[-1]['result'].get('text') or '',
            gemma_seconds=ai[-1]['result']['seconds'],flags=tr[-1]['result']['review_reasons'],image=png))
    db.rollback();db.close()
    if not cases:raise ValueError('No matching fallback cases')
    rows=[]
    for i,c in enumerate(cases):
        t=time.perf_counter();error=None
        try:
            image=cv2.imdecode(np.frombuffer(c['image'],np.uint8),cv2.IMREAD_COLOR)
            pred=list(ocr.predict(image))
            if len(pred)!=1:raise ValueError('Expected one crop result')
            result=decode(pred[0])
        except Exception as exc:
            result=dict(text='',scores=[],boxes=[],complete=False);error=str(exc)
        rows.append({k:v for k,v in c.items() if k!='image'} | dict(paddle=result['text'],
            scores=result['scores'],boxes=np.asarray(result['boxes']).tolist(),complete=result['complete'],
            seconds=time.perf_counter()-t,error=error))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(load_seconds=load,rows=rows),indent=2))
        if (i+1)%20==0:print(f'{i+1}/{len(cases)}',flush=True)
    # Only scoring, after all image inference, sees simulation labels.
    gt={str(s['student_number']):s for s in json.loads((a.dataset/'ground_truth.json').read_text())['students']}
    for r in rows:r['expected']=gt[r['student']]['responses'][r['question']]
    policies={}
    for name,threshold,agreement in [('0.90',.90,False),('0.95',.95,False),('0.98',.98,False),('0.95-and-agreement',.95,True)]:
        accepted=[r for r in rows if r['complete'] and min(r['scores'],default=0)>=threshold
                  and (not agreement or words(r['paddle'])==words(r['original']))]
        for r in rows:r['combined']=r['paddle'] if r in accepted else r['gemma']
        policies[name]=dict(accepted=len(accepted),metrics=evaluate(rows,'combined'),
            accepted_wrong=[dict(student=r['student'],question=r['question'],expected=r['expected'],paddle=r['paddle'],gemma=r['gemma']) for r in accepted if words(r['paddle'])!=words(r['expected'])],
            estimated_net_seconds_saved=sum(r['gemma_seconds'] for r in accepted)-sum(r['seconds'] for r in rows))
    seconds=[r['seconds'] for r in rows]
    summary=dict(cases=len(rows),load_seconds=load,models=['PP-OCRv5_mobile_det','en_PP-OCRv5_mobile_rec'],
        device='cpu',cpu_threads=4,failures=sum(bool(r['error']) for r in rows),
        metrics={k:evaluate(rows,k) for k in ('original','paddle','gemma')},
        timing=dict(total=sum(seconds),mean=statistics.mean(seconds),median=statistics.median(seconds),p95=float(np.percentile(seconds,95))),
        policies=policies)
    # Do not retain the last exploratory policy as an apparent actual result.
    for r in rows:r.pop('combined',None)
    a.output.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='policies'},indent=2))
    print('Policy counts', {k:dict(accepted=v['accepted'],wrong=len(v['accepted_wrong']),saving=v['estimated_net_seconds_saved']) for k,v in policies.items()})


if __name__=='__main__':main()
