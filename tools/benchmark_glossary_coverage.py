"""Generate teacher-only glossaries offline and evaluate saved answers, never save jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.question_glossaries import generate,VERSION
from core.marking import mark_question
from core.teacher_grounding import explicit_concept_match
from core.fuzzy_marking import candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--database',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        entries=[dict(id=r[0],material=json.loads(r[1]),model=r[2]) for r in db.execute(
            'SELECT id,material,model FROM question_glossaries WHERE id IN (SELECT MAX(id) FROM question_glossaries GROUP BY scope)')]
        runs=[json.loads(r[0]) for r in db.execute('SELECT result FROM structured_marking_runs WHERE sequence IN (SELECT MAX(sequence) FROM structured_marking_runs GROUP BY job)')]
    contexts={};reports=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    generation_file=args.output.with_suffix('.generation.json')
    for e in entries:
        q=e['material']['question'];key=str(q['questionId'])
        try:
            ctx,seconds=generate(e['material'],e['model']);contexts[key]=ctx
            reports.append(dict(question=key,context=ctx,seconds=seconds))
            print(f'Generated {key} in {seconds:.2f}s',flush=True)
        except Exception as exc:reports.append(dict(question=key,error=str(exc)));print(f'Failed {key}: {exc}',flush=True)
        generation_file.write_text(json.dumps(dict(version=VERSION,generation=reports),indent=2))
    # Evaluation labels are loaded only AFTER recognition/generation.
    truth=json.loads(args.labels.read_text())
    labels={(str(s['student_number']),str(q)):v for s in truth['students'] for q,v in s['expected_marks_by_question'].items()}
    originals={str(e['material']['question']['questionId']):e['material']['question'] for e in entries}
    rows=[];start=time.perf_counter()
    for run in runs:
        for r in run['results']:
            if r['display_number']<=10:continue
            key=str(r['question_id'])
            if key not in contexts:continue
            q={**r['scoring_question'],**originals[key],**contexts[key]}
            q['status']='READY';q['glossaryStatus']='draft'
            text='\n'.join(a.get('text') or '' for a in r['responses'])
            out=mark_question(q,[{'kind':'WRITTEN'}],r['responses'],r['rule'])
            if out['score'] is None:out=explicit_concept_match(q,text,r['rule']) or out
            if out['score'] is None:
                trial=candidate(q,text,r['rule'])
                if trial['accepted']:out=trial['result']
            expected=labels.get((str(r['student_id']),str(r['display_number'])))
            rows.append(dict(student=r['student_id'],question=r['display_number'],text=text,
                previous=r['score'],score=out['score'],reason=out['reason'],expected=expected))
    scored=[r for r in rows if r['score'] is not None];mismatches=[r for r in scored if r['expected'] is not None and float(r['score'])!=float(r['expected'])]
    report=dict(version=VERSION,generation=reports,rows=rows,summary=dict(answers=len(rows),scored=len(scored),
        pending=len(rows)-len(scored),mismatches=mismatches,matching_seconds=time.perf_counter()-start,
        generation_seconds=sum(e.get('seconds',0) for e in reports),reasons=dict(Counter(r['reason'] for r in rows))))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
