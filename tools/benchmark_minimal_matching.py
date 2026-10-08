"""Read-only complete-answer matcher comparison; never calls a model."""
import json
import sqlite3
from collections import Counter
from core.minimal_matching import candidate


def benchmark(db_path,labels_path):
    with sqlite3.connect('file:'+str(db_path)+'?mode=ro',uri=True) as db:
        saved=db.execute('SELECT result FROM structured_marking_runs WHERE sequence IN (SELECT MAX(sequence) FROM structured_marking_runs GROUP BY job)').fetchall()
    labels=json.load(open(labels_path))
    expected={(str(s['student_number']),str(q)):float(v) for s in labels['students'] for q,v in s['expected_marks_by_question'].items()}
    counts=Counter();bad=[];avoided_seconds=0
    for (raw,) in saved:
        run=json.loads(raw)
        originals={(m['document']['layoutId'],q['questionId']):q for m in run['manifests'] for q in m['document']['questions']}
        for r in run['results']:
            q=originals[(r['layout_id'],r['question_id'])]
            if q['responseType']!='WRITTEN':continue
            counts['written']+=1
            text='\n'.join(a['text'] for a in r['responses'])
            match=candidate(q,text)
            if not match['accepted']:continue
            counts[match['reason']]+=1
            score=float(match['result']['score']);label=expected.get((r['student_id'],str(r['display_number'])))
            if label is None:counts['missing_labels']+=1
            elif abs(score-label)>.011:bad.append(dict(student=r['student_id'],question=r['display_number'],actual=score,expected=label))
            if 'ai' in r:
                counts['potential_ai_avoided']+=1;avoided_seconds+=r['ai']['seconds']
    return dict(counts=dict(counts),incorrect_automatic_scores=bad,estimated_saved_ai_seconds=avoided_seconds)

if __name__=='__main__':
    import sys
    print(json.dumps(benchmark(*sys.argv[1:]),indent=2))
