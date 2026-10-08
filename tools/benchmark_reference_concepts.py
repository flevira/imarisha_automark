"""Offline reference-list gate experiment. No model calls or application writes."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import sqlite3
import time

from tools.benchmark_nonai_marking import evaluate
from core.written_rubric import prepare_rubric, grade_judgments
from core.written_matching import complete_criteria


def candidate(question, text, rule):
    def reject(reason):
        return dict(accepted=False, reason=reason)
    try:
        if not isinstance(rule, dict) or rule.get('mode') != 'text-exact':
            return reject('not-text-exact')
        rubric = prepare_rubric(question)
        if rubric['source'] != 'reference-list':
            return reject('not-derived-reference-list:' + rubric['source'])
        reference = question['referenceAnswers'][0]
        if re.search(r'\b(?:or|including|such as|for example)\b|e\.g\.', reference, re.I):
            return reject('alternatives-or-examples')
        # Initial trial excludes all parenthetical glosses rather than interpreting them.
        if any(c in reference for c in '()[]{}'):
            return reject('parenthetical-structure')
        # A mixed list can contain a compound item ("bread and butter, milk").
        if re.search(r'[,;\n]', reference) and re.search(r'\band\b', reference, re.I):
            return reject('mixed-list-compound-ambiguity')
        matched = complete_criteria(text, rubric['concepts'],
            case_sensitive=rule.get('caseSensitive', False),
            framing=rule.get('allowAnswerFraming', False))
        if not matched:
            return reject('incomplete-or-ambiguous-literal-coverage')
        judgments = [dict(conceptId=c['id'], status='matched' if c['id'] in matched else 'not-demonstrated',
                          segmentIds=['s1'] if c['id'] in matched else []) for c in rubric['concepts']]
        result = grade_judgments(rubric, judgments, [dict(id='s1', text=text)], question['maxMarks'])
        result['reason'] = 'benchmark-derived-reference-match'
        return dict(accepted=True, reason=result['reason'], result=result, rubric=rubric,
                    matched_concepts=[c for c in rubric['concepts'] if c['id'] in matched])
    except (ValueError, KeyError, TypeError) as exc:
        return reject('invalid-rubric:' + str(exc))


def compare(cases, candidate_fn=candidate):
    baseline = evaluate(cases)
    rows = []
    for case, base in zip(cases, baseline['rows']):
        start = time.perf_counter()
        trial = (candidate_fn(case['question'], case['studentAnswer'], case['rule']) if base['needs_ai']
                 else dict(accepted=False, reason='baseline-already-resolved-or-blocked'))
        seconds = time.perf_counter() - start
        result = trial.get('result', base['result'])
        expected = case.get('expectedScore')
        from core.marking import decimal
        agreement = None if result['score'] is None or expected is None else decimal(result['score']) == decimal(expected)
        rows.append(dict(id=case['id'], question=case['question'], studentAnswer=case['studentAnswer'],
                         expectedScore=expected, baseline=base['result'], candidate=trial, result=result,
                         agrees=agreement, added_seconds=seconds, saved_ai=case.get('saved_ai'),
                         saved_score=case.get('saved_score')))
    accepted = [r for r in rows if r['candidate']['accepted']]
    scored = [r for r in rows if r['result']['score'] is not None]
    labelled = [r for r in scored if r['expectedScore'] is not None]
    avoided = [r for r in accepted if r['saved_ai'] and not r['saved_ai'].get('reused_within_run')]
    saved = sum(r['saved_ai'].get('seconds', 0) for r in avoided)
    added = sum(r['added_seconds'] for r in rows)
    summary = dict(count=len(rows), baseline_scored=baseline['nonai_scored'], candidate_scored=len(scored),
                   newly_resolved=len(accepted), potential_ai_requests_avoided=len(avoided),
                   labelled_scored=len(labelled), exact_agreement=sum(r['agrees'] for r in labelled),
                   false_zeros=sum(float(r['result']['score']) == 0 < float(r['expectedScore']) for r in labelled),
                   excessive_credit=sum(float(r['result']['score']) > float(r['expectedScore']) for r in labelled),
                   added_matching_seconds=added, saved_ai_seconds=saved, estimated_net_seconds_saved=saved-added,
                   reasons=dict(Counter(r['candidate']['reason'] for r in rows)))
    for name,use_candidate in [('saved_workflow',False),('simulated_workflow',True)]:
        scores=[(r['result']['score'] if use_candidate and r['candidate']['accepted'] else r['saved_score'],
                 r['expectedScore']) for r in rows]
        labels=[(decimal(score),decimal(expected)) for score,expected in scores if score is not None and expected is not None]
        summary[name]=dict(labelled_scored=len(labels),pending=sum(s is None for s,e in scores),
            exact_agreement=sum(s==e for s,e in labels),false_zeros=sum(s==0<e for s,e in labels),
            excessive_credit=sum(s>e for s,e in labels))
    return dict(version='reference-list-offline-v1', summary=summary, rows=rows,
                limitations='Simulation labels are not teacher approval. Savings use saved request durations, not wall time. No live activation.')


def load_cases(database, labels):
    """Latest saved run per job in a consistent read-only transaction."""
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        saved = db.execute('SELECT result FROM structured_marking_runs WHERE sequence IN '
                           '(SELECT MAX(sequence) FROM structured_marking_runs GROUP BY job)').fetchall()
    cases = []
    for (payload,) in saved:
        run = json.loads(payload)
        for r in run['results']:
            q = r['scoring_question']
            if q['responseType'] != 'WRITTEN':
                continue
            key = (r['student_id'], str(r['display_number']))
            if key not in labels:
                continue
            # Use the original saved question, not saved AI-generated rubric/evidence.
            cases.append(dict(id=':'.join(key), question=q, rule=r['rule'],
                studentAnswer='\n'.join(a.get('text') or '' for a in r['responses']),
                expectedScore=labels[key], labelProvenance='simulation', saved_ai=r.get('ai'), saved_score=r['score']))
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Multiple attempts for the same student/question; select one dataset before comparison')
    return sorted(cases, key=lambda c:c['id'])


def html_report(report):
    esc=lambda x:escape(str(x))
    rows=[]
    for r in report['rows']:
        trial=r['candidate']
        evidence=trial.get('matched_concepts', [])
        rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['id'], r['question'].get('questionText',''),
            '; '.join(r['question'].get('referenceAnswers',[])),r['studentAnswer'],r['expectedScore'],
            r['baseline']['score'],r['result']['score'],trial['reason'],
            '; '.join(c['description'] for c in evidence),trial.get('result',{}).get('explanation','')+
            (' '+json.dumps(trial['matches'],ensure_ascii=False) if trial.get('matches') else '')])+'</tr>')
    return ('<!doctype html><meta charset="utf-8"><title>Reference concept benchmark</title>'
            '<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap}</style>'
            '<h1>Offline derived-reference concept benchmark</h1><p>'+esc(report['limitations'])+'</p><pre>'+
            esc(json.dumps(report['summary'],indent=2))+'</pre><table><tr>'+''.join('<th>'+s+'</th>' for s in
            ['Answer','Question','Reference','Extracted','Expected','Baseline','Candidate','Decision','Matched concepts','Calculation'])+
            '</tr>'+''.join(rows)+'</table>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.labels.read_text())
    labels={(s['student_number'],q):v for s in data['students'] for q,v in s['expected_marks_by_question'].items()}
    report=compare(load_cases(args.database,labels))
    if report['summary']['count']!=500:
        raise ValueError('This trial requires exactly 500 written responses; inspect the selected saved runs')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,content in [('results.json',json.dumps(report,indent=2)),('results.html',html_report(report))]:
        with (args.output/name).open('x',encoding='utf-8') as f:f.write(content)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
