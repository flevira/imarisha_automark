"""Offline gate evaluation: never calls AI or changes application data.

Input JSON cases: question, studentAnswer, optional rule and expectedScore.
Expected labels are evaluation data only; they never enter matching functions.
"""
import argparse
import json
from pathlib import Path
import time
from core.marking import mark_question, decimal
from core.teacher_grounding import approved_match, reference_examples, explicit_concept_match, normalize
from core.written_matching import VERSION, typo_suggestions

DEFAULT_RULE=dict(version=1,mode='text-exact',caseSensitive=False,collapseWhitespace=True,
                  normalizeListFormatting=True,normalizePresentation=True,allowAnswerFraming=True)


def evaluate(cases):
    rows=[]
    for case in cases:
        question=dict(case['question'])
        text=case['studentAnswer']
        examples=question.get('approvedExamples',[])
        if case.get('split','test')=='test' and any(e.get('provenance')!='teacher-reference'
                and normalize(e['text'])==normalize(text) for e in examples):
            raise ValueError('Held-out response leaked into examples')
        rule=case.get('rule') or question.get('scoringRules') or DEFAULT_RULE
        start=time.perf_counter()
        result=mark_question(question,[dict(kind='WRITTEN')],[dict(text=text,approved=True,selected=[])],rule)
        if result['reason']=='unlisted-wording-needs-review':
            grounded=approved_match(text,reference_examples(question)+examples,rule) or explicit_concept_match(question,text,rule)
            if grounded:result.update(grounded,engine='teacher-grounded-v1')
        elapsed=time.perf_counter()-start
        suggestions=typo_suggestions(text,question.get('referenceAnswers',[])+question.get('acceptedAlternatives',[]))
        expected=case.get('expectedScore')
        rows.append(dict(id=case.get('id'),question=question,studentAnswer=text,result=result,
            expectedScore=expected,labelProvenance=case.get('labelProvenance','unspecified'),seconds=elapsed,
            typo_suggestions=suggestions,needs_ai=result['reason']=='unlisted-wording-needs-review',
            matches_expected=None if expected is None or result['score'] is None else decimal(expected)==decimal(result['score'])))
    scored=[r for r in rows if r['result']['score'] is not None]
    labelled=[r for r in scored if r['expectedScore'] is not None]
    return dict(version=VERSION,count=len(rows),nonai_scored=len(scored),needs_ai=sum(r['needs_ai'] for r in rows),
        blocked_by_rules=sum(r['result']['score'] is None and not r['needs_ai'] for r in rows),
        labelled_scored=len(labelled),mismatches=sum(not r['matches_expected'] for r in labelled),
        false_zeros=sum(decimal(r['result']['score'])==0 and decimal(r['expectedScore'])>0 for r in labelled),
        excessive_credit=sum(decimal(r['result']['score'])>decimal(r['expectedScore']) for r in labelled),
        seconds=sum(r['seconds'] for r in rows),rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();report=evaluate(json.loads(args.cases.read_text()))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
