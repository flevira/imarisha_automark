"""Run local marking examples without changing saved jobs or scores.

Input: a JSON list of {question, studentAnswer, expectedScore?} objects.
Only supply expectedScore after a teacher has established the expected mark.
"""
import argparse
import json
from pathlib import Path
from core.ai_marking import LocalMarkingAssistant
from core.answer_similarity import suggest_partial_mark
from core.marking import decimal


def evaluate(cases, assistant):
    outcomes=[]
    for case in cases:
        result=assistant.assess(case['question'],case['studentAnswer'])
        expected=case.get('expectedScore')
        outcomes.append(dict(case=case, result=result,
            similarity=suggest_partial_mark(case['question'],case['studentAnswer']),
            matches_expected=None if expected is None else
                result['status']=='scored' and decimal(result['score'])==decimal(expected)))
    labelled=[o for o in outcomes if o['matches_expected'] is not None]
    return dict(cases=outcomes,resolved=sum(o['result']['status']=='scored' for o in outcomes),
                count=len(outcomes),teacher_labelled=len(labelled),
                matches_expected=sum(o['matches_expected'] for o in labelled),
                seconds=sum(o['result']['seconds'] for o in outcomes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',default='qwen3:4b-instruct-2507-q4_K_M')
    args=parser.parse_args()
    result=evaluate(json.loads(args.examples.read_text()),LocalMarkingAssistant(args.model))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
