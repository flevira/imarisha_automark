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


def evaluate(cases, assistant, *, use_draft_rubrics=False):
    outcomes=[]
    for case in cases:
        question=dict(case['question'])
        if use_draft_rubrics:question.update(case.get('proposedRubric',{}))
        from core.teacher_grounding import normalize, approved_match, reference_examples, explicit_concept_match
        if case.get('split','test')=='test' and any(normalize(e['text'])==normalize(case['studentAnswer'])
                for e in question.get('approvedExamples',[]) if e.get('provenance')!='teacher-reference'):
            raise ValueError('Held-out response leaked into teaching examples')
        result=approved_match(case['studentAnswer'],reference_examples(question)) or explicit_concept_match(question,case['studentAnswer'])
        if result is not None:result=dict(result,seconds=0)
        else:result=assistant.assess(question,case['studentAnswer'])
        expected=case.get('expectedScore') if case.get('teacherApproved',True) else None
        outcomes.append(dict(case=case, result=result,
            similarity=suggest_partial_mark(case['question'],case['studentAnswer']),
            matches_expected=None if expected is None else
                result['status']=='scored' and decimal(result['score'])==decimal(expected)))
    labelled=[o for o in outcomes if o['matches_expected'] is not None]
    resolved_labels=[o for o in labelled if o['result']['status']=='scored']
    errors=[abs(decimal(o['result']['score'])-decimal(o['case']['expectedScore'])) for o in resolved_labels]
    by_language={}
    for language in sorted({o['case'].get('language','unspecified') for o in outcomes}):
        subset=[o for o in outcomes if o['case'].get('language','unspecified')==language]
        labels=[o for o in subset if o['matches_expected'] is not None]
        by_language[language]=dict(count=len(subset),unresolved=sum(o['result']['status']!='scored' for o in subset),
            teacher_labelled=len(labels),matches_expected=sum(o['matches_expected'] for o in labels) if labels else None)
    return dict(mean_absolute_error_scored=float(sum(errors)/len(errors)) if errors else None,
                exact_agreement_all_labelled=sum(o['matches_expected'] for o in labelled)/len(labelled) if labelled else None,
                scored_accuracy=sum(o['matches_expected'] for o in resolved_labels)/len(resolved_labels) if resolved_labels else None,
                review_rate=sum(o['result']['status']!='scored' for o in outcomes)/len(outcomes) if outcomes else None,
                by_language=by_language,extraction_seconds=None,human_review_seconds=None,
                draft_rubrics_used=use_draft_rubrics,
                false_zeros=sum(o['result']['status']=='scored' and decimal(o['result']['score'])==0 and decimal(o['case']['expectedScore'])>0 for o in labelled) if labelled else None,
                unsupported_credit=sum(o['result']['status']=='scored' and decimal(o['result']['score'])>decimal(o['case']['expectedScore']) for o in labelled) if labelled else None,
                partial_credit_errors=sum(o['result']['status']=='scored' and 0<decimal(o['case']['expectedScore'])<decimal(o['case']['question']['maxMarks']) and not o['matches_expected'] for o in labelled) if labelled else None,
                unresolved=sum(o['result']['status']!='scored' for o in outcomes),
                cases=outcomes,resolved=sum(o['result']['status']=='scored' for o in outcomes),
                count=len(outcomes),teacher_labelled=len(labelled),
                matches_expected=sum(o['matches_expected'] for o in labelled) if labelled else None,
                seconds=sum(o['result']['seconds'] for o in outcomes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',default='qwen3:4b-instruct-2507-q4_K_M')
    parser.add_argument('--draft-rubrics',action='store_true',help='Evaluate proposed rubrics without approving them or modifying saved jobs')
    args=parser.parse_args()
    result=evaluate(json.loads(args.examples.read_text()),LocalMarkingAssistant(args.model),use_draft_rubrics=args.draft_rubrics)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
