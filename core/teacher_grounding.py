"""Approved examples and explicit concept matches; no model or storage dependencies."""
import hashlib
import json
import re
import unicodedata
from core.marking import decimal
from core.written_rubric import prepare_rubric, grade_judgments


def normalize(text):
    # Preserve punctuation, negation and numbers: this is not fuzzy score matching.
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def question_scope(question, rule=None):
    keys=('questionId','questionText','referenceAnswers','acceptedAlternatives','maxMarks',
          'scoringCriteria','acceptedConcepts','requiredConceptCount')
    value={k:question.get(k) for k in keys}
    value['rule']=rule
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def reference_examples(question):
    return [dict(text=text,score=str(decimal(question['maxMarks'])),approved=True,
                 provenance='teacher-reference',reviewer='Manifest author')
            for text in question.get('referenceAnswers',[])+question.get('acceptedAlternatives',[])
            if isinstance(text,str) and text.strip()]


def retrieve_examples(question,text,examples=(),limit=4):
    candidates=reference_examples(question)+[e for e in examples if e.get('approved') is True]
    tokens=set(re.findall(r'\w+',normalize(text)))
    def relevance(e):
        other=set(re.findall(r'\w+',normalize(e['text'])))
        return len(tokens & other)/max(1,len(tokens | other))
    # Lexical relevance selects teaching examples, never marks. Always include a reference.
    refs=[e for e in candidates if e['provenance']=='teacher-reference'][:1]
    ranked=sorted((e for e in candidates if e not in refs),key=relevance,reverse=True)
    return (refs+ranked)[:limit]


def approved_match(text,examples):
    matches=[e for e in examples if e.get('approved') is True and normalize(e['text'])==normalize(text)]
    if not matches:return None
    scores={str(decimal(e['score']).normalize()) for e in matches}
    if len(scores)!=1:return dict(status='review',score=None,reason='conflicting-approved-examples',examples=matches)
    return dict(status='scored',score=str(decimal(matches[0]['score'])),reason='approved-example-match',examples=matches)


def explicit_concept_match(question,text):
    rubric=prepare_rubric(question)
    if rubric['source'] not in ('explicit-concepts','explicit-criteria'):return None
    clauses=[p.strip().rstrip('.') for p in re.split(r'\n|;|\s+and\s+',text,flags=re.I) if p.strip()]
    clauses=[re.sub(r'^\s*(?:\([ivx\d]+\)|\d+[.)])\s*','',p,flags=re.I) for p in clauses]
    if not clauses:return None
    matched=set()
    for clause in clauses:
        ids=[c['id'] for c in rubric['concepts'] if normalize(clause) in
             [normalize(a.rstrip('.')) for a in [c['description']]+c.get('acceptedAlternatives',[])]]
        if len(ids)!=1:return None  # Unknown clause, negation, overlapping concepts: use semantic review.
        matched.add(ids[0])
    segments=[dict(id='s1',text=text)]
    judgments=[dict(conceptId=c['id'],status='matched' if c['id'] in matched else 'not-demonstrated',
                    segmentIds=['s1'] if c['id'] in matched else []) for c in rubric['concepts']]
    result=grade_judgments(rubric,judgments,segments,question['maxMarks'])
    result['reason']='approved-concept-match'
    return result


def uncertainty(result):
    reason=result.get('reason','')
    if result.get('status')=='scored':return None
    if reason in ('unresolved-transcription','missing-or-excluded-response'):category='extraction'
    elif any(t in reason for t in ('invalid','incomplete','unavailable','not-installed','timeout','missing-evidence')):category='technical'
    elif any(t in reason for t in ('rubric','criteria','policy','answer-key','conflicting-approved')):category='rubric'
    else:category='meaning'
    ai=result.get('ai',{})
    bounds=ai.get('score_range')
    if bounds is None and category in ('meaning','rubric') and result.get('max_marks') is not None:
        bounds=dict(lower='0',upper=str(result['max_marks']))
    return dict(category=category,reason=reason,score_range=bounds,
                disputed_concepts=ai.get('disputed_concepts',[]),confidence=None)
