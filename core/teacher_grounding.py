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
    from core.marking_context import scoring_authority
    authority=scoring_authority(question)
    glossary=authority.get('glossary') or {}
    authored=authority.get('teacherCorrections',{}).get('acceptedAlternatives',
        authority.get('teacherMaterial',{}).get('acceptedAlternatives',[]))
    refs=[dict(text=text,score=str(decimal(question['maxMarks'])),approved=True,
               provenance='teacher-reference',reviewer='Manifest author')
          for text in question.get('referenceAnswers',[]) if isinstance(text,str) and text.strip()]
    for text in question.get('acceptedAlternatives',[]):
        if not isinstance(text,str) or not text.strip():continue
        if text in authored:
            provenance='teacher-alternative'
        elif glossary.get('status')=='approved':
            provenance='teacher-approved-glossary'
        else:
            continue  # Draft suggestions are supplied separately, never as approved examples.
        refs.append(dict(text=text,score=str(decimal(question['maxMarks'])),approved=True,provenance=provenance))
    return refs


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


def approved_match(text,examples,rule=None):
    def key(value):
        if rule is not None:
            from core.marking import text_value
            return text_value(value,rule)
        return normalize(value)
    matches=[e for e in examples if e.get('approved') is True and key(e['text'])==key(text)]
    if not matches:return None
    scores={str(decimal(e['score']).normalize()) for e in matches}
    if len(scores)!=1:return dict(status='review',score=None,reason='conflicting-approved-examples',examples=matches)
    return dict(status='scored',score=str(decimal(matches[0]['score'])),reason='approved-example-match',examples=matches)


def explicit_concept_match(question,text,rule=None):
    rubric=prepare_rubric(question)
    if rubric['source'] not in ('explicit-concepts','explicit-criteria','reference-list'):return None
    from core.written_matching import complete_criteria
    matched=complete_criteria(text,rubric['concepts'],
        case_sensitive=(rule or {}).get('caseSensitive',False),
        framing=(rule or {}).get('allowAnswerFraming',True))
    if matched is None:return None
    segments=[dict(id='s1',text=text)]
    judgments=[dict(conceptId=c['id'],status='matched' if c['id'] in matched else 'not-demonstrated',
                    segmentIds=['s1'] if c['id'] in matched else []) for c in rubric['concepts']]
    result=grade_judgments(rubric,judgments,segments,question['maxMarks'])
    result['reason']='derived-reference-concept-match' if rubric['source']=='reference-list' else 'approved-concept-match'
    result['concept_source']=rubric['source']
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
