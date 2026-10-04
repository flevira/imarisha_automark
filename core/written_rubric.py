"""Named written-answer requirements, kept separate from explanatory background."""
import hashlib
import re
from decimal import Decimal, ROUND_HALF_UP
from core.marking import decimal


def validate_criteria(question):
    criteria=question.get('scoringCriteria') or []
    if not isinstance(criteria,list):raise ValueError('invalid-scoring-criteria')
    seen=set();total=Decimal(0)
    for c in criteria:
        if (not isinstance(c,dict) or not {'id','description','marks'}<=set(c) or not set(c)<={'id','description','marks','acceptedAlternatives'} or
            not isinstance(c['id'],str) or not c['id'].strip() or c['id'] in seen or
            not isinstance(c['description'],str) or not c['description'].strip()):
            raise ValueError('invalid-scoring-criteria')
        variants=c.get('acceptedAlternatives',[])
        if not isinstance(variants,list) or any(not isinstance(v,str) or not v.strip() for v in variants):raise ValueError('invalid-concept-alternatives')
        seen.add(c['id']);marks=decimal(c['marks'])
        if marks<=0 or marks.as_tuple().exponent < -2:raise ValueError('invalid-scoring-criteria')
        total+=marks
    if criteria and total!=decimal(question['maxMarks']):raise ValueError('criteria-marks-must-equal-maximum')
    return criteria


def requested_count(question):
    explicit=question.get('requiredConceptCount')
    if explicit is not None:
        if type(explicit) is not int or not 1<=explicit<=20:raise ValueError('invalid-required-concept-count')
        return explicit
    text=question.get('questionText') or ''
    numbers=dict(one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10,
                 moja=1,mbili=2,mitatu=3,minne=4,mitano=5,miwili=2,mmoja=1)
    number=r'(?:10|[1-9]|'+'|'.join(numbers)+')'
    matches=list(re.finditer(r'\b(?:state|list|name|mention|give|identify|outline|describe|explain)\s+(?:any\s+)?('+number+r')\b',text,re.I))
    matches+=list(re.finditer(r'\b(?:taja|orodhesha|eleza)\s+\w+\s+('+number+r')\b',text,re.I))
    if len(matches)!=1 or re.search(r'\b(?:each|respectively|at least|at most)\b',text,re.I):return None
    n=matches[0][1].lower()
    return int(n) if n.isdigit() else numbers[n]


def _concepts(items):
    if not isinstance(items,list) or not 1<=len(items)<=40:raise ValueError('invalid-accepted-concepts')
    seen=set();descriptions=set();result=[]
    for c in items:
        if (not isinstance(c,dict) or not {'id','description'}<=set(c) or
            not set(c)<={'id','description','acceptedAlternatives'} or
            not isinstance(c['id'],str) or not c['id'].strip() or c['id'] in seen or
            not isinstance(c['description'],str) or not c['description'].strip()):raise ValueError('invalid-accepted-concepts')
        variants=c.get('acceptedAlternatives',[])
        if not isinstance(variants,list) or any(not isinstance(v,str) or not v.strip() for v in variants):raise ValueError('invalid-concept-alternatives')
        description=' '.join(c['description'].casefold().split())
        if description in descriptions:raise ValueError('duplicate-accepted-concept')
        seen.add(c['id']);descriptions.add(description)
        result.append(dict(id=c['id'],description=c['description'],acceptedAlternatives=variants))
    return result


def prepare_rubric(question):
    """Explicit teacher rubrics win. Legacy lists are split without inventing ideas."""
    criteria=validate_criteria(question)
    if criteria:
        return dict(mode='weighted',source='explicit-criteria',concepts=criteria)
    count=requested_count(question)
    if question.get('acceptedConcepts') is not None:
        concepts=_concepts(question['acceptedConcepts'])
        if count is None:raise ValueError('concept-count-required')
        if count>len(concepts):raise ValueError('insufficient-reference-concepts')
        return dict(mode='any-n',source='explicit-concepts',requiredCount=count,concepts=concepts)
    if count:
        refs=question.get('referenceAnswers',[])
        # Multiple reference strings can mean alternative complete answers, not a pool.
        if len(refs)==1 and isinstance(refs[0],str):
            parts=[p.strip(' .\n\t') for p in re.split(r';|,\s*(?:and\s+)?|\s+and\s+|\n',refs[0]) if p.strip(' .\n\t')]
            parts=list(dict.fromkeys(parts))
            # No split within examples, parentheticals or decimal values.
            ambiguous=re.search(r'\b(?:such as|for example|including|e\.g\.)\b|[()]|\d,\d',refs[0],re.I)
            if not ambiguous and count<=len(parts)<=20:
                concepts=_concepts([dict(id='ref-'+hashlib.sha256(p.encode()).hexdigest()[:12],description=p) for p in parts])
                return dict(mode='any-n',source='reference-list',requiredCount=count,concepts=concepts)
        return dict(mode='needs-rubric',source='ambiguous-reference-list',concepts=[],requiredCount=count)
    return dict(mode='holistic',source='reference-answer',concepts=[])


def answer_segments(text, rubric=None):
    # Candidate items remain evidence, not automatically distinct credited concepts.
    listing = bool(rubric and rubric.get('mode')=='any-n')
    if not listing:
        parts=[p.strip() for p in re.split(r'\n+|(?<=[.!?;])\s+',text) if p.strip()]
    else:
        # Require punctuation after labels: ordinary words and decimal values are not labels.
        label=r'(?:\((?:[ivxlcdm]+|[a-z]|\d+)\)|(?:[ivxlcdm]+|[a-z]|\d+)[.)]|[ivxlcdm]+,)'
        marker=rf'(?:{label}\s+|[•●▪*]\s*|[-–—]\s+)'
        # Split before markers, retaining their original evidence text until label removal.
        chunks=re.split(rf'\n+|\s+(?={marker})',text,flags=re.I)
        phrases=[p.casefold() for c in rubric['concepts']
                 for p in [c['description']]+c.get('acceptedAlternatives',[])]
        parts=[]
        for chunk in chunks:
            chunk=re.sub(rf'^\s*{marker}','',chunk,flags=re.I).strip()
            if not chunk:continue
            protected=(re.search(r'\b(?:not|never|no|without|neither|between|both|si|sio|bila|hakuna)\b',chunk,re.I)
                       or any((' and ' in p or ',' in p) and p in chunk.casefold() for p in phrases))
            # Commas inside numbers (1,000 or 2,5) are not item separators.
            candidates=[chunk] if protected else re.split(r'\s+and\s+|(?:,(?!\d)|(?<!\d),)\s*(?:and\s+)?|;|(?<=[.!?])\s+',chunk,flags=re.I)
            parts.extend(p.strip(' ,;') for p in candidates if p.strip(' ,;'))
    return [dict(id=f's{i}',text=part) for i,part in enumerate(parts,1)]


def grade_judgments(rubric,judgments,segments,maximum):
    expected={c['id']:c for c in rubric['concepts']};known={s['id']:s['text'] for s in segments}
    if not isinstance(judgments,list) or len(judgments)!=len(expected):raise ValueError('ai-invalid-judgments')
    seen=set();matched=[];missing=[];uncertain=[];evidence={}
    for j in judgments:
        if (not isinstance(j,dict) or set(j)!={'conceptId','status','segmentIds'} or
            not isinstance(j['conceptId'],str) or j['conceptId'] not in expected or j['conceptId'] in seen or
            j['status'] not in ('matched','not-demonstrated','contradicted','uncertain') or
            not isinstance(j['segmentIds'],list) or any(not isinstance(x,str) or x not in known for x in j['segmentIds']) or
            len(set(j['segmentIds']))!=len(j['segmentIds'])):raise ValueError('ai-invalid-judgments')
        seen.add(j['conceptId'])
        if j['status'] in ('matched','contradicted') and not j['segmentIds']:raise ValueError('ai-missing-evidence')
        evidence[j['conceptId']]=[dict(id=s,text=known[s]) for s in j['segmentIds']]
        (matched if j['status']=='matched' else uncertain if j['status']=='uncertain' else missing).append(j['conceptId'])
    maximum=decimal(maximum)
    if rubric['mode']=='any-n':
        count=rubric['requiredCount'];score=(maximum*min(len(matched),count)/count).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        unresolved=bool(uncertain and len(matched)<count)
    else:
        score=sum((decimal(expected[i]['marks']) for i in matched),Decimal(0));unresolved=bool(uncertain)
    explanation='Credited: '+('; '.join(expected[i]['description'] for i in matched) or 'none')+'.'
    if missing:explanation+=' Not demonstrated or contradicted: '+'; '.join(expected[i]['description'] for i in missing)+'.'
    if uncertain:explanation+=' Uncertain: '+'; '.join(expected[i]['description'] for i in uncertain)+'.'
    explanation+=f' Calculated credit: {min(len(matched),rubric.get("requiredCount",len(expected)))}/{rubric.get("requiredCount",len(expected))} concepts; {score}/{maximum} marks.'
    upper=(min(maximum,score+maximum*len(uncertain)/count) if rubric['mode']=='any-n' else min(maximum,score+sum((decimal(expected[i]['marks']) for i in uncertain),Decimal(0))))
    return dict(score_range=dict(lower=str(score),upper=str(upper.quantize(Decimal('.01'),rounding=ROUND_HALF_UP))) if unresolved else None,
        disputed_concepts=[expected[i]['description'] for i in uncertain],
        status='review' if unresolved else 'scored',score=None if unresolved else str(score),
        reason='ai-uncertain-concept' if unresolved else 'ai-incorrect' if score==0 else 'ai-equivalent' if score==maximum else 'ai-partial',
        explanation=explanation,evidence=evidence,
        calculation=dict(credited_concepts=len(matched),required_concepts=rubric.get('requiredCount',len(expected)),score=str(score)))
