"""Benchmark-only bounded edit matching; never used by live marking."""
import copy
import re
from core.marking import decimal
from core.written_matching import presentation, unframe, complete_criteria
from core.written_rubric import prepare_rubric, grade_judgments

PROTECTED = set('no not never without neither nor except only all any every many more less before after instead must cannot can may should always sometimes'.split())


def distance(a,b):
    """Levenshtein distance capped at two (the trial permits at most one edit)."""
    if a==b:return 0
    if abs(len(a)-len(b))>1:return 2
    if len(a)>len(b):a,b=b,a
    i=j=errors=0
    while i<len(a) and j<len(b):
        if a[i]==b[j]:i+=1;j+=1;continue
        errors+=1
        if errors>1:return 2
        if len(a)==len(b):i+=1
        j+=1
    return min(2,errors+(i<len(a) or j<len(b)))


def pair(student,reference,rule,threshold=None):
    a=presentation(student,case_sensitive=rule.get('caseSensitive',False))
    b=presentation(reference,case_sensitive=rule.get('caseSensitive',False))
    if rule.get('allowAnswerFraming'):a=unframe(a)
    if max(len(a),len(b))>1024:return None
    # Keep negation/qualifiers, numeric values, and punctuation unchanged.
    words=lambda s:re.findall(r"\w+(?:'\w+)*",s)
    aw,bw=words(a),words(b)
    if [w.casefold() for w in aw if w.casefold() in PROTECTED]!=[w.casefold() for w in bw if w.casefold() in PROTECTED]:return None
    if re.findall(r'\d+',a)!=re.findall(r'\d+',b):return None
    if re.findall(r'[^\w\s]',a)!=re.findall(r'[^\w\s]',b):return None
    compact=lambda s:re.sub(r'\s+','',s)
    if compact(a)==compact(b):
        return dict(method='spacing',distance=0,similarity=1.0,student=student,reference=reference)
    if threshold is None or len(aw)!=len(bw) or re.search(r'\d',a+b):return None
    differences=[(x,y) for x,y in zip(aw,bw) if x!=y]
    if len(differences)!=1:return None
    x,y=differences[0]
    # Never remove a word, alter a numeric/unit-like short token, or accept a short-word typo.
    if min(len(x),len(y))<4 or not x.isalpha() or not y.isalpha():return None
    d=distance(a,b);similarity=1-d/max(len(a),len(b),1)
    if d!=1 or similarity<threshold:return None
    return dict(method='one-edit',distance=d,similarity=similarity,student=student,reference=reference)


def candidate(question,text,rule,threshold=None):
    reject=lambda why:dict(accepted=False,reason=why)
    if rule.get('mode')!='text-exact':return reject('not-text-exact')
    refs=question.get('referenceAnswers',[])+question.get('acceptedAlternatives',[])
    matches=[m for r in refs if (m:=pair(text,r,rule,threshold))]
    if len(matches)>1:return reject('ambiguous-complete-answer')
    if matches:
        return dict(accepted=True,reason='benchmark-fuzzy-full-answer',matches=matches,
            result=dict(status='scored',score=str(decimal(question['maxMarks'])),reason='benchmark-fuzzy-full-answer',
                        explanation='Complete reference matched under bounded fuzzy trial; similarity is not a mark fraction.'))
    try:rubric=prepare_rubric(question)
    except (ValueError,TypeError,KeyError):return reject('invalid-rubric')
    if rubric['source'] not in ('explicit-concepts','explicit-criteria'):
        return reject('no-approved-concepts-or-full-match')
    concepts=copy.deepcopy(rubric['concepts']);evidence=[]
    # Candidate list parts are not credited unless the complete parser consumes everything.
    parts=[p.strip() for p in re.split(r';|,|\n|\s+and\s+',text,flags=re.I) if p.strip()]
    if len(parts)>40:return reject('too-many-parts')
    for part in parts:
        found=[]
        for c in concepts:
            for phrase in [c['description']]+c.get('acceptedAlternatives',[]):
                m=pair(part,phrase,rule,threshold)
                if m:found.append((c,m))
        if len({c['id'] for c,m in found})>1:return reject('ambiguous-concept')
        if found:
            c,m=found[0];c.setdefault('acceptedAlternatives',[]).append(part);evidence.append(m)
    matched=complete_criteria(text,concepts,case_sensitive=rule.get('caseSensitive',False),framing=rule.get('allowAnswerFraming',False))
    if not matched:return reject('incomplete-coverage')
    judgments=[dict(conceptId=c['id'],status='matched' if c['id'] in matched else 'not-demonstrated',segmentIds=['s1'] if c['id'] in matched else []) for c in concepts]
    result=grade_judgments(rubric,judgments,[dict(id='s1',text=text)],question['maxMarks'])
    return dict(accepted=True,reason='benchmark-fuzzy-concepts',result=result,matches=evidence,
                matched_concepts=[c for c in rubric['concepts'] if c['id'] in matched])
