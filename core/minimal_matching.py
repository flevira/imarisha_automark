"""Complete-answer matching only; spelling similarity is not semantic confidence."""
import re
import unicodedata
from core.marking import decimal

VERSION='minimal-written-v4'
THRESHOLD=.90
# Known real-word confusion from the evaluation: never auto-correct its meaning.
UNSAFE_WORD_PAIRS={frozenset(('elected','deleted'))}
PROTECTED=set('no not never without neither nor except only all any every many more less before after instead must cannot can may should always sometimes'.split())


def normalize(text):
    value=' '.join(unicodedata.normalize('NFC',text).casefold().split())
    # Remove layout spacing only; retain punctuation, numbers and word boundaries.
    return re.sub(r'(?<=\w) +([.!?])$', r'\1', value)


def distance(a,b):
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def pair(student,reference):
    a,b=normalize(student),normalize(reference)
    if not a or not b or max(len(a),len(b))>1024:return None
    aw,bw=re.findall(r"\w+(?:'\w+)*",a),re.findall(r"\w+(?:'\w+)*",b)
    if len(aw)!=len(bw):return None
    if [w for w in aw if w in PROTECTED]!=[w for w in bw if w in PROTECTED]:return None
    if re.findall(r'\d+',a)!=re.findall(r'\d+',b):return None
    if re.findall(r'[^\w\s]',a)!=re.findall(r'[^\w\s]',b):return None
    for x,y in zip(aw,bw):
        if frozenset((x,y)) in UNSAFE_WORD_PAIRS:return None
        if x!=y and (min(len(x),len(y))<4 or not x.isalpha() or not y.isalpha()):return None
    edits=distance(a,b);similarity=1-edits/max(len(a),len(b))
    if similarity<THRESHOLD:return None
    return dict(method='levenshtein',distance=edits,similarity=similarity,student=student,reference=reference)


def candidate(question,text,rule=None):
    refs=question.get('referenceAnswers',[])+question.get('acceptedAlternatives',[])
    refs=list({normalize(r):r for r in refs if isinstance(r,str) and r.strip()}.values())
    value=normalize(text)
    if not value:return dict(accepted=False,reason='empty-answer')
    exact=[r for r in refs if normalize(r)==value]
    if exact:
        matches=[dict(method='normalized-exact',distance=0,similarity=1.,student=text,reference=exact[0])]
        reason='accepted-answer'
    else:
        matches=[m for r in refs if (m:=pair(text,r))]
        if len(matches)!=1:return dict(accepted=False,reason='ambiguous-complete-answer' if matches else 'no-complete-match')
        reason='fuzzy-full-answer'
    return dict(accepted=True,reason=reason,matches=matches,
                result=dict(status='scored',score=str(decimal(question['maxMarks'])),reason=reason))
