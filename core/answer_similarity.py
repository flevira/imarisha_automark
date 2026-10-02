"""Uncalibrated lexical suggestions, never authoritative marks."""
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import re
import unicodedata


def suggest_partial_mark(question, text):
    def tokens(value):
        return Counter(re.findall(r'\w+', unicodedata.normalize('NFC', value).casefold()))
    student=tokens(text)
    candidates=question.get('referenceAnswers',[])+question.get('acceptedAlternatives',[])
    best=Decimal(0)
    for reference in candidates:
        expected=tokens(reference)
        denominator=sum(student.values())+sum(expected.values())
        overlap=sum((student & expected).values())
        similarity=Decimal(2*overlap)/Decimal(denominator) if denominator else Decimal(0)
        best=max(best,similarity)
    maximum=Decimal(str(question['maxMarks']))
    return dict(method='token-overlap-f1-v1',similarity=float(best),
                suggested_score=str((best*maximum).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)),
                max_marks=str(maximum),requires_teacher_review=True,
                limitation='Lexical overlap only; not calibrated correctness. Negation, synonyms and translations may be misjudged.')
