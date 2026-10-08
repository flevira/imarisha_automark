"""Deterministic question-level marking. No images, OCR, AI or persistence.

An absent rule is unresolved, not a zero. Scores are decimal strings and matching
items share one question allocation. Rules are versioned and saved with each run.
"""
from decimal import Decimal, InvalidOperation, DecimalException, localcontext, ROUND_HALF_UP
import re
import unicodedata

ENGINE = 'deterministic-v2'


def decimal(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('invalid-number')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError('invalid-number') from None
    if not result.is_finite() or (result and abs(result.adjusted()) > 100) or len(result.as_tuple().digits) > 100:
        raise ValueError('invalid-number')
    return result


def text_value(value, rule):
    value = unicodedata.normalize('NFC', value)
    for key in ('caseSensitive', 'collapseWhitespace'):
        if type(rule.get(key)) is not bool:
            raise ValueError('text-normalization-policy-required')
    if rule.get('normalizeListFormatting'):
        # Only explicit line-start list labels; retain numbers, negation and maths.
        lines=value.splitlines()
        pattern=r'^\s*(?:\((?:[ivxlcdm]+|\d+)\)|(?:[ivxlcdm]+|\d+)[.)]|[•])\s+'
        labelled=[bool(re.match(pattern,line,re.I)) for line in lines if line.strip()]
        if sum(labelled)>=2:
            value=re.sub(r'(?im)'+pattern,'__ITEM__',value)
            value=' and '.join(part.strip().rstrip('.') for part in value.split('__ITEM__') if part.strip())
        value=value.strip().rstrip('.')
    if rule.get('normalizePresentation'):
        from core.written_matching import presentation
        value = presentation(value, case_sensitive=rule['caseSensitive'],
                             collapse_whitespace=rule['collapseWhitespace'])
    if rule['collapseWhitespace']:
        value = ' '.join(value.split())
    return value if rule['caseSensitive'] else value.casefold()


def written_matches(text, answers, rule):
    if rule.get('mode') == 'text-exact':
        value = text_value(text, rule)
        keys = [text_value(a, rule) for a in answers]
        if value in keys:
            return True
        if rule.get('allowAnswerFraming'):
            from core.written_matching import unframe
            return unframe(value) in keys
        return False
    if rule.get('mode') != 'numeric':
        raise ValueError('written-scoring-policy-required')
    tolerance = decimal(rule.get('tolerance'))
    unit = rule.get('unit')
    if tolerance < 0 or not isinstance(unit, str):
        raise ValueError('numeric-policy-invalid')
    # Units are literal and case-sensitive. No inferred conversions or comma rules.
    pattern = r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*' + re.escape(unit) + r'\s*'
    values = []
    for answer in answers:
        match = re.fullmatch(pattern, answer)
        if not match:
            raise ValueError('numeric-answer-key-invalid')
        values.append(decimal(match[1]))
    match = re.fullmatch(pattern, text)
    if not match:
        raise ValueError('numeric-response-needs-review')
    return any(abs(decimal(match[1]) - value) <= tolerance for value in values)


def mark_question(question, expected, responses, rule):
    with localcontext() as context:
        context.prec = 512
        return _mark_question(question, expected, responses, rule)


def _mark_question(question, expected, responses, rule):
    """responses: reviewed dicts keyed by layout page and item/part number.

    expected includes geometry and answer metadata from the saved layout.
    The caller supplies all expected parts, even when a page/response is missing.
    """
    result = dict(question_id=question['questionId'], display_number=question['displayNumber'],
                  score=None, max_marks=None, status='review', reason='', engine=ENGINE)
    try:
        maximum = decimal(question.get('maxMarks'))
        if maximum < 0:
            raise ValueError('invalid-question-marks')
        result['max_marks'] = str(maximum)
        if not expected:
            raise ValueError('missing-question-geometry')
        if len(responses) != len(expected) or any(r is None for r in responses):
            raise ValueError('missing-or-excluded-response')
        if any(not r.get('approved') for r in responses):
            raise ValueError('unresolved-transcription')
        kind = question.get('responseType')
        if any(e['kind'] != kind for e in expected):
            raise ValueError('question-type-mismatch')
        if not isinstance(rule, dict) or rule.get('version') != 1:
            raise ValueError('scoring-policy-required')
        allowed = {'version','mode'}
        if rule.get('mode') == 'text-exact':
            allowed |= {'caseSensitive','collapseWhitespace','rejectUnlisted','normalizeListFormatting',
                        'normalizePresentation','allowAnswerFraming'}
            if any(type(rule[k]) is not bool for k in allowed-{'mode','version'} if k in rule):
                raise ValueError('text-normalization-policy-required')
        elif rule.get('mode') == 'numeric':
            allowed |= {'tolerance','unit'}
        if not set(rule) <= allowed:
            raise ValueError('unsupported-scoring-policy')
        if kind == 'WRITTEN':
            if rule.get('rejectUnlisted') and (question.get('acceptedConcepts') or question.get('scoringCriteria')):
                raise ValueError('exhaustive-answer-rule-conflicts-with-partial-rubric')
            if question.get('status') not in ('READY', 'MANUAL_REVIEW'):
                raise ValueError('answer-key-needs-review')
            answers = question.get('referenceAnswers')
            alternatives = question.get('acceptedAlternatives', [])
            if not isinstance(answers, list) or not isinstance(alternatives, list):
                raise ValueError('written-answer-key-required')
            answers = answers + alternatives
            if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                raise ValueError('written-answer-key-required')
            text = '\n'.join(r['text'] for r in responses)
            # Validate rules and key even for a blank response.
            if text.strip():
                correct = written_matches(text, answers, rule)
            else:
                written_matches(answers[0], answers, rule)
                correct = False
            if not correct and text.strip() and rule.get('mode') == 'text-exact' and rule.get('rejectUnlisted') is not True:
                raise ValueError('unlisted-wording-needs-review')
        else:
            if question.get('status') != 'READY':
                raise ValueError('answer-key-needs-review')
            if kind not in ('MC', 'MA', 'TF', 'MATCH') or rule.get('mode') not in ('all-or-nothing','partial'):
                raise ValueError('unsupported-scoring-policy')
            matches = []
            fractions = []
            for expected_row, response in zip(expected, responses):
                options = {b['optionId'] for b in expected_row['bubbles']}
                key = expected_row.get('correctAnswers')
                accepted = expected_row.get('acceptableAnswers', key)
                if (expected_row.get('answerStatus') != 'READY' or not isinstance(key, list) or not key
                        or any(not isinstance(a, str) for a in key) or not set(key) <= options
                        or len(set(key)) != len(key) or (kind != 'MA' and len(key) != 1)):
                    raise ValueError('answer-key-needs-review')
                if not isinstance(accepted, list) or any(not isinstance(a, str) for a in accepted) or not set(key) <= set(accepted) <= options:
                    raise ValueError('answer-alternatives-invalid')
                if kind == 'MA' and set(accepted) != set(key):
                    raise ValueError('multiple-answer-alternatives-need-policy')
                if decimal(expected_row.get('questionMarks')) != maximum:
                    raise ValueError('question-marks-conflict')
                selected = response['selected']
                if not set(selected) <= options or len(set(selected)) != len(selected) or (kind != 'MA' and len(selected) > 1):
                    raise ValueError('invalid-reviewed-selection')
                matches.append(set(selected) == set(key) if kind == 'MA' else len(selected) == 1 and selected[0] in accepted)
                fractions.append(Decimal(max(0,len(set(selected)&set(key))-len(set(selected)-set(key))))/len(key)
                                 if kind=='MA' else Decimal(int(matches[-1])))
            correct = all(matches)
            if rule['mode']=='partial':
                score=min(maximum,(maximum*sum(fractions)/len(fractions)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
                result.update(score=str(score),status='scored',reason='partial-credit' if 0<score<maximum else
                              'accepted-answer' if correct else 'blank-or-incorrect')
                return result
        result.update(score=str(maximum if correct else Decimal(0)), status='scored',
                      reason='accepted-answer' if correct else 'blank-or-incorrect')
    except (ValueError, KeyError, TypeError, DecimalException) as exc:
        result['reason'] = str(exc) if isinstance(exc, ValueError) else 'invalid-answer-contract'
    return result
