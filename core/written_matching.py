"""Literal written-answer matching. No inferred synonyms or model calls."""
import re
import unicodedata
from functools import lru_cache

VERSION = 'nonai-written-v1'


def presentation(text, *, case_sensitive=False, collapse_whitespace=True):
    value = unicodedata.normalize('NFC', text).replace('’', "'").replace('‘', "'")
    # Remove only paired quotation marks around the entire answer.
    value = value.strip()
    if len(value) >= 2 and (value[0], value[-1]) in {('"', '"'), ('“', '”')}:
        value = value[1:-1].strip()
    # A sentence-final full stop after a letter; retain decimals, signs, !, ? and units.
    value = re.sub(r'(?<=[^\W\d_])\s*\.$', '', value)
    if collapse_whitespace:
        value = ' '.join(value.split())
    return value if case_sensitive else value.casefold()


def unframe(text):
    # Anchored, complete neutral introductory phrases, never arbitrary stopwords.
    return re.sub(r'^(?:the answer is |answer is |answer:\s*|it is )', '', text,
                  count=1, flags=re.I)


def tokens(text, case_sensitive=False):
    return tuple(re.findall(r"\w+(?:'\w+)*|[^\w\s]", presentation(text, case_sensitive=case_sensitive)))


def complete_criteria(text, concepts, *, case_sensitive=False, framing=True):
    """Return uniquely covered concept IDs, or None when any text is unexplained.

    A bounded parser consumes complete approved phrases and explicit list syntax.
    Unknown clauses, negation/contrasts not inside an approved phrase, and ambiguous
    aliases are never discarded. Repetition maps to the same concept ID.
    """
    value = presentation(unframe(text) if framing else text, case_sensitive=case_sensitive,
                         collapse_whitespace=False)
    spans = list(re.finditer(r"\w+(?:'\w+)*|[^\w\s]", value))
    source = tuple(m.group() for m in spans)
    line_boundaries = {i for i in range(1,len(spans))
                       if '\n' in value[spans[i-1].end():spans[i].start()]}
    if not source or len(source) > 512 or len(concepts) > 40:
        return None
    phrases = {}
    for concept in concepts:
        for phrase in [concept['description']] + concept.get('acceptedAlternatives', []):
            key = tokens(phrase, case_sensitive)
            if key:
                phrases.setdefault(key, set()).add(concept['id'])
    if len(phrases) > 400:
        return None
    ordered = sorted(phrases, key=len, reverse=True)
    connectors = ((',', 'and'), (';', 'and'), ('and',), (',',), (';',), ('.',))

    def label_end(pos):
        # Only at an item boundary. Require a label delimiter, never a bare number.
        for size in (3, 2, 1):
            part = source[pos:pos+size]
            if len(part) != size:
                continue
            if (size == 3 and part[0] == '(' and part[2] == ')' and
                    re.fullmatch(r'\d+|[ivxlcdm]+|[a-z]', part[1], re.I)):
                return pos + size
            if (size == 2 and part[1] in ('.', ')') and
                    re.fullmatch(r'\d+|[ivxlcdm]+|[a-z]', part[0], re.I)):
                return pos + size
            if size == 1 and part[0] in ('•', '●', '▪'):
                return pos + size
        return pos

    @lru_cache(maxsize=None)
    def parse(pos):
        # Prefer a complete authored phrase to treating its numeric prefix as a list label.
        starts = [pos]
        labelled = label_end(pos)
        if labelled != pos:
            starts.append(labelled)
        for start in starts:
            matches = [p for p in ordered if source[start:start+len(p)] == p]
            if not matches:
                continue
            longest = len(matches[0])
            outcomes = set()
            for phrase in matches:
                if len(phrase) != longest:
                    break
                owners = phrases[phrase]
                if len(owners) != 1:
                    return frozenset()
                owner = next(iter(owners));end = start + len(phrase)
                if end == len(source):
                    outcomes.add(frozenset((owner,)))
                else:
                    next_positions = set()
                    for join in connectors:
                        if source[end:end+len(join)] == join and end+len(join) < len(source):
                            next_positions.add(end+len(join))
                    if label_end(end) != end:
                        next_positions.add(end)
                    if end in line_boundaries:
                        next_positions.add(end)
                    for nxt in next_positions:
                        for rest in parse(nxt):
                            outcomes.add(rest | {owner})
                            if len(outcomes) > 1:
                                return frozenset()  # Different allocations are ambiguous.
            return frozenset(outcomes)
        return frozenset()

    outcomes = parse(0)
    return set(next(iter(outcomes))) if len(outcomes) == 1 else None


def typo_suggestions(text, alternatives):
    """Advisory one-edit single-word candidates. Never rewrite or award marks."""
    value = presentation(text)
    if not re.fullmatch(r'[^\W\d_]{4,32}', value):
        return []
    candidates = []
    for answer in alternatives:
        other = presentation(answer)
        if (other == value or not re.fullmatch(r'[^\W\d_]{4,32}', other)
                or abs(len(value)-len(other)) > 1):
            continue
        # Levenshtein distance with bounded words, not a probability of correctness.
        row = list(range(len(other)+1))
        for i, a in enumerate(value, 1):
            new = [i]
            for j, b in enumerate(other, 1):
                new.append(min(new[-1]+1, row[j]+1, row[j-1]+(a != b)))
            row = new
        if row[-1] == 1 and other not in candidates:
            candidates.append(other)
    return [dict(original=text, candidate=c, edit_distance=1, advisory_only=True)
            for c in sorted(candidates)[:3]]


def typo_note(result):
    suggestions=result.get('typo_suggestions') or []
    if not suggestions:
        return ''
    return (' Possible spelling variant: ' + ', '.join(s['candidate'] for s in suggestions)
            + '. Advisory only: original text and marks were not changed by this suggestion.')
