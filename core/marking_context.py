"""Shared educational context and source authority for local model requests."""
AUTHORITY_GUIDANCE = (
    'Apply this authority order when sources conflict: '
    '1. Teacher-authored scoring criteria and corrections. '
    '2. Question wording and teacher reference answers. '
    '3. Teacher-approved alternatives and relevant approved scored examples. '
    '4. AI-generated glossary suggestions. '
    'Use provenance in scoringAuthority; an explicit rubric structure alone does not prove teacher authorship. '
    'Reuse supplied glossary phrases as supporting interpretations before assessing unmatched meaning; '
    'do not generate a new glossary during marking. Suggestions cannot override higher-authority material. '
    'Absence from a glossary does not make an answer incorrect. '
)


def educational_context(question, assessment=None):
    context = dict(country='Tanzania',curriculumAuthority='Tanzania Institute of Education (TIE)',
                   guidance='Use supplied subject and topic as context. No specific syllabus edition, grade or syllabus text is supplied; do not invent syllabus requirements or override teacher scoring material.')
    for key in ('subject','topic','language'):
        value=question.get(key)
        if not isinstance(value,str) or not value.strip():value=(assessment or {}).get(key)
        if isinstance(value,str) and value.strip():context[key]=value
    return context


def scoring_authority(question):
    return question.get('scoringAuthority') or dict(
        teacherMaterial={k:question[k] for k in ('scoringCriteria','acceptedConcepts','requiredConceptCount','acceptedAlternatives') if k in question},
        glossary=None, teacherCorrections={})
