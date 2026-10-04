"""Compatibility helper: attach named requirements, never anonymous credit slots."""
from core.written_rubric import prepare_rubric


def with_concept_criteria(question):
    rubric=prepare_rubric(question)
    return dict(question,markingRubric=rubric,conceptCount=rubric.get('requiredCount'),criteriaSource=rubric['source'])
