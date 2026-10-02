from core.answer_similarity import suggest_partial_mark


def test_estimate_is_separate_from_grade_and_preserves_negation_limitation():
    q=dict(referenceAnswers=['water and soil'],maxMarks=5)
    exact=suggest_partial_mark(q,'Water AND soil')
    assert exact['suggested_score']=='5.00' and exact['requires_teacher_review']
    partial=suggest_partial_mark(q,'water')
    assert partial['suggested_score']=='2.50'
    assert 'score' not in partial
    assert suggest_partial_mark(q,'')['suggested_score']=='0.00'


def test_evaluation_distinguishes_unlabelled_trials_from_teacher_scores():
    from tools.evaluate_written_marking import evaluate
    class Assistant:
        def assess(self,q,text):
            return dict(status='scored',score='2',seconds=.1)
    q=dict(referenceAnswers=['water'],maxMarks=2)
    result=evaluate([dict(question=q,studentAnswer='water'),
                     dict(question=q,studentAnswer='water',expectedScore='2')],Assistant())
    assert result['count']==2 and result['teacher_labelled']==1 and result['matches_expected']==1
