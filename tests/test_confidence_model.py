from core.confidence_model import BubbleConfidenceModel


def test_flat_scores_with_close_probs_are_treated_as_blank() -> None:
    model = BubbleConfidenceModel()
    scores = {
        "A": 39.61327713382508,
        "B": 40.81348788198105,
        "C": 41.986301369863014,
        "D": 41.71022128556376,
    }

    answers, needs_review, probabilities = model.classify(["A", "B", "C", "D"], scores)

    assert answers == []
    assert needs_review is False
    assert probabilities


def test_clear_single_mark_is_not_treated_as_blank() -> None:
    model = BubbleConfidenceModel()
    scores = {
        "A": 70.0,
        "B": 28.0,
        "C": 26.0,
        "D": 25.0,
    }

    answers, needs_review, _ = model.classify(["A", "B", "C", "D"], scores)

    assert answers == ["A"]
    assert needs_review is False
