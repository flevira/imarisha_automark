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


def test_multiple_positive_answers_allowed_without_review() -> None:
    model = BubbleConfidenceModel()
    scores = {
        "A": 75.0,
        "B": 74.5,
        "C": 24.0,
        "D": 23.0,
    }

    answers, needs_review, _ = model.classify(
        ["A", "B", "C", "D"],
        scores,
        allow_multiple_answers=True,
    )

    assert answers == ["A", "B"]
    assert needs_review is False


def test_mc_multiple_allowance_discards_far_lower_probabilities() -> None:
    model = BubbleConfidenceModel()
    scores = {
        "A": 171.73318077803202,
        "B": 68.36155606407323,
        "C": 68.50114416475972,
        "D": 68.36430205949657,
    }

    answers, needs_review, probabilities = model.classify(
        ["A", "B", "C", "D"],
        scores,
        allow_multiple_answers=True,
    )

    assert answers == ["A"]
    assert needs_review is False
    assert probabilities["A"] > probabilities["B"]


def test_multiple_positive_answers_without_allowance_need_review() -> None:
    model = BubbleConfidenceModel()
    scores = {
        "A": 75.0,
        "B": 74.5,
        "C": 24.0,
        "D": 23.0,
    }

    answers, needs_review, _ = model.classify(["A", "B", "C", "D"], scores)

    assert answers == ["A", "B"]
    assert needs_review is True
