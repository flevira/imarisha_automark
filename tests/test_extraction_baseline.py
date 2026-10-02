import pytest
from tools.extraction_baseline import evaluate

WANTED = [dict(page=2, student_id="00530", question_id="10004", item_number=1, selected_options=["A", "C"])]


def rows(choices, **overrides):
    return [dict(page=2, student_id="00530", question_id="10004", item_number=1, answer=c, needs_review=False, **overrides) for c in choices]


def test_missing_all_responses_is_zero_accuracy_not_success():
    result = evaluate(WANTED, [])
    assert result["exact_response_rate"] == 0
    assert result["missing_responses"] == 1
    assert result["detected_review_rate"] is None


def test_multiple_answer_rows_are_grouped_without_hiding_duplicates():
    assert evaluate(WANTED, rows(["A", "C"]))["exact_response_rate"] == 1
    result = evaluate(WANTED, rows(["A", "A", "C"]))
    assert result["exact_response_rate"] == 0 and result["duplicate_selections"] == 1


def test_identity_subitem_or_page_mismatch_cannot_count_as_correct():
    for key, value in [("student_id", "530"), ("page", 1), ("item_number", 2), ("question_id", "other")]:
        actual = rows(["A", "C"])
        for row in actual:
            row[key] = value
        result = evaluate(WANTED, actual)
        assert result["missing_responses"] == 1 and result["unexpected_responses"] == 1
        assert result["exact_responses"] == 0


def test_correct_but_flagged_response_reports_review_separately():
    actual = rows(["A", "C"])
    actual[0]["needs_review"] = True
    result = evaluate(WANTED, actual)
    assert result["exact_responses"] == 1 and result["detected_review_rate"] == 1


def test_duplicate_oracle_identity_is_a_fixture_error():
    with pytest.raises(ValueError):
        evaluate(WANTED+WANTED, [])


def test_worker_failure_is_recorded_not_treated_as_an_empty_success(tmp_path, monkeypatch):
    import json
    import subprocess
    from tools import extraction_baseline as baseline
    fixtures = tmp_path/"fixtures"
    fixtures.mkdir()
    (fixtures/"case.pdf").write_bytes(b"fake-pdf")
    (fixtures/"catalog.json").write_text(json.dumps({"scan_dpi": 150, "cases": [
        {"id": "fake", "path": "case.pdf", "family": "test", "pages": 1, "expected": WANTED}]}))
    monkeypatch.setattr(baseline.platform, "platform", lambda: "fixture-platform")
    monkeypatch.setattr(baseline.subprocess, "check_output", lambda *a, **k: "tesseract fake\n")
    monkeypatch.setattr(baseline.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    output = tmp_path/"report"
    output.mkdir()
    (output/"fake-runs.json").write_text('[{"status":"ok","rows":[]}]')
    report = baseline.run(fixtures, output, 1, 10)
    assert report["cases"][0]["status"] == "worker-error"
    assert report["cases"][0]["runs"] == []
    assert report["cases"][0]["expected_responses"] == 1


def test_timeout_is_an_explicit_failure(tmp_path, monkeypatch):
    import json
    import subprocess
    from tools import extraction_baseline as baseline
    (tmp_path/"case.pdf").write_bytes(b"fake-pdf")
    (tmp_path/"catalog.json").write_text(json.dumps({"scan_dpi":150,"cases":[
        {"id":"timeout", "path":"case.pdf", "family":"test", "pages":1,"expected":WANTED}]}))
    monkeypatch.setattr(baseline.platform, "platform", lambda: "fixture-platform")
    monkeypatch.setattr(baseline.subprocess, "check_output", lambda *a, **k: "tesseract fake\n")
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(baseline.subprocess, "run", timeout)
    report = baseline.run(tmp_path, tmp_path/"output", 1, 1)
    assert report["cases"][0]["status"] == "timeout"
