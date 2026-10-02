from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import pytest

from core.contracts import (ExtractedResponse, LayoutRef, MarkingResult, ReadStatus,
                            RecognitionRequest, ResponseType, SourceRegion, Transcription, page_geometry)

FIXTURES = Path(__file__).parent / "fixtures/manifest"


def response(**overrides):
    values = dict(response_id="fixture-response-1", layout=LayoutRef("fixture-v4", 4, "EXAM", "42", "40"),
                  student_id="00530", question_id="10004", display_number=4, layout_page_number=2,
                  response_type=ResponseType.MATCH, item_number=2, selected_options=("A",),
                  regions=(SourceRegion("sha256:synthetic", 3, (40, 180, 230, 20), "crops/row.png"),))
    return ExtractedResponse(**{**values, **overrides})


@pytest.mark.parametrize("kind", list(ResponseType))
def test_wire_roundtrip_preserves_identifiers_and_evidence(kind):
    item = response(response_type=kind, item_number=2 if kind == ResponseType.MATCH else None,
                    raw_text="12" if kind == ResponseType.WRITTEN else None,
                    selected_options=() if kind == ResponseType.WRITTEN else ("A",))
    assert ExtractedResponse.from_json(item.to_json()) == item
    assert json.loads(item.to_json())["student_id"] == "00530"
    assert item.regions[0].page_number == 3  # source pages may be reordered
    assert item.layout_page_number == 2
    assert "score" not in json.loads(item.to_json())


def test_blank_uncertain_and_failed_are_distinct_and_persist_review_reasons():
    blank = response(selected_options=(), status=ReadStatus.BLANK)
    ambiguous = response(status=ReadStatus.UNCERTAIN, review_reasons=("erasure",))
    failed = response(selected_options=(), status=ReadStatus.ERROR, review_reasons=("model-unavailable",))
    assert not blank.needs_review
    for item in (ambiguous, failed):
        restored = ExtractedResponse.from_json(item.to_json())
        assert restored.needs_review and restored.review_reasons == item.review_reasons
    assert len({item.status for item in (blank, ambiguous, failed)}) == 3


@pytest.mark.parametrize("overrides", [
    {"student_id": 530}, {"question_id": 10004}, {"student_id": None},
    {"item_number": None}, {"layout_page_number": 0}, {"schema_version": 2},
    {"status": ReadStatus.UNCERTAIN}, {"status": ReadStatus.BLANK},
    {"selected_options": ("A", "A")}, {"regions": ()},
])
def test_invalid_or_ambiguous_records_are_not_silently_coerced(overrides):
    with pytest.raises(ValueError):
        response(**overrides)


def test_unresolved_identity_is_explicit():
    item = response(student_id=None, review_reasons=("unreadable-student-grid",))
    assert ExtractedResponse.from_json(item.to_json()).needs_review


@pytest.mark.parametrize("bbox", [(0, 0, -1, 20), (0, 0, 20, 0), (-1, 0, 20, 20), (0, 0, float("nan"), 20)])
def test_source_coordinates_must_be_usable(bbox):
    with pytest.raises(ValueError):
        SourceRegion("source", 1, bbox, "crop.png")


def test_future_fields_do_not_leak_answers_into_extraction_geometry():
    manifest = json.loads((FIXTURES/"shared-tf.json").read_text())
    page = manifest["pages"][0]
    for obj in [page, page["rows"][0], page["rows"][0]["bubbles"][0], page["studentNumber"]["columns"][0]["bubbles"][0]]:
        obj["futureTeacherSecret"] = "DO-NOT-EXPOSE"
    geometry = page_geometry(manifest, 1)
    encoded = json.dumps(geometry)
    for forbidden in ("correctAnswers", "referenceAnswers", "answerDetail", "questionMarks", "DO-NOT-EXPOSE"):
        assert forbidden not in encoded
    geometry["rows"][0]["bubbles"][0]["cx"] = 999
    assert page["rows"][0]["bubbles"][0]["cx"] != 999


def test_recognizer_contract_cannot_receive_teacher_key_and_marking_is_separate():
    class FakeRecognizer:
        def recognize(self, request):
            assert {f.name for f in fields(request)} == {"region", "image_png", "language"}
            return Transcription("12", ReadStatus.READ, (), "fake")
    item = response(response_type=ResponseType.WRITTEN, item_number=None, selected_options=(), raw_text="12")
    request = RecognitionRequest(item.regions[0], b"synthetic-crop", "en")
    assert FakeRecognizer().recognize(request).text == "12"
    original = item.to_json()
    score = MarkingResult(item.response_id, item.revision, item.layout.layout_id, item.layout.version,
                          "0", "marked", "Synthetic wrong answer", "fake-marker")
    assert MarkingResult(**json.loads(json.dumps(asdict(score)))) == score
    assert item.to_json() == original
    with pytest.raises(TypeError):
        RecognitionRequest(item.regions[0], b"crop", "en", correct_answers=["13"])


def test_fake_repository_roundtrip_needs_no_database_or_extractor():
    class MemoryRepository:
        def __init__(self):
            self.rows = {}
        def save(self, item):
            self.rows[item.response_id] = item.to_json()
        def load(self, identifier):
            return ExtractedResponse.from_json(self.rows[identifier]) if identifier in self.rows else None
    repository = MemoryRepository()
    repository.save(response())
    assert repository.load("fixture-response-1") == response()
    assert repository.load("missing") is None


def test_committed_written_response_is_an_explicit_contract_only_example():
    item = ExtractedResponse.from_json((FIXTURES/"written-response.json").read_text())
    assert item.raw_text == "12" and item.response_type == ResponseType.WRITTEN
    assert item.needs_review and item.engine == "synthetic-oracle"
    assert item.to_json() == ExtractedResponse.from_json(item.to_json()).to_json()
