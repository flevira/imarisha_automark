import json
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from tools.manifest_fixtures import FIXTURES, expected, generate, make_manifest, qr_payload


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("manifest-fixtures")
    generate(path)
    return path


@pytest.mark.parametrize("name,shared", [("personalized-mixed", False), ("shared-tf", True)])
def test_committed_manifests_and_oracles_are_reproducible(name, shared):
    manifest = json.loads((FIXTURES/f"{name}.json").read_text())
    assert make_manifest(shared) == manifest
    assert expected(manifest) == json.loads((FIXTURES/f"{name}-expected.json").read_text())
    assert manifest["assessmentGroupId"] != manifest["assessmentId"]


def test_catalog_covers_legacy_new_shared_personalized_and_multiple_pages(fixture_dir):
    catalog = json.loads((fixture_dir/"catalog.json").read_text())
    assert len(catalog["cases"]) == 6
    assert {c["family"] for c in catalog["cases"]} == {"manifest", "legacy"}
    for case in catalog["cases"]:
        with fitz.open(fixture_dir/case["path"]) as doc:
            assert len(doc) == case["pages"]
            for page in doc:
                assert bool(page.get_text().strip()) == case["id"].endswith("text")
    assert cv2.imread(str(fixture_dir/catalog["handwriting"]["path"])).shape == (100, 200, 3)
    assert catalog["handwriting"]["transcription"] == "12"
    assert "contract-only" in catalog["handwriting"]["qualification"]


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("form", ["text", "scan"])
def test_rendered_bubbles_qr_and_student_grid_match_manifest(fixture_dir, shared, form):
    name = "shared-tf" if shared else "personalized-mixed"
    manifest = json.loads((FIXTURES/f"{name}.json").read_text())
    answers = json.loads((FIXTURES/f"{name}-expected.json").read_text())
    with fitz.open(fixture_dir/f"{name}-{form}.pdf") as doc:
        for page, spec in zip(doc, manifest["pages"]):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            def dark(x, y):
                return img[int(y*2)-2:int(y*2)+3, int(x*2)-2:int(x*2)+3].mean() < 80
            qr = img[90:290, 895:1100]
            payload, _, _ = cv2.QRCodeDetector().detectAndDecode(qr)
            assert payload == qr_payload(manifest, spec["pageNumber"], shared)
            for row in spec["rows"]:
                answer = next(a for a in answers if a["question_id"] == row["questionId"] and a["item_number"] == row.get("item"))
                for bubble in row["bubbles"]:
                    assert dark(bubble["cx"], bubble["cy"]) == (bubble["optionId"] in answer["selected_options"])
            for col in spec.get("studentNumber", {}).get("columns", []):
                for bubble in col["bubbles"]:
                    assert dark(bubble["cx"], bubble["cy"]) == (bubble["value"] == "00530"[col["position"]-1])
