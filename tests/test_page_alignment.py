import json

import cv2
import fitz
import numpy as np
import pytest

from core.contracts import page_geometry
from core.manifest_layout import PreparationError, bind_sheet
from core.page_alignment import MarkerAligner, decode_sheet, transform_points
from services.manifest_resolver import CachedManifestResolver
from services.page_preparation import prepare_document, prepare_image
from tools.manifest_fixtures import FIXTURES, generate


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("m2-pdf")
    generate(path)
    return path


@pytest.fixture
def resolver(tmp_path):
    cache = CachedManifestResolver(tmp_path/"cache.db", "synthetic-fixtures-only")
    for name in ("personalized-mixed", "shared-tf"):
        cache.seed_fixture(json.loads((FIXTURES/f"{name}.json").read_text()))
    return cache


def render(path, page_number=0, dpi=150):
    with fitz.open(path) as doc:
        pix = doc[page_number].get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
        return cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)


def distorted(image, mode):
    h, w = image.shape[:2]
    if mode.startswith("rotate"):
        angle = int(mode[6:])
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        corners = np.array([[0,0], [w,0], [w,h], [0,h]], np.float32)
        transformed = cv2.transform(corners.reshape(-1,1,2), matrix).reshape(-1,2)
        low, high = transformed.min(axis=0), transformed.max(axis=0)
        matrix[:,2] -= low
        matrix = np.vstack([matrix, [0,0,1]])
        size = tuple(np.ceil(high-low).astype(int))
    elif mode == "perspective":
        matrix = cv2.getPerspectiveTransform(np.array([[0,0],[w,0],[w,h],[0,h]],np.float32),
                 np.array([[90,45],[w+30,80],[w+90,h+100],[40,h+60]],np.float32))
        size = (w+160,h+160)
    elif mode == "scaled":
        matrix = np.diag([.75,.75,1.])
        size = (round(w*.75),round(h*.75))
    else:
        matrix, size = np.eye(3), (w,h)
    return cv2.warpPerspective(image,matrix,size,borderValue=(255,255,255)), matrix


@pytest.mark.parametrize("name", ["personalized-mixed", "shared-tf"])
@pytest.mark.parametrize("mode", ["normal", "rotate90", "rotate180", "rotate270", "rotate9", "perspective", "scaled"])
def test_alignment_recovers_known_bubble_coordinates(fixture_dir, name, mode):
    manifest = json.loads((FIXTURES/f"{name}.json").read_text())
    original = render(fixture_dir/f"{name}-scan.pdf")
    scan, injected = distorted(original, mode)
    decoded = decode_sheet(scan)
    ref, _ = bind_sheet(decoded.identity, manifest)
    geometry = page_geometry(manifest, 1)
    result = MarkerAligner().align_decoded(scan, geometry, ref, decoded)
    points = np.array([[b["cx"],b["cy"]] for row in geometry["rows"] for b in row["bubbles"]],np.float32)
    if "studentNumber" in geometry:
        points = np.vstack([points, [[b["cx"],b["cy"]] for c in geometry["studentNumber"]["columns"] for b in c["bubbles"]]])
    raw_points = transform_points(points*(150/72), injected)
    recovered = transform_points(raw_points, np.array(result.pixel_to_page).reshape(3,3))
    error = np.max(np.linalg.norm(points-recovered,axis=1))
    assert error < 14.18/4, f"independent bubble coordinate error {error}"
    assert result.error_pt < 14.18/4


@pytest.mark.parametrize("defect,expected", [("marker", "markers-unreliable"), ("qr", "qr-unreadable"), ("extra-marker", "markers-unreliable")])
def test_unreliable_pages_require_review(fixture_dir, resolver, defect, expected):
    image = render(fixture_dir/"personalized-mixed-text.pdf")
    if defect == "marker":
        image[35:100,35:100] = 255
    elif defect == "qr":
        image[90:310,920:1150] = 255
    else:
        cv2.rectangle(image, (140,50), (173,83), (0,0,0), -1)
    result = prepare_image(image,"synthetic",1,resolver)
    assert result.status == "review" and result.reason == expected
    assert result.page is None


def test_reordered_pages_keep_source_and_layout_numbers_separate(fixture_dir, resolver, tmp_path):
    path = tmp_path/"reordered.pdf"
    with fitz.open(fixture_dir/"personalized-mixed-text.pdf") as source, fitz.open() as target:
        target.insert_pdf(source,from_page=1,to_page=1)
        target.insert_pdf(source,from_page=0,to_page=0)
        target.save(path)
    result = prepare_document(path,resolver)
    assert [p.status for p in result.pages] == ["prepared", "prepared"]
    assert [p.page.alignment.layout_page_number for p in result.pages] == [2,1]
    assert [p.page.source_page_number for p in result.pages] == [1,2]
    assert result.issues == []
    assert all(p.page.student_id == "00530" for p in result.pages)
    assert "correctAnswers" not in json.dumps(result.pages[0].page.geometry)


def test_missing_and_duplicate_pages_are_reported(fixture_dir, resolver, tmp_path):
    path = tmp_path/"duplicates.pdf"
    with fitz.open(fixture_dir/"personalized-mixed-text.pdf") as source, fitz.open() as target:
        for _ in range(2):
            target.insert_pdf(source,from_page=0,to_page=0)
        target.save(path)
    result = prepare_document(path,resolver)
    assert {i["code"] for i in result.issues} == {"missing-pages", "duplicate-page"}
    assert all(p.status == "review" for p in result.pages)
    assert next(i for i in result.issues if i["code"] == "missing-pages")["pages"] == [2]


def test_shared_identity_is_explicitly_pending_and_legacy_path_unchanged(fixture_dir, resolver):
    shared = prepare_document(fixture_dir/"shared-tf-text.pdf", resolver)
    assert shared.pages[0].status == "prepared" and shared.pages[0].page.student_id is None
    assert shared.issues[0]["code"] == "student-identity-pending"
    legacy = prepare_document(fixture_dir/"legacy-control-text.pdf", resolver)
    assert legacy.pages[0].status == "legacy"
    assert legacy.pages[0].reason == "legacy-sheet"


def test_wrong_target_page_is_not_aligned(fixture_dir):
    manifest = json.loads((FIXTURES/"personalized-mixed.json").read_text())
    image = render(fixture_dir/"personalized-mixed-text.pdf")
    decoded = decode_sheet(image)
    ref,_ = bind_sheet(decoded.identity,manifest)
    with pytest.raises(PreparationError) as error:
        MarkerAligner().align_decoded(image,page_geometry(manifest,2),ref,decoded)
    assert error.value.code == "identity-mismatch"


def test_distorted_marker_geometry_fails_independent_residual(fixture_dir,resolver):
    image = render(fixture_dir/"personalized-mixed-text.pdf")
    # Keep the original centre but replace one 16pt marker with a much larger square.
    cv2.rectangle(image,(37,37),(88,88),(0,0,0),-1)
    result = prepare_image(image,"synthetic",1,resolver)
    assert result.status == "review"
    assert result.reason in ("markers-unreliable","alignment-unreliable")
