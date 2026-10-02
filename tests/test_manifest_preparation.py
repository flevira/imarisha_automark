import copy
import json
from pathlib import Path
import sqlite3

import pytest

from core.manifest_layout import LegacySheet, PreparationError, bind_sheet, parse_sheet_qr, validate_manifest
from services.manifest_resolver import CachedManifestResolver, TeacherManifestClient
from tools.manifest_fixtures import FIXTURES, qr_payload


@pytest.fixture
def manifest():
    return json.loads((FIXTURES/"personalized-mixed.json").read_text())


def test_qr_preserves_student_and_page_identity(manifest):
    identity = parse_sheet_qr(qr_payload(manifest, 2, False))
    assert identity.student_id == "00530" and identity.page_number == 2
    ref, page = bind_sheet(identity, manifest)
    assert ref.assessment_group_id == "40" and page["pageNumber"] == 2


@pytest.mark.parametrize("change", [
    ";page=1", ";testId=42", ";assessmentId=99", ";studentId=530",
])
def test_qr_conflicts_are_not_guessed(manifest, change):
    with pytest.raises(PreparationError):
        parse_sheet_qr(qr_payload(manifest, 1, False)+change)


@pytest.mark.parametrize("old,new,code", [
    ("v=4", "v=8", "unsupported-version"), ("page=1", "page=0", "invalid-qr"),
    ("studentId=00530", "studentId=abc", "invalid-qr"), ("EXAM", "OTHER", "invalid-qr"),
    ("synthetic-mixed-v4", "../unsafe", "invalid-qr"), (";studentId=00530", "", "invalid-qr"),
])
def test_invalid_qr_is_explicit(manifest, old, new, code):
    with pytest.raises(PreparationError) as error:
        parse_sheet_qr(qr_payload(manifest, 1, False).replace(old, new))
    assert error.value.code == code


def test_legacy_qr_signals_legacy_without_rewriting_old_parser():
    with pytest.raises(LegacySheet):
        parse_sheet_qr("type=EXAM;examId=42;studentId=00530")


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(version=6), lambda m: m.update(unit="pixels"),
    lambda m: m["pages"][0].update(width=900), lambda m: m["pages"][0].update(markers=[]),
    lambda m: m["pages"][0]["rows"][0]["bubbles"][0].update(cx=-1),
    lambda m: m["pages"][0]["rows"][0]["bubbles"][0].update(diameter=float("nan")),
    lambda m: m["pages"].append(copy.deepcopy(m["pages"][0])),
    lambda m: m["pages"][0]["rows"][0].update(selectionMode="multiple"),
    lambda m: m["pages"][0]["rows"][0].update(kind={}),
])
def test_invalid_manifest_never_enters_cache(manifest, tmp_path, mutate):
    mutate(manifest)
    resolver = CachedManifestResolver(tmp_path/"cache.db", "school|teacher", lambda *a: manifest)
    with pytest.raises(PreparationError):
        resolver.resolve("synthetic-mixed-v4", "EXAM", "42")
    with sqlite3.connect(tmp_path/"cache.db") as db:
        assert db.execute("SELECT COUNT(*) FROM layout_cache").fetchone()[0] == 0


def test_wrong_qr_version_and_missing_page_fail_binding(manifest):
    identity = parse_sheet_qr(qr_payload(manifest, 3, False))
    with pytest.raises(PreparationError, match="page"):
        bind_sheet(identity, manifest)
    identity = parse_sheet_qr(qr_payload(manifest, 1, False).replace("v=4", "v=5").replace(";studentId=00530", ""))
    with pytest.raises(PreparationError, match="version"):
        bind_sheet(identity, manifest)


def test_fetch_once_then_work_offline_and_isolate_accounts(manifest, tmp_path):
    calls = []
    def fetch(*identity):
        calls.append(identity)
        return manifest
    path = tmp_path/"cache.db"
    online = CachedManifestResolver(path, "backend|school|teacher1", fetch)
    assert online.resolve("synthetic-mixed-v4", "EXAM", "42") == manifest
    offline = CachedManifestResolver(path, "backend|school|teacher1")
    assert offline.resolve("synthetic-mixed-v4", "EXAM", "42") == manifest
    assert len(calls) == 1
    for scope in ("backend|school|teacher2", "another-backend|school|teacher1"):
        with pytest.raises(PreparationError) as error:
            CachedManifestResolver(path, scope).resolve("synthetic-mixed-v4", "EXAM", "42")
        assert error.value.code == "layout-unavailable"


def test_refresh_denial_does_not_silently_use_old_cache(manifest, tmp_path):
    path = tmp_path/"cache.db"
    resolver = CachedManifestResolver(path, "teacher", lambda *a: manifest)
    resolver.resolve("synthetic-mixed-v4", "EXAM", "42")
    def denied(*a):
        raise PreparationError("access-denied", "Denied")
    with pytest.raises(PreparationError) as error:
        CachedManifestResolver(path, "teacher", denied, refresh=True).resolve("synthetic-mixed-v4", "EXAM", "42")
    assert error.value.code == "access-denied"


def test_authenticated_alias_is_cached_under_requested_reference(manifest, tmp_path):
    resolver = CachedManifestResolver(tmp_path/"cache.db", "teacher", lambda *a: manifest)
    assert resolver.resolve("old-printed-layout", "EXAM", "43") == manifest
    resolver.fetch = None
    assert resolver.resolve("old-printed-layout", "EXAM", "43") == manifest


def test_conflicting_content_or_corrupt_cache_is_rejected(manifest, tmp_path):
    path = tmp_path/"cache.db"
    resolver = CachedManifestResolver(path, "teacher", lambda *a: manifest)
    resolver.resolve("synthetic-mixed-v4", "EXAM", "42")
    manifest["pages"][0]["rows"][0]["correctAnswers"] = ["D"]
    resolver.refresh = True
    with pytest.raises(PreparationError) as error:
        resolver.resolve("synthetic-mixed-v4", "EXAM", "42")
    assert error.value.code == "layout-conflict"
    with sqlite3.connect(path) as db:
        db.execute("UPDATE layout_cache SET document='{}'")
    resolver.refresh = False
    with pytest.raises(PreparationError) as error:
        resolver.resolve("synthetic-mixed-v4", "EXAM", "42")
    assert error.value.code == "cache-corrupt"


def test_fixture_import_cannot_populate_production_scope(manifest, tmp_path):
    with pytest.raises(PreparationError):
        CachedManifestResolver(tmp_path/"cache.db", "teacher").seed_fixture(manifest)


@pytest.mark.parametrize("url", ["http://school.example", "https://user:password@school.example", "https://school.example?token=x", "file:///tmp"])
def test_client_rejects_unsafe_token_destination(url):
    with pytest.raises(PreparationError):
        TeacherManifestClient(url, "fake-token")


def test_contracts_and_manifest_lookup_do_not_import_legacy_ocr():
    import subprocess
    import sys
    result = subprocess.run([sys.executable,"-c",
        "import sys; import core.contracts; import services.manifest_resolver; assert 'core.extractor' not in sys.modules; assert 'pytesseract' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_legacy_public_entrypoint_still_resolves():
    from core import extract_file
    from core.extractor import extract_file as legacy
    assert extract_file is legacy
