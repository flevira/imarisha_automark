"""Run M2 separately from the unchanged legacy extractor.

Offline demonstration:
python -m tools.prepare_manifest_pages artifacts/m1-fixtures/personalized-mixed-scan.pdf --fixture-manifests tests/fixtures/manifest --output artifacts/m2-demo

Live backend: omit --fixture-manifests, pass --backend URL --account STAFF_ID.
Set AUTOMARK_API_TOKEN in the environment for online cache misses (never on CLI).
Use --refresh to reauthorize even cached entries; omit the token for offline use.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from core.manifest_layout import PreparationError
from core.page_alignment import normalized_preview
from services.manifest_resolver import CachedManifestResolver, TeacherManifestClient
from services.page_preparation import prepare_document
from services.structured_extraction import extract_prepared_document


def cache_location() -> Path:
    if sys.platform == "darwin":
        root = Path.home()/"Library/Application Support/Imarisha AutoMark"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()/"AppData/Local")))/"Imarisha AutoMark"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home()/".local/share")))/"imarisha-automark"
    return root/"manifest-cache.db"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--fixture-manifests", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--account")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/m2-preparation"))
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--extract", action="store_true", help="Read structured bubbles and student grids after preparation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fixture_manifests:
        if args.backend or args.account or args.refresh:
            parser.error("Fixture mode is separate from authenticated production caches")
        resolver = CachedManifestResolver(args.cache or args.output/"fixture-cache.db", "synthetic-fixtures-only")
        for file in args.fixture_manifests.glob("*.json"):
            manifest = json.loads(file.read_text())
            if isinstance(manifest, dict) and "layoutId" in manifest and "pages" in manifest:
                resolver.seed_fixture(manifest)
    else:
        if not args.backend or not args.account:
            parser.error("Provide --backend and --account, or use --fixture-manifests for a synthetic demo")
        token = os.environ.get("AUTOMARK_API_TOKEN")
        client = TeacherManifestClient(args.backend, token) if token else None
        resolver = CachedManifestResolver(args.cache or cache_location(), args.backend.rstrip("/")+"|"+args.account, client, refresh=args.refresh)
    result = prepare_document(args.document, resolver)
    (args.output/"preparation.json").write_text(json.dumps(result.summary(), indent=2)+"\n")
    extracted = None
    if args.extract:
        extracted = extract_prepared_document(result)
        crop_dir = args.output/"crops"
        crop_dir.mkdir(exist_ok=True)
        pages = []
        for source_page, page in extracted.pages.items():
            for ref, data in page.crops.items():
                (crop_dir/(ref.removeprefix('sha256:')+'.png')).write_bytes(data)
            pages.append(dict(source_page=source_page, student_id=page.student_id,
                identity_reasons=page.identity_reasons,
                responses=[json.loads(r.to_json()) for r in page.responses],
                evidence={key:[asdict(b) for b in bubbles] for key,bubbles in page.evidence.items()}))
        summary = dict(pages=pages, issues=extracted.issues,
            preparation_seconds=result.total_seconds, extraction_seconds=extracted.extraction_seconds,
            marking_seconds=None, review_seconds=None,
            crop_reference_format="sha256:<digest> resolves to crops/<digest>.png",
            scope="Structured extraction only; no marking or handwriting")
        (args.output/"extraction.json").write_text(json.dumps(summary, indent=2)+"\n")
    if args.previews:
        for item in result.pages:
            if item.page:
                p = item.page
                image = cv2.imdecode(np.frombuffer(p.image_png, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(str(args.output/f"page-{item.source_page_number}.png"), normalized_preview(image, p.geometry, p.alignment))
    print(json.dumps(result.summary(), indent=2))
    if extracted is not None:
        print(f"Structured extraction: {extracted.extraction_seconds:.3f}s; see extraction.json")
        return 0 if not extracted.issues and all(not r.needs_review for p in extracted.pages.values() for r in p.responses) else 1
    return 0 if all(p.status == "prepared" for p in result.pages) and not any(i["code"] != "student-identity-pending" for i in result.issues) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreparationError, OSError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
