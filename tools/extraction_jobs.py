"""Create/resume a durable M4 PDF job. Review saved jobs in the desktop app."""
import argparse
import json
import os
from pathlib import Path

from database.repository import JobRepository
from services.extraction_jobs import run_job
from services.manifest_resolver import CachedManifestResolver, ImportedManifestResolver
from tools.prepare_manifest_pages import cache_location


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document',type=Path,nargs='?')
    parser.add_argument('--job',help='Resume a saved job instead of uploading again')
    parser.add_argument('--database',type=Path,default=cache_location().parent/'structured-jobs.db')
    parser.add_argument('--fixture-manifests',type=Path)
    parser.add_argument('--manifest',type=Path,help='Import a downloaded manifest JSON')
    parser.add_argument('--handwriting-model',help='Opt-in local vision model; output always requires review')
    parser.add_argument('--language',choices=['en','sw','en-sw'],default='en-sw')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if bool(args.document)==bool(args.job):
        parser.error('Provide a PDF path or --job ID')
    if args.fixture_manifests:
        resolver=CachedManifestResolver(args.database.parent/'fixture-cache.db','synthetic-fixtures-only')
        for name in args.fixture_manifests.glob('*.json'):
            value=json.loads(name.read_text())
            if isinstance(value,dict) and 'layoutId' in value and 'pages' in value:
                resolver.seed_fixture(value)
    else:
        resolver=ImportedManifestResolver(cache_location(),JobRepository(args.database).job(args.job)['scope'] if args.job else None)
        if args.manifest:resolver.import_file(args.manifest)
    repository=JobRepository(args.database)
    job=args.job or repository.create(args.document,resolver.scope)
    from core.handwriting import OllamaHandwritingRecognizer
    recognizer=OllamaHandwritingRecognizer(args.handwriting_model) if args.handwriting_model else None
    run_job(repository,job,resolver,retry_failed=args.retry_failed,recognizer=recognizer,language=args.language)
    info=repository.job(job)
    print(json.dumps(dict(job=job,status=info['status'],pages=info['total'],
        responses=len(repository.responses(job)),issues=repository.issues(job),
        preparation_seconds=sum(p['preparation_seconds'] for p in info['pages']),
        extraction_seconds=sum(p['extraction_seconds'] for p in info['pages']),
        recognition_seconds=sum(r['transcription']['seconds'] for r in repository.responses(job) if r['transcription']),
        marking_seconds=None,review_seconds=None),indent=2))
    return 1 if any(p['status']=='failed' for p in info['pages']) else 0


if __name__=='__main__':
    raise SystemExit(main())
