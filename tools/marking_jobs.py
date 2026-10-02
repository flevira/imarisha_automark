"""Mark/re-mark a completed saved job, without scans or recognition."""
import argparse
import json
import os
from pathlib import Path

from database.structured_repository import JobRepository
from services.manifest_resolver import CachedManifestResolver, ImportedManifestResolver
from services.marking_jobs import mark_job
from tools.prepare_manifest_pages import cache_location


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--reviewer',required=True)
    parser.add_argument('--database',type=Path,default=cache_location().parent/'structured-jobs.db')
    parser.add_argument('--manifest',type=Path,help='Import a downloaded manifest JSON')
    parser.add_argument('--fixture-manifests',type=Path)
    parser.add_argument('--ai-model',help='Installed local Ollama model for ambiguous written answers')
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--structured-policy',choices=['all-or-nothing','partial'])
    choice.add_argument('--policy',type=Path,help='Explicit teacher rules JSON; retained in the marking run')
    args=parser.parse_args()
    if args.fixture_manifests:
        resolver=CachedManifestResolver(args.database.parent/'fixture-cache.db','synthetic-fixtures-only')
        for path in args.fixture_manifests.glob('*.json'):
            m=json.loads(path.read_text())
            if isinstance(m,dict) and 'layoutId' in m and 'pages' in m:resolver.seed_fixture(m)
    else:
        resolver=ImportedManifestResolver(cache_location(),JobRepository(args.database).job(args.job)['scope'])
        if args.manifest:resolver.import_file(args.manifest)
    policy=json.loads(args.policy.read_text()) if args.policy else (
        dict(version=1,structured=args.structured_policy) if args.structured_policy else None)
    from core.ai_marking import LocalMarkingAssistant
    result=mark_job(JobRepository(args.database),args.job,resolver,args.reviewer,policy,
                    LocalMarkingAssistant(args.ai_model) if args.ai_model else None)
    print(json.dumps({k:result[k] for k in ('sequence','totals','marking_seconds','review_seconds')},indent=2))


if __name__=='__main__':main()
