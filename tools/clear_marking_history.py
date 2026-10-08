"""Explicit marking-only cleanup. Never run automatically at application startup."""
import hashlib
import json
from pathlib import Path
import sqlite3
from database.structured_repository import JobRepository


def fingerprint(path):
    result={}
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        for table in ('structured_jobs','structured_pages','structured_responses','structured_reviews',
                      'structured_transcriptions','structured_sources','structured_crops','structured_issues','structured_issue_reviews'):
            h=hashlib.sha256();count=0
            for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid'):
                count+=1
                for value in row:
                    data=value if isinstance(value,bytes) else repr(value).encode()
                    h.update(len(data).to_bytes(8,'big'));h.update(data)
            result[table]=(count,h.hexdigest())
    return result


def clear(path):
    path=Path(path).resolve()
    before=fingerprint(path)
    # Existing database only: do not initialize or migrate during cleanup.
    repo=object.__new__(JobRepository);repo.path=path
    deleted=repo.clear_marking_data()
    after=fingerprint(path)
    if before!=after:raise RuntimeError('Extraction data changed during cleanup; inspect immediately')
    return dict(deleted=deleted,retained_counts={k:v[0] for k,v in after.items()},extraction_fingerprints_unchanged=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('database');parser.add_argument('--delete-marking-history',action='store_true',required=True)
    args=parser.parse_args();print(json.dumps(clear(args.database),indent=2))
