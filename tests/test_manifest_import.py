import copy
import json
import pytest
from core.manifest_layout import PreparationError
from services.manifest_resolver import ImportedManifestResolver
from test_manifest_preparation import manifest


def test_file_import_roundtrip_conflict_and_old_job_scope(tmp_path,manifest):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(format='imarisha-automark-manifest',packageVersion=1,manifest=manifest)))
    resolver=ImportedManifestResolver(tmp_path/'cache.db')
    resolver.import_file(path)
    again=ImportedManifestResolver(tmp_path/'cache.db','previous-job-scope')
    assert again.scope=='previous-job-scope'
    assert again.resolve(manifest['layoutId'],'EXAM','42')==manifest
    with pytest.raises(PreparationError,match='Import the manifest'):
        again.resolve('wrong-layout','EXAM','42')
    manifest['questions'][0]['questionText']='Changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreparationError,match='immutable'):
        resolver.import_file(path)


def test_bad_packages_fail_without_network(tmp_path,manifest):
    path=tmp_path/'manifest.json';resolver=ImportedManifestResolver(tmp_path/'cache.db')
    for data in ['not json',json.dumps(dict(manifest,answerSchemaVersion=99)),json.dumps(dict(format='imarisha-automark-manifest',packageVersion=99,manifest=manifest))]:
        path.write_text(data)
        with pytest.raises((PreparationError,ValueError)):resolver.import_file(path)


def test_imported_manifest_extracts_pdf_without_backend(tmp_path,fixture_dir):
    from tools.manifest_fixtures import FIXTURES
    from database.structured_repository import JobRepository
    from services.extraction_jobs import run_job
    resolver=ImportedManifestResolver(tmp_path/'cache.db')
    resolver.import_file(FIXTURES/'personalized-mixed.json')
    repo=JobRepository(tmp_path/'jobs.db')
    job=repo.create(fixture_dir/'personalized-mixed-text.pdf',resolver.scope)
    run_job(repo,job,resolver)
    assert repo.responses(job)
    assert all(p['status']=='extracted' for p in repo.job(job)['pages'])


from test_page_alignment import fixture_dir
