import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from core.contracts import ReadStatus, ExtractedResponse
from core.structured_extraction import StructuredExtractor
from services.page_preparation import prepare_document, prepare_image
from services.structured_extraction import extract_prepared_document
from tools.manifest_fixtures import FIXTURES
from test_page_alignment import fixture_dir, resolver, render, distorted


@pytest.mark.parametrize('name', ['personalized-mixed', 'shared-tf'])
@pytest.mark.parametrize('form', ['text', 'scan'])
def test_all_clear_answers_match_independent_oracle(fixture_dir, resolver, name, form):
    document = prepare_document(fixture_dir/f'{name}-{form}.pdf', resolver)
    output = extract_prepared_document(document)
    expected = json.loads((FIXTURES/f'{name}-expected.json').read_text())
    actual = [r for p in output.pages.values() for r in p.responses]
    assert len(actual) == len(expected)
    assert not output.issues
    for r, e in zip(actual, expected):
        assert (r.question_id,r.item_number,r.layout_page_number,r.student_id,list(r.selected_options)) == (
            e['question_id'],e['item_number'],e['page'],e['student_id'],e['selected_options'])
        assert r.status == ReadStatus.READ and not r.needs_review
        assert ExtractedResponse.from_json(r.to_json()) == r
        assert all(region.crop_ref in output.pages[r.regions[0].page_number].crops for region in r.regions)


@pytest.mark.parametrize('mode', ['rotate90','rotate180','rotate270','rotate9','perspective','scaled'])
@pytest.mark.parametrize('name', ['personalized-mixed','shared-tf'])
def test_transformed_scans_read_marks(fixture_dir,resolver,mode,name):
    scan,_ = distorted(render(fixture_dir/f'{name}-scan.pdf'),mode)
    prepared = prepare_image(scan,'synthetic',1,resolver)
    assert prepared.status == 'prepared'
    result = StructuredExtractor().extract_page(prepared.page)
    expected = json.loads((FIXTURES/f'{name}-expected.json').read_text())
    assert result.student_id == '00530'
    assert [list(r.selected_options) for r in result.responses] == [r['selected_options'] for r in expected if r['page']==1]
    assert all(not r.needs_review for r in result.responses)


def edited_page(fixture_dir,resolver,kind='personalized-mixed'):
    return prepare_document(fixture_dir/f'{kind}-text.pdf', resolver).pages[0].page


def paint(page, bubbles, values):
    image = cv2.imdecode(np.frombuffer(page.image_png,np.uint8), cv2.IMREAD_COLOR)
    inverse = np.linalg.inv(np.array(page.alignment.pixel_to_page).reshape(3,3))
    for bubble, value in zip(bubbles,values):
        point = cv2.perspectiveTransform(np.array([[[bubble['cx'],bubble['cy']]]],np.float32), inverse)[0,0]
        cv2.circle(image,tuple(np.round(point).astype(int)),round(bubble['diameter']/2*150/72*.8),(value,)*3,-1)
    return replace(page,image_png=cv2.imencode('.png',image)[1].tobytes())


@pytest.mark.parametrize('row_index', [0,1,2])
@pytest.mark.parametrize('case', ['blank','faint','erased','double'])
def test_blank_faint_erased_and_double_are_distinct(fixture_dir,resolver,row_index,case):
    page = edited_page(fixture_dir,resolver)
    row = page.geometry['rows'][row_index]
    values = [255]*len(row['bubbles'])
    if case=='faint': values[0]=150
    if case=='erased': values[0]=215
    if case=='double': values[:2]=[0,0]
    response = StructuredExtractor().extract(paint(page,row['bubbles'],values))[row_index]
    if case=='blank':
        assert response.status == ReadStatus.BLANK and not response.selected_options
    elif case=='double' and row['kind']=='MA':
        assert response.status == ReadStatus.READ and len(response.selected_options)==2
    else:
        assert response.status == ReadStatus.UNCERTAIN and response.needs_review
        assert ('multiple-selections' if case=='double' else 'ambiguous-mark') in response.review_reasons


@pytest.mark.parametrize('case',['blank','faint','double'])
def test_incomplete_student_column_never_guesses_identity(fixture_dir,resolver,case):
    page = edited_page(fixture_dir,resolver,'shared-tf')
    bubbles = page.geometry['studentNumber']['columns'][0]['bubbles']
    values = [255]*10
    if case=='faint': values[0]=150
    if case=='double': values[:2]=[0,0]
    result = StructuredExtractor().extract_page(paint(page,bubbles,values))
    assert result.student_id is None and result.identity_reasons
    assert all(r.student_id is None and r.needs_review for r in result.responses)


def test_conflicting_identity_requires_review(fixture_dir,resolver):
    page = replace(edited_page(fixture_dir,resolver,'shared-tf'), student_id='00531')
    result = StructuredExtractor().extract_page(page)
    assert result.student_id is None
    assert 'student-identity-conflict' in result.identity_reasons


def test_layout_order_and_unlisted_marks_do_not_change_reading(fixture_dir,resolver):
    page = edited_page(fixture_dir,resolver)
    geometry = dict(page.geometry,rows=list(reversed(page.geometry['rows'])))
    page = paint(replace(page,geometry=geometry),[dict(cx=310,cy=180,diameter=14.18)],[0])
    result = StructuredExtractor().extract(page)
    assert [r.question_id for r in result] == ['10003','10002','10001']
    assert result[-1].selected_options == ('B',)


def test_shared_duplicates_flag_all_copies(fixture_dir,resolver):
    document = prepare_document(fixture_dir/'shared-tf-text.pdf',resolver)
    second = replace(document.pages[0], source_page_number=2, page=replace(document.pages[0].page,source_page_number=2))
    document.pages.append(second)
    result = extract_prepared_document(document)
    assert any(i['code']=='duplicate-page' for i in result.issues)
    assert all('duplicate-page' in r.review_reasons for p in result.pages.values() for r in p.responses)


def test_preparation_review_pages_are_not_extracted(fixture_dir,resolver):
    document = prepare_document(fixture_dir/'personalized-mixed-text.pdf',resolver)
    document.pages[0].status='review'
    result = extract_prepared_document(document)
    assert list(result.pages)==[2]
    assert result.issues[0]['code']=='page-not-extracted'


@pytest.mark.parametrize('kind', ['MC','MA','TF','MATCH'])
def test_homogeneous_layouts_and_matching_invalid_selection(fixture_dir,resolver,kind):
    document=prepare_document(fixture_dir/'personalized-mixed-text.pdf',resolver)
    page=document.pages[1 if kind=='MATCH' else 0].page
    rows=[r for r in page.geometry['rows'] if r['kind']==kind]
    page=replace(page,geometry=dict(page.geometry,rows=rows))
    result=StructuredExtractor().extract(page)
    assert result and all(r.response_type.value==kind for r in result)
    if kind=='MATCH':
        changed=paint(page,rows[0]['bubbles'],[0,0,255,255])
        response=StructuredExtractor().extract(changed)[0]
        assert response.status==ReadStatus.UNCERTAIN
        assert response.item_number==1 and response.selected_options==('A','B')


def test_solid_selection_plus_erasure_stays_uncertain(fixture_dir,resolver):
    page=edited_page(fixture_dir,resolver)
    response=StructuredExtractor().extract(paint(page,page.geometry['rows'][0]['bubbles'],[0,215,255,255]))[0]
    assert response.selected_options==('A',)
    assert response.status==ReadStatus.UNCERTAIN


def test_shared_different_students_are_not_duplicates(fixture_dir,resolver):
    document=prepare_document(fixture_dir/'shared-tf-text.pdf',resolver)
    original=document.pages[0]
    page=original.page
    bubbles=page.geometry['studentNumber']['columns'][-1]['bubbles']
    second=paint(page,bubbles,[255,0]+[255]*8)
    document.pages.append(replace(original,source_page_number=2,page=replace(second,source_page_number=2)))
    result=extract_prepared_document(document)
    assert not result.issues
    assert [p.student_id for p in result.pages.values()]==['00530','00531']


def test_shared_missing_page_is_reported_after_identity_read(fixture_dir,resolver):
    document=prepare_document(fixture_dir/'shared-tf-text.pdf',resolver)
    original=document.pages[0].page
    document.pages[0].page=replace(original,geometry=dict(original.geometry,layoutPageCount=2))
    result=extract_prepared_document(document)
    assert result.issues==[dict(code='missing-pages',layout_id=original.alignment.layout.layout_id,student_id='00530',pages=[2])]


def test_corrupt_source_is_review_issue_not_batch_crash(fixture_dir,resolver):
    document=prepare_document(fixture_dir/'personalized-mixed-text.pdf',resolver)
    document.pages[0].page=replace(document.pages[0].page,image_png=b'bad')
    result=extract_prepared_document(document)
    assert list(result.pages)==[2]
    assert result.issues[0]['code']=='invalid-image'


def test_answers_cannot_influence_extraction(fixture_dir,resolver):
    page=edited_page(fixture_dir,resolver)
    baseline=StructuredExtractor().extract(page)
    polluted=replace(page,geometry=dict(page.geometry,correctAnswers=['D'],scoringPolicy='award all'))
    assert StructuredExtractor().extract(polluted)==baseline


def test_rows_can_move_to_other_column_and_bottom_of_page(fixture_dir,resolver):
    page=edited_page(fixture_dir,resolver)
    row=dict(page.geometry['rows'][0])
    row['bubbles']=[dict(b,cx=b['cx']+250,cy=750) for b in row['bubbles']]
    shifted=replace(page,geometry=dict(page.geometry,rows=[row]))
    shifted=paint(shifted,row['bubbles'],[255,255,0,255])
    result=StructuredExtractor().extract(shifted)[0]
    assert result.selected_options==('C',) and not result.needs_review
    assert all(r.bbox[1]>740 for r in result.regions)
