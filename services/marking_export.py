"""Portable reports from an immutable saved marking run; no re-marking or upload."""
import html
import json
from decimal import Decimal
from pathlib import Path
import tempfile


def matching_label(item):
    return chr(64+item) if isinstance(item,int) and 1<=item<=26 else str(item)


def matching_rows(run,result):
    return sorted([row for saved in run.get('manifests',[])
        if saved['document'].get('layoutId')==result.get('layout_id')
        for page in saved['document'].get('pages',[]) for row in page.get('rows',[])
        if str(row.get('questionId'))==str(result.get('question_id')) and row.get('kind')=='MATCH'],
        key=lambda row:row.get('item',0))


def extracted_answer(run,result):
    rows=matching_rows(run,result)
    parts=[]
    for index,response in enumerate(result.get('responses',[])):
        value=response.get('text') or ', '.join(response.get('selected',[])) or '(blank)'
        prefix=matching_label(rows[index]['item'])+'=' if index<len(rows) else ''
        parts.append(prefix+value)
    return '; '.join(parts)


def correct_answer(run,result):
    rows=matching_rows(run,result)
    if rows:return '; '.join(matching_label(row['item'])+'='+(', '.join(row.get('correctAnswers',[])) or 'Not supplied') for row in rows)
    # Bubble option IDs are the comparable values for MC/MA, not answer prose.
    for saved in run.get('manifests',[]):
        manifest=saved['document']
        if manifest.get('layoutId')!=result.get('layout_id'):continue
        for page in manifest.get('pages',[]):
            for row in page.get('rows',[]):
                if str(row.get('questionId'))==str(result.get('question_id')) and row.get('kind') in ('MC','MA'):
                    return ', '.join(row.get('correctAnswers',[])) or 'Not supplied'
    answers=result.get('reference_answers')
    if answers is None:
        for saved in run.get('manifests',[]):
            manifest=saved['document']
            if manifest.get('layoutId')!=result['layout_id']:continue
            question=next((q for q in manifest.get('questions',[]) if q['questionId']==result['question_id']),{})
            answers=question.get('referenceAnswers',[])
    return '\n'.join(answers or []) or 'Not supplied'


def export_marking_run(run, directory, *, name='Assessment', outdated=False):
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"marking-run-{int(run['sequence'])}-", dir=destination))
    esc = lambda value: html.escape(str(value))
    results = run['results']
    summaries = []
    for total in run['totals']:
        questions = [r for r in results if r['student_id'] == total['student_id']
                     and r['layout_id'] == total['layout_id']]
        maximum = sum((Decimal(r['max_marks']) for r in questions), Decimal(0))
        score = total['score'] if total['score'] is not None else 'Pending'
        summaries.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in
            [total['student_id'], total['layout_id'], score, maximum,
             total['resolved_subtotal'], total['review_count']])+'</tr>')
    details = []
    for result in results:
        responses = extracted_answer(run,result)
        ai = result.get('ai') or {}
        explanation = ai.get('explanation') or (ai.get('proposal') or {}).get('explanation', '')
        if result.get('teacher_decision'):
            explanation='Teacher: '+result['teacher_decision']['reviewer']+'. '+result['teacher_decision'].get('note','')
        if result.get('similarity_estimate') and result['score'] is None:
            estimate=result['similarity_estimate']
            explanation+=f" Similarity suggestion: {estimate['suggested_score']}/{estimate['max_marks']} (not included in totals; teacher review required)."
        details.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in
            [result['student_id'], result['display_number'], correct_answer(run,result), responses,
             result['score'] if result['score'] is not None else 'Pending',
             result['max_marks'], result['reason'], explanation])+'</tr>')
    pending = sum(r['score'] is None for r in results)
    status = 'OUTDATED — responses have changed; re-mark before using these results.' if outdated else (
        'PROVISIONAL — unresolved questions are not zero marks; final totals are pending.' if pending else
        'Completed marking run.')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>{esc(name)} — marking run {run['sequence']}</title>
<style>body{{font:15px system-ui;margin:32px;color:#17212b}}table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{border:1px solid #bbc5ce;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}}th{{background:#eef2f5}}.status{{padding:16px;background:#fff3ce}}thead{{display:table-header-group}}tr{{break-inside:avoid}}@media print{{body{{margin:0;font-size:10px}}}}</style>
<h1>{esc(name)}</h1><h2>Marking run {run['sequence']}</h2>
<p class="status">{esc(status)}</p>
<p>Marking: {run['marking_seconds']:.3f} seconds; AI: {run.get('ai_seconds',0):.3f} seconds.
Extraction and teacher review time are not included.</p>
<p>Unresolved questions: {pending}. Excluded responses: {run.get('excluded_responses',0)}.
Subtotals include only resolved questions. Excluded-only students have no score.</p>
<h2>Student results</h2><table><thead><tr><th>Student ID</th><th>Layout</th><th>Final score</th><th>Maximum represented marks</th><th>Resolved subtotal</th><th>Pending questions</th></tr></thead><tbody>{''.join(summaries)}</tbody></table>
<h2>Question results</h2><table><thead><tr><th>Student ID</th><th>Question</th><th>Correct / reference answer</th><th>Extracted answer (reviewed)</th><th>Score</th><th>Maximum</th><th>Reason</th><th>Marking explanation</th></tr></thead><tbody>{''.join(details)}</tbody></table>
<p>Saved response version: {esc(run['response_digest'])}</p>
<p>This report contains student responses. The accompanying JSON preserves the saved run, including its manifest and marking decisions. No results have been uploaded to Imarisha.</p></html>'''
    report = folder/'results.html'
    audit = folder/'marking-run.json'
    report.write_text(document, encoding='utf-8')
    audit.write_text(json.dumps(dict(assessment_name=name, outdated=outdated, run=run), ensure_ascii=False, indent=2), encoding='utf-8')
    return report, audit
