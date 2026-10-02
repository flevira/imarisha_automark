"""Portable reports from an immutable saved marking run; no re-marking or upload."""
import html
import json
from decimal import Decimal
from pathlib import Path
import tempfile


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
        responses = '; '.join(response.get('text') or ', '.join(response.get('selected', [])) or '(blank)'
                              for response in result.get('responses', []))
        ai = result.get('ai') or {}
        explanation = (ai.get('proposal') or {}).get('explanation', '')
        if result.get('similarity_estimate') and result['score'] is None:
            estimate=result['similarity_estimate']
            explanation+=f" Similarity suggestion: {estimate['suggested_score']}/{estimate['max_marks']} (not included in totals; teacher review required)."
        details.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in
            [result['student_id'], result['display_number'], responses,
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
<h2>Question results</h2><table><thead><tr><th>Student ID</th><th>Question</th><th>Reviewed response</th><th>Score</th><th>Maximum</th><th>Reason</th><th>AI explanation</th></tr></thead><tbody>{''.join(details)}</tbody></table>
<p>Saved response version: {esc(run['response_digest'])}</p>
<p>This report contains student responses. The accompanying JSON preserves the saved run, including its manifest and marking decisions. No results have been uploaded to Imarisha.</p></html>'''
    report = folder/'results.html'
    audit = folder/'marking-run.json'
    report.write_text(document, encoding='utf-8')
    audit.write_text(json.dumps(dict(assessment_name=name, outdated=outdated, run=run), ensure_ascii=False, indent=2), encoding='utf-8')
    return report, audit
