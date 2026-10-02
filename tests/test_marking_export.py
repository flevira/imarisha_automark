import json
from services.marking_export import export_marking_run


def test_report_preserves_pending_zero_ids_and_escapes_student_text(tmp_path):
    run=dict(sequence=3, marking_seconds=1.2, ai_seconds=1, response_digest='abc',
        totals=[dict(student_id='00531',layout_id='layout',score=None,resolved_subtotal='0',review_count=1)],
        results=[dict(student_id='00531',layout_id='layout',display_number=1,score=None,max_marks='5',
                      reason='ai-unresolved',responses=[dict(text='<script>bad</script>')])])
    report,audit=export_marking_run(run,tmp_path,name='<Exam>',outdated=True)
    text=report.read_text()
    assert '00531' in text and 'Pending' in text and 'OUTDATED' in text
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert json.loads(audit.read_text())['run']==run
    second,_=export_marking_run(run,tmp_path)
    assert second!=report and report.exists()
