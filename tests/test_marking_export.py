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


def test_reference_and_extracted_answers_are_adjacent_and_escaped(tmp_path):
    run=dict(sequence=1,marking_seconds=0,response_digest='abc',results=[dict(student_id='00530',question_id='q',
        layout_id='layout',display_number=20,score='2',max_marks='5',reason='teacher-mark',
        responses=[dict(text='Student <answer>')],teacher_decision=dict(reviewer='Teacher',note='Checked'))],
        totals=[],manifests=[dict(document=dict(layoutId='layout',questions=[dict(questionId='q',
        referenceAnswers=['Reference <answer>'])]))])
    report,_=export_marking_run(run,tmp_path)
    text=report.read_text()
    assert '<td>Reference &lt;answer&gt;</td><td>Student &lt;answer&gt;</td>' in text
    assert 'Teacher: Teacher. Checked' in text


def test_bubble_correct_answers_use_letters_instead_of_reference_prose():
    from services.marking_export import correct_answer
    for kind,letters in [('MC',['D']),('MA',['A','C'])]:
        run={'manifests':[{'document':{'layoutId':'layout','pages':[{'rows':[
            {'questionId':'q','kind':kind,'correctAnswers':letters}]}]}}]}
        result={'layout_id':'layout','question_id':'q','reference_answers':['Long answer text']}
        assert correct_answer(run,result)==', '.join(letters)


def test_matching_pairs_preserve_original_positions():
    from services.marking_export import correct_answer,extracted_answer
    run={'manifests':[{'document':{'layoutId':'layout','pages':[{'rows':[
        {'questionId':'q','kind':'MATCH','item':3,'correctAnswers':['D']},
        {'questionId':'q','kind':'MATCH','item':1,'correctAnswers':['B']}]}]}}]}
    result={'layout_id':'layout','question_id':'q','responses':[{'selected':['B']},{'selected':['A']}]}
    assert correct_answer(run,result)=='A=B; C=D'
    assert extracted_answer(run,result)=='A=B; C=A'
