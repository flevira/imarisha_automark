"""Saved structured extraction jobs and explicit teacher review (Flet desktop)."""
import asyncio
import base64
import json
import threading
import time
import sys
from pathlib import Path

import flet as ft

from database.structured_repository import JobRepository
from database.db import DB_PATH as LEGACY_DB_PATH
from services.job_report import export_job_report, export_batch_report
from services.marking_export import correct_answer
from services.extraction_jobs import run_job
from services.manifest_resolver import ImportedManifestResolver
from services.marking_jobs import mark_job, digest, response_snapshot, apply_teacher_mark
from tools.prepare_manifest_pages import cache_location
from app.controls.pdf_picker import choose_macos_pdf, choose_macos_manifest


def build_structured_view(page: ft.Page, repository=None):
    repository = repository or JobRepository(cache_location().parent/'structured-jobs.db')
    page.controls.clear()
    page.scroll=ft.ScrollMode.AUTO
    manifest_path=ft.TextField(label='Downloaded manifest file')
    handwriting=ft.Checkbox(label='Trial local handwriting transcription (all drafts need review)',value=False)
    model=ft.TextField(label='Installed local vision model',value='gemma3:4b-it-q4_K_M',width=350)
    language=ft.Dropdown(label='OCR / handwriting language',value='en-sw',width=220,options=[ft.dropdown.Option('en','English'),ft.dropdown.Option('sw','Kiswahili'),ft.dropdown.Option('en-sw','English and Kiswahili')])
    recognition_workers=ft.Dropdown(label='AI transcription pages at once',value='1',width=300,
        options=[ft.dropdown.Option('1','1 — sequential'),ft.dropdown.Option('2','2 — parallel trial')])
    reviewer=ft.TextField(label='Reviewer name',width=240)
    marking_policy=ft.Dropdown(label='Structured marking policy',value='manifest',width=450,options=[
        ft.dropdown.Option('manifest','Manifest answers with partial credit'),
        ft.dropdown.Option('all-or-nothing','All-or-nothing per question (including matching)')])
    ai_marking=ft.Checkbox(label='Use local AI for equivalent written answers and rubric-based partial credit',value=True)
    marking_model=ft.TextField(label='Local marking model',value='qwen3:4b-instruct-2507-q4_K_M',width=450)
    written_context={}
    source=ft.TextField(label='Answer-sheet PDF or image path',expand=True,multiline=True,
        helper_text='Select multiple files, or enter one path per line. Each file is saved as a separate job.')
    message=ft.Text('',selectable=True)
    job_message=ft.Text('',selectable=True)
    progress=ft.Text('')
    jobs_column=ft.Column()
    selected_jobs=set()
    batch_message=ft.Text('',selectable=True)
    batch_directory=ft.TextField(label='Batch results folder',value=str(Path.home()/'Downloads'))
    details=ft.Column()
    cancellation=threading.Event()
    running=False
    active_job=None
    review_page=1

    notification_text=ft.Text('')
    notification=ft.SnackBar(content=notification_text,duration=8000)
    page.overlay.append(notification)

    def notify(text):
        message.value=text
        job_message.value=text
        notification_text.value=text
        notification.open=True
        page.update()

    def resolver(job=None):
        return ImportedManifestResolver(cache_location(),repository.job(job)['scope'] if job else None)

    def import_manifest(path):
        try:
            manifest=resolver().import_file(path)
            manifest_path.value=str(path)
            notify(f"Manifest imported: {manifest['assessmentType']} {manifest['assessmentId']} · {len(manifest['pages'])} pages. Available offline for all matching student sheets.")
        except Exception as exc:
            notify(f'Manifest import failed: {exc}')

    async def choose_manifest():
        if sys.platform=='darwin':
            try:
                selected=await asyncio.to_thread(choose_macos_manifest)
                if selected:import_manifest(selected)
            except Exception as exc:notify(str(exc))
        else:
            manifest_picker.pick_files(dialog_title='Import AutoMark manifest',allow_multiple=False,
                allowed_extensions=['json'],file_type=ft.FilePickerFileType.CUSTOM)

    def select_job(job,checked):
        if checked:selected_jobs.add(job)
        else:selected_jobs.discard(job)

    def select_all_jobs(_):
        selected_jobs.update(j['id'] for j in repository.jobs())
        refresh_jobs();page.update()

    def refresh_jobs():
        jobs=repository.jobs()
        selected_jobs.intersection_update(j['id'] for j in jobs)
        jobs_column.controls=[ft.Row([ft.Checkbox(value=j['id'] in selected_jobs,disabled=running,
            on_change=lambda e,job=j['id']:select_job(job,e.control.value)),
            ft.OutlinedButton(f"{j['name']} — {j['status']} ({j['total']} pages)",
            on_click=lambda e,job=j['id']: show_job(job), disabled=running)]) for j in jobs]

    def apply_review(job,row,student,choices,note,action,written=None,feedback=None):
        try:
            reviewer.error_text=None
            note.error_text=None
            if not (reviewer.value or '').strip():
                reviewer.error_text='Required to save a review decision'
                raise ValueError('Approval not saved: enter Reviewer name at the top of this screen, then click Approve again.')
            repository.review(job,row['id'],action=action,student=student.value.strip(),
                selected=[c.data for c in choices if c.value],reviewer=reviewer.value,note=note.value or '',
                expected_sequence=row['review']['sequence'] if row['review'] else None,
                raw_text=written.value if written is not None else None)
            show_job(job)
            notify('Decision saved. Original extraction retained.')
        except Exception as exc:
            if feedback is not None:
                feedback.value=str(exc)
            notify(str(exc))

    def resolve(job,issue,note):
        try:
            repository.resolve_issue(job,issue['key'],reviewer.value,note.value)
            show_job(job)
        except Exception as exc:
            notify(str(exc))

    def complete(job):
        try:
            repository.complete(job)
            show_job(job)
            notify('Review complete. Mark / re-mark reviewed answers is now enabled. Click it to calculate scores.')
        except Exception as exc:
            notify(str(exc))

    def approve_all(job):
        try:
            result=repository.approve_all(job,reviewer.value)
            show_job(job)
            skipped=result['skipped']
            message=f"Approved {result['approved']} responses across the whole job; kept {result['retained']} previous decisions."
            if skipped:
                message+=f" {len(skipped)} responses still need attention on pages "+', '.join(map(str,sorted({r['page'] for r in skipped})))+'. Missing handwriting text is not treated as blank.'
            else:
                message+=' Click Complete review to enable marking.'
            notify(message)
        except Exception as exc:
            notify(str(exc))

    def export_results(job, run, directory):
        try:
            report=export_job_report(repository,job,directory.value,resolver(job))
            notify(f'Extraction and marking report saved: {report}. Export again after marking to update this same file.')
        except Exception as exc:
            notify(f'Export failed: {exc}')

    def request_delete(job=None):
        if running:
            notify('Wait for the current operation to finish before deleting saved data.')
            return
        name=repository.job(job)['name'] if job else 'ALL internal extraction data, imported manifests and legacy records'
        confirmation=ft.Checkbox(label='I understand these saved results will be permanently deleted',value=False)
        feedback=ft.Text('')
        def remove(_):
            nonlocal active_job,review_page
            if not confirmation.value:
                feedback.value='Tick the confirmation box to continue.';page.update();return
            try:
                result=repository.delete_saved_data(job,manifest_path=cache_location(),legacy_path=LEGACY_DB_PATH)
                if job is None or active_job==job:
                    active_job=None;review_page=1;details.controls.clear();progress.value='';job_message.value=''
                if job is None:manifest_path.value='';source.value=''
                for key in list(written_context):
                    if job is None or key[0]==job:del written_context[key]
                page.close(dialog)
                refresh_jobs()
                notify(f"Deleted {result['deleted_jobs']} saved job(s). "+
                       ('Imported manifests and legacy extraction data cleared. Original files and exported reports retained.' if job is None else 'Imported manifests and original files retained.')+
                       ('' if result['compacted'] else ' Disk space will be reclaimed later.'))
            except Exception as exc:
                feedback.value=str(exc);page.update()
        dialog=ft.AlertDialog(modal=True,title=ft.Text('Delete saved data'),
            content=ft.Column([ft.Text(f'Delete {name}? This removes saved PDF copies, page images, extracted responses, handwriting drafts, reviews and marking runs. Original files and exported reports are not deleted. '+('Imported manifests, approved marking examples and legacy records will also be deleted.' if job is None else 'Imported manifests, approved marking examples and legacy records are retained.')),confirmation,feedback],tight=True,width=500),
            actions=[ft.TextButton('Cancel',on_click=lambda _:page.close(dialog)),ft.TextButton('Delete permanently',on_click=remove)])
        page.open(dialog)

    def show_groups(_=None):
        from services.teacher_groups import review_groups,mark_group
        if running:return
        groups=review_groups(repository,selected_jobs)
        opened=time.monotonic()
        by_id={g['id']:g for g in groups}
        panels=[]
        def save(g,choices,score,note,confirm):
            if not confirm.value:notify('Confirm that you reviewed every selected response');return
            members=[m for choice,group in choices if choice.value for m in group['members']]
            try:
                outcomes=mark_group(repository,members,score.value,reviewer.value,note.value or '',time.monotonic()-opened)
                page.close(dialog)
                if active_job:show_job(active_job)
                notify(f"Group marks: {sum(o['status']=='saved' for o in outcomes)} saved. "+
                    '; '.join(o.get('reason','') for o in outcomes if o['status']!='saved'))
            except Exception as exc:notify(str(exc))
        for g in groups:
            choices=[];children=[ft.Text(g['question'] or ''),ft.Text('Reference: '+'; '.join(g['reference']))]
            for candidate in [g]+[by_id[i] for i in g['suggestions']]:
                choice=ft.Checkbox(label=f"{len(candidate['members'])} responses: {candidate['text']}",value=candidate is g)
                choices.append((choice,candidate));children.append(choice)
                children.append(ft.Text('; '.join(f"{m['name']} / {m['student']} / mark {m['score']}" for m in candidate['members'])))
            score=ft.TextField(label=f'Teacher mark out of {g["maximum"]}')
            note=ft.TextField(label='Reason / partial-credit explanation (optional)')
            confirm=ft.Checkbox(label='I reviewed every selected answer and approve this score for each',value=False)
            children.extend([score,note,confirm,ft.OutlinedButton('Save confirmed group mark',
                on_click=lambda _,g=g,cs=choices,f=score,n=note,c=confirm:save(g,cs,f,n,c))])
            panels.append(ft.ExpansionTile(title=ft.Text(f"Q{g['number']} · {len(g['members'])} identical answers"),controls=children))
        dialog=ft.AlertDialog(title=ft.Text('Review answer groups'),content=ft.Column([
            ft.Text('Similar answers are suggestions only. Individual teacher overrides are excluded. Saved scores become approved examples for this question and rubric.'),
            *panels] if panels else [ft.Text('No current marked written responses. Mark the selected scripts first.')],scroll=ft.ScrollMode.AUTO,width=750,height=550),
            actions=[ft.TextButton('Close',on_click=lambda _:page.close(dialog))])
        page.open(dialog)

    async def prepare_draft(question,fields):
        from services.rubric_drafts import draft_rubric
        notify('Preparing a rubric draft from the teacher answer and details…')
        try:
            question=dict(question,questionText=(fields[0].value or '').strip())
            draft=await asyncio.to_thread(draft_rubric,question,marking_model.value.strip())
            approved=ft.Checkbox(label='I reviewed the concepts, alternatives and partial-credit allocation',value=False)
            def accept(_):
                if not approved.value:return
                context=draft['context'];wording,criteria,concepts,count=fields
                wording.value=context['questionText']
                criteria.value='\n'.join(f"{c['marks']} | {c['description']} | {'; '.join(c.get('acceptedAlternatives',[]))}" for c in context['scoringCriteria'])
                concepts.value='\n'.join(c['description']+' | '+'; '.join(c.get('acceptedAlternatives',[])) for c in context.get('acceptedConcepts') or [])
                count.value=str(context.get('requiredConceptCount') or '')
                page.close(dialog);notify('Reviewed rubric staged. Mark / re-mark to apply and save it.')
            context=draft['context']
            preview=[ft.Text(context['questionText']),ft.Text(f"Maximum: {question['maxMarks']} marks")]
            if context.get('requiredConceptCount'):
                preview.append(ft.Text(f"Credit any {context['requiredConceptCount']} distinct concepts, equally weighted."))
            for concept in context.get('acceptedConcepts') or context['scoringCriteria']:
                preview.append(ft.Text((str(concept['marks'])+' marks: ' if 'marks' in concept else '')+concept['description']))
                if concept.get('acceptedAlternatives'):
                    preview.append(ft.Text('Accept equivalent wording: '+'; '.join(concept['acceptedAlternatives'])))
            dialog=ft.AlertDialog(title=ft.Text('Review proposed rubric'),content=ft.Column([
                ft.Text('Original teacher answer: '+'; '.join(draft['referenceAnswers'])),
                ft.Text('Answer details: '+str(draft.get('answerDetail') or '')),
                *preview,approved],width=700,height=500,scroll=ft.ScrollMode.AUTO),
                actions=[ft.TextButton('Cancel',on_click=lambda _:page.close(dialog)),ft.TextButton('Approve and use draft',on_click=accept)])
            page.open(dialog)
        except Exception as exc:notify('Rubric draft not applied: '+str(exc))

    def save_teacher_mark(job,run,result,field,note):
        try:
            sequence=apply_teacher_mark(repository,job,run['sequence'],result['student_id'],
                result['layout_id'],result['question_id'],field.value,reviewer.value,note.value or '',
                review_seconds=time.monotonic()-note.data if note.data else None)
            show_job(job)
            notify(f'Teacher mark saved in run {sequence}. Totals updated; export this latest run.')
        except Exception as exc:notify(str(exc))

    def show_job(job,number=None):
        nonlocal active_job,review_page
        if active_job!=job:
            review_page=1
            job_message.value=''
        if number is not None:
            review_page=number
        active_job=job
        info=repository.job(job)
        if not running:
            extracted=sum(p['status']=='extracted' for p in info['pages'])
            failed=sum(p['status']=='failed' for p in info['pages'])
            progress.value=(f"Extraction complete: {extracted}/{info['total']} pages extracted. "
                +('Review complete; ready for marking.' if info['status']=='complete' else 'Response review is a separate step.')
                if extracted==info['total'] else
                f"Extraction stopped: {extracted}/{info['total']} pages extracted; {failed} failed. Resume or retry failed pages.")
        individual_controls=[ft.Row([
            ft.ElevatedButton('Resume',on_click=lambda _: launch(job)),
            ft.OutlinedButton('Retry failed pages',on_click=lambda _: launch(job,True)),
            ft.OutlinedButton('Approve all saved responses',on_click=lambda _: approve_all(job)),
            ft.OutlinedButton('Complete review',on_click=lambda _: complete(job)),
            ft.OutlinedButton('Delete this job',on_click=lambda _:request_delete(job)),
            ft.OutlinedButton('Mark / re-mark reviewed answers',disabled=info['status']!='complete',
                              on_click=lambda _: page.run_task(mark,job))],wrap=True),
            ft.Text('Approve all covers every page using saved extraction values and handwriting drafts. Save individual edits first. Previous approvals/exclusions are kept; missing transcriptions and invalid responses remain for review.')]
        output_directory=ft.TextField(label='Results export folder',value=str(Path.home()/'Downloads'))
        individual_controls.extend([output_directory,ft.OutlinedButton('Export extraction and marking report',
            on_click=lambda _:export_results(job,None,output_directory))])
        details.controls=[ft.Text(f"{info['name']} — {info['status']}",size=22),job_message,
            ft.ExpansionTile(title=ft.Text('Individual script actions'),subtitle=ft.Text('Optional: retry, review, mark or export only this script.'),
                initially_expanded=False,controls=individual_controls)]
        details.controls.append(ft.Text(f'Attempt ID: {job}. Separate files may contain different attempts by the same student for the same exam.'))
        runs=repository.marking_runs(job)
        all_responses=repository.responses(job)
        written_rows=[r for r in all_responses if r['selection_mode']=='text']
        if written_rows:
            guidance=[]
            metadata={}
            if runs:
                for saved in runs[0]['manifests']:
                    doc=saved['document']
                    for question in doc.get('questions',[]):metadata[(doc['layoutId'],question['questionId'])]=question
            for row in all_responses:
                ref=row['response'].layout
                if any(k[0]==ref.layout_id for k in metadata):continue
                try:
                    doc=resolver(job).resolve(ref.layout_id,ref.assessment_type,ref.assessment_id)
                    for question in doc.get('questions',[]):metadata[(doc['layoutId'],question['questionId'])]=question
                except ValueError:pass
            previous=(runs[0].get('policy') or {}).get('writtenContext',{}) if runs else {}
            seen=set()
            for row in written_rows:
                raw=row['response'];key=(job,raw.layout.layout_id,raw.question_id)
                if key in seen:continue
                seen.add(key)
                if key not in written_context:
                    q=dict(metadata.get((key[1],key[2]),{}),**previous.get(key[1],{}).get(key[2],{}))
                    wording=ft.TextField(label=f'Question {raw.display_number}: original wording',multiline=True,value=q.get('questionText') or '')
                    criteria=ft.TextField(label='Partial-credit criteria',multiline=True,
                        helper_text='One point per line: marks | expected idea | synonyms separated by semicolons. Example: 2.5 | Identifies insufficient funding. Marks must total the question maximum.',
                        value='\n'.join(f"{c['marks']} | {c['description']} | {'; '.join(c.get('acceptedAlternatives',[]))}" for c in q.get('scoringCriteria',[])))
                    concepts=ft.TextField(label=f'Q{raw.display_number}: accepted concepts (any N)',multiline=True,
                        helper_text='One distinct idea per line. Optional synonyms after |, separated by semicolons. Leave empty when using weighted criteria.',
                        value='\n'.join(c['description']+(' | '+'; '.join(c.get('acceptedAlternatives',[])) if c.get('acceptedAlternatives') else '') for c in q.get('acceptedConcepts',[])))
                    count=ft.TextField(label=f'Q{raw.display_number}: required concept count',value=str(q.get('requiredConceptCount') or ''),
                        helper_text='For any-N concepts; otherwise inferred only from explicit question wording.')
                    written_context[key]=(wording,criteria,concepts,count)
                    for field in written_context[key]:field.data=field.value or ''
                q=dict(metadata.get((key[1],key[2]),{}),**previous.get(key[1],{}).get(key[2],{}))
                if q.get('questionText'):
                    from core.written_rubric import prepare_rubric
                    try:
                        rubric=prepare_rubric(q)
                        guidance.append(ft.Text(f"Q{raw.display_number} marking mode: {rubric['mode']}. "+'; '.join(c['description'] for c in rubric['concepts'])))
                    except ValueError as exc:guidance.append(ft.Text(f'Q{raw.display_number}: {exc}'))
                guidance.extend(written_context[key])
                guidance.append(ft.OutlinedButton(f'Draft Q{raw.display_number} rubric from teacher answer',
                    on_click=lambda _,q=q,fields=written_context[key]:page.run_task(prepare_draft,q,fields)))
            details.controls.append(ft.ExpansionTile(title=ft.Text('Written question wording and partial-credit criteria'),
                initially_expanded=False,controls=[ft.Text('Teacher reference answers already supply approved full-credit examples. Drafts require review; saving a teacher mark also saves an approved example. Edit explicit requirements, not background notes. Weighted criteria support definitions and explanations; any-N concepts support example lists. Changes apply to selected attempts with this layout and are saved with marking runs.')]+guidance))
            written_pages=sorted({r['page'] for r in written_rows})
            drafts=sum(r['transcription'] is not None for r in written_rows)
            details.controls.append(ft.Text(f"Written answers: {len(written_rows)} captured regions; {drafts} saved OCR / transcription results. Pages: "+', '.join(map(str,written_pages))))
            if drafts==0:
                details.controls.append(ft.Text('Written-answer images have been captured. Click Resume to run local OCR. Optional AI can improve missing or uncertain OCR text. You can also enter text manually. Missing text is not a blank answer.'))
            details.controls.append(ft.OutlinedButton('Review written answers',on_click=lambda _:show_job(job,written_pages[0])))
        if runs:
            latest=runs[0]
            stale=latest['response_digest']!=digest(response_snapshot(repository,job))
            details.controls.append(ft.Text(f"Marking run {latest['sequence']} · {latest['marking_seconds']:.3f}s · "+
                ('OUTDATED: complete review and re-mark.' if stale else 'Saved reviewed-response version.')))
            for total in latest['totals']:
                details.controls.append(ft.Text(f"Student {total['student_id']} · {total['layout_id']} · "+
                    (f"Score {total['score']}" if total['score'] is not None else
                     f"Needs review: {total['review_count']} questions; resolved subtotal {total['resolved_subtotal']} (not final)")))
            for result in latest['results']:
                details.controls.append(ft.Text(f"Student {result['student_id']} · Q{result['display_number']} · "+
                    (f"{result['score']}/{result['max_marks']}" if result['score'] is not None else 'Review required')+
                    f" · {result['reason']}"))
                extracted='\n'.join(r.get('text') or ', '.join(r.get('selected',[])) or '(blank)' for r in result.get('responses',[]))
                details.controls.append(ft.Row([
                    ft.Column([ft.Text('Correct / reference answer',weight=ft.FontWeight.BOLD),ft.Text(correct_answer(latest,result),selectable=True)],expand=True),
                    ft.Column([ft.Text('Extracted answer (reviewed)',weight=ft.FontWeight.BOLD),ft.Text(extracted,selectable=True)],expand=True)],vertical_alignment=ft.CrossAxisAlignment.START))
                if result.get('concept_count'):
                    details.controls.append(ft.Text(f"Question requests {result['concept_count']} distinct concepts; credit is allocated across that count, not all reference examples."))
                if result.get('scoring_question',{}).get('responseType')=='WRITTEN' or result['score'] is None or result.get('teacher_decision'):
                    mark_field=ft.TextField(label=f'Teacher mark / {result["max_marks"]}',value=result['score'] or '',width=200)
                    mark_note=ft.TextField(label='Marking note (optional)',data=time.monotonic())
                    criteria=result.get('scoring_criteria',[])
                    if criteria:
                        checks=[ft.Checkbox(label=f"{c['marks']} marks: {c['description']}",data=str(c['marks'])) for c in criteria]
                        def use_points(_,cs=checks,field=mark_field):
                            from decimal import Decimal
                            field.value=str(sum((Decimal(c.data) for c in cs if c.value),Decimal(0)));page.update()
                        details.controls.extend(checks+[ft.OutlinedButton('Use checked marking points',on_click=use_points)])
                    details.controls.extend([mark_field,mark_note,ft.OutlinedButton('Save teacher mark and approved example',disabled=stale,
                        on_click=lambda _,r=result,f=mark_field,n=mark_note:save_teacher_mark(job,latest,r,f,n))])
                if result.get('uncertainty'):
                    u=result['uncertainty'];bounds=u.get('score_range')
                    details.controls.append(ft.Text('Review category: '+u['category']+
                        (f" · Provisional range {bounds['lower']}–{bounds['upper']} marks" if bounds else '')+
                        ' · '+ '; '.join(u.get('disputed_concepts',[]))))
                if result.get('similarity_estimate') and result['score'] is None:
                    estimate=result['similarity_estimate']
                    details.controls.append(ft.Text(f"Similarity suggestion: {estimate['suggested_score']}/{estimate['max_marks']} — needs teacher review; not included in totals. Lexical overlap is not correctness."))
                if result.get('ai',{}).get('calculation'):
                    calculation=result['ai']['calculation']
                    details.controls.append(ft.Text(f"Calculated credit: {calculation['credited_concepts']}/{calculation['required_concepts']} concepts; {calculation['score']} marks."))
                if result.get('ai'):
                    proposal=result['ai'].get('proposal') or {}
                    details.controls.append(ft.Text('Local AI: '+str(result['ai'].get('explanation') or proposal.get('explanation',result['ai']['reason']))+
                                                   ' | Evidence: '+str(result['ai'].get('evidence') or proposal.get('evidence',''))))
            details.controls.append(ft.Text('Preserved marking runs: '+', '.join(str(r['sequence']) for r in runs)))
            if latest['excluded_responses']:
                details.controls.append(ft.Text(f"Excluded responses: {latest['excluded_responses']}. Excluded-only students have no score."))
        details.controls.append(ft.Dropdown(label='Review page',value=str(review_page),width=200,
            options=[ft.dropdown.Option(str(n)) for n in range(1,info['total']+1)],
            on_change=lambda e: show_job(job,int(e.control.value))))
        for p in info['pages']:
            if p['number']!=review_page:
                continue
            details.controls.append(ft.Text(f"Page {p['number']}: {p['status']} | preparation {p['preparation_seconds']:.3f}s | extraction {p['extraction_seconds']:.3f}s"))
            if p['status']=='failed':
                data=repository.page_image(job,p['number'])
                details.controls.append(ft.Text(p['reason'] or 'Page needs review'))
                if data:
                    details.controls.append(ft.Image(src_base64=base64.b64encode(data).decode(),width=500))
            meta=json.loads(p['metadata'] or '{}')
            if meta.get('preparation_message'):
                details.controls.append(ft.Text(meta['preparation_message'],selectable=True))
            if p['reason']=='layout-unavailable':
                details.controls.append(ft.Text('Import the manifest downloaded with these answer sheets, then click Retry failed pages. Saving an issue resolution does not extract a page.'))
            if meta.get('identity_reasons'):
                details.controls.append(ft.Text('Student identity needs review: '+', '.join(meta['identity_reasons'])))
                columns=[]
                for key,bubbles in meta.get('bubble_evidence',{}).items():
                    if key.startswith('student:'):
                        columns.append(ft.Column([ft.Text('Digit '+key.split(':')[1])]+[
                            ft.Row([ft.Text(b['option']),ft.Image(src_base64=base64.b64encode(repository.crop(job,b['crop'])).decode(),width=40,height=40)]) for b in bubbles]))
                details.controls.append(ft.Row(columns,wrap=True))
        for issue in repository.issues(job):
            note=ft.TextField(label='Document issue resolution (e.g. excluded duplicate, missing page requested)',value=issue['resolution'] or '')
            details.controls.append(ft.Container(content=ft.Column([
                ft.Text(issue['detail'],selectable=True),note,
                ft.OutlinedButton('Save issue resolution',on_click=lambda e,i=issue,n=note: resolve(job,i,n))]),padding=10,bgcolor=ft.Colors.AMBER_50))
        for row in repository.responses(job,review_page):
            raw=row['response']; decision=row['review']
            selected=json.loads(decision['selected']) if decision else raw.selected_options
            student=ft.TextField(label='Student ID',value=decision['student'] if decision else raw.student_id or '',width=180)
            written=ft.TextField(label='Student answer transcription (empty means blank)',multiline=True,value=(decision.get('raw_text') or '') if decision else (row.get('transcription') or {}).get('text') or raw.raw_text or '') if row['selection_mode']=='text' else None
            choices=[ft.Checkbox(label=o,data=o,value=o in selected) for o in row['options']]
            note=ft.TextField(label='Correction / review note',value='',
                              helper_text='Optional')
            row_feedback=ft.Text('',selectable=True)
            draft=row.get('transcription')
            draft_label=(f"Draft from {draft['engine']} ({draft['seconds']:.1f}s): {draft.get('text') or 'No transcription'} — teacher review required. "+', '.join(draft['review_reasons'])) if draft else 'No OCR or AI text saved.'
            history=[ft.Text(f"{h['created']} · {h['reviewer']} · {h['action']} · student {h['student']} · {h.get('raw_text') if h.get('raw_text') is not None else h['selected']} · {h['note']}")
                     for h in repository.review_history(job,row['id'])]
            if draft and draft.get('engine')=='trocr-written-v1':
                if 'ocr-unavailable' in draft.get('review_reasons',[]):
                    draft_label+=' TrOCR is unavailable. Install its dependencies and run python -m tools.prepare_trocr once to download the local model.'
                if 'ocr-language-unvalidated' in draft.get('review_reasons',[]):
                    draft_label+=' This TrOCR model is trained on English handwriting; Kiswahili accuracy is unvalidated.'
            if draft and 'ocr-language-pack-missing' in draft.get('review_reasons',[]):
                draft_label=draft_label.replace('ocr-language-pack-missing','').rstrip(' ,')
                draft_label+=' Configuration warning: a requested OCR language pack is missing; this alone does not indicate unreadable text.'
            if draft and draft.get('engine')=='paddle-trocr-agreement-v1':
                if draft['status']=='read':
                    draft_label+=' PaddleOCR corroborated the TrOCR reading; no Gemma transcription is needed.'
                elif 'paddle-unavailable' in draft.get('review_reasons',[]):
                    draft_label+=' PaddleOCR is unavailable. Install requirements-paddle.txt and run python -m tools.prepare_paddle. Optional AI fallback remains available.'
                else:
                    draft_label+=' PaddleOCR did not corroborate this reading; the TrOCR text is retained for review or optional AI fallback.'
            original=ft.Text(f"Original extraction: {raw.student_id or 'unknown student'} / {', '.join(raw.selected_options) or raw.raw_text or raw.status.value} · {', '.join(raw.review_reasons) or 'clear'}")
            audit_controls=[original]
            if written is not None:
                audit_controls.append(ft.Text(draft_label.replace('— teacher review required.', '— reviewed; original draft.') if decision else draft_label))
            audit_controls.extend(history)
            audit=ft.ExpansionTile(title=ft.Text('Extraction and review history'),
                initially_expanded=False,controls=audit_controls)
            images=[ft.Image(src_base64=base64.b64encode(repository.crop(job,r.crop_ref)).decode(),width=650 if written is not None else 60,height=160 if written is not None else 60,fit=ft.ImageFit.CONTAIN) for r in raw.regions]
            label=f"Page {row['page']} · Question {raw.display_number}"+(f" ({raw.item_number})" if raw.item_number else f" (part {raw.part_number})" if raw.part_number else '')
            details.controls.append(ft.Container(content=ft.Column([
                ft.Text(label,size=18),
                ft.Text(f"{raw.response_type.value}: "+('Review the answer image and transcription.' if written is not None else 'Choose all that apply.' if row['selection_mode']=='multiple' else 'Choose at most one option.')),
                ft.Row(images,wrap=True),student,written if written is not None else ft.Row(choices,wrap=True),
                ft.Text('Leave transcription empty only when confirming a blank answer.' if written is not None else 'Uncheck every option to explicitly approve a blank answer.'),note,
                ft.Text(draft_label if written is not None and not decision else ''),
                ft.Text('Latest decision: '+(decision['action']+' by '+decision['reviewer'] if decision else 'not reviewed')),
                ft.Text('Current review: '+('approved' if decision['action']=='approve' else 'excluded')+'.' if decision else 'Current review: needs review.'),
                audit,
                row_feedback,
                ft.Row([ft.ElevatedButton('Approve current values',on_click=lambda e,r=row,s=student,c=choices,n=note,w=written,f=row_feedback: apply_review(job,r,s,c,n,'approve',w,f)),
                        ft.OutlinedButton('Exclude response',on_click=lambda e,r=row,s=student,c=choices,n=note,w=written,f=row_feedback: apply_review(job,r,s,c,n,'exclude',w,f))])]),padding=12,border=ft.border.all(1,ft.Colors.GREY_300)))
        refresh_jobs()
        page.update()

    async def work(job=None,retry=False):
        nonlocal running,active_job,review_page
        if running:
            return
        running=True; cancellation.clear()
        message.value='';job_message.value='';progress.value='Starting extraction…'
        controls.disabled=True; details.disabled=True; cancel_button.disabled=False
        refresh_jobs(); page.update()
        try:
            inputs=[job] if job else list(dict.fromkeys(p.strip() for p in (source.value or '').splitlines() if p.strip()))
            if not inputs:raise ValueError('Choose at least one PDF or image')
            failures=[];saved=0;recognition_summaries=[]
            for index,item in enumerate(inputs,1):
                if cancellation.is_set():break
                try:
                    current_resolver=resolver(job)
                    current_job=job or await asyncio.to_thread(repository.create,Path(item).expanduser(),current_resolver.scope)
                    if active_job!=current_job:review_page=1
                    active_job=current_job
                    selected_jobs.add(current_job)
                    from core.handwriting import OllamaHandwritingRecognizer
                    recognizer=OllamaHandwritingRecognizer(model.value.strip()) if handwriting.value else None
                    recognition_state={}
                    task=asyncio.create_task(asyncio.to_thread(run_job,repository,current_job,current_resolver,cancellation,None,retry,recognizer,language.value,int(recognition_workers.value),recognition_state.update))
                    while not task.done():
                        info=await asyncio.to_thread(repository.job,current_job)
                        done=sum(p['status']!='pending' for p in info['pages'])
                        progress.value=f"File {index}/{len(inputs)} · {info['name']} · {done}/{info['total']} pages saved · {info['status']}. Cancellation saves in-flight work and skips remaining files."
                        if recognition_state:
                            state=dict(recognition_state)
                            progress.value+=f" AI: {state.get('completed',0)}/{state.get('total',0)} answers saved; active pages {state.get('active_pages',[])}; elapsed {state.get('wall_seconds',0):.1f}s."
                            if state.get('reduced_to_serial'):progress.value+=' Using one page after an inference failure.'
                        page.update()
                        await asyncio.sleep(.25)
                    recognition_summary=await task
                    if recognition_summary:
                        info=await asyncio.to_thread(repository.job,current_job)
                        progress.value=f"{info['name']} · AI {recognition_summary['completed']}/{recognition_summary['total']} answers saved in {recognition_summary['wall_seconds']:.1f}s."
                        recognition_summaries.append(recognition_summary)
                        page.update()
                    saved+=1
                except Exception as exc:
                    failures.append(f'{Path(item).name}: {exc}')
            message.value=f'{saved}/{len(inputs)} files processed. Review saved jobs below.'
            if recognition_summaries:
                ai_saved=sum(s['completed'] for s in recognition_summaries)
                ai_seconds=sum(s['wall_seconds'] for s in recognition_summaries)
                message.value+=f' AI transcription: {ai_saved} answers saved in {ai_seconds:.1f}s elapsed across files.'
                if any(s['reduced_to_serial'] for s in recognition_summaries):
                    message.value+=' Reduced to one page after an inference failure.'
            if cancellation.is_set():message.value+=' Batch cancelled; resume the current job or select remaining files to continue.'
            if failures:message.value+=' Errors: '+'; '.join(failures)
            if not saved:progress.value='No files completed. Check the error message below.'
        except Exception as exc:
            message.value=str(exc)
            progress.value='Extraction did not start.'
        finally:
            running=False; controls.disabled=False; details.disabled=False; cancel_button.disabled=True
            if active_job:
                show_job(active_job)
            else:
                refresh_jobs(); page.update()

    def policy_for(job):
        policy=dict(version=1)
        if marking_policy.value!='manifest':policy['structured']=marking_policy.value
        runs=repository.marking_runs(job)
        context=dict((runs[0].get('policy') or {}).get('writtenContext',{})) if runs else {}
        target_layouts={r['response'].layout.layout_id for r in repository.responses(job)}
        for (context_job,layout_id,qid),fields in written_context.items():
            wording,criteria,concepts,count=fields
            if layout_id not in target_layouts:continue
            if all((f.value or '')==f.data for f in fields):continue
            points=[]
            for index,line in enumerate((criteria.value or '').splitlines(),1):
                if not line.strip():continue
                marks,separator,remainder=line.partition('|')
                description,_,variants=remainder.partition('|')
                if not separator:raise ValueError('Each criterion needs marks | description')
                points.append(dict(id=f'point-{index}',marks=marks.strip(),description=description.strip(),
                    acceptedAlternatives=[v.strip() for v in variants.split(';') if v.strip()]))
            accepted=[]
            for index,line in enumerate((concepts.value or '').splitlines(),1):
                if not line.strip():continue
                description,_,variants=line.partition('|')
                accepted.append(dict(id=f'concept-{index}',description=description.strip(),
                    acceptedAlternatives=[v.strip() for v in variants.split(';') if v.strip()]))
            if points and accepted:raise ValueError('Use weighted criteria OR any-N concepts, not both')
            entry=dict(questionText=(wording.value or '').strip(),scoringCriteria=points,acceptedConcepts=None,requiredConceptCount=None)
            if accepted:entry['acceptedConcepts']=accepted
            if (count.value or '').strip():entry['requiredConceptCount']=int(count.value.strip())
            context.setdefault(layout_id,{})[qid]=entry
        if context:policy['writtenContext']=context
        if len(policy)==1:policy=None
        return policy

    async def mark(job):
        nonlocal running
        if running:
            return
        running=True;controls.disabled=True;details.disabled=True
        refresh_jobs();page.update()
        try:
            policy=policy_for(job)
            current_resolver=resolver(job)
            from core.ai_marking import LocalMarkingAssistant
            assistant=LocalMarkingAssistant(marking_model.value.strip()) if ai_marking.value else None
            marking_progress={}
            task=asyncio.create_task(asyncio.to_thread(mark_job,repository,job,current_resolver,reviewer.value,policy,assistant,marking_progress.update))
            elapsed=0
            while not task.done():
                progress.value=(f"Marking {marking_progress.get('completed',0)}/{marking_progress.get('total','?')} questions · student {marking_progress.get('student','')} · Q{marking_progress.get('question','')} · {elapsed:.0f}s · {marking_progress.get('resumed',0)} restored from saved progress.")
                page.update()
                await asyncio.sleep(.25);elapsed+=.25
            result=await task
            progress.value=f"Marking finished in {result['marking_seconds']:.3f}s; local AI {result.get('ai_seconds',0):.3f}s."
            message.value=f"Marking run {result['sequence']} saved. Extraction was not repeated."
        except Exception as exc:
            message.value=str(exc)
        finally:
            running=False;controls.disabled=False;details.disabled=False
            show_job(job)

    async def batch_action(action):
        nonlocal running
        if running:return
        jobs=sorted(selected_jobs)
        if not jobs:
            batch_message.value='Select scripts using the checkboxes or Select all scripts.'
            notify(batch_message.value);return
        if action!='export' and not (reviewer.value or '').strip():
            reviewer.error_text='Required to save review decisions'
            batch_message.value='Enter Reviewer name before batch review or marking.'
            notify(batch_message.value);return
        reviewer.error_text=None
        running=True;controls.disabled=True;details.disabled=True;batch_controls.disabled=True
        refresh_jobs();page.update()
        outcomes=[]
        try:
            from core.ai_marking import LocalMarkingAssistant
            assistant=LocalMarkingAssistant(marking_model.value.strip()) if action=='mark' and ai_marking.value else None
            if action!='export':
                for index,job in enumerate(jobs,1):
                    name=repository.job(job)['name']
                    batch_message.value=f'{action}: {index}/{len(jobs)} · {name}';page.update()
                    try:
                        if action=='approve':
                            result=await asyncio.to_thread(repository.approve_all,job,reviewer.value)
                            outcome=f"{result['approved']} approved, {result['retained']} retained, {len(result['skipped'])} need attention"
                        elif action=='complete':
                            await asyncio.to_thread(repository.complete,job);outcome='Review complete'
                        else:
                            marking_progress={}
                            task=asyncio.create_task(asyncio.to_thread(mark_job,repository,job,resolver(job),reviewer.value,policy_for(job),assistant,marking_progress.update))
                            while not task.done():
                                batch_message.value=(f"Marking script {index}/{len(jobs)} · {name} · "
                                    f"{marking_progress.get('completed',0)}/{marking_progress.get('total','?')} questions · "
                                    f"student {marking_progress.get('student','')} · Q{marking_progress.get('question','')} "
                                    f"· {marking_progress.get('resumed',0)} restored")
                                page.update();await asyncio.sleep(.25)
                            result=await task
                            pending=sum(t['review_count'] for t in result['totals'])
                            outcome=f'Marked; {pending} questions need marking review'
                        outcomes.append(name+': '+outcome)
                    except Exception as exc:outcomes.append(name+': BLOCKED — '+str(exc))
            if action in ('mark','export'):
                report=await asyncio.to_thread(export_batch_report,repository,jobs,batch_directory.value,resolver)
                outcomes.append('Combined report: '+str(report))
            batch_message.value='\n'.join(outcomes)
        except Exception as exc:batch_message.value='\n'.join(outcomes+[str(exc)])
        finally:
            running=False;controls.disabled=False;details.disabled=False;batch_controls.disabled=False
            if active_job:show_job(active_job)
            else:refresh_jobs();page.update()

    def launch(job=None,retry=False):
        page.run_task(work,job,retry)

    def picked(event):
        if event.files and event.files[0].path:
            source.value='\n'.join(dict.fromkeys(f.path for f in event.files if f.path)); page.update()
        elif event.files:
            notify('The picker did not return a local path. Enter the PDF path directly or use the desktop app.')

    async def choose_pdf():
        choose_button.disabled=True;page.update()
        try:
            if sys.platform=='darwin':
                selected=await asyncio.to_thread(choose_macos_pdf)
                if selected:
                    source.value=selected if isinstance(selected,str) else '\n'.join(dict.fromkeys(selected))
            else:
                picker.pick_files(dialog_title='Choose answer-sheet PDF or image',allow_multiple=True,
                                  allowed_extensions=['pdf','png','jpg','jpeg','tif','tiff','bmp','webp'],file_type=ft.FilePickerFileType.CUSTOM)
        except Exception as exc:
            message.value=f'Could not open file browser: {exc}. You can enter the PDF path directly.'
        finally:
            choose_button.disabled=False;page.update()

    manifest_picker=ft.FilePicker(on_result=lambda e:import_manifest(e.files[0].path) if e.files and e.files[0].path else None)
    page.overlay.append(manifest_picker)
    picker=ft.FilePicker(on_result=picked)
    page.overlay.append(picker)
    choose_button=ft.OutlinedButton('Choose PDFs / images',on_click=lambda _: page.run_task(choose_pdf))
    cancel_button=ft.OutlinedButton('Cancel after current page',disabled=True,on_click=lambda _: cancellation.set())
    controls=ft.Column([
        manifest_path,ft.Row([ft.OutlinedButton('Upload manifest',on_click=lambda _:page.run_task(choose_manifest)),ft.OutlinedButton('Import manifest path',on_click=lambda _:import_manifest(manifest_path.value))],wrap=True),handwriting,ft.Row([model,language,recognition_workers],wrap=True),reviewer,marking_policy,
        ft.Text('Two-page mode processes uncertain answer crops from separate pages of the same student. Speed depends on local model capacity; start with a small trial.'),
        ft.Text('Microsoft TrOCR reads scanned written answers one line at a time. The handwriting checkbox enables optional AI fallback for missing or uncertain OCR text.'),
        ft.Text('Partial credit: equal credit per matching item; multiple-answer credit subtracts wrong selections, with a minimum of zero. Scores round to two decimals. Written reference matches ignore case and extra whitespace.'),ai_marking,marking_model,
        ft.Row([source,choose_button]),
        ft.ElevatedButton('Create / resume extraction jobs',on_click=lambda _: launch()),
        ft.OutlinedButton('Reset all AutoMark data',on_click=lambda _:request_delete())])
    batch_controls=ft.Column([ft.Text('Batch actions apply only to checked scripts. Approval uses saved values; save individual edits first.'),
        ft.OutlinedButton('Select all scripts',on_click=select_all_jobs),
        ft.OutlinedButton('Review answer groups',on_click=show_groups),batch_directory,
        ft.Row([ft.OutlinedButton('Approve selected scripts',on_click=lambda _:page.run_task(batch_action,'approve')),
        ft.OutlinedButton('Complete selected reviews',on_click=lambda _:page.run_task(batch_action,'complete')),
        ft.OutlinedButton('Mark selected scripts',on_click=lambda _:page.run_task(batch_action,'mark')),
        ft.OutlinedButton('Export combined report',on_click=lambda _:page.run_task(batch_action,'export'))],wrap=True)])
    page.add(ft.Text('Structured extraction and review',size=28),
        ft.Text('Import the teacher manifest downloaded with the answer sheets. No backend URL, token or staff account is needed. Complete review before marking.'),
        controls,cancel_button,progress,message,ft.Text('Saved jobs',size=20),batch_controls,batch_message,jobs_column,details)
    refresh_jobs(); page.update()
