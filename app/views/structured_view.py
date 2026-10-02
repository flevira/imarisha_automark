"""Saved structured extraction jobs and explicit teacher review (Flet desktop)."""
import asyncio
import base64
import json
import threading
import sys
from pathlib import Path

import flet as ft

from database.structured_repository import JobRepository
from services.marking_export import export_marking_run
from services.extraction_jobs import run_job
from services.manifest_resolver import ImportedManifestResolver
from services.marking_jobs import mark_job, digest, response_snapshot
from tools.prepare_manifest_pages import cache_location
from app.controls.pdf_picker import choose_macos_pdf, choose_macos_manifest


def build_structured_view(page: ft.Page, repository=None):
    repository = repository or JobRepository(cache_location().parent/'structured-jobs.db')
    page.controls.clear()
    page.scroll=ft.ScrollMode.AUTO
    manifest_path=ft.TextField(label='Downloaded manifest file',expand=True)
    handwriting=ft.Checkbox(label='Trial local handwriting transcription (all drafts need review)',value=False)
    model=ft.TextField(label='Installed local vision model',value='gemma3:4b-it-q4_K_M',width=350)
    language=ft.Dropdown(label='Handwriting language',value='en-sw',width=220,options=[ft.dropdown.Option('en','English'),ft.dropdown.Option('sw','Kiswahili'),ft.dropdown.Option('en-sw','English and Kiswahili')])
    reviewer=ft.TextField(label='Reviewer name',width=240)
    marking_policy=ft.Dropdown(label='Structured marking policy',value='manifest',width=450,options=[
        ft.dropdown.Option('manifest','Manifest answers with partial credit'),
        ft.dropdown.Option('all-or-nothing','All-or-nothing per question (including matching)')])
    ai_marking=ft.Checkbox(label='Use local AI for equivalent written answers and rubric-based partial credit',value=True)
    marking_model=ft.TextField(label='Local marking model',value='qwen3:4b-instruct-2507-q4_K_M',width=450)
    written_context={}
    source=ft.TextField(label='Answer-sheet PDF path',expand=True)
    message=ft.Text('',selectable=True)
    job_message=ft.Text('',selectable=True)
    progress=ft.Text('')
    jobs_column=ft.Column()
    details=ft.Column()
    cancellation=threading.Event()
    running=False
    active_job=None
    review_page=1

    def notify(text):
        message.value=text
        job_message.value=text
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

    def refresh_jobs():
        jobs_column.controls=[ft.OutlinedButton(f"{j['name']} — {j['status']} ({j['total']} pages)",
            on_click=lambda e,job=j['id']: show_job(job), disabled=running) for j in repository.jobs()]

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
            stale=run['response_digest']!=digest(response_snapshot(repository,job))
            report,audit=export_marking_run(run,directory.value,
                name=repository.job(job)['name'],outdated=stale)
            notify(f'Results exported. Open {report} in a browser to read or print. Saved audit: {audit}')
        except Exception as exc:
            notify(f'Export failed: {exc}')

    def show_job(job,number=None):
        nonlocal active_job,review_page
        if active_job!=job:
            review_page=1
            job_message.value=''
        if number is not None:
            review_page=number
        active_job=job
        info=repository.job(job)
        details.controls=[ft.Text(f"{info['name']} — {info['status']}",size=22),ft.Row([
            ft.ElevatedButton('Resume',on_click=lambda _: launch(job)),
            ft.OutlinedButton('Retry failed pages',on_click=lambda _: launch(job,True)),
            ft.OutlinedButton('Approve all saved responses',on_click=lambda _: approve_all(job)),
            ft.OutlinedButton('Complete review',on_click=lambda _: complete(job)),
            ft.OutlinedButton('Mark / re-mark reviewed answers',disabled=info['status']!='complete',
                              on_click=lambda _: page.run_task(mark,job))],wrap=True),job_message,
            ft.Text('Approve all covers every page using saved extraction values and handwriting drafts. Save individual edits first. Previous approvals/exclusions are kept; missing transcriptions and invalid responses remain for review.')]
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
            previous=(runs[0].get('policy') or {}).get('writtenContext',{}) if runs else {}
            seen=set()
            for row in written_rows:
                raw=row['response'];key=(job,raw.layout.layout_id,raw.question_id)
                if key in seen:continue
                seen.add(key)
                if key not in written_context:
                    q=previous.get(key[1],{}).get(key[2],metadata.get((key[1],key[2]),{}))
                    wording=ft.TextField(label=f'Question {raw.display_number}: original wording',multiline=True,value=q.get('questionText') or '')
                    criteria=ft.TextField(label='Partial-credit criteria',multiline=True,
                        helper_text='One point per line: marks | expected idea. Example: 2.5 | Identifies insufficient funding. Marks must total the question maximum.',
                        value='\n'.join(f"{c['marks']} | {c['description']}" for c in q.get('scoringCriteria',[])))
                    written_context[key]=(wording,criteria)
                guidance.extend(written_context[key])
            details.controls.append(ft.ExpansionTile(title=ft.Text('Written question wording and partial-credit criteria'),
                initially_expanded=False,controls=[ft.Text('Optional for older sheets. These apply to every student with this layout and are saved with the next marking run.')]+guidance))
            written_pages=sorted({r['page'] for r in written_rows})
            drafts=sum(r['transcription'] is not None for r in written_rows)
            details.controls.append(ft.Text(f"Written answers: {len(written_rows)} captured regions; {drafts} saved transcription drafts. Pages: "+', '.join(map(str,written_pages))))
            if drafts==0:
                details.controls.append(ft.Text('No handwriting drafts have been saved. Enable Trial local handwriting transcription above and click Resume to transcribe saved crops, or review their images and enter text manually.'))
            details.controls.append(ft.OutlinedButton('Review written answers',on_click=lambda _:show_job(job,written_pages[0])))
        if runs:
            latest=runs[0]
            stale=latest['response_digest']!=digest(response_snapshot(repository,job))
            details.controls.append(ft.Text(f"Marking run {latest['sequence']} · {latest['marking_seconds']:.3f}s · "+
                ('OUTDATED: complete review and re-mark.' if stale else 'Saved reviewed-response version.')))
            output_directory=ft.TextField(label='Results export folder',value=str(repository.path.parent/'exports'))
            details.controls.extend([output_directory,
                ft.OutlinedButton('Export results report and audit',
                    on_click=lambda _:export_results(job,latest,output_directory))])
            for total in latest['totals']:
                details.controls.append(ft.Text(f"Student {total['student_id']} · {total['layout_id']} · "+
                    (f"Score {total['score']}" if total['score'] is not None else
                     f"Needs review: {total['review_count']} questions; resolved subtotal {total['resolved_subtotal']} (not final)")))
            for result in latest['results']:
                details.controls.append(ft.Text(f"Student {result['student_id']} · Q{result['display_number']} · "+
                    (f"{result['score']}/{result['max_marks']}" if result['score'] is not None else 'Review required')+
                    f" · {result['reason']}"))
                if result.get('similarity_estimate') and result['score'] is None:
                    estimate=result['similarity_estimate']
                    details.controls.append(ft.Text(f"Similarity suggestion: {estimate['suggested_score']}/{estimate['max_marks']} — needs teacher review; not included in totals. Lexical overlap is not correctness."))
                if result.get('ai'):
                    proposal=result['ai'].get('proposal') or {}
                    details.controls.append(ft.Text('Local AI: '+str(proposal.get('explanation',result['ai']['reason']))+
                                                   ' | Evidence: '+str(proposal.get('evidence',''))))
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
            draft_label=(f"Draft from {draft['engine']} ({draft['seconds']:.1f}s): {draft.get('text') or 'No transcription'} — teacher review required. "+', '.join(draft['review_reasons'])) if draft else 'No handwriting model draft.'
            history=[ft.Text(f"{h['created']} · {h['reviewer']} · {h['action']} · student {h['student']} · {h.get('raw_text') if h.get('raw_text') is not None else h['selected']} · {h['note']}")
                     for h in repository.review_history(job,row['id'])]
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
        controls.disabled=True; details.disabled=True; cancel_button.disabled=False
        refresh_jobs(); page.update()
        try:
            current_resolver=resolver(job)
            if job is None:
                job=await asyncio.to_thread(repository.create,Path(source.value.strip()).expanduser(),current_resolver.scope)
            if active_job!=job:
                review_page=1
            active_job=job
            from core.handwriting import OllamaHandwritingRecognizer
            recognizer=OllamaHandwritingRecognizer(model.value.strip()) if handwriting.value else None
            task=asyncio.create_task(asyncio.to_thread(run_job,repository,job,current_resolver,cancellation,None,retry,recognizer,language.value))
            while not task.done():
                info=await asyncio.to_thread(repository.job,job)
                done=sum(p['status']!='pending' for p in info['pages'])
                progress.value=f"{done}/{info['total']} pages saved · {info['status']}. Cancellation stops after the current page or recognition request."
                page.update()
                await asyncio.sleep(.25)
            await task
            message.value='Job saved. Review responses and document issues below.'
        except Exception as exc:
            message.value=str(exc)
        finally:
            running=False; controls.disabled=False; details.disabled=False; cancel_button.disabled=True
            if active_job:
                show_job(active_job)
            else:
                refresh_jobs(); page.update()

    async def mark(job):
        nonlocal running
        if running:
            return
        running=True;controls.disabled=True;details.disabled=True
        refresh_jobs();page.update()
        try:
            policy=dict(version=1)
            if marking_policy.value!='manifest':policy['structured']=marking_policy.value
            context={}
            for (context_job,layout_id,qid),(wording,criteria) in written_context.items():
                if context_job!=job:continue
                if not (wording.value or '').strip() and not (criteria.value or '').strip():continue
                points=[]
                for index,line in enumerate((criteria.value or '').splitlines(),1):
                    if not line.strip():continue
                    marks,separator,description=line.partition('|')
                    if not separator:raise ValueError('Each criterion needs marks | description')
                    points.append(dict(id=f'point-{index}',marks=marks.strip(),description=description.strip()))
                context.setdefault(layout_id,{})[qid]=dict(questionText=(wording.value or '').strip(),scoringCriteria=points)
            if context:policy['writtenContext']=context
            if len(policy)==1:policy=None
            current_resolver=resolver(job)
            from core.ai_marking import LocalMarkingAssistant
            assistant=LocalMarkingAssistant(marking_model.value.strip()) if ai_marking.value else None
            task=asyncio.create_task(asyncio.to_thread(mark_job,repository,job,current_resolver,reviewer.value,policy,assistant))
            elapsed=0
            while not task.done():
                progress.value=f'Marking saved responses · {elapsed:.0f}s. Local AI checks only unresolved written answers.'
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

    def launch(job=None,retry=False):
        page.run_task(work,job,retry)

    def picked(event):
        if event.files and event.files[0].path:
            source.value=event.files[0].path; page.update()
        elif event.files:
            notify('The picker did not return a local path. Enter the PDF path directly or use the desktop app.')

    async def choose_pdf():
        choose_button.disabled=True;page.update()
        try:
            if sys.platform=='darwin':
                selected=await asyncio.to_thread(choose_macos_pdf)
                if selected:
                    source.value=selected
            else:
                picker.pick_files(dialog_title='Choose answer-sheet PDF',allow_multiple=False,
                                  allowed_extensions=['pdf'],file_type=ft.FilePickerFileType.CUSTOM)
        except Exception as exc:
            message.value=f'Could not open file browser: {exc}. You can enter the PDF path directly.'
        finally:
            choose_button.disabled=False;page.update()

    manifest_picker=ft.FilePicker(on_result=lambda e:import_manifest(e.files[0].path) if e.files and e.files[0].path else None)
    page.overlay.append(manifest_picker)
    picker=ft.FilePicker(on_result=picked)
    page.overlay.append(picker)
    choose_button=ft.OutlinedButton('Choose PDF',on_click=lambda _: page.run_task(choose_pdf))
    cancel_button=ft.OutlinedButton('Cancel after current page',disabled=True,on_click=lambda _: cancellation.set())
    def back(_):
        from app.views.home_view import build_home_view
        page.overlay.remove(picker)
        page.overlay.remove(manifest_picker)
        build_home_view(page)
    controls=ft.Column([ft.OutlinedButton('Back to legacy extraction',on_click=back),
        ft.Row([manifest_path,ft.OutlinedButton('Upload manifest',on_click=lambda _:page.run_task(choose_manifest)),ft.OutlinedButton('Import manifest path',on_click=lambda _:import_manifest(manifest_path.value))],wrap=True),handwriting,ft.Row([model,language],wrap=True),reviewer,marking_policy,
        ft.Text('Partial credit: equal credit per matching item; multiple-answer credit subtracts wrong selections, with a minimum of zero. Scores round to two decimals. Written reference matches ignore case and extra whitespace.'),ai_marking,marking_model,
        ft.Row([source,choose_button]),
        ft.ElevatedButton('Create / resume extraction job',on_click=lambda _: launch())])
    page.add(ft.Text('Structured extraction and review',size=28),
        ft.Text('Import the teacher manifest downloaded with the answer sheets. No backend URL, token or staff account is needed. Complete review before marking.'),
        controls,cancel_button,progress,message,ft.Text('Saved jobs',size=20),jobs_column,details)
    refresh_jobs(); page.update()
