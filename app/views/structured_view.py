"""Saved structured extraction jobs and explicit teacher review (Flet desktop)."""
import asyncio
import base64
import json
import threading
from pathlib import Path

import flet as ft

from database.structured_repository import JobRepository
from services.extraction_jobs import run_job
from services.manifest_resolver import CachedManifestResolver, TeacherManifestClient
from tools.prepare_manifest_pages import cache_location


def build_structured_view(page: ft.Page, repository=None):
    repository = repository or JobRepository(cache_location().parent/'structured-jobs.db')
    page.controls.clear()
    page.scroll=ft.ScrollMode.AUTO
    backend=ft.TextField(label='Imarisha backend URL',value='http://127.0.0.1:18080',width=350)
    account=ft.TextField(label='Staff account ID',width=220)
    token=ft.TextField(label='Teacher access token (optional for cached layouts)',password=True,can_reveal_password=True,width=420)
    handwriting=ft.Checkbox(label='Trial local handwriting transcription (all drafts need review)',value=False)
    model=ft.TextField(label='Installed local vision model',value='gemma3:4b-it-q4_K_M',width=350)
    language=ft.Dropdown(label='Handwriting language',value='en-sw',width=220,options=[ft.dropdown.Option('en','English'),ft.dropdown.Option('sw','Kiswahili'),ft.dropdown.Option('en-sw','English and Kiswahili')])
    reviewer=ft.TextField(label='Reviewer name',width=240)
    source=ft.TextField(label='Answer-sheet PDF path',expand=True)
    message=ft.Text('',selectable=True)
    progress=ft.Text('')
    jobs_column=ft.Column()
    details=ft.Column()
    cancellation=threading.Event()
    running=False
    active_job=None
    review_page=1

    def notify(text):
        message.value=text
        page.update()

    def resolver():
        if not account.value.strip():
            raise ValueError('Enter the staff account ID used for these layouts')
        url=backend.value.strip().rstrip('/')
        client=TeacherManifestClient(url,token.value) if token.value else None
        return CachedManifestResolver(cache_location(),url+'|'+account.value.strip(),client)

    def refresh_jobs():
        jobs_column.controls=[ft.OutlinedButton(f"{j['name']} — {j['status']} ({j['total']} pages)",
            on_click=lambda e,job=j['id']: show_job(job), disabled=running) for j in repository.jobs()]

    def apply_review(job,row,student,choices,note,action,written=None):
        try:
            repository.review(job,row['id'],action=action,student=student.value.strip(),
                selected=[c.data for c in choices if c.value],reviewer=reviewer.value,note=note.value,
                expected_sequence=row['review']['sequence'] if row['review'] else None,
                raw_text=written.value if written is not None else None)
            show_job(job)
            notify('Decision saved. Original extraction retained.')
        except Exception as exc:
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
            notify('Review complete. No marking was performed.')
        except Exception as exc:
            notify(str(exc))

    def show_job(job,number=None):
        nonlocal active_job,review_page
        if active_job!=job:
            review_page=1
        if number is not None:
            review_page=number
        active_job=job
        info=repository.job(job)
        details.controls=[ft.Text(f"{info['name']} — {info['status']}",size=22),ft.Row([
            ft.ElevatedButton('Resume',on_click=lambda _: launch(job)),
            ft.OutlinedButton('Retry failed pages',on_click=lambda _: launch(job,True)),
            ft.OutlinedButton('Complete review',on_click=lambda _: complete(job))])]
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
            note=ft.TextField(label='Correction / review note',value=decision['note'] if decision else '')
            draft=row.get('transcription')
            draft_label=(f"Draft from {draft['engine']} ({draft['seconds']:.1f}s): {draft.get('text') or 'No transcription'} — teacher review required. "+', '.join(draft['review_reasons'])) if draft else 'No handwriting model draft.'
            history=[ft.Text(f"{h['created']} · {h['reviewer']} · {h['action']} · student {h['student']} · {h.get('raw_text') if h.get('raw_text') is not None else h['selected']} · {h['note']}")
                     for h in repository.review_history(job,row['id'])]
            images=[ft.Image(src_base64=base64.b64encode(repository.crop(job,r.crop_ref)).decode(),width=650 if written is not None else 60,height=160 if written is not None else 60,fit=ft.ImageFit.CONTAIN) for r in raw.regions]
            label=f"Page {row['page']} · Question {raw.display_number}"+(f" ({raw.item_number})" if raw.item_number else f" (part {raw.part_number})" if raw.part_number else '')
            details.controls.append(ft.Container(content=ft.Column([
                ft.Text(label,size=18),ft.Text(f"Original: {raw.student_id or 'unknown student'} / {', '.join(raw.selected_options) or raw.raw_text or raw.status.value} · {', '.join(raw.review_reasons) or 'clear'}"),
                ft.Text(f"{raw.response_type.value}: "+('Review the answer image and transcription.' if written is not None else 'Choose all that apply.' if row['selection_mode']=='multiple' else 'Choose at most one option.')),
                ft.Row(images,wrap=True),student,written if written is not None else ft.Row(choices,wrap=True),
                ft.Text('Leave transcription empty only when confirming a blank answer.' if written is not None else 'Uncheck every option to explicitly approve a blank answer.'),note,
                ft.Text(draft_label),
                ft.Text('Latest decision: '+(decision['action']+' by '+decision['reviewer'] if decision else 'not reviewed')),
                ft.Column(history),
                ft.Row([ft.ElevatedButton('Approve current values',on_click=lambda e,r=row,s=student,c=choices,n=note,w=written: apply_review(job,r,s,c,n,'approve',w)),
                        ft.OutlinedButton('Exclude response',on_click=lambda e,r=row,s=student,c=choices,n=note,w=written: apply_review(job,r,s,c,n,'exclude',w))])]),padding=12,border=ft.border.all(1,ft.Colors.GREY_300)))
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
            current_resolver=resolver()
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

    def launch(job=None,retry=False):
        page.run_task(work,job,retry)

    def picked(event):
        if event.files and event.files[0].path:
            source.value=event.files[0].path; page.update()

    picker=ft.FilePicker(on_result=picked)
    page.overlay.append(picker)
    cancel_button=ft.OutlinedButton('Cancel after current page',disabled=True,on_click=lambda _: cancellation.set())
    def back(_):
        from app.views.home_view import build_home_view
        page.overlay.remove(picker)
        build_home_view(page)
    controls=ft.Column([ft.OutlinedButton('Back to legacy extraction',on_click=back),
        ft.Row([backend,account],wrap=True),token,handwriting,ft.Row([model,language],wrap=True),reviewer,
        ft.Row([source,ft.OutlinedButton('Choose PDF',on_click=lambda _: picker.pick_files(allow_multiple=False,allowed_extensions=['pdf'],file_type=ft.FilePickerFileType.CUSTOM))]),
        ft.ElevatedButton('Create / resume extraction job',on_click=lambda _: launch())])
    page.add(ft.Text('Structured extraction and review',size=28),
        ft.Text('Saved locally. No marking. Teacher access is needed only to fetch uncached layouts.'),
        controls,cancel_button,progress,message,ft.Text('Saved jobs',size=20),jobs_column,details)
    refresh_jobs(); page.update()
