"""One replaceable HTML report per job, available before or after marking."""
import html
import json
import os
import tempfile
from pathlib import Path
from services.marking_jobs import digest, response_snapshot
from services.marking_export import correct_answer, matching_label


def export_job_report(repository, job, directory, resolver=None):
    with repository.lock(job):
        info=repository.job(job)
        rows=repository.responses(job)
        runs=repository.marking_runs(job)
        latest=runs[0] if runs else None
        stale=bool(latest and latest['response_digest']!=digest(response_snapshot(repository,job)))
        manifests=list(latest.get('manifests',[])) if latest else []
        if resolver:
            known={m['document']['layoutId'] for m in manifests}
            for row in rows:
                layout=row['response'].layout
                if layout.layout_id in known:continue
                try:document=resolver.resolve(layout.layout_id,layout.assessment_type,layout.assessment_id)
                except (ValueError,RuntimeError):continue
                manifests.append({'document':document});known.add(layout.layout_id)
        esc=lambda v:html.escape(str(v if v is not None else ''))
        grouped={}
        for row in rows:
            raw=row['response'];review=row['review'];draft=row['transcription']
            student=review['student'] if review else raw.student_id
            key=(student,raw.layout.layout_id,raw.question_id)
            group=grouped.setdefault(key,dict(number=raw.display_number,answers=[],reviews=[],pages=set()))
            text=(review['raw_text'] if review else raw.raw_text or (draft or {}).get('text')) if row['selection_mode']=='text' else ', '.join(json.loads(review['selected']) if review else raw.selected_options)
            prefix=matching_label(raw.item_number)+'=' if raw.item_number is not None else ''
            group['answers'].append(prefix+(text or '(blank / no text saved)'))
            group['reviews'].append(review['action'] if review else 'Not reviewed');group['pages'].add(row['page'])
        questions={(m['document']['layoutId'],str(q['questionId'])):q for m in manifests for q in m['document'].get('questions',[])}
        results={(r['student_id'],r['layout_id'],r['question_id']):r for r in latest['results']} if latest else {}
        cells=[]
        for key,g in grouped.items():
            result=results.get(key)
            reference=correct_answer({'manifests':manifests},dict(student_id=key[0],layout_id=key[1],question_id=key[2]))
            score='Outdated — re-mark' if stale else ('Not marked yet' if not result else result['score'] if result['score'] is not None else 'Pending review')
            explanation=result.get('reason','') if result else ''
            if result and result.get('teacher_decision'):explanation+=': '+result['teacher_decision'].get('note','')
            elif result and result.get('ai'):explanation+=': '+(result['ai'].get('explanation') or (result['ai'].get('proposal') or {}).get('explanation',''))
            calculation=(result or {}).get('ai',{}).get('calculation')
            if calculation:explanation+=f" Calculated credit: {calculation['credited_concepts']}/{calculation['required_concepts']} concepts; {calculation['score']} marks."
            u=(result or {}).get('uncertainty')
            if u:
                explanation+=' Review category: '+u['category']+'.'
                if u.get('score_range'):
                    explanation+=f" Provisional range: {u['score_range']['lower']}–{u['score_range']['upper']} marks (not confidence)."
                if u.get('disputed_concepts'):explanation+=' Disputed: '+'; '.join(u['disputed_concepts'])
            context=questions.get((key[1],str(key[2])),{})
            wording=(result or {}).get('question_text') or context.get('questionText','')
            detail=(result or {}).get('answer_detail') or context.get('answerDetail','')
            values=[key[0],g['number'],', '.join(map(str,sorted(g['pages']))),wording,detail,reference,'\n'.join(g['answers']),', '.join(sorted(set(g['reviews']))),score,result['max_marks'] if result else questions.get((key[1],str(key[2])),{}).get('maxMarks',''),explanation]
            cells.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
        summaries=[]
        if latest and not stale:
            for total in latest['totals']:
                summaries.append(f"Student {esc(total['student_id'])}: {esc(total['score'] if total['score'] is not None else 'Pending')} · resolved subtotal {esc(total['resolved_subtotal'])} · unresolved questions {total['review_count']}")
        status='Marking outdated; re-mark current responses.' if stale else 'Not marked yet.' if not latest else 'Latest saved marking results; pending scores are not zero.'
        headers=['Student','Question','Pages','Question wording','Answer details / context','Correct / reference answer','Extracted answer (latest reviewed value)','Extraction review','Mark','Maximum','Marking explanation']
        document='<!doctype html><html><meta charset="utf-8"><title>AutoMark results</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbc5ce;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf2f7}thead{display:table-header-group}</style>'
        document+=f"<h1>{esc(info['name'])}</h1><p>Attempt ID: {esc(job)}</p><p>Extraction: {sum(p['status']=='extracted' for p in info['pages'])}/{info['total']} pages. {len(rows)} response records.</p><p>{status}</p>"
        document+='<p>'+'</p><p>'.join(summaries)+'</p><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></html>'
    destination=Path(directory).expanduser();destination.mkdir(parents=True,exist_ok=True)
    import hashlib
    report=destination/f'automark-results-{hashlib.sha256(job.encode()).hexdigest()[:16]}.html'
    fd,tmp=tempfile.mkstemp(dir=destination,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:stream.write(document)
        os.replace(tmp,report)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return report


def export_batch_report(repository,jobs,directory,resolver_factory):
    """Replace one combined report for the same explicit set of attempts."""
    import hashlib
    jobs=sorted(set(jobs))
    if not jobs:raise ValueError('Select at least one script')
    sections=[]
    with tempfile.TemporaryDirectory() as temp:
        for index,job in enumerate(jobs,1):
            report=export_job_report(repository,job,temp,resolver_factory(job))
            content=report.read_text(encoding='utf-8')
            content=content[content.index('<h1>'):].removesuffix('</html>')
            sections.append((repository.job(job)['name'],f'<section id="script-{index}">{content}</section>'))
    nav='<ul>'+''.join(f'<li><a href="#script-{i}">{html.escape(name)}</a></li>' for i,(name,_) in enumerate(sections,1))+'</ul>'
    document='<!doctype html><html><meta charset="utf-8"><title>AutoMark batch results</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbc5ce;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf2f7}section{break-before:page;margin-top:48px}</style><h1>Batch extraction and marking results</h1>'+nav+''.join(s for _,s in sections)+'</html>'
    destination=Path(directory).expanduser();destination.mkdir(parents=True,exist_ok=True)
    report=destination/('automark-batch-'+hashlib.sha256('|'.join(jobs).encode()).hexdigest()[:16]+'.html')
    fd,temp=tempfile.mkstemp(dir=destination,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:stream.write(document)
        os.replace(temp,report)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return report
