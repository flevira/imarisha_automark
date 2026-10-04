"""Teacher-confirmed groups; no automatic fuzzy scoring or replacement of overrides."""
from difflib import SequenceMatcher
from core.teacher_grounding import normalize
from core.marking import decimal
from services.marking_jobs import digest, response_snapshot, apply_teacher_mark


def review_groups(repository,jobs):
    groups={}
    for job in sorted(set(jobs)):
        runs=repository.marking_runs(job)
        if not runs or repository.job(job)['status']!='complete':continue
        run=runs[0]
        if run['response_digest']!=digest(response_snapshot(repository,job)):continue
        for r in run['results']:
            if r.get('scoring_question',{}).get('responseType')!='WRITTEN':continue
            # Explicit individual marks are never included in group changes.
            if r.get('teacher_decision'):continue
            text='\n'.join(x.get('text','') for x in r['responses'])
            if not text.strip():continue
            key=(r['example_scope'],normalize(text))
            group=groups.setdefault(key,dict(scope=key[0],text=text,question=r['question_text'],
                number=r['display_number'],maximum=r['max_marks'],reference=r['reference_answers'],members=[]))
            group['members'].append(dict(job=job,sequence=run['sequence'],student=r['student_id'],
                layout=r['layout_id'],qid=r['question_id'],text=text,name=repository.job(job)['name'],score=r['score']))
    result=list(groups.values())
    for i,g in enumerate(result):
        g['id']=str(i)
        g['suggestions']=[str(j) for j,h in enumerate(result) if j!=i and h['scope']==g['scope']
            and SequenceMatcher(None,normalize(g['text']),normalize(h['text'])).ratio()>=.75]
    return sorted(result,key=lambda g:(g['number'],-len(g['members']),g['id']))


def mark_group(repository,members,score,reviewer,note='',review_seconds=None):
    """Validate the displayed snapshots before applying; return per-member outcomes."""
    if not reviewer.strip() or not members:raise ValueError('Select responses and enter Reviewer name')
    value=decimal(score)
    scope=None
    for member in members:
        runs=repository.marking_runs(member['job'])
        if not runs or runs[0]['sequence']!=member['sequence']:raise ValueError('Results changed; refresh groups')
        run=runs[0]
        if run['response_digest']!=digest(response_snapshot(repository,member['job'])):raise ValueError('Responses changed; re-mark first')
        r=next(r for r in run['results'] if (r['student_id'],r['layout_id'],r['question_id'])==
               (member['student'],member['layout'],member['qid']))
        if r.get('teacher_decision'):raise ValueError('Individual teacher override must be preserved')
        if scope is not None and scope!=r['example_scope']:raise ValueError('Groups must share the same question and rubric')
        scope=r['example_scope']
        if not 0<=value<=decimal(r['max_marks']) or value.as_tuple().exponent < -2:raise ValueError('Invalid group mark')
    outcomes=[]
    versions={m['job']:m['sequence'] for m in members}
    for member in members:
        try:
            seq=versions[member['job']]
            saved=apply_teacher_mark(repository,member['job'],seq,member['student'],member['layout'],member['qid'],
                value,reviewer,note or 'Teacher-confirmed answer group',
                review_seconds=review_seconds/len(members) if review_seconds is not None else None)
            versions[member['job']]=saved
            outcomes.append(dict(job=member['job'],sequence=saved,status='saved'))
        except Exception as exc:outcomes.append(dict(job=member['job'],status='blocked',reason=str(exc)))
    return outcomes
