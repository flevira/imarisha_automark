"""Mark saved teacher-approved responses without opening a PDF or invoking OCR."""
from dataclasses import asdict
import hashlib
import json
import time
from decimal import Decimal, localcontext

from core.marking import ENGINE, mark_question
from core.manifest_layout import validate_manifest


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def response_snapshot(repository, job):
    return [dict(response=asdict(row['response']),review=row['review']) for row in repository.responses(job)]


def score_sum(results):
    with localcontext() as context:
        context.prec = 512
        return str(sum((Decimal(r['score']) for r in results if r['score'] is not None),Decimal(0)))


def mark_job(repository, job, resolver, reviewer, policy=None, assistant=None):
    """Use manifest keys with the user-agreed partial-credit fallback.

    {version:1, structured:'all-or-nothing', written:{questionId:rule}}
    Per-question manifest scoringRules take precedence. Existing placeholder
    The legacy ASSESSMENT_POLICY_REQUIRED placeholder uses the agreed fallback.
    """
    if not reviewer.strip():
        raise ValueError('Supply the teacher responsible for this marking run')
    if policy is not None and (not isinstance(policy, dict) or policy.get('version') != 1
            or not set(policy) <= {'version','structured','written','writtenDefault','writtenContext'}
            or not isinstance(policy.get('written', {}), dict)):
        raise ValueError('Unsupported teacher scoring policy')
    started = time.perf_counter()
    with repository.lock(job):
        info = repository.job(job)
        if info['scope'] != resolver.scope:
            raise ValueError('Use the same backend and staff account as extraction')
        if info['status'] != 'complete':
            raise ValueError('Complete extraction review before marking')
        snapshot = response_snapshot(repository, job)
        rows = repository.responses(job)
        manifests, groups = {}, {}
        for row in rows:
            raw, review = row['response'], row['review']
            if not review:
                raise ValueError('Review every response before marking')
            ref = raw.layout
            key = (ref.assessment_type, ref.assessment_id, ref.layout_id, ref.version, ref.assessment_group_id)
            if key not in manifests:
                m = validate_manifest(resolver.resolve(ref.layout_id,ref.assessment_type,ref.assessment_id))
                if (m['assessmentType'],m['assessmentId'],m['layoutId'],m['version'],m['assessmentGroupId']) != key:
                    raise ValueError('Marking manifest does not match the extracted layout')
                manifests[key] = m
            if review['action'] == 'exclude':
                continue
            group = groups.setdefault((key,review['student']), {})
            row_key = (raw.question_id, raw.layout_page_number, raw.item_number, raw.part_number)
            page=manifests[key]['pages'][raw.layout_page_number-1]
            source_rows=page['rows']+page.get('writtenRegions',[])
            if not any((r['questionId'],r.get('item'),r.get('order'),r['kind'],r['number']) ==
                       (raw.question_id,raw.item_number,raw.part_number,raw.response_type.value,raw.display_number)
                       for r in source_rows):
                raise ValueError('Saved response does not match manifest geometry')
            if row_key in group:
                raise ValueError('Duplicate approved response')
            group[row_key] = dict(approved=True, selected=json.loads(review['selected']),
                                 text=review['raw_text'] or '', response_id=raw.response_id,
                                 response_revision=raw.revision, review_sequence=review['sequence'])
        contexts=(policy or {}).get('writtenContext',{})
        if not isinstance(contexts,dict):raise ValueError('Written context must be keyed by layout ID')
        known={m['layoutId']:m for m in manifests.values()}
        from core.ai_marking import validate_criteria
        for layout_id,entries in contexts.items():
            if layout_id not in known or not isinstance(entries,dict):raise ValueError('Unknown written-context layout')
            qs={q['questionId']:q for q in known[layout_id].get('questions',[])}
            for qid,context in entries.items():
                if (qid not in qs or qs[qid]['responseType']!='WRITTEN' or not isinstance(context,dict)
                        or not set(context)<={'questionText','scoringCriteria'}):
                    raise ValueError('Invalid written question context')
                if not isinstance(context.get('questionText'),str) or not context['questionText'].strip():
                    raise ValueError('Supply the original question wording for written criteria')
                validate_criteria(dict(qs[qid],**context))
        results, totals, ai_cache = [], [], {}
        for (key, student), group in groups.items():
            manifest = manifests[key]
            questions = manifest.get('questions', [])
            if (manifest.get('answerSchemaVersion') != 1 or not questions
                    or len({q['questionId'] for q in questions}) != len(questions)):
                raise ValueError('Manifest needs a unique question list and answer schema version 1')
            geometry = {}
            for page in manifest['pages']:
                for entry in page['rows'] + page.get('writtenRegions', []):
                    geometry.setdefault(entry['questionId'], []).append(dict(entry, pageNumber=page['pageNumber']))
            if set(geometry) != {q['questionId'] for q in questions}:
                raise ValueError('Answer metadata does not cover the layout questions')
            student_results = []
            for question in questions:
                question=dict(question,**contexts.get(manifest['layoutId'],{}).get(question['questionId'],{}))
                qid = question['questionId']
                expected = sorted(geometry[qid],key=lambda r:(r.get('order', r.get('item', 0)),r['pageNumber']))
                if any(r['number'] != question['displayNumber'] for r in expected):
                    raise ValueError('Question numbering conflicts with the saved layout')
                answers = [group.get((qid,r['pageNumber'],r.get('item'),r.get('order'))) for r in expected]
                rule = question.get('scoringRules')
                if rule is None and policy:
                    if question['responseType'] == 'WRITTEN':
                        rule = policy.get('written', {}).get(qid,policy.get('writtenDefault'))
                    elif policy.get('structured'):
                        rule = dict(version=1,mode=policy['structured'])
                if rule is None and isinstance(manifest.get('scoringPolicy'), dict):
                    rule = manifest['scoringPolicy']
                if rule is None and manifest.get('scoringPolicy') in (None,'ASSESSMENT_POLICY_REQUIRED'):
                    rule=(dict(version=1,mode='text-exact',caseSensitive=False,collapseWhitespace=True)
                          if question['responseType']=='WRITTEN' else dict(version=1,mode='partial'))
                result = mark_question(question,expected,answers,rule)
                if result['reason']=='unlisted-wording-needs-review':
                    from core.answer_similarity import suggest_partial_mark
                    result['similarity_estimate']=suggest_partial_mark(question,'\n'.join(r['text'] for r in answers))
                if assistant and result['reason']=='unlisted-wording-needs-review':
                    text='\n'.join(r['text'] for r in answers)
                    if '[illegible]' not in text.lower():
                        ai_key=digest(dict(question=question,text=text,rule=rule))
                        if ai_key in ai_cache:
                            assistance=dict(ai_cache[ai_key],seconds=0,reused_within_run=True)
                        else:
                            try:
                                assistance=assistant.assess(question,text)
                            except Exception:
                                assistance=dict(status='review',score=None,reason='ai-unavailable',seconds=0)
                            ai_cache[ai_key]=assistance
                        result['ai']=assistance
                        if assistance.get('status')=='scored':
                            from core.marking import decimal
                            try:
                                score=decimal(assistance.get('score'))
                                if not 0<=score<=decimal(question['maxMarks']):raise ValueError()
                                result.update(score=str(score),status='scored',reason=assistance['reason'],engine='local-ai-v1')
                            except (ValueError,KeyError):
                                result['reason']='ai-invalid-score'
                        else:
                            result['reason']=assistance.get('reason','ai-unresolved')
                result.update(student_id=student,layout_id=key[2],layout_version=key[3],
                              assessment_type=key[0],assessment_id=key[1],assessment_group_id=key[4],
                              manifest_digest=digest(manifest),rule=rule,
                              responses=[r for r in answers if r is not None])
                results.append(result); student_results.append(result)
            resolved = all(r['status'] == 'scored' for r in student_results)
            totals.append(dict(student_id=student,layout_id=key[2],assessment_type=key[0],assessment_group_id=key[4],
                               status='scored' if resolved else 'review',
                               score=score_sum(student_results) if resolved else None,
                               resolved_subtotal=score_sum(student_results),
                               review_count=sum(r['status'] != 'scored' for r in student_results)))
        result = dict(engine=ENGINE,reviewer=reviewer.strip(),policy=policy,
                      response_digest=digest(snapshot),response_snapshot=snapshot,
                      manifests=[dict(digest=digest(m),document=m) for m in manifests.values()],
                      results=results,totals=totals,document_issues=repository.issues(job),
                      excluded_responses=sum(r['review']['action']=='exclude' for r in rows),
                      ai_seconds=sum(r.get('ai',{}).get('seconds',0) for r in results),
                      marking_seconds=time.perf_counter()-started,review_seconds=None)
        sequence = repository.save_marking_run(job,result)
        return dict(sequence=sequence,**result)
