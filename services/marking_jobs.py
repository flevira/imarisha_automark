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


def mark_job(repository, job, resolver, reviewer, policy=None, assistant=None, progress=None):
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
                        or not set(context)<={'questionText','scoringCriteria','acceptedConcepts','requiredConceptCount'}):
                    raise ValueError('Invalid written question context')
                if not isinstance(context.get('questionText'),str) or not context['questionText'].strip():
                    raise ValueError('Supply the original question wording for written criteria')
                from core.written_rubric import prepare_rubric
                prepare_rubric(dict(qs[qid],**context))
        previous_runs=repository.marking_runs(job)
        previous_marks={(r['student_id'],r['layout_id'],r['question_id']):r for r in (previous_runs[0]['results'] if previous_runs else []) if r.get('teacher_decision')}
        results, totals, ai_cache = [], [], {}
        total_questions=sum(len(manifests[key]['questions']) for key,student in groups)
        resumed_questions=0;checkpoint_seconds=0
        # Exact input identity prevents old marks surviving response/rubric/example changes.
        run_identity=dict(version='marking-checkpoint-v1-written-v15',snapshot=digest(snapshot),
                          policy=policy,reviewer=reviewer.strip(),engine=ENGINE,
                          assistant=type(assistant).__qualname__ if assistant else None,
                          model=getattr(getattr(assistant,'client',None),'model',None),
                          weights=assistant.checkpoint_identity() if hasattr(assistant,'checkpoint_identity') else None)

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
                if question['responseType']=='WRITTEN':
                    from core.concept_criteria import with_concept_criteria
                    question=with_concept_criteria(question)
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
                    rule=(dict(version=1,mode='text-exact',caseSensitive=False,collapseWhitespace=True,normalizeListFormatting=True)
                          if question['responseType']=='WRITTEN' else dict(version=1,mode='partial'))
                from core.teacher_grounding import question_scope, reference_examples, approved_match, explicit_concept_match, uncertainty
                scope=question_scope(question,rule)
                examples=repository.examples(scope) if question['responseType']=='WRITTEN' else []
                question=dict(question,approvedExamples=examples)
                checkpoint_key=digest(dict(run=run_identity,manifest=manifest,student=student,
                                           question=question,answers=answers,rule=rule,
                                           previous=previous_marks.get((student,key[2],qid))))
                cached=repository.marking_checkpoint(job,checkpoint_key)
                if cached:
                    result=cached['result'];checkpoint_seconds+=cached['seconds'];resumed_questions+=1
                    results.append(result);student_results.append(result)
                    if progress:progress(dict(completed=len(results),total=total_questions,student=student,
                                              question=question['displayNumber'],resumed=resumed_questions))
                    continue
                question_started=time.perf_counter()
                if progress:progress(dict(completed=len(results),total=total_questions,student=student,
                                          question=question['displayNumber'],resumed=resumed_questions))
                result = mark_question(question,expected,answers,rule)
                previous=previous_marks.get((student,key[2],qid))
                if previous and previous.get('example_scope')==scope and previous.get('responses')==[r for r in answers if r is not None]:
                    result.update(score=previous['score'],status='scored',reason='teacher-mark',engine='teacher',teacher_decision=previous['teacher_decision'])
                if result['reason']=='unlisted-wording-needs-review':
                    text='\n'.join(r['text'] for r in answers)
                    grounded=approved_match(text,reference_examples(question)+examples) or explicit_concept_match(question,text)
                    if grounded:
                        result.update(grounded,engine='teacher-grounded-v1')

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
                result['uncertainty']=uncertainty(result)
                result.update(example_scope=scope,scoring_question={k:v for k,v in question.items() if k!='approvedExamples'},student_id=student,layout_id=key[2],layout_version=key[3],
                              assessment_type=key[0],assessment_id=key[1],assessment_group_id=key[4],
                              manifest_digest=digest(manifest),rule=rule,question_text=question.get('questionText'),answer_detail=question.get('answerDetail'),concept_count=question.get('conceptCount'),criteria_source=question.get('criteriaSource'),reference_answers=question.get('referenceAnswers',[]),scoring_criteria=question.get('scoringCriteria',[]),marking_rubric=question.get('markingRubric'),
                              responses=[r for r in answers if r is not None])
                repository.save_marking_checkpoint(job,checkpoint_key,
                    dict(result=result,seconds=time.perf_counter()-question_started))
                results.append(result); student_results.append(result)
                if progress:progress(dict(completed=len(results),total=total_questions,student=student,
                                          question=question['displayNumber'],resumed=resumed_questions))
            resolved = all(r['status'] == 'scored' for r in student_results)
            totals.append(dict(student_id=student,layout_id=key[2],assessment_type=key[0],assessment_group_id=key[4],
                               status='scored' if resolved else 'review',
                               score=score_sum(student_results) if resolved else None,
                               resolved_subtotal=score_sum(student_results),
                               review_count=sum(r['status'] != 'scored' for r in student_results)))
        result = dict(engine=ENGINE,attempt_id=job,reviewer=reviewer.strip(),policy=policy,
                      response_digest=digest(snapshot),response_snapshot=snapshot,
                      manifests=[dict(digest=digest(m),document=m) for m in manifests.values()],
                      results=results,totals=totals,document_issues=repository.issues(job),
                      excluded_responses=sum(r['review']['action']=='exclude' for r in rows),
                      ai_seconds=sum(r.get('ai',{}).get('seconds',0) for r in results),
                      marking_seconds=time.perf_counter()-started+checkpoint_seconds,
                      resumed_questions=resumed_questions,review_seconds=None)
        sequence = repository.save_marking_run(job,result)
        return dict(sequence=sequence,**result)


def apply_teacher_mark(repository,job,sequence,student,layout,qid,score,reviewer,note='', *, save_example=True, review_seconds=None):
    """Create a new immutable result version; never modify the original run."""
    import copy
    from core.marking import decimal
    if not reviewer.strip():raise ValueError('Enter Reviewer name before saving a mark')
    with repository.lock(job):
        if repository.job(job)['status']!='complete':
            raise ValueError('Complete extraction review before saving a mark')
        runs=repository.marking_runs(job)
        if not runs or runs[0]['sequence']!=sequence:raise ValueError('Results changed; reopen the latest run')
        run=copy.deepcopy(runs[0])
        if run['response_digest']!=digest(response_snapshot(repository,job)):
            raise ValueError('Responses changed; complete review and re-mark first')
        result=next((r for r in run['results'] if (r['student_id'],r['layout_id'],r['question_id'])==(student,layout,qid)),None)
        if result is None:raise ValueError('Question result not found')
        value=decimal(score)
        if not 0<=value<=decimal(result['max_marks']) or value.as_tuple().exponent < -2:
            raise ValueError('Enter a mark between zero and the maximum, with up to two decimal places')
        result.update(score=str(value),status='scored',reason='teacher-mark',engine='teacher',uncertainty=None,
                      teacher_decision=dict(reviewer=reviewer.strip(),note=note,previous_score=result['score']))
        for total in run['totals']:
            rr=[r for r in run['results'] if (r['student_id'],r['layout_id'])==(total['student_id'],total['layout_id'])]
            pending=sum(r['status']!='scored' for r in rr)
            total.update(review_count=pending,status='review' if pending else 'scored',
                         resolved_subtotal=score_sum(rr),score=None if pending else score_sum(rr))
        run.pop('sequence',None);run.pop('created',None)
        run.update(parent_run=sequence,reviewer=reviewer.strip(),operation='teacher-mark')
        if review_seconds is not None:
            duration=decimal(review_seconds)
            if duration<0:raise ValueError('Invalid review duration')
            run['review_seconds']=float(decimal(run.get('review_seconds') or 0)+duration)
            run['review_timing_method']='elapsed-review-interface-time'
        example=None
        if save_example and result.get('example_scope') and result.get('scoring_question',{}).get('responseType')=='WRITTEN':
            text='\n'.join(r.get('text','') for r in result['responses'])
            if text.strip():example=dict(scope=result['example_scope'],text=text,score=str(value),reviewer=reviewer.strip())
        return repository.save_marking_run(job,run,example=example)
