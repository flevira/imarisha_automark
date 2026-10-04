"""Durable M4 jobs and append-only teacher decisions, separate from legacy rows."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

import fitz

from core.contracts import ExtractedResponse


class JobRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS teacher_examples(
              id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, text TEXT NOT NULL,
              score TEXT NOT NULL, reviewer TEXT NOT NULL, provenance TEXT NOT NULL,
              source_job TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS structured_marking_runs(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL,
              result TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS structured_transcriptions(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, response TEXT NOT NULL,
              result TEXT NOT NULL, language TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS structured_sources(digest TEXT PRIMARY KEY, data BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS structured_jobs(
              id TEXT PRIMARY KEY, digest TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending', total INTEGER NOT NULL,
              created TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(digest,scope));
            CREATE TABLE IF NOT EXISTS structured_pages(
              job TEXT NOT NULL, number INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
              reason TEXT, image BLOB, metadata TEXT, preparation_seconds REAL DEFAULT 0,
              extraction_seconds REAL DEFAULT 0, PRIMARY KEY(job,number));
            CREATE TABLE IF NOT EXISTS structured_responses(
              job TEXT NOT NULL, id TEXT NOT NULL, page INTEGER NOT NULL, raw TEXT NOT NULL,
              options TEXT NOT NULL, selection_mode TEXT NOT NULL, PRIMARY KEY(job,id));
            CREATE TABLE IF NOT EXISTS structured_crops(
              job TEXT NOT NULL, ref TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY(job,ref));
            CREATE TABLE IF NOT EXISTS structured_reviews(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, response TEXT NOT NULL,
              action TEXT NOT NULL, student TEXT, selected TEXT NOT NULL, reviewer TEXT NOT NULL,
              note TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS structured_issues(
              job TEXT NOT NULL, key TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY(job,key));
            CREATE TABLE IF NOT EXISTS structured_issue_reviews(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, issue TEXT NOT NULL,
              reviewer TEXT NOT NULL, note TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            ''')
            columns={r['name'] for r in db.execute('PRAGMA table_info(structured_reviews)')}
            if 'raw_text' not in columns:
                db.execute('ALTER TABLE structured_reviews ADD COLUMN raw_text TEXT')
        if os.name != 'nt':
            self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def lock(self, job):
        # OS releases the lock after process death, so interrupted jobs can resume.
        filename = self.path.parent / (hashlib.sha256((str(self.path.resolve())+job).encode()).hexdigest()+'.joblock')
        with filename.open('a+b') as stream:
            if os.name == 'nt':
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b'0'); stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise ValueError('This job is already running') from None
            else:
                import fcntl
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError('This job is already running') from None
            try:
                yield
            finally:
                if os.name == 'nt':
                    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def create(self, source: Path, scope: str) -> str:
        if not scope:
            raise ValueError('An account/backend scope is required')
        source = Path(source)
        if source.stat().st_size > 200*1024*1024:
            raise ValueError('Upload limit is 200 MiB per job')
        data = source.read_bytes()
        from core.source_document import open_source
        with open_source(data) as doc:
            total = len(doc)
            if doc.needs_pass or not 1 <= total <= 200:
                raise ValueError('Use an unlocked PDF or supported image containing 1–200 pages')
        digest = hashlib.sha256(data).hexdigest()
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO structured_sources VALUES (?,?)', (digest,data))
            db.execute('INSERT OR IGNORE INTO structured_jobs(id,digest,scope,name,total) VALUES (?,?,?,?,?)',
                       (uuid.uuid4().hex,digest,scope,source.name,total))
            job = db.execute('SELECT id FROM structured_jobs WHERE digest=? AND scope=?',(digest,scope)).fetchone()[0]
            db.executemany('INSERT OR IGNORE INTO structured_pages(job,number) VALUES (?,?)',((job,n) for n in range(1,total+1)))
        return job

    def jobs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM structured_jobs ORDER BY created DESC,id')]

    def job(self, job):
        with self.connect() as db:
            row = db.execute('SELECT * FROM structured_jobs WHERE id=?',(job,)).fetchone()
            if row is None:
                raise ValueError('Unknown extraction job')
            result = dict(row)
            result['pages'] = [dict(r) for r in db.execute('SELECT number,status,reason,metadata,preparation_seconds,extraction_seconds FROM structured_pages WHERE job=? ORDER BY number',(job,))]
            return result

    def source(self, job):
        with self.connect() as db:
            return db.execute('SELECT data FROM structured_sources JOIN structured_jobs USING(digest) WHERE id=?',(job,)).fetchone()[0]

    def status(self, job, status):
        with self.connect() as db:
            db.execute('UPDATE structured_jobs SET status=? WHERE id=?',(status,job))

    def save_page(self, job, number, status, reason, image, metadata, preparation_seconds, extraction_seconds, result=None, geometry=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT status FROM structured_pages WHERE job=? AND number=?',(job,number)).fetchone()
            if old[0] == 'extracted':
                return  # Completed pages and their raw responses are immutable on retry.
            if result:
                rows=list(geometry['rows'])+[dict(bubbles=[],selectionMode='text') for r in sorted(geometry.get('writtenRegions',[]),key=lambda r:r['order'])]
                if len(result.responses) != len(rows):
                    raise ValueError('Every manifest row must have a response record')
                for response,row in zip(result.responses,rows):
                    db.execute('INSERT INTO structured_responses VALUES (?,?,?,?,?,?)',
                        (job,response.response_id,number,response.to_json(),json.dumps([b['optionId'] for b in row['bubbles']]),row['selectionMode']))
                db.executemany('INSERT OR IGNORE INTO structured_crops VALUES (?,?,?)',((job,ref,data) for ref,data in result.crops.items()))
            db.execute('UPDATE structured_pages SET status=?,reason=?,image=?,metadata=?,preparation_seconds=?,extraction_seconds=? WHERE job=? AND number=?',
                (status,reason,image,json.dumps(metadata),preparation_seconds,extraction_seconds,job,number))

    def responses(self, job, page_number=None):
        with self.connect() as db:
            sql = 'SELECT * FROM structured_responses WHERE job=?'
            params = (job,)
            if page_number is not None:
                sql += ' AND page=?'
                params += (page_number,)
            rows = [dict(r) for r in db.execute(sql+' ORDER BY page,rowid',params)]
            for row in rows:
                row['response'] = ExtractedResponse.from_json(row['raw'])
                row['options'] = json.loads(row['options'])
                review = db.execute('SELECT * FROM structured_reviews WHERE job=? AND response=? ORDER BY sequence DESC LIMIT 1',(job,row['id'])).fetchone()
                row['review'] = dict(review) if review else None
                draft=db.execute('SELECT result,language FROM structured_transcriptions WHERE job=? AND response=? ORDER BY sequence DESC LIMIT 1',(job,row['id'])).fetchone()
                row['transcription'] = dict(json.loads(draft['result']),language=draft['language']) if draft else None
            return rows

    def save_transcription(self, job, response_id, result, language):
        from dataclasses import asdict
        with self.connect() as db:
            db.execute('INSERT INTO structured_transcriptions(job,response,result,language) VALUES (?,?,?,?)',
                       (job,response_id,json.dumps(asdict(result)),language))

    def save_example(self, scope, text, score, reviewer, source_job=None):
        from core.marking import decimal
        if not scope or not text.strip() or not reviewer.strip():
            raise ValueError('Example needs question, answer and teacher')
        with self.connect() as db:
            db.execute('INSERT INTO teacher_examples(scope,text,score,reviewer,provenance,source_job) VALUES(?,?,?,?,?,?)',
                (scope,text,str(decimal(score)),reviewer.strip(),'teacher-score',source_job))

    def examples(self, scope):
        with self.connect() as db:
            rows=[dict(r,approved=True) for r in db.execute(
                'SELECT * FROM teacher_examples WHERE scope=? ORDER BY id DESC',(scope,))]
        # Latest explicit correction to the same text replaces it for future retrieval.
        from core.teacher_grounding import normalize
        latest={}
        for row in rows:latest.setdefault(normalize(row['text']),row)
        return list(latest.values())

    def save_marking_run(self, job, result, example=None):
        """Caller holds the job lock; runs are append-only and never extraction data."""
        with self.connect() as db:
            sequence=db.execute('INSERT INTO structured_marking_runs(job,result) VALUES (?,?)',
                              (job,json.dumps(result,allow_nan=False))).lastrowid
            if example:
                db.execute('INSERT INTO teacher_examples(scope,text,score,reviewer,provenance,source_job) VALUES(?,?,?,?,?,?)',
                    (example['scope'],example['text'],example['score'],example['reviewer'],'teacher-score',job))
            return sequence

    def marking_runs(self, job):
        with self.connect() as db:
            return [dict(sequence=r['sequence'],created=r['created'],**json.loads(r['result']))
                    for r in db.execute('SELECT * FROM structured_marking_runs WHERE job=? ORDER BY sequence DESC',(job,))]

    def crop(self, job, ref):
        with self.connect() as db:
            return db.execute('SELECT data FROM structured_crops WHERE job=? AND ref=?',(job,ref)).fetchone()[0]

    def review_history(self, job, response_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM structured_reviews WHERE job=? AND response=? ORDER BY sequence',(job,response_id))]

    def page_image(self, job, number):
        with self.connect() as db:
            return db.execute('SELECT image FROM structured_pages WHERE job=? AND number=?',(job,number)).fetchone()[0]

    def review(self, job, response_id, *, action, student, selected, reviewer, note='', expected_sequence=None, raw_text=None):
        if action not in ('approve','exclude') or not reviewer.strip():
            raise ValueError('Choose approve/exclude and supply a reviewer name')
        with self.lock(job), self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            state = db.execute('SELECT status FROM structured_jobs WHERE id=?',(job,)).fetchone()
            if state is None or state[0] not in ('review','complete'):
                raise ValueError('Resume the job to finish its document checks before review')
            row = db.execute('SELECT * FROM structured_responses WHERE job=? AND id=?',(job,response_id)).fetchone()
            if row is None:
                raise ValueError('Unknown response')
            latest = db.execute('SELECT MAX(sequence) FROM structured_reviews WHERE job=? AND response=?',(job,response_id)).fetchone()[0]
            if latest != expected_sequence:
                raise ValueError('This response changed; reload it before saving')
            raw = ExtractedResponse.from_json(row['raw'])
            if row['selection_mode']=='text':
                if raw_text is None:
                    raw_text = raw.raw_text or ''
                if not isinstance(raw_text,str) or len(raw_text)>20000 or selected:
                    raise ValueError('Written responses need text, not option selections')
            elif raw_text is not None:
                raise ValueError('Bubble responses cannot contain written text')
            if action == 'approve':
                meta = json.loads(db.execute('SELECT metadata FROM structured_pages WHERE job=? AND number=?',(job,row['page'])).fetchone()[0])
                if meta.get('identity_digit_count') and len(student or '') != meta['identity_digit_count']:
                    raise ValueError('Shared sheets require exactly five student digits')
                if not student or not student.isascii() or not student.isdigit():
                    raise ValueError('Enter the complete numeric student ID, including leading zeros')
                if len(selected)!=len(set(selected)) or not set(selected)<=set(json.loads(row['options'])):
                    raise ValueError('Choose only listed options, without duplicates')
                if row['selection_mode']=='single' and len(selected)>1:
                    raise ValueError('This row permits at most one answer; blank must be explicit')
            db.execute('INSERT INTO structured_reviews(job,response,action,student,selected,reviewer,note,raw_text) VALUES (?,?,?,?,?,?,?,?)',
                (job,response_id,action,student,json.dumps(selected),reviewer.strip(),note.strip(),raw_text))
            db.execute("UPDATE structured_jobs SET status='review' WHERE id=?",(job,))

    def approve_all(self, job, reviewer):
        """Approve saved values across the job, preserving prior teacher decisions.

        Missing/unreadable responses are not silently turned into blanks. One
        transaction records the batch; original extraction and drafts are retained.
        """
        if not (reviewer or '').strip():
            raise ValueError('Enter Reviewer name before approving all responses')
        with self.lock(job):
            info=self.job(job)
            if info['status'] not in ('review','complete'):
                raise ValueError('Finish extraction before approving responses')
            metadata={p['number']:json.loads(p['metadata'] or '{}') for p in info['pages']}
            pending=[];skipped=[];retained=0
            for row in self.responses(job):
                if row['review']:
                    retained+=1;continue
                raw=row['response'];student=raw.student_id;selected=list(raw.selected_options);text=None
                reason=None
                digits=metadata[row['page']].get('identity_digit_count')
                if not student or not student.isascii() or not student.isdigit() or (digits and len(student)!=digits):
                    reason='student identity needs review'
                elif row['selection_mode']=='text':
                    draft=row['transcription']
                    text=(draft.get('text') if draft and draft.get('status')!='error' else None) or raw.raw_text
                    if not text and raw.status.value!='blank':reason='written answer needs transcription'
                    text=text or ''
                    if len(text)>20000:reason='written answer is too long'
                elif (not set(selected)<=set(row['options']) or
                      (row['selection_mode']=='single' and len(selected)>1) or
                      (not selected and raw.status.value!='blank')):
                    reason='selection needs review'
                if reason:
                    skipped.append(dict(page=row['page'],question=raw.display_number,reason=reason));continue
                pending.append((job,row['id'],'approve',student,json.dumps(selected),reviewer.strip(),
                                'Bulk approval of saved responses',text))
            with self.connect() as db:
                db.executemany('INSERT INTO structured_reviews(job,response,action,student,selected,reviewer,note,raw_text) VALUES (?,?,?,?,?,?,?,?)',pending)
                if pending:db.execute("UPDATE structured_jobs SET status='review' WHERE id=?",(job,))
            return dict(approved=len(pending),retained=retained,skipped=skipped)

    def set_issues(self, job, issues):
        with self.connect() as db:
            db.execute('DELETE FROM structured_issues WHERE job=?',(job,))
            for issue in issues:
                detail = json.dumps(issue,sort_keys=True)
                key = hashlib.sha256(detail.encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO structured_issues VALUES (?,?,?)',(job,key,detail))

    def issues(self, job):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT i.*, (SELECT note FROM structured_issue_reviews r
              WHERE r.job=i.job AND r.issue=i.key ORDER BY sequence DESC LIMIT 1) resolution
              FROM structured_issues i WHERE i.job=?''',(job,))]

    def resolve_issue(self, job, key, reviewer, note):
        if not reviewer.strip() or not note.strip():
            raise ValueError('A reviewer and resolution note are required')
        with self.lock(job), self.connect() as db:
            if not db.execute('SELECT 1 FROM structured_issues WHERE job=? AND key=?',(job,key)).fetchone():
                raise ValueError('Unknown issue')
            db.execute('INSERT INTO structured_issue_reviews(job,issue,reviewer,note) VALUES (?,?,?,?)',(job,key,reviewer,note))

    def complete(self, job):
        with self.lock(job):
            if self.job(job)['status'] not in ('review','complete'):
                raise ValueError('Resume the job to finish its document checks first')
            if any(p['status']=='pending' for p in self.job(job)['pages']):
                raise ValueError('Resume unfinished pages first')
            unreviewed = [r for r in self.responses(job) if r['review'] is None]
            unresolved = [i for i in self.issues(job) if i['resolution'] is None]
            if unreviewed or unresolved:
                pages = ', '.join(str(p) for p in sorted({r['page'] for r in unreviewed}))
                written_pending=sum(r['selection_mode']=='text' for r in unreviewed)
                raise ValueError(f'Review not complete: {len(unreviewed)} responses still need approval or exclusion'
                                 + (f' on pages {pages}' if pages else '')
                                 + f'; {len(unresolved)} document issues need resolution. Use Review page to check the remaining responses.'
                                 + (f' {written_pending} written responses need review: enter their text manually and approve, or exclude them. AI is optional.' if written_pending else ''))
            identities, seen = {}, set()
            for row in self.responses(job):
                decision, raw = row['review'], row['response']
                if decision['action'] == 'exclude':
                    continue
                student = decision['student']
                if identities.setdefault(row['page'],student) != student:
                    raise ValueError('Approved responses on one page must use the same student ID')
                key = (raw.layout.layout_id,student,raw.question_id,raw.item_number,raw.part_number)
                if key in seen:
                    raise ValueError('Exclude duplicate student/question responses before completing review')
                seen.add(key)
            self.status(job,'complete')


    def delete_saved_data(self, job=None, *, manifest_path=None, legacy_path=None):
        """Delete a selected job or all jobs while refusing active job locks.

        For a full reset, optional manifest/legacy databases are cleared in the same transaction.
        """
        with ExitStack() as locks:
            jobs=self.jobs()
            targets=[r['id'] for r in jobs if job is None or r['id']==job]
            if job is not None and not targets:
                raise ValueError('Saved job no longer exists')
            for target in sorted(targets):
                locks.enter_context(self.lock(target))
            with self.connect() as db:
                attached=[]
                if job is None:
                    for alias,path,table in [('manifest_store',manifest_path,'layout_cache'),('legacy_store',legacy_path,'extractions')]:
                        if path is not None and Path(path).exists():
                            db.execute(f'ATTACH DATABASE ? AS {alias}',(str(Path(path).resolve()),))
                            attached.append((alias,table))
                db.execute('BEGIN IMMEDIATE')
                current={r[0] for r in db.execute('SELECT id FROM structured_jobs')}
                if job is None and current!=set(targets):
                    raise ValueError('Saved jobs changed; refresh and try again')
                for target in targets:
                    for table in ('structured_marking_runs','structured_transcriptions',
                                  'structured_pages','structured_responses','structured_crops',
                                  'structured_reviews','structured_issues','structured_issue_reviews'):
                        db.execute(f'DELETE FROM {table} WHERE job=?',(target,))
                    db.execute('DELETE FROM structured_jobs WHERE id=?',(target,))
                if job is None:
                    # Include any orphaned historical records as part of a full reset.
                    for table in ('structured_marking_runs','structured_transcriptions','structured_pages',
                                  'structured_responses','structured_crops','structured_reviews',
                                  'structured_issues','structured_issue_reviews','structured_sources','teacher_examples'):
                        db.execute(f'DELETE FROM {table}')
                    for alias,table in attached:
                        if db.execute(f"SELECT 1 FROM {alias}.sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
                            db.execute(f'DELETE FROM {alias}.{table}')
                else:
                    db.execute('DELETE FROM structured_sources WHERE digest NOT IN (SELECT digest FROM structured_jobs)')
            # VACUUM cannot run inside the deletion transaction. Do not report
            # a completed deletion as failed just because reclamation is busy.
            compacted=True
            try:
                with self.connect() as db:db.execute('VACUUM')
                if job is None:
                    for path in (manifest_path,legacy_path):
                        if path is not None and Path(path).exists():
                            with sqlite3.connect(path,timeout=30) as db:db.execute('VACUUM')
            except sqlite3.OperationalError:
                compacted=False
            return dict(deleted_jobs=len(targets),compacted=compacted)
