"""Read-only saved-crop experiment. Never imported by the live extraction path."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import time
import unicodedata

import cv2
import numpy as np
from core.local_text_ocr import TrOCRBackend, MODEL_ID, split_lines
from tools.handwriting_benchmark import distance


def selected_confidence(ids, log_probs, eos):
    """Scores already follow the selected beam's ancestry; ignore EOS/padding."""
    eos = [eos] if isinstance(eos, int) else (eos or [])
    values = []
    for token, log_prob in zip(ids, log_probs):
        if int(token) in eos:
            return (math.exp(sum(values) / len(values)) if values else 0.,
                    math.exp(min(values)) if values else 0., True)
        values.append(float(log_prob))
    return (math.exp(sum(values) / len(values)) if values else 0.,
            math.exp(min(values)) if values else 0., False)


class RetryBackend(TrOCRBackend):
    def retry_line(self, image):
        from PIL import Image
        with self.lock, self.torch.inference_mode():
            pixels = self.processor(images=Image.fromarray(image).convert('RGB'),
                                    return_tensors='pt').pixel_values.to(self.device)
            output = self.model.generate(pixels, max_new_tokens=128, do_sample=False,
                num_beams=3, num_return_sequences=1, max_time=20.,
                return_dict_in_generate=True, output_scores=True)
            scores = self.model.compute_transition_scores(output.sequences, output.scores,
                beam_indices=output.beam_indices, normalize_logits=True)
            ids = output.sequences[0, -scores.shape[1]:]
            avg, low, done = selected_confidence(ids.tolist(), scores[0].tolist(),
                                                self.model.generation_config.eos_token_id)
            text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
            return text, avg, low, done


def passes(value):
    return bool(value[0].strip()) and value[1] >= .90 and value[2] >= .50 and value[3]


def candidate(d):
    return (d.get('engine') == 'trocr-written-v1' and d.get('status') == 'uncertain'
            and set(d.get('review_reasons', [])) == {'ocr-low-confidence'})


def trial(png, backend):
    start = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return dict(accepted=False, error='invalid-image', lines=[], baseline='', retry='', retry_seconds=0.)
    lines, flags = split_lines(image)
    records = []
    for line in lines:
        t = time.perf_counter()
        baseline = backend.read_line(line)
        base_seconds = time.perf_counter() - t
        value = baseline
        retry_seconds = 0.
        error = None
        # Only one retry for low-confidence lines, never truncation/empty/split failures.
        retried = bool(not flags and baseline[0].strip() and baseline[3]
                       and (baseline[1] < .90 or baseline[2] < .50))
        if retried:
            t = time.perf_counter()
            try:
                value = backend.retry_line(line)
            except Exception as exc:
                error = type(exc).__name__
            retry_seconds = time.perf_counter() - t
        records.append(dict(baseline=baseline, retry=value, retried=retried, error=error,
                            baseline_seconds=base_seconds, retry_seconds=retry_seconds))
    accepted = bool(records and not flags and all(not r['error'] and passes(r['retry']) for r in records))
    return dict(accepted=accepted, flags=flags, lines=records,
                baseline='\n'.join(r['baseline'][0] for r in records),
                retry='\n'.join(r['retry'][0] for r in records),
                retry_seconds=sum(r['retry_seconds'] for r in records),
                baseline_seconds=sum(r['baseline_seconds'] for r in records),
                wall_seconds=time.perf_counter()-start)


def normalize(s):
    return ' '.join(unicodedata.normalize('NFC', s or '').casefold().split())


def words(s):
    return re.findall(r'\w+', normalize(s))


def metrics(rows, key, lexical=False):
    ce = we = nc = nw = exact = 0
    for r in rows:
        a, b = normalize(r['expected']), normalize(r[key])
        if lexical:
            a, b = ' '.join(words(a)), ' '.join(words(b))
        ce += distance(a, b); we += distance(a.split(), b.split())
        nc += len(a); nw += len(a.split()); exact += a == b
    return dict(n=len(rows), exact=exact, CER=ce/nc if nc else None, WER=we/nw if nw else None)


def timing(values):
    return dict(total=sum(values), mean=statistics.mean(values) if values else 0,
                median=statistics.median(values) if values else 0,
                p95=float(np.percentile(values, 95)) if values else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    # Freeze a consistent read snapshot without writing or holding up WAL writers.
    db.execute('BEGIN')
    jobs = {r['id']:dict(r) for r in db.execute('select * from structured_jobs')}
    history = {}
    for r in db.execute('select * from structured_transcriptions order by sequence'):
        history.setdefault((r['job'],r['response']), []).append(dict(r, result=json.loads(r['result'])))
    cases = []
    for r in db.execute('select * from structured_responses'):
        h = history.get((r['job'],r['id']), [])
        ordinary = [t for t in h if t['result']['engine']=='trocr-written-v1']
        if not ordinary or not candidate(ordinary[-1]['result']) or ordinary[-1]['language']!='en':
            continue
        ai = [t for t in h if t['result']['engine'].startswith('ollama:')]
        if not ai:
            raise ValueError('Candidate lacks saved AI comparison')
        job = jobs[r['job']]; source = args.dataset/'student_scripts'/job['name']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=job['digest']:
            raise ValueError('Source dataset does not match saved job')
        raw = json.loads(r['raw'])
        if len(raw['regions'])!=1:
            raise ValueError('Benchmark expects one region per saved response')
        png = db.execute('select data from structured_crops where job=? and ref=?',
                         (r['job'],raw['regions'][0]['crop_ref'])).fetchone()[0]
        cases.append(dict(job=r['job'],response=r['id'],student=raw['student_id'],
            question=str(raw['display_number']),image=png,original=ordinary[-1]['result']['text'],
            gemma=ai[-1]['result'].get('text') or '',gemma_seconds=ai[-1]['result']['seconds']))
    db.rollback(); db.close()
    if not cases:
        raise ValueError('No eligible low-confidence English answers')
    t = time.perf_counter(); backend = RetryBackend(os.environ.get('AUTOMARK_TROCR_MODEL', MODEL_ID))
    load_seconds = time.perf_counter()-t
    rows=[]
    for i,c in enumerate(cases):
        prediction=trial(c['image'],backend)
        rows.append({k:v for k,v in c.items() if k!='image'} | prediction)
        print(f'{i+1}/{len(cases)} accepted={prediction["accepted"]} retry={prediction["retry_seconds"]:.2f}s',flush=True)
        # Save progress separately; no prediction is written to the app database.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(load_seconds=load_seconds,device=backend.device,rows=rows),indent=2))
    # Labels are loaded only after model inference is over.
    labels={str(s['student_number']):s for s in json.loads((args.dataset/'ground_truth.json').read_text())['students']}
    for r in rows:
        r['expected']=labels[r['student']]['responses'][r['question']]
        r['simulated_final']=r['retry'] if r['accepted'] else r['gemma']
    keys=('original','baseline','retry','gemma','simulated_final')
    accepted=[r for r in rows if r['accepted']]
    summary=dict(candidates=len(rows),accepted=len(accepted),baseline_reproduced=sum(r['original']==r['baseline'] for r in rows),
        load_seconds=load_seconds,device=backend.device,
        strict={k:metrics(rows,k) for k in keys},lexical={k:metrics(rows,k,True) for k in keys},
        baseline_timing=timing([r['baseline_seconds'] for r in rows]),
        retry_timing=timing([r['retry_seconds'] for r in rows]),
        saved_gemma_timing=timing([r['gemma_seconds'] for r in rows]),
        estimated_net_seconds_saved=sum(r['gemma_seconds'] for r in accepted)-sum(r['retry_seconds'] for r in rows),
        corrected=sum(words(r['original'])!=words(r['expected']) and words(r['retry'])==words(r['expected']) for r in rows),
        regressed=sum(words(r['original'])==words(r['expected']) and words(r['retry'])!=words(r['expected']) for r in rows),
        accepted_mismatches=[dict(student=r['student'],question=r['question'],expected=r['expected'],original=r['original'],retry=r['retry'],gemma=r['gemma']) for r in accepted if words(r['retry'])!=words(r['expected'])])
    args.output.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
