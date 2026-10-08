"""Compare spacing-only and 95/97/99% one-edit gates offline."""
import argparse
from functools import partial
from html import escape
import json
from pathlib import Path
from tools.benchmark_reference_concepts import load_cases,compare,html_report
from tools.fuzzy_marking_trial import candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.labels.read_text())
    labels={(s['student_number'],q):v for s in data['students'] for q,v in s['expected_marks_by_question'].items()}
    cases=load_cases(args.database,labels)
    reports={}
    for name,threshold in [('spacing',None),('edit95',.95),('edit97',.97),('edit99',.99)]:
        report=compare(cases,partial(candidate,threshold=threshold))
        report['version']='fuzzy-offline-v1:'+name
        report['limitations']+=' Similarity is uncalibrated. Spacing and one-character edits can change meaning.'
        reports[name]=report
    args.output.mkdir(parents=True,exist_ok=True)
    for name,report in reports.items():
        for ext,content in [('json',json.dumps(report,indent=2)),('html',html_report(report).replace('Offline derived-reference concept benchmark','Offline fuzzy benchmark: '+name))]:
            with (args.output/(name+'.'+ext)).open('x',encoding='utf-8') as f:f.write(content)
    summary={k:v['summary'] for k,v in reports.items()}
    with (args.output/'index.html').open('x',encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><h1>Offline fuzzy marking comparison</h1><pre>'+escape(json.dumps(summary,indent=2))+'</pre>'+''.join('<p><a href="'+n+'.html">'+n+'</a></p>' for n in reports))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
