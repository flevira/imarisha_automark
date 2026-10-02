"""Synthetic M5 blank, typed and image-answer papers; no HTR accuracy claim."""
import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

from tools.manifest_fixtures import FIXTURES,expected,render,rasterize


def generate(output,shared=False,continuation=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((FIXTURES/f'written-v{7 if shared else 6}.json').read_text())
    if continuation:
        region=manifest['pages'][-1]['writtenRegions'][-1]
        region.update(regionId='2-2',questionId='2',number=2,order=2)
    render(manifest,expected(manifest),output/'blank.pdf',shared)
    ink=np.full((80,200,3),255,np.uint8)
    cv2.polylines(ink,[np.array([(20,30),(30,20),(30,65)])],False,(0,0,0),3)
    cv2.polylines(ink,[np.array([(65,30),(75,20),(95,25),(95,40),(65,65),(100,65)])],False,(0,0,0),3)
    with fitz.open(output/'blank.pdf') as doc:
        for spec in manifest['pages']:
            for region in spec.get('writtenRegions',[]):
                x,y,w,h=(region[k] for k in ('x','y','width','height'))
                if region['questionId']=='2':
                    doc[spec['pageNumber']-1].insert_text((x+10,y+20),'First line\nSecond line' if region['order']==1 else 'Continued response',fontsize=12)
                elif region['questionId']=='4':
                    doc[spec['pageNumber']-1].insert_image(fitz.Rect(x+10,y+10,x+110,y+50),stream=cv2.imencode('.png',ink)[1].tobytes())
        doc.save(output/'answers.pdf')
    rasterize(output/'answers.pdf',output/'scan.pdf')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/m5-written'));p.add_argument('--shared',action='store_true')
    args=p.parse_args();generate(args.output,args.shared)
