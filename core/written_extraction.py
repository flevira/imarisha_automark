"""M5 region capture and visible PDF text; handwriting is never guessed here."""
import hashlib

import cv2
import fitz
import numpy as np

from core.contracts import ExtractedResponse, ReadStatus, ResponseType, SourceRegion
from core.page_alignment import normalized_preview, transform_points


def has_local_ink(dark):
    """Keep small written answers out of the blank fast path, regardless of box size."""
    count,_,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8)
    height,width=dark.shape
    for x,y,w,h,area in stats[1:]:
        # Ignore isolated scan specks and thin region-border remnants.
        if area<8 or w<2 or h<3:continue
        if (w>=width*.95 and h<=3) or (h>=height*.95 and w<=3):continue
        return True
    return False


def capture_written(page, result, pdf_page=None):
    if not page.geometry.get('writtenRegions'):
        return result
    image=cv2.imdecode(np.frombuffer(page.image_png,np.uint8),cv2.IMREAD_COLOR)
    normalized=normalized_preview(image,page.geometry,page.alignment)
    matrix=np.array(page.alignment.pixel_to_page).reshape(3,3)
    # FitZ text positions are unrotated page points. Map through the rendered scan.
    def normalized_box(box):
        rect=fitz.Rect(box)
        corners=[rect.tl,rect.tr,rect.br,rect.bl]
        if pdf_page.rotation:
            corners=[p*pdf_page.rotation_matrix for p in corners]
        points=transform_points(np.array([[p.x*150/72,p.y*150/72] for p in corners],np.float32),matrix)
        return fitz.Rect(*points.min(axis=0),*points.max(axis=0))
    words=[]; images=[]
    if pdf_page is not None:
        visible=[fitz.Rect(s['bbox']) for s in pdf_page.get_texttrace() if s['type']!=3 and s.get('opacity',1)>0]
        words=[(normalized_box(w[:4]),w) for w in pdf_page.get_text('words',sort=True)
               if any(v.intersects(fitz.Rect(w[:4])) for v in visible)]
        images=[normalized_box(info['bbox']) for info in pdf_page.get_image_info()]
    responses=list(result.responses)
    for region in sorted(page.geometry['writtenRegions'],key=lambda r:r['order']):
        x,y,w,h=(region[k] for k in ('x','y','width','height'))
        box=fitz.Rect(x,y,x+w,y+h)
        crop=normalized[round(y*2):round((y+h)*2),round(x*2):round((x+w)*2)]
        encoded=cv2.imencode('.png',crop)[1].tobytes()
        ref='sha256:'+hashlib.sha256(encoded).hexdigest()
        result.crops[ref]=encoded
        source=SourceRegion(page.source_id,page.source_page_number,(x,y,w,h),ref)
        contained=[word for bbox,word in words if box.contains(bbox)]
        partial=any(box.intersects(bbox) and not box.contains(bbox) for bbox,_ in words)
        has_image=any(box.intersects(bbox) for bbox in images)
        lines={}
        for word in contained:
            lines.setdefault((word[5],word[6]),[]).append(word[4])
        text='\n'.join(' '.join(parts) for parts in lines.values()).strip()
        dark=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)<200
        ink=float(np.mean(dark))
        residual=dark.copy()
        for bbox,_ in words:
            if box.contains(bbox):
                left=max(0,int((bbox.x0-x-1)*2));top=max(0,int((bbox.y0-y-1)*2))
                right=min(crop.shape[1],int((bbox.x1-x+1)*2)+1);bottom=min(crop.shape[0],int((bbox.y1-y+1)*2)+1)
                residual[top:bottom,left:right]=False
        unexplained_ink=float(np.mean(residual))>.002
        reasons=list(result.identity_reasons)
        raw=None
        if text and not has_image and not partial and not unexplained_ink:
            raw=text;status=ReadStatus.READ;engine='pdf-visible-text-v1'
        elif ink<=.002 and not has_local_ink(dark) and not text and not partial:
            status=ReadStatus.BLANK;engine='written-region-capture-v1'
        else:
            status=ReadStatus.UNCERTAIN;engine='written-region-capture-v1'
            reasons.append('text-crosses-answer-boundary' if partial else 'handwriting-recognition-pending')
        response_id=hashlib.sha256(f'{page.source_id}|{page.source_page_number}|{page.alignment.layout.layout_id}|written:{region["regionId"]}'.encode()).hexdigest()
        responses.append(ExtractedResponse(response_id,page.alignment.layout,result.student_id,
            region['questionId'],region['number'],page.alignment.layout_page_number,ResponseType.WRITTEN,
            (source,),raw_text=raw,status=status,review_reasons=tuple(reasons),engine=engine,part_number=region['order']))
    result.responses=tuple(responses)
    return result
