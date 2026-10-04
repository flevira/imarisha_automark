"""Bounded Tesseract OCR of answer regions; no model download or answer keys."""
import time
import cv2
import numpy as np
import pytesseract
from core.contracts import Transcription, ReadStatus


class LocalTextOCR:
    def __init__(self):
        self.languages=None

    def recognize(self,request):
        start=time.perf_counter();engine='tesseract-written-v1'
        try:
            if self.languages is None:self.languages=set(pytesseract.get_languages(config=''))
            wanted={'en':['eng'],'sw':['swa'],'en-sw':['eng','swa']}[request.language]
            available=[lang for lang in wanted if lang in self.languages]
            reasons=[]
            if len(available)!=len(wanted):reasons.append('ocr-language-pack-missing')
            if not available:
                if 'eng' in self.languages:available=['eng']
                else:raise ValueError('ocr-language-unavailable')
            image=cv2.imdecode(np.frombuffer(request.image_png,np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError('ocr-invalid-image')
            if image.size>20_000_000:raise ValueError('ocr-region-too-large')
            image=cv2.copyMakeBorder(image,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
            data=pytesseract.image_to_data(image,lang='+'.join(available),config='--psm 6',
                output_type=pytesseract.Output.DICT,timeout=20)
            lines={};conf=[]
            for n,word in enumerate(data['text']):
                word=word.strip()
                if not word:continue
                lines.setdefault((data['block_num'][n],data['par_num'][n],data['line_num'][n]),[]).append(word)
                conf.append(float(data['conf'][n]))
            text='\n'.join(' '.join(words) for words in lines.values())
            confidence=sum(conf)/len(conf) if conf else 0
            if not text:reasons.append('ocr-no-text')
            elif confidence<80 or min(conf)<45:reasons.append('ocr-low-confidence')
            return Transcription(text or None,ReadStatus.UNCERTAIN if reasons else ReadStatus.READ,
                tuple(reasons),engine,time.perf_counter()-start,confidence=confidence)
        except Exception as exc:
            reason='ocr-unavailable' if isinstance(exc,pytesseract.TesseractNotFoundError) else 'ocr-failed'
            return Transcription(None,ReadStatus.ERROR,(reason,),engine,time.perf_counter()-start)
