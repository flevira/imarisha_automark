"""Open PDFs and raster images through the same page pipeline."""
from io import BytesIO
import fitz
from PIL import Image, ImageOps


def open_source(data):
    if data[:1024].lstrip().startswith(b'%PDF-'):
        return fitz.open(stream=data,filetype='pdf')
    document=fitz.open()
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {'PNG','JPEG','TIFF','BMP','WEBP'}:
                raise ValueError('Use PDF, PNG, JPEG, TIFF, BMP or WebP')
            if getattr(image,'n_frames',1)>200:raise ValueError('Maximum 200 image frames')
            for index in range(getattr(image,'n_frames',1)):
                image.seek(index)
                if image.width*image.height>40_000_000:raise ValueError('Image exceeds 40 megapixels')
                frame=ImageOps.exif_transpose(image).convert('RGB')
                # Fit long edge to A4 length without changing its aspect ratio.
                scale=842/max(frame.size)
                page=document.new_page(width=frame.width*scale,height=frame.height*scale)
                buf=BytesIO();frame.save(buf,format='PNG')
                page.insert_image(page.rect,stream=buf.getvalue())
        return document
    except BaseException:
        document.close();raise
