"""Explicit one-time download of TrOCR weights; no student data is transmitted."""
from core.local_text_ocr import MODEL_ID


def main():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, allow_patterns=[
        '*.json', 'merges.txt', 'vocab.*', '*.safetensors', 'README.md'])
    print(f'TrOCR model cached at {path}')


if __name__ == '__main__':
    main()
