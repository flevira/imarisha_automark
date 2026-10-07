"""Explicit preparation of local Paddle weights; never uploads student data."""
import argparse
from pathlib import Path
import shutil
from core.paddle_text_ocr import MODELS, model_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=model_directory())
    parser.add_argument('--source', type=Path, help='Reuse already downloaded weights')
    args = parser.parse_args()
    for model in MODELS:
        destination = args.directory / model
        if args.source:
            destination.mkdir(parents=True, exist_ok=True)
            for name in ('inference.json', 'inference.pdiparams', 'inference.yml'):
                shutil.copy2(args.source / model / name, destination / name)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download('PaddlePaddle/' + model, allow_patterns=['inference.*'],
                              local_dir=destination)
    print(f'PaddleOCR weights prepared at {args.directory}')


if __name__ == '__main__':
    main()
