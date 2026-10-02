"""Measure the unchanged extractor; zero detections count as failures, not accuracy.

python -m tools.extraction_baseline --fixtures artifacts/m1-fixtures --output artifacts/m1-baseline
Each case runs in a fresh subprocess; optional repeats in that process show warm runs.
No marking, source persistence, physical scanning or upload is benchmarked in M1.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time


def evaluate(expected: list[dict], rows: list[dict]) -> dict:
    def key(row: dict) -> tuple:
        return (row.get("page"), row.get("student_id"), row.get("question_id"), row.get("item_number"))
    actual = defaultdict(list)
    for row in rows:
        actual[key(row)].append(row)
    expected_keys = {key(row) for row in expected}
    if len(expected_keys) != len(expected):
        raise ValueError("Duplicate expected response identity")
    exact = missing = review = duplicates = 0
    for wanted in expected:
        found = actual.get(key(wanted), [])
        if not found:
            missing += 1
            continue
        choices = [str(r["answer"]).lower() if str(r["answer"]).lower() in ("true", "false") else str(r["answer"])
                   for r in found if r.get("answer") not in (None, "")]
        duplicates += len(choices) - len(set(choices))
        exact += set(choices) == set(wanted["selected_options"]) and len(choices) == len(set(choices))
        review += any(r.get("needs_review", False) for r in found)
    return dict(expected_responses=len(expected), exact_responses=exact, missing_responses=missing,
                unexpected_responses=len(set(actual)-expected_keys), duplicate_selections=duplicates,
                matched_responses_needing_review=review,
                detected_responses_needing_review=sum(any(r.get("needs_review", False) for r in group) for group in actual.values()),
                exact_response_rate=exact/len(expected) if expected else None,
                detected_review_rate=sum(any(r.get("needs_review", False) for r in group) for group in actual.values())/len(actual) if actual else None,
                detected_responses=len(actual), output_rows=len(rows))


def worker(source: Path, output: Path, repeats: int) -> None:
    # Use installed EasyOCR weights only. The benchmark must never download a model.
    from functools import lru_cache
    from core import ocr_engine
    from core.extractor import extract_file_with_report
    @lru_cache(maxsize=1)
    def offline_easyocr():
        try:
            import easyocr
            return easyocr.Reader(["en"], gpu=False, verbose=False, download_enabled=False)
        except Exception:
            return None
    ocr_engine._get_easyocr_reader = offline_easyocr
    # This M1 profile compares Tesseract + optional cached EasyOCR on every machine.
    # Optional Paddle initialization can download models, so it is explicitly excluded.
    ocr_engine._get_paddle_reader = lambda: None
    runs = []
    for index in range(repeats):
        start = time.perf_counter()
        try:
            rows, report = extract_file_with_report(str(source), "auto")
            result = dict(status="ok", rows=rows, workflow_report=report)
        except Exception as exc:
            result = dict(status="error", rows=[], error=f"{type(exc).__name__}: {exc}")
        result.update(run=index+1, cache_state="cold-process" if index == 0 else "warm-process",
                      extraction_seconds=time.perf_counter()-start)
        runs.append(result)
        # Save each completed repeat so a later failure doesn't hide earlier results.
        output.write_text(json.dumps(runs, indent=2, default=str)+"\n")


def run(fixtures: Path, output: Path, repeats: int, timeout: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((fixtures / "catalog.json").read_text())
    packages = {}
    for name in ("PyMuPDF", "opencv-python-headless", "pytesseract", "easyocr", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        tesseract = subprocess.check_output(["tesseract", "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        tesseract = "unavailable"
    report = dict(schema_version=1, synthetic=True, python=platform.python_version(),
                  cached_easyocr_weights_present={name: (Path.home()/".EasyOCR/model"/name).is_file() for name in ("craft_mlt_25k.pth", "english_g2.pth")}, platform=platform.platform(),
                  machine=platform.machine(), packages=packages, tesseract=tesseract,
                  extractor_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/"core").glob("*.py")) if p.name != "contracts.py"},
                  ocr_profile="Tesseract + cached English EasyOCR CPU; Paddle disabled; no model downloads",
                  timing_scope="Extractor call incl. lazy model loading; process wall includes Python imports. No marking or SQLite persistence.",
                  scan_dpi=catalog["scan_dpi"], cases=[])
    for case in catalog["cases"]:
        data_file = output / f'{case["id"]}-runs.json'
        # Do not interpret leftovers from a previous run as this run's result.
        data_file.unlink(missing_ok=True)
        start = time.perf_counter()
        with (output / f'{case["id"]}.log').open("w") as log:
            try:
                proc = subprocess.run([sys.executable, "-m", "tools.extraction_baseline", "--worker", str(fixtures/case["path"]),
                                       "--output", str(data_file), "--repeats", str(repeats)], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                status = "ok" if proc.returncode == 0 else "worker-error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        runs = json.loads(data_file.read_text()) if data_file.exists() else []
        for result in runs:
            result["metrics"] = evaluate(case["expected"], result["rows"])
        item = dict(id=case["id"], family=case["family"], pages=case["pages"], expected_responses=len(case["expected"]), status=status,
                    process_seconds=time.perf_counter()-start, runs=runs,
                    input_sha256=hashlib.sha256((fixtures/case["path"]).read_bytes()).hexdigest(),
                    oracle_sha256=hashlib.sha256(json.dumps(case["expected"], sort_keys=True).encode()).hexdigest())
        if len(runs) != repeats or any(r["status"] != "ok" for r in runs):
            item["status"] = status if status != "ok" else "incomplete"
        report["cases"].append(item)
        (output / "summary.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
        print(case["id"], item["status"], round(item["process_seconds"], 2), "seconds", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("artifacts/m1-fixtures"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1-baseline"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    if args.worker:
        worker(args.worker, args.output, args.repeats)
    else:
        report = run(args.fixtures, args.output, args.repeats, args.timeout)
        sys.exit(0 if all(c["status"] == "ok" for c in report["cases"]) else 1)
