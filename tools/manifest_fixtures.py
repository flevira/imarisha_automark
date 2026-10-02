"""Generate synthetic M1 fixtures; no student records or production manifests.

Run: python -m tools.manifest_fixtures --output artifacts/m1-fixtures
PDF/image artifacts are regenerated, not committed. JSON oracles are committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/manifest"
W, H = 595.28, 841.89


def make_manifest(shared: bool) -> dict:
    version = 5 if shared else 4
    kinds = [[("TF", "10005", None)]] if shared else [
        [("MC", "10001", None), ("MA", "10002", None), ("TF", "10003", None)],
        [("MATCH", "10004", 1), ("MATCH", "10004", 2)],
    ]
    manifest = dict(layoutId=f"synthetic-{'shared' if shared else 'mixed'}-v{version}", version=version,
                    assessmentType="TEST" if shared else "EXAM", assessmentId="42", assessmentGroupId="40",
                    unit="pt", coordinateOrigin="top-left", answerSchemaVersion=1, sourceRequestId="synthetic-request",
                    totalMarks=2 if shared else 8, scoringPolicy="ASSESSMENT_POLICY_REQUIRED", questions=[], pages=[])
    for pn, specs in enumerate(kinds, 1):
        page = dict(pageNumber=pn, width=W, height=H, headings=[], rows=[], markers=[
            dict(id=name, cx=x, cy=y, size=16) for name, x, y in
            [("TL", 30, 30), ("TR", 565.28, 30), ("BR", 565.28, 811.89), ("BL", 30, 811.89)]])
        for i, (kind, qid, item) in enumerate(specs):
            options = ["true", "false"] if kind == "TF" else list("ABCD")
            y = (300 if shared else 180) + i * 20
            centers = [130, 210] if kind == "TF" else [110, 160, 210, 260]
            row = dict(questionId=qid, number=1 if shared else int(qid[-1]), kind=kind, x=40, y=y,
                       optionIndices=list(range(len(options))), centers=centers,
                       selectionMode="multiple" if kind == "MA" else "single",
                       bubbles=[dict(optionId=o, cx=x, cy=y, diameter=14.18) for o, x in zip(options, centers)],
                       correctAnswers=(["A", "C"] if kind == "MA" else ["A" if item == 1 else "B"] if kind == "MATCH" else [options[0]]),
                       questionMarks=2, answerStatus="READY")
            if item:
                row["item"] = item
            page["rows"].append(row)
            if not any(q["questionId"] == qid for q in manifest["questions"]):
                manifest["questions"].append(dict(questionId=qid, displayNumber=row["number"], maxMarks=2,
                    responseType=kind, status="READY", referenceAnswers=(["A=A", "B=B"] if kind == "MATCH" else row["correctAnswers"]), answerDetail="Teacher-only fixture rubric"))
        if shared:
            page["studentNumber"] = dict(field="studentId", digitCount=5, order="left-to-right",
                selectionMode="single-per-column", preserveLeadingZeros=True, columns=[
                    dict(position=c+1, bubbles=[dict(value=str(d), cx=68+c*28, cy=82+d*20, diameter=14.18)
                                               for d in range(10)]) for c in range(5)])
        manifest["pages"].append(page)
    return manifest


def qr_payload(manifest: dict, page: int, shared: bool) -> str:
    typ = manifest["assessmentType"]
    return (f'v={manifest["version"]};type={typ};assessmentId=42;{typ.lower()}Id=42;'
            f'layoutId={manifest["layoutId"]};page={page}' + ("" if shared else ";studentId=00530"))


def expected(manifest: dict) -> list[dict]:
    responses = []
    for page in manifest["pages"]:
        for row in page["rows"]:
            # Deliberately differ from the correct-answer key: extraction isn't marking.
            selected = {"MC": ["B"], "MA": ["A", "D"], "TF": ["false"], "MATCH": ["B" if row.get("item") == 1 else "A"]}[row["kind"]]
            responses.append(dict(page=page["pageNumber"], student_id="00530", question_id=row["questionId"],
                                  item_number=row.get("item"), selected_options=selected))
    return responses


def draw_qr(page: fitz.Page, payload: str) -> None:
    image = cv2.QRCodeEncoder_create().encode(payload)
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    page.insert_image(fitz.Rect(451, 48, 545, 142), stream=cv2.imencode(".png", image)[1].tobytes())


def rasterize(pdf: Path, dest: Path) -> None:
    with fitz.open(pdf) as source, fitz.open() as out:
        for page in source:
            image = page.get_pixmap(dpi=150, alpha=False)
            target = out.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=image.tobytes("png"))
        out.save(dest, deflate=True)


def render(manifest: dict, answers: list[dict], dest: Path, shared: bool) -> None:
    with fitz.open() as doc:
        for spec in manifest["pages"]:
            page = doc.new_page(width=W, height=H)
            page.insert_text((225 if shared else 40, 50), "SYNTHETIC TEST SHEET", fontsize=11)
            page.insert_text((225 if shared else 40, 70), "No real student data", fontsize=9)
            if not shared:
                page.insert_text((40, 95), "Student: 00530", fontsize=10)
            draw_qr(page, qr_payload(manifest, spec["pageNumber"], shared))
            for marker in spec["markers"]:
                x, y, size = marker["cx"], marker["cy"], marker["size"]/2
                page.draw_rect(fitz.Rect(x-size, y-size, x+size, y+size), fill=(0, 0, 0))
            if shared:
                page.insert_text((40, 48), "STUDENT NUMBER", fontsize=9)
                for column in spec["studentNumber"]["columns"]:
                    page.insert_text((column["bubbles"][0]["cx"]-3, 66), str(column["position"]), fontsize=9)
                    for b in column["bubbles"]:
                        if column["position"] == 1:
                            page.insert_text((40, b["cy"]+3), b["value"], fontsize=9)
                        filled = b["value"] == "00530"[column["position"]-1]
                        page.draw_circle((b["cx"], b["cy"]), b["diameter"]/2, fill=(0, 0, 0) if filled else None)
            for row in spec["rows"]:
                answer = next(a for a in answers if a["question_id"] == row["questionId"] and a["item_number"] == row.get("item"))
                label = str(row["number"]) + ({1: "(i)", 2: "(ii)"}.get(row.get("item"), ""))
                if row["kind"] == "MA":
                    label += " *"
                page.insert_text((row["x"], row["y"]+3), label, fontsize=10)
                for b in row["bubbles"]:
                    page.insert_text((b["cx"]-(35 if row["kind"] == "TF" else 20), b["cy"]+3), b["optionId"], fontsize=9)
                    page.draw_circle((b["cx"], b["cy"]), b["diameter"]/2,
                                     fill=(0, 0, 0) if b["optionId"] in answer["selected_options"] else None)
            for region in spec.get("writtenRegions", []):
                x,y,w,h = (region[k] for k in ("x","y","width","height"))
                page.insert_text((x,y-10),f"Question {region['number']} (answer below)",fontsize=10)
                page.draw_rect(fitz.Rect(x-4,y-4,x+w+4,y+h+4),width=.5)
            page.insert_text((40, 784), "* Choose all that apply.", fontsize=9)
            page.insert_text((225, 799), f'Page {spec["pageNumber"]} | Synthetic fixture', fontsize=9)
        doc.save(dest, deflate=True)


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, shared in [("personalized-mixed", False), ("shared-tf", True)]:
        manifest = json.loads((FIXTURES / f"{name}.json").read_text())
        oracle = json.loads((FIXTURES / f"{name}-expected.json").read_text())
        native = output / f"{name}-text.pdf"
        render(manifest, oracle, native, shared)
        rasterize(native, output / f"{name}-scan.pdf")
        for form in ("text", "scan"):
            cases.append(dict(id=f"{name}-{form}", path=f"{name}-{form}.pdf", manifest=f"{name}.json",
                              expected=oracle, pages=len(manifest["pages"]), family="manifest"))
    # A separate legacy control makes failures on the new layout interpretable.
    native = output / "legacy-control-text.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=W, height=H)
        page.insert_text((40, 70), "SYNTHETIC LEGACY CONTROL", fontsize=12)
        draw_qr(page, "type=EXAM;examId=42;studentId=00530")
        page.insert_text((40, 320), "MULTIPLE", fontsize=12)
        for x, label in [(40, "NO"), (65, "QID"), (135, "A"), (175, "B"), (215, "C"), (255, "D")]:
            page.insert_text((x, 350), label, fontsize=10)
        page.insert_text((40, 380), "1", fontsize=10)
        page.insert_text((65, 380), "10001", fontsize=10)
        for x in (138, 178, 218, 258):
            page.draw_circle((x, 376), 7, fill=(0, 0, 0) if x == 178 else None)
        doc.save(native, deflate=True)
    rasterize(native, output / "legacy-control-scan.pdf")
    for form in ("text", "scan"):
        cases.append(dict(id=f"legacy-control-{form}", path=f"legacy-control-{form}.pdf", pages=1, family="legacy",
            expected=[dict(page=1, student_id="00530", question_id="10001", item_number=None, selected_options=["B"])]))
    # Hand-drawn synthetic digit strokes are only a contract fixture, not an HTR benchmark.
    crop = np.full((100, 200, 3), 255, np.uint8)
    cv2.polylines(crop, [np.array([(35, 35), (49, 20), (50, 79)])], False, (0, 0, 0), 3)
    cv2.polylines(crop, [np.array([(85, 30), (99, 20), (121, 23), (129, 35), (122, 46), (85, 80), (132, 80)])], False, (0, 0, 0), 3)
    cv2.imwrite(str(output / "synthetic-written-12.png"), crop)
    catalog = dict(schema_version=1, synthetic=True, scan_dpi=150, cases=cases,
                   handwriting={"path": "synthetic-written-12.png", "transcription": "12", "qualification": "contract-only; not real handwriting accuracy"})
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1-fixtures"))
    args = parser.parse_args()
    generate(args.output)
    print(args.output / "catalog.json")
