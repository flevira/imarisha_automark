# =============================
# INSTALL
# =============================
!pip install pymupdf opencv-python pandas pytesseract pillow

# =============================
# IMPORTS
# =============================
import re
import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
from google.colab import files

# =============================
# USER INPUT
# =============================
print("Upload the PDF answer sheet file")
uploaded = files.upload()
pdf_name = list(uploaded.keys())[0]

print("\nSelect Student ID Source:")
print("1 - QR only")
print("2 - TEXT only")
print("3 - AUTO (QR first, then text)")
choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    STUDENT_ID_MODE = "qr"
elif choice == "2":
    STUDENT_ID_MODE = "text"
else:
    STUDENT_ID_MODE = "auto"

# =============================
# CONFIG
# =============================
RENDER_ZOOM = 3.5
BUBBLE_EXTRA_MARGIN = 10.0
BUBBLE_REVIEW_MARGIN = 4.0
INCLUDE_OPEN_ENDED = True

MATCH_LEFT_LABELS = {1: "A", 2: "B", 3: "C", 4: "D"}

# =============================
# QR HELPERS
# =============================
def decode_qr_robust(img_bgr):
    detector = cv2.QRCodeDetector()
    attempts = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    attempts.append(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    attempts.append(clahe.apply(gray))

    attempts.append(
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    )

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(otsu)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    attempts.append(cv2.filter2D(gray, -1, sharpen_kernel))

    attempts.append(cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

    h, w = gray.shape
    qr_crop = gray[int(0.10 * h):int(0.38 * h), int(0.72 * w):int(0.98 * w)]
    attempts.append(cv2.resize(qr_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))

    for attempt in attempts:
        data, _, _ = detector.detectAndDecode(attempt)
        if data:
            return data

    return None

def parse_qr_data(data):
    parsed = {
        "type": None,
        "studentid": None,
        "examid": None,
        "testid": None,
        "assessmentid": None,
    }
    if not data:
        return parsed

    for part in data.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            parsed[k.strip().lower()] = v.strip()
    return parsed

# =============================
# RENDER / TEXT HELPERS
# =============================
def render_page_to_bgr(page, zoom=RENDER_ZOOM):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_page_text(page):
    return page.get_text("text", sort=True)

def get_page_words(page):
    return page.get_text("words", sort=True)

def search_text_rects(page, phrase):
    return page.search_for(phrase, quads=False)

def group_words_into_lines(words, y_tolerance=4.0):
    lines = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        if not lines:
            lines.append([word])
            continue

        current_y = np.mean([w[1] for w in lines[-1]])
        if abs(word[1] - current_y) <= y_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines

# =============================
# STUDENT ID HELPERS
# =============================
def extract_student_id_printed_text(page):
    text = get_page_text(page)
    m = re.search(r"Student ID:\s*(\d+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None

def extract_student_id_handwritten_ocr(img_bgr):
    h, w = img_bgr.shape[:2]
    x0 = int(0.02 * w)
    x1 = int(0.48 * w)
    y0 = int(0.16 * h)
    y1 = int(0.30 * h)

    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    variants = []

    variants.append(gray)
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(th1)
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    variants.append(th2)
    variants.append(cv2.bitwise_not(th1))

    best_digits = None
    best_conf = -1.0

    for variant in variants:
        data = pytesseract.image_to_data(
            variant,
            config="--psm 6 -c tessedit_char_whitelist=0123456789[]:",
            output_type=pytesseract.Output.DICT,
        )

        tokens = []
        confs = []

        for txt, conf_str in zip(data["text"], data["conf"]):
            txt = (txt or "").strip()
            try:
                conf_val = float(conf_str)
            except:
                conf_val = -1.0

            if txt:
                tokens.append(txt)
                confs.append(conf_val)

        joined = " ".join(tokens)
        digit_groups = re.findall(r"\d+", joined)

        if digit_groups:
            candidate = max(digit_groups, key=len)
            avg_conf = np.mean([c for c in confs if c >= 0]) if confs else -1.0
            if len(candidate) >= 1 and avg_conf > best_conf:
                best_digits = candidate
                best_conf = avg_conf

    if best_digits:
        low_conf = best_conf < 45
        return best_digits, low_conf

    return None, True

def resolve_student_id(qr_id, printed_id, handwritten_id, mode):
    # QR-only mode: ignore text IDs completely
    if mode == "qr":
        return {
            "student_id": qr_id,
            "student_id_all": qr_id if qr_id else None,
            "student_id_qr": qr_id,
            "student_id_text": None,
            "student_id_handwritten": None,
            "student_id_source": "qr",
            "id_conflict": False,
            "needs_review": qr_id is None,
        }

    # TEXT-only mode
    if mode == "text":
        detected = []
        if printed_id:
            detected.append(printed_id)
        if handwritten_id and handwritten_id not in detected:
            detected.append(handwritten_id)

        conflict = len(set(detected)) > 1
        chosen = printed_id or handwritten_id

        return {
            "student_id": chosen,
            "student_id_all": ",".join(detected) if detected else None,
            "student_id_qr": qr_id,
            "student_id_text": printed_id,
            "student_id_handwritten": handwritten_id,
            "student_id_source": "text",
            "id_conflict": conflict,
            "needs_review": True,
        }

    # AUTO mode
    detected = []
    if qr_id:
        detected.append(qr_id)
    if printed_id and printed_id not in detected:
        detected.append(printed_id)
    if handwritten_id and handwritten_id not in detected:
        detected.append(handwritten_id)

    conflict = len(set(detected)) > 1

    if qr_id:
        chosen = qr_id
        source = "qr"
    elif printed_id:
        chosen = printed_id
        source = "text_printed"
    elif handwritten_id:
        chosen = handwritten_id
        source = "text_handwritten"
    else:
        chosen = None
        source = "missing"

    return {
        "student_id": chosen,
        "student_id_all": ",".join(detected) if detected else None,
        "student_id_qr": qr_id,
        "student_id_text": printed_id,
        "student_id_handwritten": handwritten_id,
        "student_id_source": source,
        "id_conflict": conflict,
        "needs_review": chosen is None or conflict or source.startswith("text"),
    }

# =============================
# SECTION DETECTION
# =============================
def detect_section_blocks(page):
    heading_specs = [
        ("MC", "MULTIPLE CHOICE"),
        ("TF", "TRUE / FALSE"),
        ("MATCH", "MATCHING"),
        ("OPEN_ENDED", "OPEN ENDED"),
    ]

    found = []
    for section_type, phrase in heading_specs:
        rects = search_text_rects(page, phrase)
        for rect in rects:
            found.append((section_type, rect))

    found.sort(key=lambda x: x[1].y0)

    if not found:
        return []

    page_bottom = page.rect.y1
    blocks = []

    for idx, (section_type, rect) in enumerate(found):
        y0 = rect.y1 + 6
        if idx + 1 < len(found):
            y1 = found[idx + 1][1].y0 - 6
        else:
            y1 = page_bottom - 10

        if y1 > y0 + 10:
            blocks.append({
                "section_type": section_type,
                "y0": y0,
                "y1": y1,
            })

    return blocks

def find_option_centers(page, section):
    words = get_page_words(page)
    top_zone_y1 = section["y0"] + 80

    labels_by_section = {
        "MC": ["A", "B", "C", "D"],
        "TF": ["T", "F"],
        "MATCH": ["A", "B", "C", "D"],
    }

    labels = labels_by_section.get(section["section_type"], [])
    centers = {}

    for x0, y0, x1, y1, text, *_ in words:
        if y0 < section["y0"] or y1 > top_zone_y1:
            continue
        clean = text.strip().upper()
        if clean in labels and clean not in centers:
            centers[clean] = (x0 + x1) / 2.0

    return centers

# =============================
# ROW EXTRACTION
# =============================
def extract_rows_for_section(page, section):
    words = get_page_words(page)
    section_words = [w for w in words if w[1] >= section["y0"] and w[3] <= section["y1"]]
    lines = group_words_into_lines(section_words)

    rows = []

    for line in lines:
        texts = [w[4].strip() for w in line if w[4].strip()]
        if not texts:
            continue

        joined = " ".join(texts).upper()

        if (
            "ID A B C D" in joined
            or "ID T F" in joined
            or "QID ITEM A B C D" in joined
            or "QUESTION STUDENT ANSWER" in joined
        ):
            continue

        xs = [w[0] for w in line]
        ys0 = [w[1] for w in line]
        xs1 = [w[2] for w in line]
        ys1 = [w[3] for w in line]
        bbox = (min(xs), min(ys0), max(xs1), max(ys1))
        y_center = (bbox[1] + bbox[3]) / 2.0

        if section["section_type"] in {"MC", "TF", "OPEN_ENDED"}:
            qids = [t for t in texts if re.fullmatch(r"\d{5,6}", t)]
            if qids:
                rows.append({
                    "section_type": section["section_type"],
                    "question_id": qids[0],
                    "item_number": None,
                    "y_center": y_center,
                    "row_bbox": bbox,
                })

        elif section["section_type"] == "MATCH":
            qids = [t for t in texts if re.fullmatch(r"\d{5,6}", t)]
            items = [t for t in texts if re.fullmatch(r"[1-4]", t)]
            if qids and items:
                rows.append({
                    "section_type": section["section_type"],
                    "question_id": qids[0],
                    "item_number": int(items[0]),
                    "y_center": y_center,
                    "row_bbox": bbox,
                })

    return rows

# =============================
# BUBBLE SCORING
# =============================
def page_to_image_scale(page, img_bgr):
    sx = img_bgr.shape[1] / page.rect.width
    sy = img_bgr.shape[0] / page.rect.height
    return sx, sy

def sample_bubble_score(gray, cx, cy, radius):
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(gray.shape[1], cx + radius)
    y1 = min(gray.shape[0], cy + radius)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    rr = min(roi.shape[0], roi.shape[1]) / 2.2
    cy0 = roi.shape[0] / 2.0
    cx0 = roi.shape[1] / 2.0
    mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= rr ** 2

    pixels = roi[mask]
    if pixels.size == 0:
        pixels = roi.flatten()

    return 255.0 - float(np.mean(pixels))

def extract_row_answers(page, img_bgr, section, row, option_centers):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx, sy = page_to_image_scale(page, img_bgr)

    if section["section_type"] == "MC":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    elif section["section_type"] == "TF":
        ordered_labels = ["T", "F"]
        output_map = {"T": "True", "F": "False"}
    elif section["section_type"] == "MATCH":
        ordered_labels = ["A", "B", "C", "D"]
        output_map = {"A": "A", "B": "B", "C": "C", "D": "D"}
    else:
        return [], True, {}

    row_h_page = max(10.0, row["row_bbox"][3] - row["row_bbox"][1])
    radius = max(10, int((row_h_page * sy) * 0.38))
    cy = int(row["y_center"] * sy)

    scores = {}
    for label in ordered_labels:
        if label not in option_centers:
            scores[label] = 0.0
            continue
        cx = int(option_centers[label] * sx)
        scores[label] = sample_bubble_score(gray, cx, cy, radius)

    score_values = list(scores.values())
    if not score_values:
        return [], True, scores

    mean_score = float(np.mean(score_values))
    max_score = float(np.max(score_values))
    min_score = float(np.min(score_values))

    filled_labels = [
        label for label in ordered_labels
        if scores[label] >= mean_score + BUBBLE_EXTRA_MARGIN
    ]

    needs_review = False

    if not filled_labels:
        best_label = max(scores, key=scores.get)
        sorted_vals = sorted(score_values, reverse=True)
        if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) >= (BUBBLE_EXTRA_MARGIN + 2):
            filled_labels = [best_label]
        else:
            needs_review = True

    if (max_score - min_score) < BUBBLE_REVIEW_MARGIN:
        needs_review = True

    if section["section_type"] == "TF" and len(filled_labels) != 1:
        needs_review = True

    if section["section_type"] == "MATCH" and len(filled_labels) != 1:
        needs_review = True

    answers = [output_map[label] for label in filled_labels if label in output_map]
    return answers, needs_review, scores

# =============================
# OUTPUT HELPERS
# =============================
def build_matching_pair(item_number, answer):
    if item_number is None or answer is None:
        return None
    left = MATCH_LEFT_LABELS.get(item_number)
    if not left:
        return None
    return f"{left}={answer}"

def append_output_rows(output_rows, base, row, answers, needs_review, scores):
    if row["section_type"] == "OPEN_ENDED":
        output_rows.append({
            **base,
            "section": "OPEN_ENDED",
            "question_id": row["question_id"],
            "item_number": None,
            "answer": None,
            "matching_pair": None,
            "manual_marking": True,
            "needs_review": True,
            "score_debug": None,
        })
        return

    if not answers:
        output_rows.append({
            **base,
            "section": row["section_type"],
            "question_id": row["question_id"],
            "item_number": row["item_number"],
            "answer": None,
            "matching_pair": None,
            "manual_marking": False,
            "needs_review": True,
            "score_debug": str(scores),
        })
        return

    for ans in answers:
        output_rows.append({
            **base,
            "section": row["section_type"],
            "question_id": row["question_id"],
            "item_number": row["item_number"],
            "answer": ans,
            "matching_pair": build_matching_pair(row["item_number"], ans) if row["section_type"] == "MATCH" else None,
            "manual_marking": False,
            "needs_review": bool(needs_review),
            "score_debug": str(scores),
        })

# =============================
# MAIN PROCESSING
# =============================
doc = fitz.open(stream=uploaded[pdf_name], filetype="pdf")
all_rows = []

for page_idx in range(len(doc)):
    page = doc[page_idx]
    img_bgr = render_page_to_bgr(page)
    page_number = page_idx + 1

    # QR is mandatory
    qr_raw = decode_qr_robust(img_bgr)
    if not qr_raw:
        all_rows.append({
            "page": page_number,
            "student_id": None,
            "student_id_all": None,
            "student_id_qr": None,
            "student_id_text": None,
            "student_id_handwritten": None,
            "student_id_source": "missing",
            "id_conflict": False,
            "sheet_type": None,
            "exam_id": None,
            "test_id": None,
            "section": None,
            "question_id": None,
            "item_number": None,
            "answer": None,
            "matching_pair": None,
            "manual_marking": False,
            "needs_review": True,
            "score_debug": None,
            "qr_raw": None,
        })
        continue

    qr_parsed = parse_qr_data(qr_raw)
    qr_student_id = qr_parsed.get("studentid")
    qr_type = (qr_parsed.get("type") or "").upper() or None

    exam_id = None
    test_id = None
    if qr_type == "EXAM":
        exam_id = qr_parsed.get("examid") or qr_parsed.get("assessmentid")
    elif qr_type == "TEST":
        test_id = qr_parsed.get("testid") or qr_parsed.get("assessmentid")

    printed_id = extract_student_id_printed_text(page)
    handwritten_id, _ = extract_student_id_handwritten_ocr(img_bgr)

    id_info = resolve_student_id(
        qr_id=qr_student_id,
        printed_id=printed_id,
        handwritten_id=handwritten_id,
        mode=STUDENT_ID_MODE,
    )

    base = {
        "page": page_number,
        "student_id": id_info["student_id"],
        "student_id_all": id_info["student_id_all"],
        "student_id_qr": id_info["student_id_qr"],
        "student_id_text": id_info["student_id_text"],
        "student_id_handwritten": id_info["student_id_handwritten"],
        "student_id_source": id_info["student_id_source"],
        "id_conflict": id_info["id_conflict"],
        "sheet_type": qr_type,
        "exam_id": exam_id if qr_type == "EXAM" else None,
        "test_id": test_id if qr_type == "TEST" else None,
        "qr_raw": qr_raw,
    }

    sections = detect_section_blocks(page)

    for section in sections:
        option_centers = find_option_centers(page, section)
        rows = extract_rows_for_section(page, section)

        if section["section_type"] == "OPEN_ENDED":
            if INCLUDE_OPEN_ENDED:
                for row in rows:
                    append_output_rows(
                        output_rows=all_rows,
                        base=base,
                        row=row,
                        answers=[],
                        needs_review=True,
                        scores={},
                    )
            continue

        for row in rows:
            answers, row_review, scores = extract_row_answers(
                page=page,
                img_bgr=img_bgr,
                section=section,
                row=row,
                option_centers=option_centers,
            )

            review_flag = row_review or id_info["needs_review"] or (row["question_id"] is None)

            append_output_rows(
                output_rows=all_rows,
                base=base,
                row=row,
                answers=answers,
                needs_review=review_flag,
                scores=scores,
            )

# =============================
# SAVE RESULTS
# =============================
df = pd.DataFrame(all_rows)

if not df.empty:
    sort_cols = [c for c in ["student_id", "page", "section", "question_id", "item_number", "answer"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

output_name = "final_extracted_results.csv"
df.to_csv(output_name, index=False)

print("\n✅ Extraction complete")
print(df.head(40).to_string(index=False))

files.download(output_name)
