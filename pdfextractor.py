import re
import fitz
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
import pdfplumber
import os

def is_real_table(tbl):
    if len(tbl) < 2:
        return False
    cols = len(tbl[0])
    non_empty = 0
    for c in range(cols):
        for r in tbl[1:]:
            if r[c] and r[c].strip():
                non_empty += 1
                break
    return non_empty > 1

def extract_page_blocks(pl_page, fmz_page, page_num):
    out = f"--- Page {page_num} ---\n"

    # 1) detect real data-table
    tables = pl_page.extract_tables({
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5
    })
    real = next((t for t in tables if is_real_table(t)), None)

    if real:
        # split into before/table/after using regex on raw text
        raw = pl_page.extract_text() or ""
        header_rx = re.compile(r"Version\s+Créateur\s*/\s*Modificateur.*Contenu", re.I)
        lines = raw.split("\n")
        pre, post, in_table = [], [], False
        for ln in lines:
            if not in_table and header_rx.search(ln):
                in_table = True; continue
            if not in_table:
                pre.append(ln)
            else:
                post.append(ln)
        if pre:
            out += "[Text – before table]\n" + "\n".join(pre).strip() + "\n\n"
        out += "[Table]\n"
        for row in real:
            out += " | ".join(cell.strip() if cell else "" for cell in row) + "\n"
        out += "\n"
        if post:
            out += "[Text – after table]\n" + "\n".join(post).strip() + "\n\n"

    else:
        # NO real table → full page via pdfplumber fixed layout
        text = pl_page.extract_text(layout=True)
        out += "[Text]\n" + text + "\n"

    return out

def extract_text_with_layout(pdf_path):
    out = ""
    with pdfplumber.open(pdf_path) as pdf:
        doc = fitz.open(pdf_path)
        for i, pl_pg in enumerate(pdf.pages):
            fmz_pg = doc.load_page(i)
            out += extract_page_blocks(pl_pg, fmz_pg, i+1)
    return out

def extract_ocr(pdf_path):
    imgs = convert_from_path(pdf_path)
    ocr = []
    for i, img in enumerate(imgs):
        ocr.append(f"--- OCR Page {i+1} ---\n")
        ocr.append(pytesseract.image_to_string(img, lang='eng+deu+fra+spa+ita'))
    return "".join(ocr)

def extract_text_to_file(pdf_path, txt_path):
    text = extract_text_with_layout(pdf_path)
    if not text.strip():
        print("Layout extraction failed, trying OCR…")
        text = extract_ocr(pdf_path)
    try:
        print("Detected language:", detect(text))
    except:
        pass
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved to", txt_path)

if __name__ == "__main__":
    input_pdf  = r"path to the pdf doc.pdf"
    output_txt = os.path.splitext(input_pdf)[0] + "_final.txt"
    extract_text_to_file(input_pdf, output_txt)