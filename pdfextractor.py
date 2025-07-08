import re
import fitz
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect_langs
import pdfplumber
import os
from io import StringIO
from pdfplumber.table import TableSettings

def is_real_table(tbl):
    if not tbl or len(tbl) < 2:
        return False
    cols = len(tbl[0])
    non_empty = 0
    for c in range(cols):
        for r in tbl[1:]:
            if r[c] and r[c].strip():
                non_empty += 1
                break
    return non_empty > 1

def format_table(table):
    if not table:
        return ""
    # Calculate max width for each column to preserve alignment
    col_widths = [0] * len(table[0])
    for row in table:
        for i, cell in enumerate(row):
            cell = cell.strip() if cell else ""
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Format table with aligned columns
    out = []
    for row in table:
        formatted_row = []
        for i, cell in enumerate(row):
            cell = cell.strip() if cell else ""
            # Pad cell to align with max column width
            formatted_row.append(cell.ljust(col_widths[i]))
        out.append(" | ".join(formatted_row))
    return "\n".join(out) + "\n"

def extract_page_blocks(pl_page, fmz_page, page_num):
    out = StringIO()
    out.write(f"--- Page {page_num} ---\n")

    # Configure table extraction with stricter settings
    table_settings = {
        "vertical_strategy": "lines_strict",
        "horizontal_strategy": "lines_strict",
        "intersection_tolerance": 3,
        "min_words_vertical": 3,
        "min_words_horizontal": 3
    }
    tables = pl_page.extract_tables(table_settings)
    real_table = next((t for t in tables if is_real_table(t)), None)

    if real_table:
        # Extract text with layout preservation
        raw_text = pl_page.extract_text(layout=True, keep_blank_chars=True) or ""
        lines = raw_text.split("\n")
        header_rx = re.compile(r"Version\s+Créateur\s*/\s*Modificateur.*Contenu", re.I | re.MULTILINE)
        
        # Split text into before and after table
        pre, post, in_table = [], [], False
        for ln in lines:
            if not in_table and header_rx.search(ln):
                in_table = True
                continue
            if not in_table:
                pre.append(ln)
            else:
                post.append(ln)
        
        # Write pre-table text
        if pre and any(ln.strip() for ln in pre):
            out.write("[Text – before table]\n")
            out.write("\n".join(ln.rstrip() for ln in pre if ln.strip()) + "\n\n")
        
        # Write formatted table
        out.write("[Table]\n")
        out.write(format_table(real_table))
        out.write("\n")
        
        # Write post-table text
        if post and any(ln.strip() for ln in post):
            out.write("[Text – after table]\n")
            out.write("\n".join(ln.rstrip() for ln in post if ln.strip()) + "\n\n")
    else:
        # No table found, extract text with layout preservation
        text = pl_page.extract_text(layout=True, keep_blank_chars=True)
        if text.strip():
            out.write("[Text]\n")
            out.write(text.rstrip() + "\n\n")
        else:
            # Fallback to character-level extraction for complex layouts
            chars = pl_page.chars
            if chars:
                out.write("[Text]\n")
                current_line = []
                last_y = None
                for char in sorted(chars, key=lambda c: (c['y0'], c['x0'])):
                    if last_y is not None and abs(char['y0'] - last_y) > 5:
                        out.write("".join(current_line).rstrip() + "\n")
                        current_line = []
                    current_line.append(char['text'])
                    last_y = char['y0']
                if current_line:
                    out.write("".join(current_line).rstrip() + "\n")
                out.write("\n")

    result = out.getvalue()
    out.close()
    return result

def extract_text_with_layout(pdf_path):
    out = StringIO()
    with pdfplumber.open(pdf_path) as pdf:
        doc = fitz.open(pdf_path)
        for i, pl_page in enumerate(pdf.pages):
            fmz_page = doc.load_page(i)
            out.write(extract_page_blocks(pl_page, fmz_page, i+1))
    result = out.getvalue()
    out.close()
    return result

def extract_ocr(pdf_path):
    imgs = convert_from_path(pdf_path, fmt="png", dpi=300)
    ocr = StringIO()
    for i, img in enumerate(imgs):
        ocr.write(f"--- OCR Page {i+1} ---\n")
        # Use Tesseract with layout preservation
        custom_config = r'--oem 1 --psm 6 -l eng+deu+fra+spa+ita'
        text = pytesseract.image_to_string(img, config=custom_config)
        ocr.write(text.rstrip() + "\n\n")
    result = ocr.getvalue()
    ocr.close()
    return result

def extract_languages(text):
    try:
        languages = detect_langs(text)
        detected_languages = ", ".join([lang.lang for lang in languages if lang.prob > 0.1])
        return detected_languages or "No languages detected"
    except:
        return "Language detection failed"

def extract_text_to_file(pdf_path, txt_path):
    text = extract_text_with_layout(pdf_path)
    if not text.strip():
        print("Layout extraction failed, attempting OCR...")
        text = extract_ocr(pdf_path)
    detected_languages = extract_languages(text)
    print(f"Detected languages: {detected_languages}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {txt_path}")

if __name__ == "__main__":
    input_pdf = r"path to the pdf file"
    output_txt = os.path.splitext(input_pdf)[0] + "_final.txt"
    extract_text_to_file(input_pdf, output_txt)
