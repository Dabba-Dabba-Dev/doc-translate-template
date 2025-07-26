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

def extract_page_blocks(pl_page):
    out = StringIO()

    # Configure table extraction
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
        # Extract full page text (layout-preserved)
        text = pl_page.extract_text(layout=True, keep_blank_chars=True) or ""
        
        if text.strip():
            out.write(text.strip() + "\n\n")  # Write all text
        out.write(format_table(real_table) + "\n")  # Write formatted table
    else:
        # No table found, try layout-preserved or fallback to chars
        text = pl_page.extract_text(layout=True, keep_blank_chars=True)
        if text and text.strip():
            out.write(text.strip() + "\n\n")
        else:
            chars = pl_page.chars
            if chars:
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
        for i, pl_page in enumerate(pdf.pages):
            out.write(extract_page_blocks(pl_page))
    result = out.getvalue()
    out.close()
    return result

def extract_ocr(pdf_path):
    imgs = convert_from_path(pdf_path, fmt="png", dpi=300)
    #call saja ocr extraction

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
    input_pdf = r"C:/Users/lenovo/Desktop/dabba dabba/doc-translate-template/contrat.pdf"
    output_txt = os.path.splitext(input_pdf)[0] + "_final.txt"
    extract_text_to_file(input_pdf, output_txt)