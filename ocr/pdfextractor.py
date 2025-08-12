from pdf2image import convert_from_path
import pdfplumber
import os
from io import StringIO

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

def has_extractable_text(pdf_path):
    """Check if PDF has extractable text content"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:  # Check first 3 pages
                text = page.extract_text()
                if text and text.strip():
                    return True
        return False
    except:
        return False

def extract_text_with_layout(pdf_path):
    out = StringIO()
    with pdfplumber.open(pdf_path) as pdf:
        for i, pl_page in enumerate(pdf.pages):
            out.write(extract_page_blocks(pl_page))
    result = out.getvalue()
    out.close()
    return result

def extract_text_to_file(pdf_path, txt_path, lang=None):
    """
    Extract text from PDF to file.
    
    Args:
        pdf_path: Path to input PDF file
        txt_path: Path to output text file
        lang: Language parameter (optional, for future OCR fallback)
    """
    text = extract_text_with_layout(pdf_path)
    
    # Note: OCR fallback removed - this should be handled by the calling code
    if not text.strip():
        print("No extractable text found in PDF")
        text = ""
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {txt_path}")
    
    return text
