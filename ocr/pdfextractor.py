import re
import fitz
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect_langs
import pdfplumber
import os
from io import StringIO

# Imports for DOCX export
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

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
  """Formats a table (list of lists) into a string with aligned columns."""
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
  """
  Extracts content from a single PDF page into a list of structured blocks.
  Each block is a dictionary with 'type' ('text' or 'table') and 'content'.
  """
  page_blocks = []

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
      
      # Add pre-table text block
      pre_text_content = "\n".join(ln.rstrip() for ln in pre if ln.strip())
      if pre_text_content:
          page_blocks.append({'type': 'text', 'content': pre_text_content})
      
      # Add table block
      page_blocks.append({'type': 'table', 'content': real_table})
      
      # Add post-table text block
      post_text_content = "\n".join(ln.rstrip() for ln in post if ln.strip())
      if post_text_content:
          page_blocks.append({'type': 'text', 'content': post_text_content})
  else:
      # No table found, extract text with layout preservation
      text = pl_page.extract_text(layout=True, keep_blank_chars=True)
      if text.strip():
          page_blocks.append({'type': 'text', 'content': text.rstrip()})
      else:
          # Fallback to character-level extraction for complex layouts
          chars = pl_page.chars
          if chars:
              current_line_chars = []
              last_y = None
              char_extracted_lines = []
              for char in sorted(chars, key=lambda c: (c['y0'], c['x0'])):
                  if last_y is not None and abs(char['y0'] - last_y) > 5:
                      char_extracted_lines.append("".join(current_line_chars).rstrip())
                      current_line_chars = []
                  current_line_chars.append(char['text'])
                  last_y = char['y0']
              if current_line_chars:
                  char_extracted_lines.append("".join(current_line_chars).rstrip())
              
              if char_extracted_lines:
                  page_blocks.append({'type': 'text', 'content': "\n".join(char_extracted_lines)})

  return page_blocks

def export_to_txt(all_blocks_per_page, output_txt_path):
    """
    Saves the extracted content (text and tables) into a plain .txt file.
    """
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for page_num, blocks in enumerate(all_blocks_per_page):
            f.write(f"--- Page {page_num + 1} ---\n")
            for block in blocks:
                if block['type'] == 'text':
                    f.write(block['content'].rstrip() + "\n\n")
                elif block['type'] == 'table':
                    f.write(format_table(block['content']) + "\n\n")

def extract_ocr(pdf_path, lang=None):
  """Extracts text from PDF using OCR (Tesseract)."""
  imgs = convert_from_path(pdf_path, fmt="png", dpi=300)
  ocr = StringIO()
  for i, img in enumerate(imgs):
      ocr.write(f"--- OCR Page {i+1} ---\n")
      # Use Tesseract with layout preservation
      custom_config = r'--oem 1 --psm 6'
      if lang:
          custom_config += f' -l {lang}'
      text = pytesseract.image_to_string(img, config=custom_config)
      ocr.write(text.rstrip() + "\n\n")
  result = ocr.getvalue()
  ocr.close()
  return result

def extract_languages(text):
  """Detects languages in the given text."""
  try:
      languages = detect_langs(text)
      detected_languages = ", ".join([lang.lang for lang in languages if lang.prob > 0.1])
      return detected_languages or "No languages detected"
  except:
      return "Language detection failed"

def extract_text_from_pdf(pdf_path, lang=None, output_docx_path=None, output_txt_path=None):
  """
  Extracts text and tables from a PDF, attempting layout-based extraction first,
  then falling back to OCR if no significant text is found.
  Optionally exports the content to a DOCX file.
  
  Args:
      pdf_path: Path to the PDF file.
      lang: Optional language code for OCR (e.g., 'eng', 'fra', 'deu', etc.).
      output_docx_path: Optional path to save the extracted content as a DOCX file.
  
  Returns:
      A tuple containing:
      - The extracted text as a single string (for backward compatibility).
      - A string of detected languages.
  """
  all_extracted_blocks_per_page = []
  with pdfplumber.open(pdf_path) as pdf:
      doc = fitz.open(pdf_path)
      for i, pl_page in enumerate(pdf.pages):
          fmz_page = doc.load_page(i)
          page_blocks = extract_page_blocks(pl_page, fmz_page, i+1)
          all_extracted_blocks_per_page.append(page_blocks)
  
  # Check if any content was extracted via layout-based method
  has_content = any(any(block['content'].strip() for block in page_blocks if block['type'] == 'text') or 
                    any(block['type'] == 'table' for block in page_blocks) 
                    for page_blocks in all_extracted_blocks_per_page)

  # If no significant content from layout, try OCR
  if not has_content:
      print("Layout extraction failed, attempting OCR...")
      ocr_text = extract_ocr(pdf_path, lang)
      # For OCR, we'll treat the entire OCR output as one text block per page for DOCX
      all_extracted_blocks_per_page = []
      for i, page_ocr_text in enumerate(ocr_text.split("--- OCR Page ")[1:]):
          # Clean up the page header from OCR output
          page_ocr_text = page_ocr_text.split('\n', 1)[1].strip()
          if page_ocr_text:
              all_extracted_blocks_per_page.append([{'type': 'text', 'content': page_ocr_text}])
          else:
              all_extracted_blocks_per_page.append([]) # Empty page

  # Export to DOCX if path is provided
  if output_docx_path:
        export_to_txt(all_extracted_blocks_per_page, output_txt_path)

  # Reconstruct the text string from blocks for the original return value
  text_output_string = StringIO()
  for page_num, blocks in enumerate(all_extracted_blocks_per_page):
      text_output_string.write(f"--- Page {page_num + 1} ---\n")
      for block in blocks:
          if block['type'] == 'text':
              text_output_string.write("[Text]\n")
              text_output_string.write(block['content'] + "\n\n")
          elif block['type'] == 'table':
              text_output_string.write("[Table]\n")
              text_output_string.write(format_table(block['content']) + "\n\n")
  
  final_text_string = text_output_string.getvalue()
  text_output_string.close()

  detected_languages = extract_languages(final_text_string)
  return final_text_string, detected_languages
