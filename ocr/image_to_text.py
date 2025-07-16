import cv2
import pytesseract
import requests
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.shared import Pt  # For setting font size in points
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class EnhancedOCRWithDocx:
    def __init__(self):
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Bullet symbols from original code
        self.bullet_symbols = {
            'circle': '•',
            'square': '■',
            'check': '✓',
            'cross': '✗'
        }
        
        # Common meaningless patterns for OCR errors
        self.meaningless_patterns = [
            r'^([a-zA-Z])\1\{2,\}$',  # Any letter repeated 3+ times
            r'^[^a-zA-Z0-9\s]+$',   # Only special character
        ]
        
        # Sentence ending punctuation
        self.sentence_endings = r'[.!?:;]'

    def _detect_orientation(self, img: np.ndarray) -> str:
        """Detect if the image/document is landscape or portrait"""
        height, width = img.shape[:2]
        return "landscape" if width > height else "portrait"
        
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)
    
    def _extract_lines_with_alignment(self, img: np.ndarray, lang: str = None) -> List[Dict[str, Any]]:
        """Improved alignment detection that better handles right-aligned contact info"""
        results = []
        h_img, w_img = img.shape[:2]
        
        # Conservative thresholds
        CENTER_MARGIN_PX = 40          # Strict center threshold
        RIGHT_MARGIN_PX = 70           # Minimum right offset to count as right-aligned
        MIN_RIGHT_ALIGN_WIDTH = 0.4    # Minimum width for right-aligned blocks
        
        # Build custom config with language if provided
        custom_config = r'--oem 3 --psm 6'
        if lang:
            custom_config += f' -l {lang}'
            
        data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

        current_line = {
            'text': '',
            'left': w_img,  # Initialize with max value
            'right': 0      # Initialize with min value
        }
        prev_line_num = -1

        for i in range(len(data['text'])):
            if int(data['conf'][i]) < 0:
                continue

            text = data['text'][i].strip()
            if not text:
                continue

            line_num = data['line_num'][i]
            left = data['left'][i]
            width = data['width'][i]
            right = left + width

            if line_num != prev_line_num:
                if current_line['text']:
                    # Calculate alignment with improved rules
                    line_width = current_line['right'] - current_line['left']
                    line_center = current_line['left'] + (line_width / 2)
                    
                    # Check for right-aligned text first (contact info, dates etc.)
                    right_offset = w_img - current_line['right']
                    if (right_offset <= RIGHT_MARGIN_PX and 
                        line_width < w_img * MIN_RIGHT_ALIGN_WIDTH):
                        alignment = "right"
                    # Then check for centered text
                    elif abs(line_center - (w_img / 2)) <= CENTER_MARGIN_PX:
                        alignment = "center"
                    else:
                        alignment = "left"
                        
                    results.append({
                        "text": current_line['text'].strip(),
                        "alignment": alignment
                    })
                
                # Start new line
                current_line = {
                    'text': text + ' ',
                    'left': left,
                    'right': right
                }
                prev_line_num = line_num
            else:
                # Continue current line
                current_line['text'] += text + ' '
                current_line['left'] = min(current_line['left'], left)
                current_line['right'] = max(current_line['right'], right)

        # Process the last line
        if current_line['text']:
            line_width = current_line['right'] - current_line['left']
            line_center = current_line['left'] + (line_width / 2)
            right_offset = w_img - current_line['right']
            
            if (right_offset <= RIGHT_MARGIN_PX and 
                line_width < w_img * MIN_RIGHT_ALIGN_WIDTH):
                alignment = "right"
            elif abs(line_center - (w_img / 2)) <= CENTER_MARGIN_PX:
                alignment = "center"
            else:
                alignment = "left"
                
            results.append({
                "text": current_line['text'].strip(),
                "alignment": alignment
            })

        return results
    
    def _fix_line_continuations(self, text: str) -> Tuple[str, int]:
        """Fix line continuations:
        - Merge lines if the current ends with a hyphen (forced merge)
        - Or if the next line starts with lowercase and the current line doesn't end with punctuation
        """
        lines = text.split('\n')
        if not lines:
            return text, 0

        hyphens = {'-', '\u00AD', '–', '—', '‐', '‑'}
        end_punctuation = {'.', '!', '?', ':', ';'}
        continuation_count = 0
        fixed_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]
            current_stripped = current_line.strip()

            # Preserve empty lines
            if not current_stripped:
                fixed_lines.append(current_line)
                i += 1
                continue

            ends_with_hyphen = any(current_stripped.endswith(h) for h in hyphens)

            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()

                # 1. Merge if ends with hyphen
                if ends_with_hyphen and next_stripped:
                    continuation_count += 1
                    before_hyphen = current_stripped[:-1]
                    merged_line = before_hyphen + next_stripped
                    indent = current_line[:len(current_line) - len(current_line.lstrip())]
                    current_line = indent + merged_line
                    i += 1  # Skip next line

                # 2. Merge if next starts with lowercase and current line doesn't end with punctuation
                elif (
                    next_stripped
                    and next_stripped[0].islower()
                    and current_stripped[-1] not in end_punctuation
                ):
                    continuation_count += 1
                    merged_line = current_stripped + ' ' + next_stripped
                    indent = current_line[:len(current_line) - len(current_line.lstrip())]
                    current_line = indent + merged_line
                    i += 1  # Skip next line

            fixed_lines.append(current_line)
            i += 1

        return '\n'.join(fixed_lines), continuation_count

    def save_to_docx(self, lines: List[Dict[str, Any]], output_path: str, orientation: str = "portrait"):
        doc = Document()
        section = doc.sections[0]

        # Set narrow margins (0.5 inches on all sides)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.header_distance = Inches(0.25)
        section.footer_distance = Inches(0.25)

        if orientation == "landscape":
            section.orientation = WD_ORIENT.LANDSCAPE
            section.page_width, section.page_height = section.page_height, section.page_width
        
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'  # Or your preferred font
        font.size = Pt(10)     # Set default font size to 10
        
        for line in lines:
            text = line.get('text', '')
            alignment = line.get('alignment', 'left')

            if not text.strip():
                doc.add_paragraph()
                continue

            p = doc.add_paragraph()
            p.add_run(text)

            # Set alignment
            if alignment == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif alignment == "right":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        doc.save(output_path)
        print(f"✅ Saved aligned document to {output_path}")
    
    def extract_text(self, image_path: str, lang: str = None) -> Dict[str, Any]:
        """Main extraction method. Supports image files and PDFs (first page)."""
        # Detect if PDF
        if image_path.lower().endswith('.pdf'):
            if convert_from_path is None:
                raise ImportError("pdf2image is required for PDF support. Install with 'pip install pdf2image'.")
            pages = convert_from_path(image_path, dpi=150, first_page=1, last_page=1)
            if not pages:
                raise FileNotFoundError(f"No pages found in PDF '{image_path}'.")
            img = np.array(pages[0])
            if img.shape[2] == 4:  # RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image file '{image_path}' not found.")

        # Detect orientation
        orientation = self._detect_orientation(img)

        # Extract lines with alignment information
        lines = self._extract_lines_with_alignment(img, lang)

        # Combine text for other processing
        combined_text = "\n".join([line['text'] for line in lines])
        
        # Fix line continuations and hyphenated words
        fixed_text, continuation_count = self._fix_line_continuations(combined_text)

        return {
            "lines": lines,  # This is what save_to_docx needs
            "text": fixed_text,  # This is the combined text
            "orientation": orientation,
            "stats": {
                "line_continuations": continuation_count,
                "sentence_merges": 0,
                "total_merges": continuation_count
            }
        }
    
    def process_image(self, image_path: str, output_docx: str = None, lang: str = None) -> Dict[str, Any]:
        """Process image and optionally save to DOCX with grammar correction"""
        result = self.extract_text(image_path, lang)
        
        final_text = result["text"]
        
        # Save to DOCX if output path provided
        if output_docx:
            self.save_to_docx(
                result["lines"],  # Pass the lines with alignment info, not the text string
                output_docx,
                result["orientation"]
            )
        
        return result