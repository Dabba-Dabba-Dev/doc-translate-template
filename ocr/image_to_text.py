import cv2
import pytesseract
import numpy as np
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from spellchecker import SpellChecker
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from io import StringIO

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class EnhancedOCRProcessor:
    def __init__(self):
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Bullet symbols
        self.bullet_symbols = {
            'circle': '•',
            'square': '■',
            'check': '✓',
            'cross': '✗'
        }
        
        # Common meaningless patterns for OCR errors
        self.meaningless_patterns = [
            r'^([a-zA-Z])\1\{2,\}$',  # Any letter repeated 3+ times
            r'^([a-zA-Z])\1$',
            r'^[^a-zA-Z0-9\s]+$',   # Only special character
        ]
        
        # Sentence ending punctuation
        self.sentence_endings = r'[.!?:;]'

    def _is_pdf(self, file_path: str) -> bool:
        """Check if the file is a PDF"""
        return file_path.lower().endswith('.pdf')

    def _is_image(self, file_path: str) -> bool:
        """Check if the file is a supported image"""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return os.path.splitext(file_path.lower())[1] in image_exts

    def _detect_orientation(self, img: np.ndarray) -> str:
        """Detect if the image/document is landscape or portrait"""
        height, width = img.shape[:2]
        return "landscape" if width > height else "portrait"
        
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)
    
    
    def _fix_line_continuations(self, text: str) -> Tuple[str, int]:
        """Fix line continuations while preserving original spacing"""
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

            # Preserve empty lines exactly as they are
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

    def _extract_lines_with_alignment(self, img: np.ndarray, lang: str = None) -> List[Dict[str, Any]]:
        """Improved alignment detection that preserves original spacing"""
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
            'right': 0,     # Initialize with min value
            'spaces': []     # Track spaces between words
        }
        prev_line_num = -1
        prev_right = 0      # Track right edge of previous word

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
                        "alignment": alignment,
                        "left": current_line['left'],
                        "spaces": current_line['spaces']  # Save space information
                    })

                # Start new line
                current_line = {
                    'text': text,
                    'left': left,
                    'right': right,
                    'spaces': []
                }
                prev_line_num = line_num
                prev_right = right
            else:
                # Calculate space between words
                space = left - prev_right
                current_line['spaces'].append(space)

                # Continue current line
                current_line['text'] += ' ' + text
                current_line['left'] = min(current_line['left'], left)
                current_line['right'] = max(current_line['right'], right)
                prev_right = right

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
                "alignment": alignment,
                "left": current_line['left'],
                "spaces": current_line['spaces']
            })

        return results

    def save_to_txt(self, all_pages_results: List[Dict[str, Any]], output_path: str):
        """Save OCR results to a text file with more natural preserved spacing"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_num, page_result in enumerate(all_pages_results):
                # Add page separator for multi-page documents
                if len(all_pages_results) > 1:
                    f.write(f"--- Page {page_num + 1} ---\n")
                
                for line in page_result["lines"]:
                    text_parts = line['text'].split()
                    spaces = line.get('spaces', [])
                    
                    if not text_parts:
                        f.write('\n')  # Preserve blank lines
                        continue
                    
                    # Calculate indentation based on alignment
                    if line['alignment'] == 'center':
                        # For centered text, calculate appropriate padding
                        line_width = sum(len(part) for part in text_parts) + len(spaces)
                        indent = max(0, (80 - line_width) // 2)  # Assuming 80-char width
                        f.write(' ' * indent)
                    elif line['alignment'] == 'right':
                        # For right-aligned text, calculate appropriate padding
                        line_width = sum(len(part) for part in text_parts) + len(spaces)
                        indent = max(0, 80 - line_width)  # Right-align within 80 chars
                        f.write(' ' * indent)
                    else:
                        # For left-aligned, use more subtle indentation
                        indent = max(0, line['left'] // 8)  # More gentle conversion (8px per space)
                        f.write(' ' * indent)
                    
                    # Write first word
                    f.write(text_parts[0])
                    
                    # Write remaining words with proportional spacing
                    for i in range(1, len(text_parts)):
                        if i-1 < len(spaces):
                            # More natural space calculation (10px per space)
                            space_count = max(1, min(4, spaces[i-1] // 10))
                        else:
                            space_count = 1  # Default to single space
                        f.write(' ' * space_count + text_parts[i])
                    
                    f.write('\n')
                
                # Add separation between pages
                if len(all_pages_results) > 1:
                    f.write('\n')
        print(f"✅ Saved text to {output_path}")
    def _process_single_image(self, img_path: str, lang: str = None) -> Dict[str, Any]:
        """Process a single image file"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{img_path}' not found or could not be read.")
        
        orientation = self._detect_orientation(img)
        processed_img = self._preprocess_image(img)
        lines = self._extract_lines_with_alignment(processed_img, lang)
        combined_text = "\n".join([line['text'] for line in lines])
        fixed_text, continuation_count = self._fix_line_continuations(combined_text)
        
        return {
            "lines": lines,
            "text": fixed_text,
            "orientation": orientation,
            "stats": {
                "line_continuations": continuation_count,
                "sentence_merges": 0,
                "total_merges": continuation_count
            },
            "page_number": 1
        }

    def _process_pdf(self, pdf_path: str, lang: str = None) -> List[Dict[str, Any]]:
        """Process all pages of a PDF file"""
        if convert_from_path is None:
            raise ImportError("pdf2image is required for PDF support. Install with 'pip install pdf2image'.")
        
        all_pages_results = []
        pages = convert_from_path(pdf_path, dpi=300, thread_count=4)
        
        for page_num, page in enumerate(pages):
            img = np.array(page)
            if img.shape[2] == 4:  # RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            orientation = self._detect_orientation(img)
            processed_img = self._preprocess_image(img)
            lines = self._extract_lines_with_alignment(processed_img, lang)
            combined_text = "\n".join([line['text'] for line in lines])
            fixed_text, continuation_count = self._fix_line_continuations(combined_text)
            
            all_pages_results.append({
                "lines": lines,
                "text": fixed_text,
                "orientation": orientation,
                "stats": {
                    "line_continuations": continuation_count,
                    "sentence_merges": 0,
                    "total_merges": continuation_count
                },
                "page_number": page_num + 1
            })
        
        return all_pages_results

    def extract_text(self, file_path: str, lang: str = None) -> List[Dict[str, Any]]:
        """
        Main extraction method that handles both image files and PDFs.
        Returns a list of page results (even for single images for consistency).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        
        if self._is_pdf(file_path):
            return self._process_pdf(file_path, lang)
        elif self._is_image(file_path):
            return [self._process_single_image(file_path, lang)]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def process_file(self, file_path: str, output_txt: str = None, lang: str = None) -> List[Dict[str, Any]]:
        """
        Process any supported file (image or PDF) with optional text output.
        Returns list of page results.
        """
        results = self.extract_text(file_path, lang)
        
        if output_txt:
            self.save_to_txt(results, output_txt)
            
        return results