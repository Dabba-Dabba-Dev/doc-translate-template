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
            r'^([a-zA-Z])\1{2,}$',  # Any letter repeated 3+ times
            r'^[^a-zA-Z0-9\s]+$',   # Only special characters
            r'^[0-9]+[a-zA-Z]{1,2}$', # Numbers followed by 1-2 letters
            r'^[a-zA-Z]{1,2}[0-9]+$', # 1-2 letters followed by numbers
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
    
    def _extract_lines_with_alignment(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Improved alignment detection that better handles right-aligned contact info"""
        results = []
        h_img, w_img = img.shape[:2]
        
        # Conservative thresholds
        CENTER_MARGIN_PX = 50          # Strict center threshold
        RIGHT_MARGIN_PX = 70           # Minimum right offset to count as right-aligned
        MIN_RIGHT_ALIGN_WIDTH = 0.4    # Minimum width for right-aligned blocks
        
        custom_config = r'--oem 3 --psm 6'
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
    
    def _extract_with_layout_improved(self, img: np.ndarray) -> str:
        """Extract text using the simpler, more effective approach"""
        try:
            # Use the same preprocessing as the original simple method
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
            
            # Optional: Remove small noise (from original simple method)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 100:
                    cv2.drawContours(thresh, [cnt], -1, 0, -1)
            
            # Use simple OCR config that works well
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            # Fallback to basic OCR
            return pytesseract.image_to_string(img, config='--oem 3 --psm 6')

    def _fix_line_continuations(self, text: str) -> Tuple[str, int]:
        """Fix line continuations with hyphenated words and tracking"""
        lines = [line for line in text.split('\n')]
        if not lines:
            return text, 0

        # Define hyphen characters (including soft hyphens and common OCR line-break hyphens)
        hyphens = {'-', '\u00AD', '–', '—', '‐', '‑'}
        continuation_count = 0
        fixed_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]
            # Skip empty lines (preserve them as-is)
            if not current_line.strip():
                fixed_lines.append(current_line)
                i += 1
                continue

            # Check if the line ends with a hyphen (ignore trailing spaces)
            stripped_current = current_line.rstrip()
            ends_with_hyphen = any(stripped_current.endswith(h) for h in hyphens)

            # Force merge if hyphen is found (no validity checks)
            if ends_with_hyphen and i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.lstrip()
                if next_stripped:
                    continuation_count += 1
                    # Remove the hyphen and merge with the next line's content
                    before_hyphen = stripped_current[:-1]  # Remove the hyphen
                    merged_line = before_hyphen + next_stripped  # Force merge
                    # Preserve original indentation of the current line
                    indent = current_line[:len(current_line) - len(current_line.lstrip())]
                    current_line = indent + merged_line
                    i += 1  # Skip the next line since we merged it

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
    
    def correct_with_languagetool(self, text: str) -> str:
        """Apply grammar correction using LanguageTool"""
        try:
            response = requests.post(
                "https://api.languagetool.org/v2/check",
                data={
                    "text": text,
                    "language": "auto"  # auto-detect language
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"LanguageTool API error: {response.status_code}")
                return text
            
            # Apply corrections
            suggested_text = text
            offset_correction = 0
            
            for match in response.json()["matches"]:
                if match['replacements']:
                    start = match['offset'] + offset_correction
                    end = start + match['length']
                    replacement = match['replacements'][0]['value']
                    suggested_text = suggested_text[:start] + replacement + suggested_text[end:]
                    offset_correction += len(replacement) - match['length']
            
            return suggested_text
            
        except Exception as e:
            print(f"LanguageTool correction failed: {e}")
            return text

    def extract_text(self, image_path: str) -> Dict[str, Any]:
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
        lines = self._extract_lines_with_alignment(img)

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
    
    def process_image(self, image_path: str, output_docx: str = None, apply_grammar_correction: bool = False) -> Dict[str, Any]:
        """Process image and optionally save to DOCX with grammar correction"""
        result = self.extract_text(image_path)
        
        # Apply grammar correction if requested
        final_text = result["text"]
        if apply_grammar_correction:
            print("✅ Applying grammar correction...")
            final_text = self.correct_with_languagetool(result["text"])
            result["corrected_text"] = final_text
        
        # Save to DOCX if output path provided
        if output_docx:
            self.save_to_docx(
                result["lines"],  # Pass the lines with alignment info, not the text string
                output_docx,
                result["orientation"]
            )
        
        return result

# === Main execution ===
if __name__ == "__main__":
    try:
        # Initialize enhanced OCR with DOCX support
        ocr = EnhancedOCRWithDocx()
        
        # Process the image/PDF with DOCX saving and optional grammar correction
        result = ocr.process_image(
            "diplome licence allemand.pdf",  # Fixed: consistent quotes
            "ocr_output.docx",
            apply_grammar_correction=True
        )
        
        print("🔍 Enhanced Extracted Text:")
        print("=" * 50)
        print(result["text"])
        print("=" * 50)
        
        print("\n📊 Processing Statistics:")
        print(f" - Line continuations fixed: {result['stats']['line_continuations']}")
        print(f" - Document orientation: {result['orientation']}")
        
        if 'corrected_text' in result:
            print("\n📝 Grammar Corrected Text:")
            print("=" * 50)
            print(result["corrected_text"])
            print("=" * 50)
        
        # Save results to text file as backup
        with open("enhanced_output_with_docx.txt", "w", encoding="utf-8") as f:
            f.write("=== ENHANCED EXTRACTED TEXT ===\n")
            f.write(result["text"])
            if 'corrected_text' in result:
                f.write("\n\n=== GRAMMAR CORRECTED TEXT ===\n")
                f.write(result["corrected_text"])
            f.write(f"\n\n=== PROCESSING STATISTICS ===\n")
            f.write(f"Line continuations fixed: {result['stats']['line_continuations']}\n")
            f.write(f"Document orientation: {result['orientation']}\n")
        
        print("\n💾 Results saved to:")
        print("  - enhanced_processed_document.docx (formatted document)")
        print("  - enhanced_output_with_docx.txt (text backup)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Possible solutions:")
        print("1. Make sure the image file exists and path is correct")
        print("2. Install required packages:")
        print("   pip install opencv-python pytesseract requests pyspellchecker python-docx")
        print("3. For PDF support: pip install pdf2image")
        print("4. Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        print("5. Check internet connection for LanguageTool API")