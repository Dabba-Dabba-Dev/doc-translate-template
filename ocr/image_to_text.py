import cv2
import pytesseract
import numpy as np
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
import re
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION, WD_ORIENT

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class ImageToText:
    def __init__(self):
        self.bullet_symbols = {
            'circle': '•',
            'square': '■',
            'check': '✓',
            'cross': '✗'
        }
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Common meaningless patterns for OCR errors
        self.meaningless_patterns = [
            r'^([a-zA-Z])\1{2,}$', # Any letter repeated 3+ times
            r'^[^a-zA-Z0-9\s]+$', # Only special characters
            r'^[0-9]+[a-zA-Z]{1,2}$', # Numbers followed by 1-2 letters
            r'^[a-zA-Z]{1,2}[0-9]+$', # 1-2 letters followed by numbers
        ]
        
        # Sentence ending punctuation
        self.sentence_endings = r'[.!?:;]'
        
    def _detect_orientation(self, img: np.ndarray) -> str:
        """Detect if the image/document is landscape or portrait"""
        height, width = img.shape[:2]
        return "landscape" if width > height else "portrait"
        
    def _fix_line_continuations(self, text: str) -> str:
        lines = [line for line in text.split('\n')]
        if not lines:
            return text

        # Define hyphen characters (including soft hyphens and common OCR line-break hyphens)
        hyphens = {'-', '\u00AD', '–', '—', '‐', '‑'}

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
                    # Remove the hyphen and merge with the next line's content
                    before_hyphen = stripped_current[:-1]  # Remove the hyphen
                    merged_line = before_hyphen + next_stripped  # Force merge

                    # Preserve original indentation of the current line
                    indent = current_line[:len(current_line) - len(current_line.lstrip())]
                    current_line = indent + merged_line
                    i += 1  # Skip the next line since we merged it

            fixed_lines.append(current_line)
            i += 1

        return '\n'.join(fixed_lines)
    
    def _looks_like_valid_word(self, word: str) -> bool:
        """
        Basic heuristic to check if a word looks valid based on vowel/consonant ratio.
        """
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for char in word if char in vowels)
        
        if vowel_count == 0:
            return False
        
        vowel_ratio = vowel_count / len(word)
        return 0.1 <= vowel_ratio <= 0.8

    def _merge_continuous_sentences(self, lines: List[str]) -> List[str]:
        """
        Merge lines with prioritized conditions:
        1. If current line has only one word -> merge
        2. If not ending with punctuation AND next starts lowercase -> merge
        3. If last word uppercase AND next starts uppercase -> merge
        4. If not ending with punctuation AND next starts uppercase -> merge (lowest priority)
        """
        if not lines:
            return lines

        merged = []
        i = 0

        while i < len(lines):
            original_line = lines[i]
            stripped_line = original_line.strip()
            
            if not stripped_line:
                merged.append(original_line)
                i += 1
                continue

            # Initialize conditions
            ends_with_punct = any(stripped_line.endswith(p) for p in {'.', '!', '?', ':', ';', ')', ']', '}', '"', "'"})
            has_one_word = len(stripped_line.split()) == 1
            current_words = stripped_line.split()
            last_word_upper = bool(current_words) and current_words[-1][0].isupper()

            merged_line = original_line

            while i + 1 < len(lines):
                next_original = lines[i + 1]
                next_stripped = next_original.strip()
                
                if not next_stripped:
                    break

                next_starts_lower = next_stripped[0].islower()
                next_starts_upper = next_stripped[0].isupper()

                # Check conditions in priority order
                if has_one_word:
                    # Highest priority - single word lines always merge
                    should_merge = True
                elif not ends_with_punct and next_starts_lower:
                    # Second priority - standard sentence continuation
                    should_merge = True
                elif last_word_upper and next_starts_upper:
                    # Third priority - uppercase sequences
                    should_merge = True
                elif not ends_with_punct and next_starts_upper:
                    # Lowest priority - uppercase after non-punctuation
                    should_merge = True
                else:
                    should_merge = False

                if should_merge:
                    # Merge with exactly one space
                    merged_line = merged_line.rstrip() + ' ' + next_stripped
                    i += 1
                    
                    # Update conditions for next iteration
                    stripped_line = merged_line.strip()
                    ends_with_punct = any(stripped_line.endswith(p) for p in {'.', '!', '?', ':', ';', ')', ']', '}', '"', "'"})
                    has_one_word = False
                    current_words = stripped_line.split()
                    last_word_upper = bool(current_words) and current_words[-1][0].isupper()
                else:
                    break

            merged.append(merged_line)
            i += 1

        return merged

    def _looks_like_new_sentence(self, line: str) -> bool:
        """
        Check if a line looks like it starts a new sentence
        """
        line = line.strip()
        if not line:
            return False
            
        # Check if it starts with a capital letter
        if line[0].isupper():
            return True
            
        # Check if it starts with a bullet point
        if line.startswith(('•', '■', '✓', '✗', '-', '*')):
            return True
            
        # Check if it starts with a number (numbered list)
        if re.match(r'^\d+\.', line):
            return True
            
        return False

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)

    def _is_meaningless_word(self, word: str) -> bool:
        """Check if a word is likely a meaningless OCR error"""
        if not word or len(word) < 2:
            return False
            
        # Check against meaningless patterns
        for pattern in self.meaningless_patterns:
            if re.match(pattern, word):
                return True
        
        # Check for words with unusual character repetition
        if len(word) >= 3:
            char_counts = {}
            for char in word.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # If any character appears more than 50% of the time and word is short
            max_char_ratio = max(char_counts.values()) / len(word)
            if max_char_ratio > 0.5 and len(word) <= 5:
                return True
        
        return False

    def _filter_nonsense_words(self, text: str) -> str:
        """
        Enhanced filtering focusing on short meaningless words and OCR errors
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Skip lines that are:
            # 1. Empty, OR
            # 2. Single non-alphanumeric character (like |, /), OR
            # 3. Single character that's not a meaningful word (like "a", "I" are kept)
            if (not stripped_line or 
                (len(stripped_line) == 1 and not stripped_line.isalnum()) or
                (len(stripped_line) == 1 and stripped_line.isalpha() and stripped_line.lower() not in {'a', 'i'})):
                continue
            # Split into words and non-words (punctuation, spaces)
            tokens = re.findall(r'(\w+|\W+)', line)
            cleaned_tokens = []
            
            for token in tokens:
                # Keep non-word tokens as-is (spaces, punctuation)
                if not token.strip() or not re.match(r'\w+', token):
                    cleaned_tokens.append(token)
                    continue
                
                # Check if word should be kept
                lower_token = token.lower()
                
                # Check if it's a meaningless word
                if self._is_meaningless_word(token):
                    cleaned_tokens.append('')  # Remove meaningless words
                    continue
                
                # For other words, apply original logic
                is_valid = (
                    len(token) >= 3 or  # Keep words 3+ characters
                    token.istitle() or  # proper nouns
                    token.isupper() or  # acronyms
                    token in self.spell or  # in dictionary
                    any(char.isdigit() for char in token) or  # alphanumeric
                    (len(token) > 2 and token.lower() in self.spell)  # check lowercase
                )
                
                if is_valid:
                    cleaned_tokens.append(token)
                else:
                    # Replace nonsense words with empty string
                    cleaned_tokens.append('')
                    
            # Reconstruct the line while cleaning up extra spaces
            cleaned_line = ''.join(cleaned_tokens)
            # Clean up multiple spaces
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
            cleaned_lines.append(cleaned_line)
            
        return '\n'.join(cleaned_lines)

    def _get_text_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Helper to get text boxes from pytesseract"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')
            return [(data['left'][i], data['top'][i], 
                     data['left'][i] + data['width'][i], 
                     data['top'][i] + data['height'][i]) 
                    for i, word in enumerate(data['text']) if word.strip()]
        except Exception:
            return []

    def _overlaps_text(self, box: Tuple[int, int, int, int], 
                      text_boxes: List[Tuple[int, int, int, int]], 
                      threshold: float = 0.3) -> bool:
        """Check if box overlaps with text regions"""
        x1, y1, x2, y2 = box
        for tx1, ty1, tx2, ty2 in text_boxes:
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih > threshold * (x2 - x1) * (y2 - y1):
                return True
        return False

    def _filter_overlapping_boxes(self, boxes: List[Dict[str, Any]], max_items: int = 2) -> List[Dict[str, Any]]:
        """Filter overlapping boxes keeping the largest ones"""
        def overlap(a, b):
            return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

        filtered = []
        for i, item in enumerate(boxes):
            keep = True
            for j, other in enumerate(boxes):
                if i != j and overlap(item['box'], other['box']):
                    a_area = (item['box'][2]-item['box'][0])*(item['box'][3]-item['box'][1])
                    b_area = (other['box'][2]-other['box'][0])*(other['box'][3]-other['box'][1])
                    if a_area < b_area:
                        keep = False
                        break
            if keep and (max_items is None or len(filtered) < max_items):
                filtered.append(item)
        return filtered

    def detect_bullet_points(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect circular, square, check, and cross bullet points"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        bullet_boxes = []

        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=7, maxRadius=20
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                area = np.pi * r * r
                if area > 120:  # Filter out small circles
                    bullet_boxes.append({'box': [x - r, y - r, x + r, y + r], 'type': 'circle'})

        # Detect squares using contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and area > 60 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h
                if 0.8 < aspect < 1.2 and w > 10 and h > 10:
                    bullet_boxes.append({'box': [x, y, x + w, y + h], 'type': 'square'})

        # Detect check marks and crosses using template matching
        templates = [
            ('check', np.array([[0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]], dtype=np.uint8) * 255),
            ('cross', np.array([[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1]], dtype=np.uint8) * 255)
        ]
        for typ, template in templates:
            for scale in [15, 20, 25]:
                tpl = cv2.resize(template, (scale, scale), interpolation=cv2.INTER_NEAREST)
                res = cv2.matchTemplate(thresh, tpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res > 0.7)
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    if scale > 10:
                        bullet_boxes.append({'box': [x, y, x + scale, y + scale], 'type': typ})

        # Remove boxes that overlap with text boxes
        text_boxes = self._get_text_boxes(img)
        filtered = []
        for item in bullet_boxes:
            if not self._overlaps_text(tuple(item['box']), text_boxes, threshold=0.5):
                filtered.append(item)
        return self._filter_overlapping_boxes(filtered, max_items=None)

    def _insert_bullet_points(self, text: str, img: np.ndarray, bullet_items: List[Dict[str, Any]]) -> str:
        """Insert bullet point symbols into the text"""
        lines = text.splitlines()
        h, _ = img.shape[:2]
        
        # Get line positions from OCR data or estimate
        try:
            processed = self._preprocess_image(img)
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')
            line_positions = []
            for i, word in enumerate(data['text']):
                if word.strip() and (not line_positions or abs(data['top'][i] - line_positions[-1]) > 10):
                    line_positions.append(data['top'][i])
        except Exception:
            line_positions = [int(h * i / max(1, len(lines))) for i in range(len(lines))]

        # Map bullets to lines
        bullet_lines = {}
        for item in bullet_items:
            x1, y1, x2, y2 = item['box']
            bullet_y = (y1 + y2) // 2
            closest_line = min(enumerate(line_positions), key=lambda x: abs(x[1] - bullet_y), default=(0, 0))[0]
            bullet_lines[closest_line] = item['type']

        # Insert bullet symbols
        for idx, typ in bullet_lines.items():
            if 0 <= idx < len(lines):
                symbol = self.bullet_symbols.get(typ, '•')
                if not lines[idx].strip().startswith(symbol):
                    lines[idx] = f"{symbol} " + lines[idx].lstrip()

        return "\n".join(lines)

    def _extract_with_layout_improved(self, img: np.ndarray) -> str:
        try:
            processed = self._preprocess_image(img)
            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
            )
            
            # Dictionary to hold lines {y_position: [words]}
            lines = {}
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    y_pos = data['top'][i]
                    line_key = y_pos // 10  # Group words into lines (adjust 10 as needed)
                    
                    if line_key not in lines:
                        lines[line_key] = []
                    
                    lines[line_key].append({
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'width': data['width'][i]
                    })

            # Sort lines by vertical position
            sorted_lines = sorted(lines.items(), key=lambda x: x[0])
            
            output_lines = []
            for line_key, words in sorted_lines:
                # Sort words horizontally
                words.sort(key=lambda x: x['left'])
                
                # Calculate indentation (first word's position)
                if words:
                    first_word_pos = words[0]['left']
                    indent_spaces = first_word_pos // 10  # Convert pixels to spaces
                    
                    # Build the line
                    line_text = ' ' * indent_spaces
                    prev_end = first_word_pos
                    
                    for word in words:
                        # Add spacing between words
                        gap = word['left'] - prev_end
                        if gap > 5:  # Only add spaces if significant gap
                            line_text += ' ' * max(1, gap // 10)
                        line_text += word['text']
                        prev_end = word['left'] + word['width']
                    
                    output_lines.append(line_text)

            return '\n'.join(output_lines)
            
        except Exception as e:
            print(f"Layout extraction failed: {e}")
            return pytesseract.image_to_string(img, config='--oem 3 --psm 6')
    def save_to_docx(self, text: str, output_path: str,  orientation: str = "portrait"):
        """Save the extracted text to a DOCX file with proper formatting and orientation"""
        doc = Document()
        
        # Set page orientation
        section = doc.sections[0]
        if orientation == "landscape":
            section.orientation = WD_ORIENT.LANDSCAPE
            # Swap width and height for landscape
            section.page_width, section.page_height = section.page_height, section.page_width
        
        
        # Add extracted text with preserved spacing
        lines = text.split('\n')
        for line in lines:
            # Handle empty lines
            if not line.strip():
                doc.add_paragraph()
                continue
            
            # Create paragraph
            p = doc.add_paragraph()
            
            # Handle bullet points
            if line.strip().startswith(('•', '■', '✓', '✗')):
                p.style = 'List Bullet'
                p.add_run(line.strip()[1:].strip())  # Remove bullet symbol
            else:
                # Calculate indentation from leading spaces
                stripped_line = line.lstrip()
                leading_spaces = len(line) - len(stripped_line)
                
                # Set indentation (each 4 spaces = 0.25 inch)
                if leading_spaces > 0:
                    indent_inches = (leading_spaces / 4) * 0.25
                    p.paragraph_format.left_indent = Inches(indent_inches)
                
                # Add the text, preserving internal spacing
                p.add_run(stripped_line)
        
        # Save the document
        doc.save(output_path)
        print(f"✅ Document saved to: {output_path} (Orientation: {orientation})")

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

        # Extract text with improved layout preservation
        text = self._extract_with_layout_improved(img)

        # Fix line continuations and hyphenated words
        text = self._fix_line_continuations(text)

        # Filter out nonsense words
        text = self._filter_nonsense_words(text)

        # Detect bullet points
        bullet_items = self.detect_bullet_points(img)
        text_with_bullets = self._insert_bullet_points(text, img, bullet_items)

        return {
            "text": text_with_bullets.strip(),
            "bullets": [item['box'] for item in bullet_items],
            "orientation": orientation
        }

    def process_image(self, image_path: str, output_docx: str = None) -> Dict[str, Any]:
        """Process image and optionally save to DOCX"""
        result = self.extract_text(image_path)
        
        # Save to DOCX if output path provided
        if output_docx:
            self.save_to_docx(
                result["text"], 
                output_docx, 
                result["orientation"]
            )
        
        return result
    
if __name__ == "__main__":
    ocr = ImageToText()
    
    # Process the image/PDF
    result = ocr.process_image('Screenshot 2025-07-06 095609.png', 'processed_document.docx')
    
    print("🔍 Extracted Text:\n", result["text"])
    print("📄 Detected bullet points:", result["bullets"])
    print("🔄 Document orientation:", result["orientation"])
    print("💾 Text also saved to 'processed_document.docx'")
    
    # Still save to txt for backup
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])