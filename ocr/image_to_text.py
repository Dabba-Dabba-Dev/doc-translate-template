import cv2
import pytesseract
import numpy as np
import requests
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class EnhancedOCRProcessor:
    """Enhanced OCR processor with advanced extraction capabilities"""
    
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
            r'^([a-zA-Z])\1{2,}$',  # Any letter repeated 3+ times
            r'^[^a-zA-Z0-9\s]+$',   # Only special characters
            r'^[0-9]+[a-zA-Z]{1,2}$', # Numbers followed by 1-2 letters
            r'^[a-zA-Z]{1,2}[0-9]+$', # 1-2 letters followed by numbers
        ]
        
        # Sentence ending punctuation
        self.sentence_endings = r'[.!?:;]'

    def extract_text_advanced(self, img: np.ndarray, languages: str = "eng") -> str:
        """
        Advanced text extraction with enhanced preprocessing and multi-language support
        Args:
            img: Input image as numpy array
            languages: Language codes (e.g., 'eng', 'ara', 'chi_sim', 'fra+eng')
        Returns:
            Extracted text as string
        """
        try:
            # Enhanced preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Denoising (helps with scanned docs)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Sharpening kernel for better text clarity
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Optimal thresholding
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Multi-language OCR config with layout preservation
            custom_config = f'''
                --oem 3 
                --psm 11 
                -l {languages}
                -c preserve_interword_spaces=1
                -c tessedit_do_invert=0
            '''
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"Advanced OCR extraction failed: {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(img, languages)

    def _fallback_extraction(self, img: np.ndarray, languages: str = "eng") -> str:
        """Fallback extraction method"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Basic preprocessing
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Basic OCR config
            custom_config = f'''
                --oem 3 
                --psm 11 
                -l {languages}
                -c preserve_interword_spaces=1
                -c tessedit_do_invert=0
            '''
            
            text = pytesseract.image_to_string(thresh, config=custom_config)
            return text
            
        except Exception as e:
            print(f"Fallback extraction failed: {e}")
            return ""

    def extract_text(self, image_path: str, languages: str = "eng", use_advanced: bool = True) -> str:
        """
        Main extraction method with multi-language support and advanced processing
        Args:
            image_path: Path to input image file
            languages: Language codes (e.g., 'eng', 'ara', 'chi_sim', 'fra+eng')
            use_advanced: Whether to use advanced extraction method
        Returns:
            Extracted text as string
        """
        try:
            # Handle PDF files
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
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")

            # Use advanced or basic extraction
            if use_advanced:
                text = self.extract_text_advanced(img, languages)
            else:
                text = self._fallback_extraction(img, languages)
            
            # Apply post-processing
            text = self._post_process_text(text)
            
            return text
        
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            return ""

    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text to improve quality"""
        if not text:
            return text
        
        # Fix line continuations
        text, _ = self._fix_line_continuations(text)
        
        
        # Remove meaningless patterns
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not self._is_meaningless_line(line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _is_meaningless_line(self, line: str) -> bool:
        """Check if a line contains meaningless OCR artifacts"""
        for pattern in self.meaningless_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _fix_line_continuations(self, text: str) -> Tuple[str, int]:
        """Fix line continuations with hyphenated words"""
        lines = [line for line in text.split('\n')]
        if not lines:
            return text, 0

        # Define hyphen characters
        hyphens = {'-', '\u00AD', '–', '—', '‐', '‑'}
        continuation_count = 0
        fixed_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]

            # Check if the line ends with a hyphen
            stripped_current = current_line.rstrip()
            ends_with_hyphen = any(stripped_current.endswith(h) for h in hyphens)

            # Merge if hyphen is found
            if ends_with_hyphen and i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.lstrip()
                if next_stripped:
                    continuation_count += 1
                    # Remove the hyphen and merge
                    before_hyphen = stripped_current[:-1]
                    merged_line = before_hyphen + next_stripped
                    # Preserve original indentation
                    indent = current_line[:len(current_line) - len(current_line.lstrip())]
                    current_line = indent + merged_line
                    i += 1  # Skip the next line

            fixed_lines.append(current_line)
            i += 1

        return '\n'.join(fixed_lines), continuation_count

    def _detect_orientation(self, img: np.ndarray) -> str:
        """Detect if the image/document is landscape or portrait"""
        height, width = img.shape[:2]
        return "landscape" if width > height else "portrait"

    def _extract_lines_with_alignment(self, img: np.ndarray, languages: str = "eng") -> List[Dict[str, Any]]:
        """Extract lines with alignment detection"""
        results = []
        h_img, w_img = img.shape[:2]

        # Use advanced preprocessing
        processed_img = self._preprocess_image(img)

        # Alignment detection thresholds
        CENTER_MARGIN_PX = 50
        RIGHT_MARGIN_PX = 70
        MIN_RIGHT_ALIGN_WIDTH = 0.4

        # OCR config for data extraction
        custom_config = f'''
            --oem 3 
            --psm 11 
            -l {languages}
            -c preserve_interword_spaces=1 
            -c tessedit_do_invert=0
        '''
        
        data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)

        current_line = {
            'text': '',
            'left': w_img,
            'right': 0
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
                    # Calculate alignment
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
                        "text": current_line['text'],
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
                "text": current_line['text'],
                "alignment": alignment
            })

        return results

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Advanced image preprocessing for OCR"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def save_to_docx(self, lines: List[Dict[str, Any]], output_path: str, orientation: str = "portrait"):
        """Save extracted lines to DOCX with proper alignment"""
        doc = Document()
        section = doc.sections[0]

        if orientation == "landscape":
            section.orientation = WD_ORIENT.LANDSCAPE
            section.page_width, section.page_height = section.page_height, section.page_width

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

    
    def process_image(self, image_path: str, languages: str = "eng", output_docx: str = None, use_advanced_method: bool = True) -> Dict[str, Any]:
        """
        Process image with full functionality
        Args:
            image_path: Path to input image
            languages: Language codes
            output_docx: Path to save DOCX file
            use_advanced_method: Whether to use advanced extraction
        Returns:
            Dictionary with extraction results
        """
        try:
            # Load image
            if image_path.lower().endswith('.pdf'):
                if convert_from_path is None:
                    raise ImportError("pdf2image is required for PDF support.")
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

            # Extract text using selected method
            if use_advanced_method:
                print("🚀 Using advanced extraction method...")
                combined_text = self.extract_text_advanced(img, languages)
                lines = self._extract_lines_with_alignment(img, languages)
            else:
                print("📝 Using standard extraction method...")
                lines = self._extract_lines_with_alignment(img, languages)
                combined_text = "\n".join([line['text'] for line in lines])
            
            # Fix line continuations
            fixed_text, continuation_count = self._fix_line_continuations(combined_text)

            # Apply grammar correction if requested
            final_text = fixed_text

            # Save to DOCX if requested
            if output_docx:
                self.save_to_docx(lines, output_docx, orientation)

            return {
                "lines": lines,
                "text": final_text,
                "raw_text": combined_text,
                "orientation": orientation,
                "languages": languages,
                "extraction_method": "advanced" if use_advanced_method else "standard",
                "stats": {
                    "line_continuations": continuation_count,
                    "word_count": len(final_text.split()),
                    "char_count": len(final_text)
                }
            }

        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                "error": str(e),
                "text": "",
                "lines": [],
                "stats": {}
            }

    def compare_extraction_methods(self, image_path: str, languages: str = "eng") -> Dict[str, Any]:
        """Compare advanced vs standard extraction methods"""
        print("🔍 Comparing extraction methods...")
        
        # Extract with both methods
        advanced_result = self.process_image(image_path, languages, use_advanced_method=True)
        standard_result = self.process_image(image_path, languages, use_advanced_method=False)
        
        return {
            "advanced": {
                "text": advanced_result["text"],
                "word_count": advanced_result["stats"]["word_count"],
                "char_count": advanced_result["stats"]["char_count"],
                "line_continuations": advanced_result["stats"]["line_continuations"]
            },
            "standard": {
                "text": standard_result["text"],
                "word_count": standard_result["stats"]["word_count"],
                "char_count": standard_result["stats"]["char_count"],
                "line_continuations": standard_result["stats"]["line_continuations"]
            }
        }


# Example usage
if __name__ == "__main__":
    try:
        # Initialize enhanced OCR processor
        ocr = EnhancedOCRProcessor()
        
        # Process with multiple languages (English + German example)
        result = ocr.process_image(
            "Screenshot 2025-07-06 095609.png",
            languages="eng+deu",
            output_docx="enhanced_output.docx",
            use_advanced_method=True
        )
        
        print("Enhanced OCR Results:")
        print("=" * 50)
        print(result["text"])
        print("=" * 50)
        
        print(f"\nStatistics:")
        print(f"- Words: {result['stats']['word_count']}")
        print(f"- Characters: {result['stats']['char_count']}")
        print(f"- Line continuations fixed: {result['stats']['line_continuations']}")
        print(f"- Orientation: {result['orientation']}")
        print(f"- Languages: {result['languages']}")
        print(f"- Method: {result['extraction_method']}")
        print(f"- Grammar corrected: {result['grammar_corrected']}")
        
    except Exception as e:
        print(f"Error: {e}")

# Common language codes for reference:
# 'eng' - English
# 'ara' - Arabic  
# 'chi_sim' - Chinese Simplified
# 'chi_tra' - Chinese Traditional
# 'fra' - French
# 'deu' - German
# 'spa' - Spanish
# 'rus' - Russian
# 'jpn' - Japanese
# 'kor' - Korean
# 'hin' - Hindi
# 'ita' - Italian
# 'por' - Portuguese
# 'tur' - Turkish
# 'pol' - Polish
# 'nld' - Dutch
# 'swe' - Swedish
# 'dan' - Danish
# 'nor' - Norwegian
# 'fin' - Finnish
# 'ces' - Czech
# 'hun' - Hungarian
# 'ron' - Romanian
# 'bul' - Bulgarian
# 'hrv' - Croatian
# 'slv' - Slovenian
# 'slk' - Slovak
# 'est' - Estonian
# 'lav' - Latvian
# 'lit' - Lithuanian
# 'ell' - Greek
# 'heb' - Hebrew
# 'tha' - Thai
# 'vie' - Vietnamese
# 'ind' - Indonesian
# 'msa' - Malay
# 'tgl' - Tagalog
# 'ukr' - Ukrainian
# 'bel' - Belarusian
# 'cat' - Catalan
# 'eus' - Basque
# 'glg' - Galician
# 'aze' - Azerbaijani
# 'uzb' - Uzbek
# 'kaz' - Kazakh
# 'kir' - Kyrgyz
# 'tgk' - Tajik
# 'mon' - Mongolian
# 'nep' - Nepali
# 'ben' - Bengali
# 'guj' - Gujarati
# 'pan' - Punjabi
# 'tam' - Tamil
# 'tel' - Telugu
# 'kan' - Kannada
# 'mal' - Malayalam
# 'ori' - Odia
# 'asm' - Assamese
# 'mar' - Marathi
# 'sin' - Sinhala
# 'mya' - Myanmar
# 'khm' - Khmer
# 'lao' - Lao
# 'bod' - Tibetan
# 'dzo' - Dzongkha
# 'fas' - Persian
# 'pus' - Pashto
# 'urd' - Urdu
# 'snd' - Sindhi
# 'amh' - Amharic
# 'tir' - Tigrinya
# 'orm' - Oromo
# 'som' - Somali
# 'swa' - Swahili
# 'afr' - Afrikaans