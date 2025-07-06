import cv2
import pytesseract
import requests
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker

class EnhancedOCR:
    def __init__(self):
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Common meaningless patterns for OCR errors
        self.meaningless_patterns = [
            r'^([a-zA-Z])\1{2,}$',  # Any letter repeated 3+ times
            r'^[^a-zA-Z0-9\s]+$',   # Only special characters
        ]
        
        # Sentence ending punctuation
        self.sentence_endings = r'[.!?:;]'
        
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for OCR (from reference code)"""
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
        while preserving EXACT original spacing and indentation.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip completely empty lines (preserve newlines)
            if not line.strip():
                cleaned_lines.append(line)
                continue
                
            # Track original character positions
            cleaned_chars = []
            i = 0
            n = len(line)
            
            while i < n:
                # Handle non-word characters (spaces, punctuation)
                if not line[i].isalnum():
                    cleaned_chars.append(line[i])
                    i += 1
                    continue
                    
                # Extract full word starting at position i
                word_start = i
                while i < n and line[i].isalnum():
                    i += 1
                word = line[word_start:i]
                
                # Check if word should be kept
                keep_word = False
                if word:  # Only process if we actually found a word
                    lower_word = word.lower()
                    
                    # Check if it's a meaningful word
                    if not self._is_meaningless_word(word):
                        keep_word = (
                            len(word) >= 3 or
                            word.istitle() or
                            word.isupper() or
                            word in self.spell or
                            any(char.isdigit() for char in word) or
                            (len(word) > 2 and lower_word in self.spell)
                        )
                
                # Preserve either the word or its original spacing
                if keep_word:
                    cleaned_chars.append(word)
                else:
                    # Replace word with empty string BUT PRESERVE POSITION
                    # This maintains alignment of surrounding text
                    cleaned_chars.append(' ' * len(word) if line[word_start].isspace() else '')
                    
            # Reconstruct line exactly with original spacing
            cleaned_line = ''.join(cleaned_chars)
            
            # Only clean redundant spaces if they're not part of indentation
            if cleaned_line.strip():
                # Collapse only MID-LINE multiple spaces
                cleaned_line = re.sub(r'(?<=\S)\s+(?=\S)', ' ', cleaned_line)
            
            cleaned_lines.append(cleaned_line)
            
        return '\n'.join(cleaned_lines)

    def _extract_with_layout_improved(self, img: np.ndarray) -> str:
        """Extract text with improved layout preservation (from reference code)"""
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

    def _fix_line_continuations(self, text: str) -> Tuple[str, int]:
        """Fix line continuations with hyphenated words and tracking"""
        lines = [line for line in text.split('\n') if line.strip()]
        if not lines:
            return text, 0

        hyphens = {'-', '\u00AD', '–', '—', '‐', '‑', '~'}
        continuation_count = 0

        fixed_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i].rstrip()
            
            if not current_line:
                fixed_lines.append(lines[i])
                i += 1
                continue

            # Check for hyphen at end (ignore trailing spaces)
            stripped_current = current_line.rstrip()
            ends_with_hyphen = any(stripped_current.endswith(h) for h in hyphens)
            
            # Also check for comma continuation
            ends_with_comma = stripped_current.endswith(',')
            
            # Check if next line starts lowercase
            starts_lower = (i + 1 < len(lines) and 
                        lines[i+1].strip() and 
                        lines[i+1].strip()[0].islower())

            if (ends_with_hyphen or ends_with_comma or starts_lower) and i + 1 < len(lines):
                next_line = lines[i+1].lstrip()
                if next_line:
                    # Count this continuation
                    continuation_count += 1
                    
                    # Remove the hyphen if present
                    if ends_with_hyphen:
                        before_hyphen = stripped_current[:-1]
                    else:
                        before_hyphen = stripped_current
                    
                    # Merge with next line
                    merged_line = before_hyphen + ' ' + next_line
                    fixed_lines.append(merged_line)
                    i += 2  # Skip next line
                    continue

            fixed_lines.append(lines[i])
            i += 1

        return '\n'.join(fixed_lines), continuation_count

    def _merge_continuous_sentences(self, lines: List[str]) -> Tuple[List[str], int]:
        """Merge continuous sentences with tracking"""
        if not lines:
            return lines, 0

        merged = []
        merge_count = 0
        i = 0

        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                merged.append(lines[i])
                i += 1
                continue

            merged_line = lines[i]  # Preserve original formatting
            original_i = i

            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                if not next_line:
                    break

                # Enhanced merging conditions
                should_merge = self._should_merge_lines(merged_line, next_line)
                
                if should_merge:
                    merge_count += 1
                    original_indent = len(lines[i]) - len(lines[i].lstrip())
                    merged_line = lines[i][:original_indent] + lines[i].lstrip() + ' ' + next_line
                    i += 1
                else:
                    break

            merged.append(merged_line)
            if i == original_i:  # Only advance if no merge occurred
                i += 1

        return merged, merge_count

    def _should_merge_lines(self, current_line: str, next_line: str) -> bool:
        """Determine if two lines should be merged"""
        current_stripped = current_line.strip()
        next_stripped = next_line.strip()
        
        if not current_stripped or not next_stripped:
            return False

        # Current line conditions
        ends_with_punct = any(current_stripped.endswith(p) for p in {'.', '!', '?', ':', ';', ',', ')', ']', '}', '"', "'"})
        has_few_words = len(current_stripped.split()) <= 2
        current_words = current_stripped.split()
        last_word_upper = bool(current_words) and current_words[-1][0].isupper()
        last_word_short = bool(current_words) and len(current_words[-1]) <= 3

        # Next line conditions
        next_starts_lower = next_stripped[0].islower()
        next_starts_upper = next_stripped[0].isupper()
        next_short = len(next_stripped.split()) <= 3
        next_starts_with_article = next_stripped.lower().startswith(('the ', 'a ', 'an ', 'this ', 'that '))

        # Merge conditions
        return (
            has_few_words or
            (not ends_with_punct and next_starts_lower) or
            (last_word_upper and next_starts_upper) or
            (last_word_short and next_starts_lower) or
            (ends_with_punct and next_starts_with_article) or
            (current_stripped.endswith(',') and next_starts_lower)
        )

    def _post_process_text(self, text: str) -> str:
        """Final cleanup of merged text"""
        # Fix common OCR errors
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix space-after-hyphen
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([a-zA-Z])\s+-\s+([a-zA-Z])', r'\1-\2', text)  # Fix spaced hyphens
        return text

    def _analyze_text_alignment(self, text: str) -> str:
        """Analyze and adjust text alignment based on spacing patterns"""
        lines = text.split('\n')
        if not lines:
            return text
        
        # Analyze indentation patterns
        line_indents = []
        for line in lines:
            if line.strip():  # Only non-empty lines
                indent = len(line) - len(line.lstrip())
                line_indents.append(indent)
        
        if not line_indents:
            return text
        
        # Find common indentation levels
        unique_indents = list(set(line_indents))
        unique_indents.sort()
        
        # Determine if text appears to be centered or left-aligned
        avg_indent = sum(line_indents) / len(line_indents)
        max_indent = max(line_indents)
        
        processed_lines = []
        for line in lines:
            if not line.strip():
                processed_lines.append('')
                continue
            
            current_indent = len(line) - len(line.lstrip())
            content = line.strip()
            
            # Apply alignment rules
            if len(content) < 50 and current_indent > avg_indent:
                # Likely a title or header - center it
                processed_lines.append(content)
            elif current_indent > max_indent * 0.5:
                # Significantly indented - reduce to reasonable level
                new_indent = min(8, current_indent // 2)
                processed_lines.append(' ' * new_indent + content)
            else:
                # Normal text - minimal or no indentation
                processed_lines.append(content)
        
        return '\n'.join(processed_lines)

    def extract_text_enhanced(self, image_path: str) -> Dict[str, Any]:
        """Enhanced text extraction with layout preservation"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        
        # Use the improved layout extraction from reference code
        text = self._extract_with_layout_improved(img)
        
        # Apply nonsense word filtering
        text = self._filter_nonsense_words(text)
        
        # Fix line continuations
        text, continuation_count = self._fix_line_continuations(text)
        
        # Merge continuous sentences
        lines = text.split('\n')
        merged_lines, merge_count = self._merge_continuous_sentences(lines)
        text = '\n'.join(merged_lines)
        
        # Analyze and adjust text alignment
        text = self._analyze_text_alignment(text)
        
        # Final post-processing
        text = self._post_process_text(text)
        
        return {
            "text": text.strip(),
            "stats": {
                "line_continuations": continuation_count,
                "sentence_merges": merge_count,
                "total_merges": continuation_count + merge_count
            }
        }
    
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

# === Main execution ===
if __name__ == "__main__":
    try:
        # Initialize enhanced OCR
        ocr = EnhancedOCR()
        
        # Extract text with enhanced layout preservation
        result = ocr.extract_text_enhanced('Screenshot 2025-06-25 224812.png')
        
        print("🔍 Enhanced Extracted Text:")
        print("=" * 50)
        print(result["text"])
        print("=" * 50)
        
        print("\n📊 Processing Statistics:")
        print(f" - Line continuations fixed: {result['stats']['line_continuations']}")
        print(f" - Sentence merges: {result['stats']['sentence_merges']}")
        print(f" - Total improvements: {result['stats']['total_merges']}")
        
        # Apply grammar correction
        print("\n✅ Applying grammar correction...")
        corrected_text = ocr.correct_with_languagetool(result["text"])
        
        print("\n📝 Final Corrected Text:")
        print("=" * 50)
        print(corrected_text)
        print("=" * 50)
        
        # Save results
        with open("enhanced_output.txt", "w", encoding="utf-8") as f:
            f.write("=== ENHANCED EXTRACTED TEXT ===\n")
            f.write(result["text"])
            f.write("\n\n=== GRAMMAR CORRECTED TEXT ===\n")
            f.write(corrected_text)
        
        print("\n💾 Results saved to 'enhanced_output.txt'")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Possible solutions:")
        print("1. Make sure the image file exists and path is correct")
        print("2. Install required packages: pip install opencv-python pytesseract requests pyspellchecker")
        print("3. Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        print("4. Check internet connection for LanguageTool API")