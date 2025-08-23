import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import os
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import colorsys
import math
import re

@dataclass
class TextStyle:
    """Enhanced text styling with typography features"""
    font_size: int
    font_weight: str = "normal" 
    text_color: str = "black"
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: int = 0
    padding: int = 5
    alignment: str = "left" 
    line_spacing: float = 1.2
    letter_spacing: float = 0.0

@dataclass
class PlacedBlock:
    """Enhanced block tracking with typography info"""
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    style_category: str = "default"
    font_size: int = 34
    block_id: str = ""

class EnhancedDocumentReconstructor:
    def __init__(self, 
                 canvas_width: int = 2480, 
                 canvas_height: int = 3508,
                 background_color: str = "white",
                 dpi: int = 300):
        
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.background_color = background_color
        self.dpi = dpi
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.font_cache = {}
        self.placed_blocks = []
        
        
        self.dominant_font_family = None
        self.font_usage_stats = {}
        
        
        self.left_margin_threshold = 0.15    
        self.right_margin_threshold = 0.85   
        self.center_threshold = 0.1          
        
        
        self.layout_rows = {} 
        self.min_vertical_gap = 15  
        self.overlap_tolerance = 5  
        
       
        self.document_text_analysis = {}
        self.paragraph_patterns = []
        self.normal_text_indicators = []
        
        
        self._load_enhanced_font_family()
        self._setup_enhanced_style_patterns()
        self._setup_smart_paragraph_detection()
    
    def _setup_smart_paragraph_detection(self):
        """Setup intelligent paragraph detection patterns"""
        
       
        self.normal_paragraph_patterns = [
            # Common paragraph starters
            r'^(The|This|That|In|On|At|With|For|By|To|From|Of|As|An|A)\s+\w+',
            r'^(We|I|You|They|He|She|It)\s+(are|am|is|have|has|will|would|should|can|may)',
            r'^(After|Before|During|Since|Until|While|When|Where|How|Why)\s+',
            r'^(However|Moreover|Furthermore|Therefore|Additionally|Meanwhile|Nevertheless)',
            r'^\w+ing\s+\w+',  # Gerund starts
            r'^\w+ed\s+\w+',   # Past tense starts
            
            # Sentence continuation patterns
            r'\.\s+[A-Z]\w+',  # Sentence after period
            r',\s+and\s+\w+',  # Continuation with 'and'
            r',\s+but\s+\w+',  # Continuation with 'but'
            r',\s+which\s+\w+', # Relative clause
            r',\s+that\s+\w+',  # That clause
            
            # Common body text phrases
            r'(is|are|was|were|has|have|had)\s+(been|being)',
            r'(will|would|could|should|may|might)\s+(be|have)',
            r'according to',
            r'in order to',
            r'due to the fact',
            r'it is important',
            r'please note',
            r'as mentioned',
            r'for more information',
        ]
        
        # Patterns that definitely indicate headers/titles (SHOULD be bold)
        self.header_patterns = [
            r'^[A-Z]{3,}',  # All caps (3+ letters)
            r'^[A-Z][A-Z\s]{10,}$',  # Mostly caps
            r'^\d+\.\s*[A-Z]',  # Numbered sections
            r'^(CHAPTER|SECTION|PART|ARTICLE)\s*\d*',
            r'^(REPUBLIC|EMBASSY|INVITATION|CERTIFICATE|AUTHORIZATION)',
            r'OŚWIADCZENIE|STATEMENT|CONFIRMATION',
        ]
        
        
        self.normal_text_keywords = [
            'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'his', 
            'from', 'they', 'she', 'her', 'been', 'than', 'its', 'who', 'did', 'yes', 'get',
            'has', 'had', 'him', 'old', 'see', 'now', 'way', 'may', 'say', 'each', 'which',
            'their', 'time', 'will', 'about', 'would', 'could', 'should', 'there', 'what',
            'your', 'when', 'him', 'my', 'me', 'will', 'if', 'no', 'do', 'would', 'who',
            'so', 'about', 'out', 'many', 'then', 'them', 'these', 'some', 'her', 'would',
            'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know',
            'just', 'first', 'get', 'over', 'think', 'also', 'your', 'work', 'life', 'only',
            'new', 'years', 'way', 'may', 'say', 'each', 'which', 'she', 'do', 'how'
        ]
    
    def _analyze_document_text_characteristics(self, ocr_data: List[Dict]):
        """Analyze the entire document to understand text patterns"""
        all_texts = []
        text_lengths = []
        word_counts = []
        sentence_counts = []
        
        for block in ocr_data:
            text = block.get('block_text', '').strip()
            if text:
                all_texts.append(text)
                text_lengths.append(len(text))
                word_counts.append(len(text.split()))
                sentence_counts.append(len([s for s in text.split('.') if s.strip()]))
        
        if not all_texts:
            return
        
        
        self.document_text_analysis = {
            'total_blocks': len(all_texts),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'avg_sentence_count': sum(sentence_counts) / len(sentence_counts),
            'long_text_threshold': sorted(text_lengths)[int(len(text_lengths) * 0.7)],  # 70th percentile
            'short_text_threshold': sorted(text_lengths)[int(len(text_lengths) * 0.3)],  # 30th percentile
        }
        
        self.logger.info(f"📊 Document analysis: avg_length={self.document_text_analysis['avg_text_length']:.1f}, "
                        f"long_threshold={self.document_text_analysis['long_text_threshold']}, "
                        f"short_threshold={self.document_text_analysis['short_text_threshold']}")
    
    def _is_normal_paragraph_text(self, text: str, block_data: Dict) -> bool:
        """Intelligent detection of normal paragraph text (should not be bold)"""
        if not text or not text.strip():
            return False
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        text_length = len(text_clean)
        word_count = len(text_clean.split())
        
        
        paragraph_score = 0
        header_score = 0
        
       
        if hasattr(self, 'document_text_analysis') and self.document_text_analysis:
            if text_length > self.document_text_analysis.get('long_text_threshold', 100):
                paragraph_score += 3
            elif text_length > self.document_text_analysis.get('avg_text_length', 50):
                paragraph_score += 1
            
            if text_length < self.document_text_analysis.get('short_text_threshold', 30):
                header_score += 2  
        
        # 2. Word count analysis
        if word_count > 15:  
            paragraph_score += 3
        elif word_count > 8:
            paragraph_score += 1
        elif word_count < 5:
            header_score += 1
        
       
        for pattern in self.normal_paragraph_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                paragraph_score += 2
                break
        
        
        for pattern in self.header_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                header_score += 4
                break
        
       
        caps_ratio = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if text_clean else 0
        
        if caps_ratio > 0.7:  
            header_score += 3
        elif caps_ratio > 0.5:
            header_score += 1
        elif caps_ratio < 0.15: 
            paragraph_score += 2
        
        
        words = text_lower.split()
        common_word_count = sum(1 for word in words if word in self.normal_text_keywords)
        common_word_ratio = common_word_count / len(words) if words else 0
        
        if common_word_ratio > 0.3: 
            paragraph_score += 2
        elif common_word_ratio > 0.2:
            paragraph_score += 1
        
        
        sentences = [s.strip() for s in text_clean.split('.') if s.strip()]
        if len(sentences) > 1: 
            paragraph_score += 2
        
        
        punctuation_count = sum(1 for c in text_clean if c in '.,;:!?()[]{}"-')
        punctuation_ratio = punctuation_count / len(text_clean) if text_clean else 0
        
        if punctuation_ratio > 0.05: 
            paragraph_score += 1
        
        
        non_bold_indicators = [
            'duration of employment', 'working hours', 'salary', 'monthly gross',
            'the employee', 'the employer', 'according to', 'based on',
            'in accordance with', 'pursuant to', 'subject to', 'provided that',
            'terms and conditions', 'rights and obligations', 'furthermore',
            'moreover', 'however', 'therefore', 'additionally', 'please note',
            'for your information', 'should you have', 'do not hesitate',
            'looking forward', 'thank you', 'best regards', 'sincerely'
        ]
        
        for indicator in non_bold_indicators:
            if indicator.lower() in text_lower:
                paragraph_score += 2
                break
        
        
        box = block_data.get('block_box', [[0, 0], [1, 1]])
        left_x = box[0][0]
        
        if 0.1 < left_x < 0.9:  
            paragraph_score += 1
        
        
        is_paragraph = paragraph_score > header_score and paragraph_score >= 3
        
        
        if abs(paragraph_score - header_score) <= 2:
            self.logger.debug(f"📝 Borderline text analysis: '{text_clean[:50]}...' "
                            f"P-score: {paragraph_score}, H-score: {header_score}, "
                            f"Decision: {'PARAGRAPH' if is_paragraph else 'HEADER'}")
        
        return is_paragraph
    
    def _load_enhanced_font_family(self):
        """Enhanced font loading with professional fallbacks"""
        self.fonts = {
            'regular': self._load_font_with_fallback("arial.ttf"),
            'bold': self._load_font_with_fallback("arialbd.ttf"),
            'italic': self._load_font_with_fallback("ariali.ttf"),
            'light': self._load_font_with_fallback("arialne.ttf"),
        }
        
        
        self.professional_fonts = {
            'times': self._load_font_with_fallback("times.ttf"),
            'times_bold': self._load_font_with_fallback("timesbd.ttf"),
            'calibri': self._load_font_with_fallback("calibri.ttf"),
            'calibri_bold': self._load_font_with_fallback("calibrib.ttf"),
            'georgia': self._load_font_with_fallback("georgia.ttf"),
            'georgia_bold': self._load_font_with_fallback("georgiab.ttf"),
            'verdana': self._load_font_with_fallback("verdana.ttf"),
            'verdana_bold': self._load_font_with_fallback("verdanab.ttf"),
        }
    
    def _load_font_with_fallback(self, preferred_font: str) -> ImageFont.ImageFont:
        """Enhanced font loading with comprehensive fallbacks"""
        font_paths = [
           
            f"C:/Windows/Fonts/{preferred_font}",
           
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
          
            f"/usr/share/fonts/truetype/dejavu/{preferred_font}",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 40)
               
                font_name = os.path.basename(font_path)
                self.font_usage_stats[font_name] = self.font_usage_stats.get(font_name, 0) + 1
                self.logger.debug(f"Successfully loaded font: {font_name}")
                return font
            except (OSError, IOError):
                continue
        
        self.logger.warning(f"Could not load {preferred_font}, using default font")
        return ImageFont.load_default()
    
    def _detect_dominant_font_family(self, ocr_data: List[Dict]) -> str:
        """Enhanced AI-powered dominant font detection"""
        font_characteristics = {}
        document_type_indicators = []
        
        # Analyze text content for document type
        all_text = ""
        for block in ocr_data:
            block_text = block.get('block_text', '')
            all_text += block_text + " "
            
            # Collect document type indicators
            if any(word in block_text.upper() for word in ['REPUBLIC', 'EMBASSY', 'INVITATION', 'CERTIFICATE']):
                document_type_indicators.append('formal_official')
            elif any(word in block_text.upper() for word in ['AUTHORIZATION', 'PERMISSION', 'EMPLOYMENT']):
                document_type_indicators.append('legal_business')
        
        # Analyze font characteristics
        for block in ocr_data:
            if 'lines' not in block:
                continue
                
            for line in block['lines']:
                estimated_height = line.get('estimated_font_height', 40)
                boldness = line.get('boldness_score', 0)
                
                # Create font signature based on characteristics
                font_signature = f"h{int(estimated_height/5)*5}_b{int(boldness*10)}"
                font_characteristics[font_signature] = font_characteristics.get(font_signature, 0) + 1
        
        # Enhanced font selection logic based on document analysis
        if font_characteristics:
            dominant_signature = max(font_characteristics, key=font_characteristics.get)
            self.logger.info(f"Detected dominant font characteristics: {dominant_signature}")
            self.logger.info(f"Document type indicators: {set(document_type_indicators)}")
            
            # Choose font based on document type and characteristics
            if 'formal_official' in document_type_indicators:
                if "h35" in dominant_signature or "h40" in dominant_signature:
                    return "times"  # Times for formal official documents
                else:
                    return "georgia"  # Georgia for readable formal text
            elif 'legal_business' in document_type_indicators:
                return "calibri"  # Modern professional look
            else:
                # Default analysis
                if "h35" in dominant_signature or "h40" in dominant_signature:
                    return "times"
                elif "h30" in dominant_signature:
                    return "calibri"
                elif "h25" in dominant_signature:
                    return "verdana"
                else:
                    return "times"  # Default to Times for better readability
        
        return "times"  # Default to Times New Roman for professional appearance
    
    def _get_font(self, size: int, weight: str = "regular") -> ImageFont.ImageFont:
        """Enhanced font retrieval with intelligent selection and weight support"""
        cache_key = f"{size}_{weight}_{self.dominant_font_family}"
        if cache_key not in self.font_cache:
            try:
                
                if self.dominant_font_family and weight == "bold":
                    font_key = f"{self.dominant_font_family}_bold"
                    if font_key in self.professional_fonts:
                        base_font = self.professional_fonts[font_key]
                    elif self.dominant_font_family in self.professional_fonts:
                        base_font = self.professional_fonts[self.dominant_font_family]
                    else:
                        base_font = self.fonts.get('bold', self.fonts['regular'])
                elif self.dominant_font_family and self.dominant_font_family in self.professional_fonts:
                    base_font = self.professional_fonts[self.dominant_font_family]
                else:
                    base_font = self.fonts.get(weight, self.fonts['regular'])
                
                if hasattr(base_font, 'path') and base_font.path:
                    self.font_cache[cache_key] = ImageFont.truetype(base_font.path, size)
                else:
                    self.font_cache[cache_key] = ImageFont.load_default()
                    
            except Exception as e:
                self.logger.warning(f"Font loading failed for {cache_key}: {e}")
                self.font_cache[cache_key] = ImageFont.load_default()
        
        return self.font_cache[cache_key]
    
    def _setup_enhanced_style_patterns(self):
        """Enhanced style patterns with better typography"""
        self.style_patterns = {
            'document_title': {
                'keywords': ['REPUBLIC OF POLAND', 'INVITATION', 'AUTHORIZATION', 'CERTIFICATE'],
                'style': TextStyle(
                    font_size=38, 
                    font_weight="bold", 
                    text_color="black",
                    padding=6,
                    alignment="center",
                    line_spacing=1.15
                )
            },
            'main_header': {
                'keywords': ['STATEMENT', 'OŚWIADCZENIE', 'EMPLOYER', 'CONFIRMATION', 'EMBASSY'],
                'style': TextStyle(
                    font_size=36, 
                    font_weight="bold", 
                    text_color="black",
                    padding=5,
                    alignment="center",
                    line_spacing=1.1
                )
            },
            'sub_header': {
                'keywords': ['INVITATION', 'EMPLOY', 'EMPLOYMENT', 'TUNISIA', 'Szanowny', 'Dear Sir', 'Honorable', 'Based on art'],
                'style': TextStyle(
                    font_size=32, 
                    font_weight="bold", 
                    text_color="black",
                    padding=4,
                    alignment="left",
                    line_spacing=1.1
                )
            },
            'company_info': {
                'keywords': ['Personnel', 'SAL', 'Sp.', 'NIP:', 'REGON:', 'KRS:', 'Human Resources', 'Voivode', 'Government Security'],
                'style': TextStyle(
                    font_size=26, 
                    text_color="black",
                    padding=3,
                    alignment="left",
                    line_spacing=1.15
                )
            },
            'body_text': {
                'keywords': ['Duration', 'Working hours', 'grants permission', 'citizen of', 'capacity of', 'Upon consideration'],
                'style': TextStyle(
                    font_size=30, 
                    text_color="black",
                    padding=4,
                    alignment="left",
                    line_spacing=1.2
                )
            },
            'official_text': {
                'keywords': ['The inviting person', 'You are invited', 'Official stamp', 'Permission to work'],
                'style': TextStyle(
                    font_size=28, 
                    text_color="black",
                    padding=3,
                    alignment="left",
                    line_spacing=1.2
                )
            },
            'signature_area': {
                'keywords': ['Deputy Head', 'Katarzyna', 'signature', 'authorized person'],
                'style': TextStyle(
                    font_size=28, 
                    text_color="black",
                    padding=3,
                    alignment="left",
                    line_spacing=1.1
                )
            },
            'date_info': {
                'keywords': ['valid from', 'Invitation valid', 'Date and place', 'Date and evidence'],
                'style': TextStyle(
                    font_size=26, 
                    text_color="black",
                    padding=2,
                    alignment="center",
                    line_spacing=1.1
                )
            }
        }
    def _detect_content_alignment_within_block(self, block_data: Dict, text: str) -> str:
        """Detect alignment based on content positioning within the block"""
        box = block_data['block_box']
        left_x = box[0][0]
        right_x = box[1][0]
        width = right_x - left_x
        
        # Check if we have line-level data from OCR
        if 'lines' in block_data and block_data['lines']:
            return self._detect_alignment_from_lines(block_data['lines'], width)
        
        # Fallback: analyze text characteristics
        return "left"

    def _detect_alignment_from_lines(self, lines: List[Dict], block_width: float) -> str:
        """Detect alignment based on line positioning within the block"""
        left_margins = []
        right_margins = []
        center_deviations = []
        
        for line in lines:
            if 'normalized_box' in line:
                line_left = line['normalized_box'][0][0]
                line_right = line['normalized_box'][1][0]
                line_center = (line_left + line_right) / 2
                block_center = 0.5  # Normalized block center
                
                left_margins.append(line_left)
                right_margins.append(1 - line_right)  # Distance from right edge
                center_deviations.append(abs(line_center - block_center))
        
        if not left_margins:
            return "left"  # Default
        
        # Calculate average positioning
        avg_left_margin = sum(left_margins) / len(left_margins)
        avg_right_margin = sum(right_margins) / len(right_margins)
        avg_center_deviation = sum(center_deviations) / len(center_deviations)
        
        # Alignment detection thresholds
        left_aligned_threshold = 0.05  # Small left margin
        right_aligned_threshold = 0.05  # Small right margin
        center_aligned_threshold = 0.05  # Small center deviation
        
        # Check for right alignment (small right margin, large left margin)
        if avg_right_margin < right_aligned_threshold and avg_left_margin > 0.2:
            return "right"
        
        # Check for center alignment (small center deviation)
        if avg_center_deviation < center_aligned_threshold:
            return "center"
        
        # Check for left alignment (small left margin)
        if avg_left_margin < left_aligned_threshold:
            return "left"
        
        # Default to left if uncertain
        return "left"
    
    def _detect_text_style_enhanced(self, text: str, block_data: Dict) -> TextStyle:
        """IMPROVED text style detection with smart paragraph detection"""
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Analyze block position and content positioning
        box = block_data['block_box']
        left_x = box[0][0]
        right_x = box[1][0]
        width = right_x - left_x
        
        # Get line-level alignment information from OCR data
        detected_alignment = self._detect_content_alignment_within_block(block_data, text)
        
        # SMART PARAGRAPH DETECTION - Check if this is normal paragraph text
        is_paragraph = self._is_normal_paragraph_text(text, block_data)
        
       
        style_priority = [
            'document_title', 'main_header', 'sub_header', 
            'official_text', 'body_text', 'company_info', 
            'signature_area', 'date_info'
        ]
        
        matched_style = None
        for style_name in style_priority:
            if style_name in self.style_patterns:
                pattern = self.style_patterns[style_name]
                keywords = pattern['keywords']
                
                for keyword in keywords:
                    if keyword.upper() in text_upper or keyword.lower() in text_lower:
                        matched_style = pattern['style']
                        break
                
                if matched_style:
                    break
        
        # Apply smart paragraph detection logic
        if matched_style:
            # SMART OVERRIDE: If detected as paragraph, force normal weight
            final_weight = matched_style.font_weight
            if is_paragraph and matched_style.font_weight == "bold":
                final_weight = "normal"
                self.logger.debug(f"Override: Changed '{text[:50]}...' from bold to normal (paragraph detected)")
            
            
            final_alignment = matched_style.alignment
            if matched_style.alignment != "left":  # Only override if style has specific alignment
                final_alignment = matched_style.alignment
            
            return TextStyle(
                font_size=matched_style.font_size,
                font_weight=final_weight,  # Use smart weight
                text_color=matched_style.text_color,
                padding=matched_style.padding,
                alignment=final_alignment,
                line_spacing=matched_style.line_spacing
            )

        
        fallback_size = 30
        if len(text) < 50:  
            fallback_size = 32
        elif len(text) > 200: 
            fallback_size = 28
        
        # SMART WEIGHT DETECTION for fallback
        fallback_weight = "normal"  # Default to normal
        
        # Only make it bold if it's clearly a header/title
        if not is_paragraph:
            # Check if it looks like a title/header
            if (len(text) < 100 and  # Short text
                (text.isupper() or  # All uppercase
                 text.count(' ') < 5 or  # Few words
                 any(word in text.upper() for word in ['TITLE', 'HEADER', 'SECTION', 'CHAPTER']))):
                fallback_weight = "bold"
        
        return TextStyle(
            font_size=fallback_size,
            font_weight=fallback_weight,  # Smart weight detection
            text_color="black",
            padding=4,
            alignment=detected_alignment,
            line_spacing=1.2
        )
    
    def _calculate_layout_rows(self, ocr_data: List[Dict]):
        """Calculate layout rows for better spacing management"""
        self.layout_rows = {}
        
        # Group blocks by approximate Y position
        y_positions = []
        for block in ocr_data:
            box = block['block_box']
            center_y = (box[0][1] + box[1][1]) / 2
            y_positions.append((center_y, block))
        
        # Sort by Y position
        y_positions.sort(key=lambda x: x[0])
        
        # Create row groups with tolerance
        current_row = 0
        last_y = -1
        row_threshold = 0.03  # 3% of canvas height tolerance
        
        for y_pos, block in y_positions:
            if last_y == -1 or abs(y_pos - last_y) > row_threshold:
                current_row += 1
                last_y = y_pos
            
            if current_row not in self.layout_rows:
                self.layout_rows[current_row] = []
            
            self.layout_rows[current_row].append(block)
        
        self.logger.info(f"Organized content into {len(self.layout_rows)} layout rows")
    
    def _get_smart_position(self, block_data: Dict, text_width: int, text_height: int, style: TextStyle) -> Tuple[int, int]:
        """IMPROVED positioning that follows the better alignment logic from second code"""
        box = block_data['block_box']
        
        # Use original coordinates as the primary reference (like in second code)
        original_x1 = int(box[0][0] * self.canvas_width)
        original_y1 = int(box[0][1] * self.canvas_height)
        original_x2 = int(box[1][0] * self.canvas_width)
        
        # IMPROVED X position calculation 
        if style.alignment == "center":
           
            canvas_center = self.canvas_width // 2
            x_position = canvas_center - text_width // 2
        elif style.alignment == "right":
            
            x_position = min(original_x2 - text_width - 10, self.canvas_width - text_width - 20)
        else:  
            x_position = max(original_x1, 20)  
        
        # Ensure X is within bounds
        x_position = max(20, min(x_position, self.canvas_width - text_width - 20))
        
        # Smart Y positioning to avoid overlaps (keep the sophisticated system from first code)
        y_position = original_y1
        max_attempts = 20
        
        for attempt in range(max_attempts):
            overlaps = False
            test_y = y_position + (attempt * self.min_vertical_gap)
            
            for placed_block in self.placed_blocks:
                if not (x_position + text_width + self.overlap_tolerance < placed_block.x1 or 
                       x_position - self.overlap_tolerance > placed_block.x2 or 
                       test_y + text_height + self.overlap_tolerance < placed_block.y1 or 
                       test_y - self.overlap_tolerance > placed_block.y2):
                    overlaps = True
                    break
            
            if not overlaps:
                y_position = test_y
                break
            
            if overlaps and attempt > 10:
                for placed_block in self.placed_blocks:
                    if (x_position < placed_block.x2 and x_position + text_width > placed_block.x1):
                        y_position = max(y_position, placed_block.y2 + self.min_vertical_gap)
        
        
        y_position = max(20, min(y_position, self.canvas_height - text_height - 20))
        
        return x_position, y_position
    
    def _wrap_text_smart(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        """Smart text wrapping with better line break handling"""
        lines = []
        
        
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            
            
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                
                try:
                    bbox = font.getbbox(test_line)
                    line_width = bbox[2] - bbox[0]
                except:
                    line_width = len(test_line) * (font.size * 0.6)
                
                if line_width <= max_width or not current_line:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
        
        return [line for line in lines if line is not None]
    
    def _draw_enhanced_text_block(self, draw: ImageDraw.Draw, text: str, block_data: Dict, style: TextStyle):
        """Enhanced text drawing with improved positioning and typography"""
        if not text.strip():
            return
        
        font_weight = "bold" if style.font_weight == "bold" else "regular"
        font = self._get_font(style.font_size, font_weight)
        
        box = block_data['block_box']
        original_width = int((box[1][0] - box[0][0]) * self.canvas_width)
        max_width = max(original_width, 200)  
        lines = self._wrap_text_smart(text, font, max_width)
        if not lines:
            return
        
        
        total_width = 0
        line_heights = []
        
        for line in lines:
            if line.strip():
                try:
                    bbox = font.getbbox(line)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                    total_width = max(total_width, line_width)
                    line_heights.append(max(line_height, style.font_size))
                except:
                    line_width = len(line) * (style.font_size * 0.6)
                    line_height = style.font_size
                    total_width = max(total_width, line_width)
                    line_heights.append(line_height)
            else:
                line_heights.append(style.font_size // 2)
        
        
        if line_heights:
            base_line_height = max(line_heights) if line_heights else style.font_size
            spacing = int(base_line_height * (style.line_spacing - 1.0))
            total_height = sum(line_heights) + max(0, (len(lines) - 1) * spacing)
        else:
            total_height = style.font_size
        
        
        final_x, final_y = self._get_smart_position(block_data, total_width, total_height, style)
        
        block_id = block_data.get('block_id', f"block_{len(self.placed_blocks)}")
        self.placed_blocks.append(PlacedBlock(
            final_x, final_y, final_x + total_width, final_y + total_height, 
            text, font_size=style.font_size, block_id=block_id
        ))
        
        # Draw text with enhanced quality
        current_y = final_y
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                current_y += line_heights[i] if i < len(line_heights) else style.font_size // 2
                continue
            
            # Calculate line positioning
            try:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = line_heights[i] if i < len(line_heights) else style.font_size
            except:
                line_width = len(line) * (style.font_size * 0.6)
                line_height = style.font_size
            
            if style.alignment == "center":
                line_x = final_x + (total_width - line_width) // 2
            elif style.alignment == "right":
                line_x = final_x + total_width - line_width
            else:  
                line_x = final_x
            
            
            if style.font_weight == "bold":
                offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
                for dx, dy in offsets:
                    draw.text((line_x + dx, current_y + dy), line, font=font, fill=style.text_color)
            else:
                draw.text((line_x, current_y), line, font=font, fill=style.text_color)
            
            
            if i < len(lines) - 1:
                current_y += line_height + spacing
    
    def _create_robust_translation_mapping(self, ocr_data: List[Dict], translation_data: List[Dict]) -> Dict[str, str]:
        """Create robust translation mapping with fuzzy matching"""
        translation_map = {}
        
    
        for trans_item in translation_data:
            original = trans_item.get('original', '').strip()
            translated = trans_item.get('translated', '').strip()
            if original and translated:
                translation_map[original] = translated
        
       
        for block in ocr_data:
            block_text = block.get('block_text', '').strip()
            if block_text and block_text not in translation_map:
                # Try fuzzy matching
                best_match = self._find_best_translation_match(block_text, translation_data)
                if best_match:
                    translation_map[block_text] = best_match
                else:
                    translation_map[block_text] = block_text  # Keep original
        
        return translation_map
    
    def _find_best_translation_match(self, text: str, translation_data: List[Dict]) -> Optional[str]:
        """Find best translation match using fuzzy logic"""
        import difflib
        
        best_ratio = 0
        best_translation = None
        
        for trans_item in translation_data:
            original = trans_item.get('original', '').strip()
            if not original:
                continue
            
           
            ratio = difflib.SequenceMatcher(None, text.lower(), original.lower()).ratio()
            
            if ratio > best_ratio and ratio > 0.8: 
                best_ratio = ratio
                best_translation = trans_item.get('translated', '').strip()
        
        return best_translation if best_translation else None
    
    def _sort_blocks_by_layout(self, ocr_data: List[Dict]) -> List[Dict]:
        """Sort blocks by natural document layout with improved logic from second code"""
        def get_layout_key(block):
            box = block['block_box']
            center_y = (box[0][1] + box[1][1]) / 2
            left_x = box[0][0]
            
           
            row_group = int(center_y * 25)  
            
            return (row_group, left_x)
        
        return sorted(ocr_data, key=get_layout_key)
    
    def reconstruct_document(self, 
                           ocr_data: List[Dict], 
                           translation_data: List[Dict],
                           output_path: str = "enhanced_reconstructed_document.png",
                           debug_mode: bool = False) -> str:
        """Enhanced document reconstruction with improved alignment and smart paragraph detection"""
        try:
            
            self.placed_blocks = []
            
           
            self.logger.info("Analyzing document structure, typography, and paragraph patterns...")
            self._analyze_document_text_characteristics(ocr_data)
            self.dominant_font_family = self._detect_dominant_font_family(ocr_data)
            self._calculate_layout_rows(ocr_data)
            
            self.logger.info(f"Selected font family: {self.dominant_font_family}")
            self.logger.info(f"Organized into {len(self.layout_rows)} layout rows")
            self.logger.info(f" Document stats: {self.document_text_analysis.get('total_blocks', 0)} blocks, "
                           f"avg length: {self.document_text_analysis.get('avg_text_length', 0):.1f}")
            
            
            translation_map = self._create_robust_translation_mapping(ocr_data, translation_data)
            
            
            canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), self.background_color)
            draw = ImageDraw.Draw(canvas)
            
            sorted_blocks = self._sort_blocks_by_layout(ocr_data)
            processed_blocks = 0
            font_usage_summary = {}
            paragraph_detection_stats = {'paragraphs': 0, 'headers': 0, 'overrides': 0}
            
            for block in sorted_blocks:
                block_text = block.get('block_text', '').strip()
                if not block_text:
                    continue
                
                translated_text = translation_map.get(block_text, block_text)
                
               
                style = self._detect_text_style_enhanced(translated_text, block)
                
               
                is_paragraph = self._is_normal_paragraph_text(translated_text, block)
                if is_paragraph:
                    paragraph_detection_stats['paragraphs'] += 1
                else:
                    paragraph_detection_stats['headers'] += 1
                
                if style.font_weight == "normal" and not is_paragraph:
                    paragraph_detection_stats['overrides'] += 1
                font_key = f"{self.dominant_font_family}_{style.font_size}_{style.font_weight}"
                font_usage_summary[font_key] = font_usage_summary.get(font_key, 0) + 1
                
                self._draw_enhanced_text_block(draw, translated_text, block, style)
                
                if debug_mode:
                    
                    box = block['block_box']
                    x1, y1 = int(box[0][0] * self.canvas_width), int(box[0][1] * self.canvas_height)
                    x2, y2 = int(box[1][0] * self.canvas_width), int(box[1][1] * self.canvas_height)
                    
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
                    
                    debug_font = self._get_font(14)
                    debug_text = f"{processed_blocks+1}|{style.alignment}|{style.font_size}|{'P' if is_paragraph else 'H'}"
                    draw.text((x1, y1-25), debug_text, font=debug_font, fill="blue")
                
                processed_blocks += 1
                
            
            self.logger.info("Applying quality enhancements...")
            canvas = self._apply_quality_enhancements(canvas)
            
            canvas.save(output_path, format='PNG', quality=100, dpi=(self.dpi, self.dpi), 
                       optimize=False, compress_level=0)
            
           
            self.logger.info(f"Enhanced document reconstructed: {output_path}")
            self.logger.info(f" Successfully processed {processed_blocks} blocks")
            self.logger.info(f" Layout: {len(self.layout_rows)} rows, {len(self.placed_blocks)} placed blocks")
            self.logger.info(f"Typography Summary:")
            for font_combo, count in sorted(font_usage_summary.items()):
                self.logger.info(f"   {font_combo}: {count} blocks")
            
            
            self.logger.info(f"Smart Paragraph Detection Results:")
            self.logger.info(f"Detected paragraphs (normal weight): {paragraph_detection_stats['paragraphs']}")
            self.logger.info(f"Detected headers/titles (bold eligible): {paragraph_detection_stats['headers']}")
            self.logger.info(f"Bold-to-normal overrides: {paragraph_detection_stats['overrides']}")
            
            paragraph_accuracy = (paragraph_detection_stats['paragraphs'] / processed_blocks * 100) if processed_blocks > 0 else 0
            self.logger.info(f"Paragraph detection rate: {paragraph_accuracy:.1f}%")
            
            if self.font_usage_stats:
                self.logger.info(f"Font Files Used: {list(self.font_usage_stats.keys())}")
            
            
            overlaps = self._detect_final_overlaps()
            if overlaps:
                self.logger.warning(f"Found {len(overlaps)} potential overlaps - consider manual review")
            else:
                self.logger.info("No overlaps detected - clean layout achieved")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f" Error in enhanced reconstruction: {e}")
            raise
    
    def _apply_quality_enhancements(self, canvas: Image.Image) -> Image.Image:
        """Apply quality enhancements to the final document"""
        enhanced = canvas.copy()
        
        try:
            # Subtle contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # Slight sharpness boost
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Minimal brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.02)
            
            self.logger.info("✨ Quality enhancements applied successfully")
            
        except Exception as e:
            self.logger.warning(f"Quality enhancement failed: {e}, using original")
            return canvas
        
        return enhanced
    
    def _detect_final_overlaps(self) -> List[Tuple[PlacedBlock, PlacedBlock]]:
        """Detect any remaining overlaps in the final layout"""
        overlaps = []
        
        for i, block1 in enumerate(self.placed_blocks):
            for j, block2 in enumerate(self.placed_blocks[i+1:], i+1):
                # Check if blocks overlap
                if not (block1.x2 < block2.x1 or block1.x1 > block2.x2 or 
                       block1.y2 < block2.y1 or block1.y1 > block2.y2):
                    overlaps.append((block1, block2))
        
        return overlaps
    
    def load_data(self, ocr_json_path: str, translation_json_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load OCR and translation data from JSON files"""
        try:
            with open(ocr_json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            with open(translation_json_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
            
            self.logger.info(f"Loaded {len(ocr_data)} OCR blocks and {len(translation_data)} translations")
            return ocr_data, translation_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def create_analysis_report(self, ocr_data: List[Dict], translation_data: List[Dict]) -> Dict:
        """Create detailed analysis report of the document reconstruction process"""
        
        # Analyze font characteristics and paragraph patterns
        self._analyze_document_text_characteristics(ocr_data)
        self.dominant_font_family = self._detect_dominant_font_family(ocr_data)
        translation_map = self._create_robust_translation_mapping(ocr_data, translation_data)
        
        analysis = {
            'document_info': {
                'total_blocks': len(ocr_data),
                'total_translations': len(translation_data),
                'successful_mappings': sum(1 for k, v in translation_map.items() if k != v),
                'dominant_font': self.dominant_font_family
            },
            'layout_analysis': {
                'estimated_rows': len(self.layout_rows) if hasattr(self, 'layout_rows') else 0,
                'alignment_distribution': {},
                'font_size_distribution': {},
                'paragraph_distribution': {'paragraphs': 0, 'headers': 0}
            },
            'quality_metrics': {
                'translation_coverage': 0,
                'style_detection_rate': 0,
                'paragraph_detection_accuracy': 0,
                'estimated_overlaps': 0
            },
            'recommendations': []
        }
        
        # Analyze each block with paragraph detection
        alignment_counts = {}
        font_size_counts = {}
        style_matches = 0
        paragraph_count = 0
        
        for block in ocr_data:
            block_text = block.get('block_text', '').strip()
            if not block_text:
                continue
                
            translated_text = translation_map.get(block_text, block_text)
            style = self._detect_text_style_enhanced(translated_text, block)
            is_paragraph = self._is_normal_paragraph_text(translated_text, block)
            
            # Count alignments and sizes
            alignment_counts[style.alignment] = alignment_counts.get(style.alignment, 0) + 1
            font_size_counts[style.font_size] = font_size_counts.get(style.font_size, 0) + 1
            
            # Count paragraph detection
            if is_paragraph:
                paragraph_count += 1
                analysis['layout_analysis']['paragraph_distribution']['paragraphs'] += 1
            else:
                analysis['layout_analysis']['paragraph_distribution']['headers'] += 1
            
            # Check if style was detected (not fallback)
            if any(keyword.lower() in translated_text.lower() 
                  for pattern in self.style_patterns.values() 
                  for keyword in pattern['keywords']):
                style_matches += 1
        
        analysis['layout_analysis']['alignment_distribution'] = alignment_counts
        analysis['layout_analysis']['font_size_distribution'] = font_size_counts
        
        # Calculate quality metrics
        total_blocks = analysis['document_info']['total_blocks']
        if total_blocks > 0:
            analysis['quality_metrics']['translation_coverage'] = analysis['document_info']['successful_mappings'] / total_blocks
            analysis['quality_metrics']['style_detection_rate'] = style_matches / total_blocks
            analysis['quality_metrics']['paragraph_detection_accuracy'] = paragraph_count / total_blocks
        
        # Generate enhanced recommendations
        if analysis['quality_metrics']['translation_coverage'] < 0.8:
            analysis['recommendations'].append("Consider improving translation coverage - some text may not be translated")
        
        if analysis['quality_metrics']['style_detection_rate'] < 0.5:
            analysis['recommendations'].append("Style detection could be improved - consider adding more keywords to style patterns")
        
        if analysis['quality_metrics']['paragraph_detection_accuracy'] < 0.3:
            analysis['recommendations'].append("Low paragraph detection rate - most text detected as headers (may result in excessive bold)")
        elif analysis['quality_metrics']['paragraph_detection_accuracy'] > 0.8:
            analysis['recommendations'].append("High paragraph detection rate - good balance between headers and body text")
        
        if len(font_size_counts) > 8:
            analysis['recommendations'].append("Many different font sizes detected - document may benefit from typography harmonization")
        
        return analysis

