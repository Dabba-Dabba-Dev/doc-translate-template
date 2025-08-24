import json
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import difflib
from collections import Counter

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
        
              
        
        
        self.layout_rows = {} 
        self.min_vertical_gap = 5  
        self.overlap_tolerance = 10  
        
       
        self.document_text_analysis = {}
        self.paragraph_patterns = []
        self.normal_text_indicators = []
        
        
        self._load_enhanced_font_family()
    
    
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
        
        self.logger.info(f"Ã°Å¸â€œÅ  Document analysis: avg_length={self.document_text_analysis['avg_text_length']:.1f}, "
                        f"long_threshold={self.document_text_analysis['long_text_threshold']}, "
                        f"short_threshold={self.document_text_analysis['short_text_threshold']}")
    
    
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
    def _detect_alignment_for_docx(self, block_data: Dict, text: str) -> str:
        """Enhanced alignment detection with tighter thresholds and better logic"""
        box = block_data['block_box']
        left_x = box[0][0]
        right_x = box[1][0]
        center_x = (left_x + right_x) / 2
        
        # Much tighter thresholds - most documents have clear left/right margins
        left_threshold = 0.15   # Only truly left-aligned content
        right_threshold = 0.85  # Only truly right-aligned content
        center_low_threshold = 0.45   # Narrower center detection
        center_high_threshold = 0.55  # Narrower center detection
        
        # Check if we have line-level data for more accurate detection
        if 'lines' in block_data and block_data['lines']:
            line_alignments = []
            line_positions = []
            
            for line in block_data['lines']:
                if 'normalized_box' in line:
                    line_left = line['normalized_box'][0][0]
                    line_right = line['normalized_box'][1][0]
                    line_center = (line_left + line_right) / 2
                    line_positions.append((line_left, line_right, line_center))
                    
                    # Use tighter thresholds for line-level detection
                    if line_center <= center_low_threshold:
                        line_alignments.append("left")
                    elif line_center >= center_high_threshold:
                        line_alignments.append("right")
                    elif center_low_threshold < line_center < center_high_threshold:
                        line_alignments.append("center")
                    else:
                        # Default ambiguous cases to left
                        line_alignments.append("left")
            
            if line_alignments:
                alignment_counts = Counter(line_alignments)
                most_common_alignment, count = alignment_counts.most_common(1)[0]
                
                # Require stronger consensus for center alignment
                consensus_threshold = 0.8 if most_common_alignment == "center" else 0.6
                
                if count / len(line_alignments) >= consensus_threshold:
                    return most_common_alignment
                
                # If no strong consensus, analyze line positioning patterns
                if line_positions:
                    left_positions = [pos[0] for pos in line_positions]
                    right_positions = [pos[1] for pos in line_positions]
                    
                    # Check if lines are consistently left-aligned (similar left margins)
                    left_std = self._calculate_std(left_positions)
                    if left_std < 0.02 and min(left_positions) < 0.2:  # Consistent left margins
                        return "left"
                    
                    # Check if lines are consistently right-aligned (similar right margins)
                    right_std = self._calculate_std(right_positions)
                    if right_std < 0.02 and max(right_positions) > 0.8:  # Consistent right margins
                        return "right"
        
        # Fall back to block-level detection with conservative thresholds
        if center_x < left_threshold:
            return "left"
        elif center_x > right_threshold:
            return "right"
        elif center_low_threshold <= center_x <= center_high_threshold:
            # Additional validation for center alignment
            block_width = right_x - left_x
            # Only consider it centered if it's not a full-width block
            if block_width < 0.8:  # Less than 80% of page width
                return "center"
            else:
                return "left"  # Full-width blocks are typically left-aligned
        
        # Default to left alignment for ambiguous cases
        return "left"

    def _calculate_std(self, values):
        """Calculate standard deviation of a list of values"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _detect_alignment_from_lines(self, lines: List[Dict], block_width: float) -> str:
        """Improved line-based alignment detection"""
        if not lines:
            return "left"
        
        left_margins = []
        right_margins = []
        centers = []
        
        for line in lines:
            if 'normalized_box' in line:
                line_left = line['normalized_box'][0][0]
                line_right = line['normalized_box'][1][0]
                line_center = (line_left + line_right) / 2
                
                left_margins.append(line_left)
                right_margins.append(1 - line_right)  # Distance from right edge
                centers.append(line_center)
        
        if not left_margins:
            return "left"
        
        # Calculate consistency of positioning
        left_std = self._calculate_std(left_margins)
        right_std = self._calculate_std(right_margins)
        center_std = self._calculate_std(centers)
        
        avg_left = sum(left_margins) / len(left_margins)
        avg_right = sum(right_margins) / len(right_margins)
        avg_center = sum(centers) / len(centers)
        
        # Check for consistent left alignment
        if left_std < 0.03 and avg_left < 0.15:
            return "left"
        
        # Check for consistent right alignment  
        if right_std < 0.03 and avg_right < 0.15:
            return "right"
        
        # Check for center alignment (most restrictive)
        if (center_std < 0.02 and 0.45 <= avg_center <= 0.55 and
            avg_left > 0.1 and avg_right > 0.1):  # Significant margins on both sides
            return "center"
        
        # Default to left
        return "left"
    def get_structured_content(self, ocr_data: List[Dict], translation_data: List[Dict]) -> List[Dict]:
        """Get structured content for DOCX generation instead of image rendering"""
        try:
            self.placed_blocks = []
            
            # Analysis (same as before)
            self._analyze_document_text_characteristics(ocr_data)
            self.dominant_font_family = self._detect_dominant_font_family(ocr_data)
            self._calculate_layout_rows(ocr_data)
            
            translation_map = self._create_robust_translation_mapping(ocr_data, translation_data)
            sorted_blocks = self._sort_blocks_by_layout(ocr_data)
            
            structured_content = []
            
            for block in sorted_blocks:
                block_text = block.get('block_text', '').strip()
                if not block_text:
                    continue
                    
                translated_text = translation_map.get(block_text, block_text)
                style = self._detect_text_style_enhanced(translated_text, block)
                
                # Create structured block data
                content_block = {
                    'text': translated_text,
                    'font_size': 12,
                    'is_bold': style.font_weight == "bold",
                    'alignment': style.alignment,
                    'line_spacing': style.line_spacing,
                    'original_y': block['block_box'][0][1],  
                    'block_id': block.get('block_id', f"block_{len(structured_content)}")
                }
                
                structured_content.append(content_block)
                
            self.logger.info(f"Generated structured content with {len(structured_content)} blocks")
            return structured_content
            
        except Exception as e:
            self.logger.error(f"Error in structured content generation: {e}")
            raise
    def _is_text_bold(self, block_data: Dict) -> bool:
        """Simple bold detection using OCR data"""
        if 'lines' in block_data:
            bold_scores = []
            for line in block_data['lines']:
                if 'boldness_score' in line:
                    bold_scores.append(line['boldness_score'])
            
            if bold_scores:
                avg_boldness = sum(bold_scores) / len(bold_scores)
                return avg_boldness > 800  # Simple threshold
        
        return False
    def _get_font(self, size: int, weight: str = "regular") -> ImageFont.ImageFont:
        """Enhanced font retrieval with intelligent selection and weight support"""
        size = int(round(size))
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
    
    def _detect_content_alignment_within_block(self, block_data: Dict, text: str) -> str:
        """Detect alignment based on content positioning within the block"""
        box = block_data['block_box']
        left_x = box[0][0]
        right_x = box[1][0]
        width = right_x - left_x
        
        if 'lines' in block_data and block_data['lines']:
            return self._detect_alignment_from_lines(block_data['lines'], width)
        
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
                block_center = 0.5  
                
                left_margins.append(line_left)
                right_margins.append(1 - line_right)  
                center_deviations.append(abs(line_center - block_center))
        
        if not left_margins:
            return "left" 
        
        # Calculate average positioning
        avg_left_margin = sum(left_margins) / len(left_margins)
        avg_right_margin = sum(right_margins) / len(right_margins)
        avg_center_deviation = sum(center_deviations) / len(center_deviations)
        
        # Alignment detection thresholds
        left_aligned_threshold = 0.03  
        right_aligned_threshold = 0.05  
        center_aligned_threshold = 0.05 
        
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
    
    def _detect_dominant_font_family(self, ocr_data: List[Dict]) -> str:
        """Enhanced AI-powered dominant font detection using block-level font height"""
        font_characteristics = {}
        document_type_indicators = []
        
        # Analyze text content for document type
        all_text = ""
        for block in ocr_data:
            block_text = block.get('block_text', '')
            all_text += block_text + " "
        
        # Analyze font characteristics using block-level font height
        for block in ocr_data:
            estimated_height = 40
            boldness = 0
            
            # Try to get boldness from lines if available
            if 'lines' in block and block['lines']:
                line_boldness = [line.get('boldness_score', 0) for line in block['lines']]
                boldness = sum(line_boldness) / len(line_boldness) if line_boldness else 0
            
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
                    return "times" 
                else:
                    return "georgia"  
            elif 'legal_business' in document_type_indicators:
                return "calibri" 
            else:
                # Default analysis
                if "h35" in dominant_signature or "h40" in dominant_signature:
                    return "times"
                elif "h30" in dominant_signature:
                    return "calibri"
                elif "h25" in dominant_signature:
                    return "verdana"
                else:
                    return "times" 
        
        return "times" 

    def _detect_text_style_enhanced(self, text: str, block_data: Dict) -> TextStyle:
        """Use consistent font sizing based on block content"""
        
        
        # Get the estimated font height from OCR data - THIS IS THE KEY CHANGE
        block_font_size = 40
        
        # Determine alignment
        detected_alignment = self._detect_alignment_for_docx(block_data, text)
        
        # Check if this should be bold
        is_bold = self._is_text_bold(block_data)
        
        return TextStyle(
            font_size=int(block_font_size),
            font_weight="bold" if is_bold else "normal",
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
        row_threshold = 0.03  
        
        for y_pos, block in y_positions:
            if last_y == -1 or abs(y_pos - last_y) > row_threshold:
                current_row += 1
                last_y = y_pos
            
            if current_row not in self.layout_rows:
                self.layout_rows[current_row] = []
            
            self.layout_rows[current_row].append(block)
        
        self.logger.info(f"Organized content into {len(self.layout_rows)} layout rows")
    
    def _get_smart_position(self, block_data: Dict, text_width: int, text_height: int, style: TextStyle) -> Tuple[int, int]:
        """Improved positioning that maintains original vertical placement while handling overlaps"""
        box = block_data['block_box']
        
        # Use original coordinates as the primary reference
        original_x1 = int(box[0][0] * self.canvas_width)
        original_y1 = int(box[0][1] * self.canvas_height)
        original_x2 = int(box[1][0] * self.canvas_width)
        original_y2 = int(box[1][1] * self.canvas_height)
        
        # Calculate original block height for reference
        original_height = original_y2 - original_y1
        
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
        
        # START with original Y position - this is the key fix
        y_position = original_y1
        
        # Only adjust Y if there's an actual overlap, don't preemptively add gaps
        for placed_block in self.placed_blocks:
            if (x_position < placed_block.x2 and x_position + text_width > placed_block.x1 and
                y_position < placed_block.y2 and y_position + text_height > placed_block.y1):
                # There's an overlap, move this block below the overlapping one
                y_position = placed_block.y2 + 5  
                break
        
        # Ensure Y is within bounds
        y_position = max(20, min(y_position, self.canvas_height - text_height - 20))
        
        return x_position, y_position
    def _wrap_text_smart(self, text: str, font: ImageFont.ImageFont, font_size: int, max_width: int) -> List[str]:
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
                    # Fallback calculation using font size
                    line_width = len(test_line) * (font_size * 0.6)
                
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

        # Expand width if less than 70% of document width
        threshold_width = int(self.canvas_width * 0.7)
        if original_width < threshold_width:
            if style.alignment == "left":
                max_width = threshold_width
            elif style.alignment == "right":
                max_width = threshold_width
            elif style.alignment == "center":
                max_width = threshold_width
            else:
                max_width = original_width
        else:
            max_width = original_width

        max_width = max(max_width, 200)

        lines = self._wrap_text_smart(text, font, style.font_size, max_width)

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
                best_match = self._find_best_translation_match(block_text, translation_data)
                if best_match:
                    translation_map[block_text] = best_match
                else:
                    translation_map[block_text] = block_text 
        
        return translation_map
    
    def _find_best_translation_match(self, text: str, translation_data: List[Dict]) -> Optional[str]:
        """Find best translation match using fuzzy logic"""
        
        
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
            
            for block in sorted_blocks:
                block_text = block.get('block_text', '').strip()
                if not block_text:
                    continue
                
                translated_text = translation_map.get(block_text, block_text)
                
               
                style = self._detect_text_style_enhanced(translated_text, block)
                
               
                
                font_key = f"{self.dominant_font_family}_{style.font_size}_{style.font_weight}"
                font_usage_summary[font_key] = font_usage_summary.get(font_key, 0) + 1
                
                self._draw_enhanced_text_block(draw, translated_text, block, style)
                
                if debug_mode:
                    
                    box = block['block_box']
                    x1, y1 = int(box[0][0] * self.canvas_width), int(box[0][1] * self.canvas_height)
                    x2, y2 = int(box[1][0] * self.canvas_width), int(box[1][1] * self.canvas_height)
                    
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
                    
                    debug_font = self._get_font(12)
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
            
            self.logger.info("Ã¢Å“Â¨ Quality enhancements applied successfully")
            
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
    