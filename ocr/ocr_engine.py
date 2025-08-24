import os
import tempfile
import json
from PIL import Image, ImageDraw, ImageOps
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
import re
import cv2

def add_layout_newlines(text):
    # Already contains '\n' where lines were short; just clean extra spaces
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()

ocr_model = ocr_predictor(pretrained=True)


def calculate_line_spacing(lines):
    spacings = []
    for i in range(1, len(lines)):
        prev_bottom = lines[i-1]["box"][3]
        curr_top = lines[i]["box"][1]
        spacing = curr_top - prev_bottom
        if spacing > 0:
            spacings.append(spacing)
    return np.median(spacings) if spacings else 10

def get_alignment_group(x_left, tolerance=20):
    return round(x_left / tolerance) * tolerance

def should_merge_blocks(block1, block2, font_height, typical_spacing, alignment_tolerance=20):
    last_line = block1[-1]
    first_line = block2[0]

    vertical_gap = first_line["box"][1] - last_line["box"][3]
    alignment_diff = abs(last_line["box"][0] - first_line["box"][0])

    max_paragraph_gap = font_height * 0.8
    max_section_gap = font_height * 1.5

    condition1 = (vertical_gap <= max_paragraph_gap and alignment_diff <= alignment_tolerance)
    condition2 = vertical_gap <= font_height * 0.3

    if vertical_gap > max_section_gap:
        return False

    return condition1 or condition2

def clean_text(text):
    """Basic text cleaning without sentence segmentation"""
    text = re.sub(r'-\n', '', text)  # Remove hyphenated line breaks
    text = re.sub(r'(\d+)\.\n', r'\1. ', text)  # Fix numbered lists
    return text

def extract_line_text(line):
    """Extract text including symbols and bullet points"""
    text_parts = []
    for word in line.words:
        # Include low-confidence elements that might be symbols
        if word.confidence > 0.1:
            text_parts.append(word.value)
    return " ".join(text_parts)

def calculate_boldness_score_improved(pil_image, line_data, font_height):
    """
    Improved boldness calculation using multiple approaches:
    1. Pixel density analysis
    2. Edge density
    3. Variance-based approach
    """
    x_min, y_min, x_max, y_max = line_data["box"]
    
    # Add padding to ensure we capture the text properly
    padding = 2
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(pil_image.width, x_max + padding)
    y_max = min(pil_image.height, y_max + padding)
    
    # Extract the text region
    try:
        text_region = pil_image.crop((x_min, y_min, x_max, y_max))
        
        # Convert to grayscale
        gray_region = text_region.convert('L')
        region_array = np.array(gray_region)
        
        if region_array.size == 0:
            return {"boldness_score": 0, "confidence": 0}
        
        # Method 1: Pixel Density Analysis
        # Count dark pixels (assuming text is darker than background)
        # Use adaptive threshold based on image statistics
        mean_intensity = np.mean(region_array)
        threshold = mean_intensity * 0.8  # Adjust threshold based on image brightness
        
        dark_pixels = np.sum(region_array < threshold)
        total_pixels = region_array.size
        pixel_density = dark_pixels / total_pixels if total_pixels > 0 else 0
        
        # Method 2: Edge Density (more edges = thicker strokes)
        edges = cv2.Canny(region_array, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels if total_pixels > 0 else 0
        
        # Method 3: Standard Deviation of Intensities
        # Bold text typically has higher variance in intensities
        intensity_std = np.std(region_array)
        normalized_std = intensity_std / 255.0  # Normalize to 0-1
        
        # Method 4: Morphological Analysis
        # Apply closing operation to fill text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(region_array, cv2.MORPH_CLOSE, kernel)
        
        # Compare original vs closed to estimate stroke thickness
        diff = np.abs(closed.astype(float) - region_array.astype(float))
        stroke_thickness_indicator = np.mean(diff) / 255.0
        
        # Combine all methods with weights
        boldness_score = (
            pixel_density * 40 +           # 40% weight
            edge_density * 25 +            # 25% weight
            normalized_std * 20 +          # 20% weight
            stroke_thickness_indicator * 15 # 15% weight
        ) * 100  # Scale to 0-100
        
        # Calculate confidence based on text region size and clarity
        region_height = y_max - y_min
        region_width = x_max - x_min
        min_size_threshold = 10  # Minimum reasonable text size
        
        size_confidence = min(1.0, (region_height * region_width) / (min_size_threshold ** 2))
        clarity_confidence = min(1.0, intensity_std / 64.0)  # Higher std = clearer text
        
        overall_confidence = (size_confidence + clarity_confidence) / 2
        
        return {
            "boldness_score": boldness_score,
            "confidence": overall_confidence,
            "pixel_density": pixel_density,
            "edge_density": edge_density,
            "intensity_std": normalized_std,
            "stroke_thickness": stroke_thickness_indicator
        }
        
    except Exception as e:
        print(f"Warning: Error calculating boldness for region ({x_min}, {y_min}, {x_max}, {y_max}): {e}")
        return {"boldness_score": 0, "confidence": 0}

def extract_blocks_with_boxes(pil_image, image_path="output_overlay.png", alignment_tolerance=20):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        doc = DocumentFile.from_images(tmp_path)
    finally:
        os.remove(tmp_path)

    result = ocr_model(doc)

    draw = ImageDraw.Draw(pil_image)
    w, h = pil_image.size
    all_lines = []

    # Extract lines with bounding boxes
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                x_min = int(line.geometry[0][0] * w)
                y_min = int(line.geometry[0][1] * h)
                x_max = int(line.geometry[1][0] * w)
                y_max = int(line.geometry[1][1] * h)
                line_text = extract_line_text(line)
                
                alignment_group = get_alignment_group(x_min, alignment_tolerance)

                all_lines.append({
                    "text": line_text,
                    "box": (x_min, y_min, x_max, y_max),
                    "center_y": (y_min + y_max) / 2,
                    "alignment_group": alignment_group
                })

    # Sort AFTER collecting all lines
    all_lines.sort(key=lambda x: x["center_y"])
    if not all_lines:
        return []

    font_height = 40
    typical_spacing = calculate_line_spacing(all_lines)


    # Calculate boldness scores for all lines (now with font_height context)
    for line in all_lines:
        boldness_info = calculate_boldness_score_improved(pil_image, line, font_height)
        line.update(boldness_info)

    # Group lines into blocks by vertical spacing
    blocks = []
    current_block = [all_lines[0]]

    for i in range(1, len(all_lines)):
        current_line = all_lines[i]
        prev_line = all_lines[i-1]
        vertical_gap = current_line["box"][1] - prev_line["box"][3]

        if vertical_gap <= font_height * 0.4:
            current_block.append(current_line)
        else:
            blocks.append(current_block)
            current_block = [current_line]

    if current_block:
        blocks.append(current_block)

    # Merge blocks based on vertical gap & alignment
    merged_blocks = [blocks[0]] if blocks else []
    
    for i in range(1, len(blocks)):
        current_block = blocks[i]
        last_merged_block = merged_blocks[-1]
        if should_merge_blocks(last_merged_block, current_block, font_height, typical_spacing, alignment_tolerance):
            merged_blocks[-1].extend(current_block)
            print(f"[INFO] Merged blocks: '{last_merged_block[0]['text'][:30]}...' with '{current_block[0]['text'][:30]}...'")
        else:
            merged_blocks.append(current_block)

    # Helper: check if line fills block width
    def line_fills_block(line, block_width, threshold=0.9):
        line_width = line["box"][2] - line["box"][0]
        return (line_width / block_width) >= threshold

    output = []
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    # Assemble final text with width-aware merging
    for idx, block in enumerate(merged_blocks):
        block_x_min = min(line["box"][0] for line in block)
        block_x_max = max(line["box"][2] for line in block)
        block_width = block_x_max - block_x_min
        
        block_lines_text = []
        for line in block:
            if line_fills_block(line, block_width) and not line["text"].strip().endswith(('.', ',', ';', ':', '!', '?')):
                block_lines_text.append(line["text"])  # merge with next line
            else:
                block_lines_text.append(line["text"] + "\n")  # keep newline for layout

        raw_text = " ".join(block_lines_text)
        block_text = add_layout_newlines(raw_text)

        # Calculate average boldness for the block
        valid_boldness_scores = [line["boldness_score"] for line in block if line.get("confidence", 0) > 0.3]
        avg_boldness = np.mean(valid_boldness_scores) if valid_boldness_scores else 0

        # Draw overlay with boldness indication
        x_min = block_x_min
        y_min = min(line["box"][1] for line in block)
        x_max = block_x_max
        y_max = max(line["box"][3] for line in block)

        color = colors[idx % len(colors)]
        # Make border thicker for bold text
        border_width = 5 if avg_boldness > 30 else 3
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=border_width)
        
        # Add boldness info to label
        label = f"Block {idx+1} (Bold: {avg_boldness:.1f})"
        draw.text((x_min, y_min - 20), label, fill=color)

        output.append({
            "block_id": idx + 1,
            "block_text": block_text,
            "block_box": [
                [x_min / w, y_min / h],
                [x_max / w, y_max / h]
            ],
            "line_count": len(block),
            "average_boldness": avg_boldness,
            "lines": [
                {
                    "text": line["text"],
                    "normalized_box": [
                        [line["box"][0] / w, line["box"][1] / h],
                        [line["box"][2] / w, line["box"][3] / h]
                    ],
                    "boldness_score": line.get("boldness_score", 0),
                    "boldness_confidence": line.get("confidence", 0),
                    "alignment_group": line["alignment_group"]
                }
                for line in block
            ]
        })

    pil_image.save(image_path)
    print(f"[âœ”] Overlay image saved to {image_path}")
    print(f"[âœ”] Created {len(output)} blocks from {len(all_lines)} lines")

    # Print boldness summary
    print("\n[BOLDNESS ANALYSIS]")
    for block in output:
        print(f"Block {block['block_id']}: Average boldness = {block['average_boldness']:.1f}")
        for line in block['lines']:
            if line['boldness_confidence'] > 0.3:  # Only show confident predictions
                print(f"  Line: '{line['text'][:50]}...' - Bold: {line['boldness_score']:.1f}")

    return output