import os
import tempfile
import json
from PIL import Image, ImageDraw, ImageOps
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
import re

def add_layout_newlines(text):
    # Already contains '\n' where lines were short; just clean extra spaces
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


ocr_model = ocr_predictor(pretrained=True)

def calculate_font_height(lines):
    heights = [line["box"][3] - line["box"][1] for line in lines]
    return np.median(heights) if heights else 20

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

                line_text = " ".join([word.value for word in line.words])
                thickness = np.mean([word.confidence for word in line.words]) if line.words else 0.5
                height = y_max - y_min
                boldness_score = thickness * height
                alignment_group = get_alignment_group(x_min, alignment_tolerance)

                all_lines.append({
                    "text": line_text,
                    "box": (x_min, y_min, x_max, y_max),
                    "center_y": (y_min + y_max) / 2,
                    "boldness_score": boldness_score,
                    "alignment_group": alignment_group
                })

    all_lines.sort(key=lambda x: x["center_y"])
    if not all_lines:
        return []

    font_height = calculate_font_height(all_lines)
    typical_spacing = calculate_line_spacing(all_lines)

    print(f"[INFO] Estimated font height: {font_height:.1f}px")
    print(f"[INFO] Typical line spacing: {typical_spacing:.1f}px")

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
            if line_fills_block(line, block_width):
                block_lines_text.append(line["text"])  # merge with next line
            else:
                block_lines_text.append(line["text"] + "\n")  # keep newline for layout

        raw_text = " ".join(block_lines_text)
        block_text = add_layout_newlines(raw_text)


        # Draw overlay
        x_min = block_x_min
        y_min = min(line["box"][1] for line in block)
        x_max = block_x_max
        y_max = max(line["box"][3] for line in block)

        color = colors[idx % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min, y_min - 20), f"Block {idx+1}", fill=color)

        output.append({
            "block_id": idx + 1,
            "block_text": block_text,
            "block_box": [
                [x_min / w, y_min / h],
                [x_max / w, y_max / h]
            ],
            "line_count": len(block),
            "estimated_font_height": font_height,
            "lines": [
                {
                    "text": line["text"],
                    "normalized_box": [
                        [line["box"][0] / w, line["box"][1] / h],
                        [line["box"][2] / w, line["box"][3] / h]
                    ],
                    "boldness_score": line["boldness_score"],
                    "alignment_group": line["alignment_group"]
                }
                for line in block
            ]
        })

    pil_image.save(image_path)
    print(f"[✔] Overlay image saved to {image_path}")
    print(f"[✔] Created {len(output)} blocks from {len(all_lines)} lines")

    return output