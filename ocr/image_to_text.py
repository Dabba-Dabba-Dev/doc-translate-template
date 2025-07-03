import cv2
import pytesseract
import numpy as np
import glob
import os
from typing import List, Dict, Any, Tuple
import mimetypes

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class ImageToText:
    def __init__(self):
        # ...existing code...
        self.bullet_symbols = {
            'circle': '•',
            'square': '■',
            'check': '✓',
            'cross': '✗'
        }
        # ...existing code...

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)

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
            param1=50, param2=30, minRadius=7, maxRadius=20  # minRadius increased to avoid small letters
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                area = np.pi * r * r
                # Filter out small circles (likely not bullets)
                if area > 120:
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
                # Filter out small or non-square shapes
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
                loc = np.where(res > 0.7)  # stricter threshold
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    # Only accept if the region is not too small
                    if scale > 10:
                        bullet_boxes.append({'box': [x, y, x + scale, y + scale], 'type': typ})

        # Remove boxes that overlap with text boxes (likely not bullets)
        text_boxes = self._get_text_boxes(img)
        filtered = []
        for item in bullet_boxes:
            if not self._overlaps_text(tuple(item['box']), text_boxes, threshold=0.5):
                filtered.append(item)
        return self._filter_overlapping_boxes(filtered, max_items=None)

    def detect_frames(self, img: np.ndarray) -> List[List[int]]:
        """Detect rectangular frames using contours"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [list(cv2.boundingRect(approx)) for cnt in contours 
                if len(approx := cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4 
                and cv2.contourArea(cnt) > 1000]

    def detect_signature_region(self, img: np.ndarray) -> List[List[int]]:
        """Detect up to 2 signature-like regions using contours, heuristics, and text density filtering"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_boxes = self._get_text_boxes(img)
        h, w = img.shape[:2]

        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect = cw / max(ch, 1)
            inkiness = np.sum(thresh[y:y+ch, x:x+cw] == 255) / max(cw * ch, 1)
            if (ch > 20 and cw > 100 and area > 500 and 2.5 < aspect < 10 and 
                y > h * 0.6 and inkiness > 0.10 and 
                not self._overlaps_text((x, y, x+cw, y+ch), text_boxes)):
                candidates.append((area, [x, y, x + cw, y + ch]))

        candidates.sort(reverse=True)
        filtered = self._filter_overlapping_boxes([{'box': box} for _, box in candidates], max_items=2)
        return [item['box'] for item in filtered]

    def detect_round_stamps_and_logos(self, img: np.ndarray) -> List[List[int]]:
        """Detect up to 2 most color-distinct round stamps/logos, capturing their full shape."""
        small = cv2.resize(img, (300, 300)) if max(img.shape[:2]) > 400 else img.copy()
        data = small.reshape((-1, 3)).astype(np.float32)
        K = 3
        _, labels, centers = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        text_color = centers[np.argmax([np.linalg.norm(c - bg_color) for c in centers if not np.array_equal(c, bg_color)])]
        diff_bg = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
        diff_text = np.linalg.norm(img.astype(np.float32) - text_color, axis=2)
        mask = np.logical_and(diff_bg > 60, diff_text > 60).astype(np.uint8) * 255
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=1)

        try:
            ocr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(ocr_gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
            text_boxes = [(data['left'][i], data['top'][i], data['left'][i]+data['width'][i], data['top'][i]+data['height'][i])
                          for i, word in enumerate(data['text']) if word.strip()]
        except Exception:
            text_boxes = []

        def overlaps_text(x1, y1, x2, y2):
            for tx1, ty1, tx2, ty2 in text_boxes:
                ix1, iy1 = max(x1, tx1), max(y1, ty1)
                ix2, iy2 = min(x2, tx2), min(y2, ty2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                if iw * ih > 0.2 * (x2 - x1) * (y2 - y1):
                    return True
            return False

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stamp_candidates = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (np.pi * (radius ** 2)) if radius > 0 else 0
            if 0.7 < circularity < 1.2 and radius > 35:
                x1, y1, w, h = cv2.boundingRect(cnt)
                x2, y2 = x1 + w, y1 + h
                if overlaps_text(x1, y1, x2, y2):
                    continue
                mean_color = cv2.mean(img, mask=cv2.drawContours(np.zeros(mask.shape, np.uint8), [cnt], -1, 255, -1))[:3]
                dist_bg = np.linalg.norm(np.array(mean_color) - bg_color)
                dist_text = np.linalg.norm(np.array(mean_color) - text_color)
                score = dist_bg + dist_text
                stamp_candidates.append((score, [x1, y1, x2, y2]))
        stamp_candidates.sort(reverse=True)
        stamp_boxes = []
        for _, box in stamp_candidates:
            if len(stamp_boxes) == 0 or all(
                box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]
                for b in stamp_boxes
            ):
                stamp_boxes.append(box)
            if len(stamp_boxes) == 2:
                break
        # Fallback to HoughCircles if none found
        if not stamp_boxes:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            blurred = cv2.medianBlur(masked_gray, 7)
            h, w = img.shape[:2]
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                param1=60, param2=30, minRadius=40, maxRadius=int(min(h, w) * 0.5)
            )
            circle_candidates = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if r > 35 and x > r and y > r and x < w - r and y < h - r:
                        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                        area = (x2 - x1) * (y2 - y1)
                        if area > 2000 and not overlaps_text(x1, y1, x2, y2):
                            mean_color = cv2.mean(img[y1:y2, x1:x2])[:3]
                            dist_bg = np.linalg.norm(np.array(mean_color) - bg_color)
                            dist_text = np.linalg.norm(np.array(mean_color) - text_color)
                            score = dist_bg + dist_text
                            circle_candidates.append((score, [x1, y1, x2, y2]))
                circle_candidates.sort(reverse=True)
                for _, box in circle_candidates:
                    if len(stamp_boxes) == 0 or all(
                        box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]
                        for b in stamp_boxes
                    ):
                        stamp_boxes.append(box)
                    if len(stamp_boxes) == 2:
                        break
        return stamp_boxes

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

    def _save_regions(self, img: np.ndarray, boxes: List[List[int]], prefix: str):
        """Save detected regions as images"""
        # Remove old files
        for f in glob.glob(f"{prefix}_*.png"):
            try:
                os.remove(f)
            except Exception:
                pass
        
        # Save new regions
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            cv2.imwrite(f"{prefix}_{i}.png", img[y1:y2, x1:x2])

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Main extraction method. Supports image files and PDFs (first page)."""
        # Detect if PDF
        if image_path.lower().endswith('.pdf'):
            if convert_from_path is None:
                raise ImportError("pdf2image is required for PDF support. Install with 'pip install pdf2image'.")
            # Use single page (first page) and lower DPI for speed
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

        # Extract text
        text = pytesseract.image_to_string(self._preprocess_image(img), config='--oem 3 --psm 6')

        # Detect elements
        bullet_items = self.detect_bullet_points(img)
        text_with_bullets = self._insert_bullet_points(text, img, bullet_items)
        frame_boxes = self.detect_frames(img)
        signature_boxes = self.detect_signature_region(img)
        stamp_boxes = self.detect_round_stamps_and_logos(img)

        # Save regions
        self._save_regions(img, signature_boxes, "signature")
        self._save_regions(img, stamp_boxes, "stamp")

        # Debug visualization
        debug_img = img.copy()
        colors = {'circle': (0, 255, 0), 'square': (0, 128, 255), 
                 'check': (255, 0, 255), 'cross': (0, 0, 255)}
        for item in bullet_items:
            cv2.rectangle(debug_img, tuple(item['box'][:2]), tuple(item['box'][2:4]), 
                         colors.get(item['type'], (0, 255, 0)), 2)
        for box in frame_boxes:
            cv2.rectangle(debug_img, tuple(box[:2]), tuple(box[2:4]), (255, 0, 0), 2)
        for box in signature_boxes:
            cv2.rectangle(debug_img, tuple(box[:2]), tuple(box[2:4]), (0, 0, 255), 2)
        for box in stamp_boxes:
            cv2.rectangle(debug_img, tuple(box[:2]), tuple(box[2:4]), (255, 0, 255), 2)
        cv2.imwrite("debug_detected.png", debug_img)

        return {
            "text": text_with_bullets.strip(),
            "bullets": [item['box'] for item in bullet_items],
            "frames": frame_boxes,
            "signatures": signature_boxes,
            "stamps": stamp_boxes
        }

    def process_image(self, image_path: str) -> Dict[str, Any]:
        return self.extract_text(image_path)

if __name__ == "__main__":
    ocr = ImageToText()
    result = ocr.process_image('Screenshot 2025-07-03 001911.png')
    print("🔍 Extracted Text:\n", result["text"])
    print("Detected bullet points:", result["bullets"])
    print("Detected frames:", result["frames"])
    print("Detected signatures:", result["signatures"])
    with open ("output.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])