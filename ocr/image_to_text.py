import cv2
import pytesseract
import numpy as np
import glob
import os

class ImageToText:
    def __init__(self):
        pass  # No YOLO needed for lightweight detection

    def _preprocess_image(self, img):
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)

    def detect_bullet_points(self, img):
        """Detect circular, square, check, and cross bullet points"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        bullet_boxes = []

        # Detect circles (as before)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=20
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                bullet_boxes.append({'box': [x - r, y - r, x + r, y + r], 'type': 'circle'})

        # Detect squares using contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 30 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect = float(w) / h
                if 0.8 < aspect < 1.2:  # roughly square
                    bullet_boxes.append({'box': [x, y, x + w, y + h], 'type': 'square'})

        # Detect check marks and crosses using template matching
        # Define simple binary templates for check and cross
        check_templates = [
            np.array([
                [0,0,1,0,0],
                [0,1,0,0,0],
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0]
            ], dtype=np.uint8) * 255,
            np.array([
                [0,0,0,1,0,0],
                [0,0,1,0,0,0],
                [0,1,0,0,0,0],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0]
            ], dtype=np.uint8) * 255
        ]
        cross_templates = [
            np.array([
                [1,0,0,0,1],
                [0,1,0,1,0],
                [0,0,1,0,0],
                [0,1,0,1,0],
                [1,0,0,0,1]
            ], dtype=np.uint8) * 255
        ]

        # Resize templates for matching
        for template, mark_type in [(check_templates[0], 'check'), (cross_templates[0], 'cross')]:
            for scale in [15, 20, 25]:
                tpl = cv2.resize(template, (scale, scale), interpolation=cv2.INTER_NEAREST)
                res = cv2.matchTemplate(thresh, tpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res > 0.6)
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    bullet_boxes.append({'box': [x, y, x + scale, y + scale], 'type': mark_type})

        # Also try the second check template
        for scale in [15, 20, 25]:
            tpl = cv2.resize(check_templates[1], (scale, scale), interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(thresh, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res > 0.6)
            for pt in zip(*loc[::-1]):
                x, y = pt
                bullet_boxes.append({'box': [x, y, x + scale, y + scale], 'type': 'check'})

        # Remove overlapping boxes (keep largest)
        def overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)
        filtered = []
        for i, item in enumerate(bullet_boxes):
            keep = True
            for j, other in enumerate(bullet_boxes):
                if i != j and overlap(item['box'], other['box']):
                    if (item['box'][2]-item['box'][0])*(item['box'][3]-item['box'][1]) < (other['box'][2]-other['box'][0])*(other['box'][3]-other['box'][1]):
                        keep = False
                        break
            if keep:
                filtered.append(item)
        return filtered

    def detect_frames(self, img):
        """Detect rectangular frames using contours"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_boxes = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(approx)
                frame_boxes.append([x, y, x + w, y + h])
        return frame_boxes

    def detect_signature_region(self, img):
        """Detect signature-like regions using contours, heuristics, and text density filtering"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        signature_boxes = []
        # Get text boxes from pytesseract to filter out text regions
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
            text_boxes = []
            for i, word in enumerate(data['text']):
                if word.strip() != "":
                    x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text_boxes.append((x, y, x + bw, y + bh))
        except Exception:
            text_boxes = []

        def overlaps_text(x1, y1, x2, y2):
            for tx1, ty1, tx2, ty2 in text_boxes:
                # If more than 30% of the region overlaps with text, consider it text
                ix1, iy1 = max(x1, tx1), max(y1, ty1)
                ix2, iy2 = min(x2, tx2), min(y2, ty2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                if iw * ih > 0.3 * (x2 - x1) * (y2 - y1):
                    return True
            return False

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect = cw / float(ch) if ch > 0 else 0
            roi = thresh[y:y+ch, x:x+cw]
            inkiness = np.sum(roi == 255) / (cw * ch + 1e-5)
            # Heuristics: wide, not too tall, enough area, near bottom, "inky" (not sparse), not text
            if (
                ch > 20 and cw > 100 and area > 500 and
                aspect > 2.5 and aspect < 10 and
                y > h * 0.6 and inkiness > 0.10 and
                not overlaps_text(x, y, x + cw, y + ch)
            ):
                signature_boxes.append([x, y, x + cw, y + ch])
        return signature_boxes

    def detect_round_stamps_and_logos(self, img):
        """Detect the single most color-distinct round stamp/logo, capturing its full shape."""
        # Resize for faster color clustering if large
        small = cv2.resize(img, (300, 300)) if max(img.shape[:2]) > 400 else img.copy()
        data = small.reshape((-1, 3)).astype(np.float32)
        # K-means to find main colors (background, text, stamp)
        K = 3
        _, labels, centers = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        # Sort by brightness (background is usually brightest or darkest)
        brightness = np.sum(centers, axis=1)
        bg_idx = np.argmax(np.bincount(labels.flatten()))
        bg_color = centers[bg_idx]
        # Assume text is the most different from background
        text_idx = np.argmax([np.linalg.norm(c - bg_color) for c in centers if not np.array_equal(c, bg_color)])
        text_color = centers[text_idx]
        # Create mask for pixels far from both background and text color
        diff_bg = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
        diff_text = np.linalg.norm(img.astype(np.float32) - text_color, axis=2)
        mask = np.logical_and(diff_bg > 60, diff_text > 60).astype(np.uint8) * 255
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=1)

        # Get text boxes from pytesseract to filter out text regions
        try:
            ocr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(ocr_gray, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
            text_boxes = []
            for i, word in enumerate(data['text']):
                if word.strip() != "":
                    x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text_boxes.append((x, y, x + bw, y + bh))
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

        # Find all contours in the mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1
        best_box = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circ_area = np.pi * (radius ** 2)
            circularity = area / circ_area if circ_area > 0 else 0
            if 0.7 < circularity < 1.2 and radius > 35:
                x1, y1, w, h = cv2.boundingRect(cnt)
                x2, y2 = x1 + w, y1 + h
                if overlaps_text(x1, y1, x2, y2):
                    continue
                # Compute mean color inside the contour
                mask_cnt = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                mean_color = cv2.mean(img, mask=mask_cnt)[:3]
                dist_bg = np.linalg.norm(np.array(mean_color) - bg_color)
                dist_text = np.linalg.norm(np.array(mean_color) - text_color)
                score = dist_bg + dist_text
                if score > best_score:
                    best_score = score
                    best_box = [x1, y1, x2, y2]

        # If not found by contour, fallback to HoughCircles as before
        if best_box is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            blurred = cv2.medianBlur(masked_gray, 7)
            h, w = img.shape[:2]
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                param1=60, param2=30, minRadius=40, maxRadius=int(min(h, w) * 0.5)
            )
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                best_score = -1
                for (x, y, r) in circles:
                    if r > 35 and x > r and y > r and x < w - r and y < h - r:
                        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                        area = (x2 - x1) * (y2 - y1)
                        if area > 2000 and not overlaps_text(x1, y1, x2, y2):
                            roi = img[y1:y2, x1:x2]
                            mean_color = cv2.mean(roi)[:3]
                            dist_bg = np.linalg.norm(np.array(mean_color) - bg_color)
                            dist_text = np.linalg.norm(np.array(mean_color) - text_color)
                            score = dist_bg + dist_text
                            if score > best_score:
                                best_score = score
                                best_box = [x1, y1, x2, y2]

        # Only return the most color-distinct stamp, as a list of one box (or empty if none)
        return [best_box] if best_box is not None else []

    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        processed = self._preprocess_image(img)
        text = pytesseract.image_to_string(processed, config=r'--oem 3 --psm 6')

        # Detect special elements
        bullet_items = self.detect_bullet_points(img)
        bullet_boxes = [item['box'] for item in bullet_items]
        bullet_types = [item['type'] for item in bullet_items]
        frame_boxes = self.detect_frames(img)
        signature_boxes = self.detect_signature_region(img)
        stamp_boxes = self.detect_round_stamps_and_logos(img)

        # Insert bullet points into text based on y-coordinate and type
        lines = text.splitlines()
        line_positions = []
        h, w = img.shape[:2]
        try:
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=r'--oem 3 --psm 6')
            for i, word in enumerate(data['text']):
                if word.strip() != "":
                    top = data['top'][i]
                    if not line_positions or abs(top - line_positions[-1]) > 10:
                        line_positions.append(top)
        except Exception:
            line_positions = [int(h * i / max(1, len(lines))) for i in range(len(lines))]

        bullet_lines = {}
        for idx, (box, typ) in enumerate(zip(bullet_boxes, bullet_types)):
            x1, y1, x2, y2 = box
            bullet_y = (y1 + y2) // 2
            min_dist = float('inf')
            min_idx = -1
            for i, ly in enumerate(line_positions):
                dist = abs(bullet_y - ly)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_idx != -1:
                bullet_lines[min_idx] = typ

        bullet_symbols = {
            'circle': '•',
            'square': '■',
            'check': '✓',
            'cross': '✗'
        }
        for idx, typ in bullet_lines.items():
            if 0 <= idx < len(lines):
                symbol = bullet_symbols.get(typ, '•')
                if not lines[idx].strip().startswith(symbol):
                    lines[idx] = f"{symbol} " + lines[idx].lstrip()

        text_with_bullets = "\n".join(lines)

        # Save each detected signature as an image
        for i, (x1, y1, x2, y2) in enumerate(signature_boxes):
            sig_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"signature_{i+1}.png", sig_crop)

        # Remove old stamp images before saving new ones
        for f in glob.glob("stamp_*.png"):
            try:
                os.remove(f)
            except Exception:
                pass

        # Save each detected round stamp/logo as an image
        for i, (x1, y1, x2, y2) in enumerate(stamp_boxes):
            stamp_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"stamp_{i+1}.png", stamp_crop)

        # Optionally, draw detected regions for visualization
        debug_img = img.copy()
        for item in bullet_items:
            x1, y1, x2, y2 = item['box']
            color = (0, 255, 0) if item['type'] == 'circle' else (0, 128, 255) if item['type'] == 'square' else (255, 0, 255) if item['type'] == 'check' else (0, 0, 255)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        for x1, y1, x2, y2 in frame_boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x1, y1, x2, y2 in signature_boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for x1, y1, x2, y2 in stamp_boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imwrite("debug_detected.png", debug_img)

        # Return text and detected regions
        result = {
            "text": text_with_bullets.strip(),
            "bullets": bullet_boxes,
            "frames": frame_boxes,
            "signatures": signature_boxes,
            "stamps": stamp_boxes
        }
        return result

    def process_image(self, image_path):
        return self.extract_text(image_path)

if __name__ == "__main__":
    ocr = ImageToText()
    result = ocr.process_image('APznzabTslW0F8B-lNrlhK5bsjtgWK5VyQ8weIjU5gczWaAIVD__xqJjUllf9qHoXfhbbqK_wAFFKg1VnBR-BU1UIFFN5kzAYOCKKWahY3P0-yvleEqpau1e9IU3riRQdxgFzBz0GofKZ0I7G19bvI-Jy09FTHxQoUWTX5gOIsS_xTuwerVC9ahgPu6e_page-0001.jpg')
    print("🔍 Extracted Text:\n", result["text"])
    print("Detected bullet points:", result["bullets"])
    print("Detected frames:", result["frames"])
    print("Detected signatures:", result["signatures"])
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])