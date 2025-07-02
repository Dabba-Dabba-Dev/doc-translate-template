import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

class ImageToText:
    def __init__(self):
        # Load YOLO model (text detection)
        self.text_detector = YOLO("yolov8n.pt")  # Lightweight model
        # self.text_detector = YOLO("yolov8x.pt")  # More accurate but slower

    def _preprocess_image(self, img):
        """Basic image preprocessing for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.GaussianBlur(thresh, (3, 3), 0)

    def _extract_with_yolo(self, img):
        """Detect text regions using YOLO"""
        results = self.text_detector(img)  # Class 80 = text in some models
        return [map(int, box) for box in results[0].boxes.xyxy.tolist()]

    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        # Detect text regions with YOLO
        text_boxes = self._extract_with_yolo(img)
        full_text = []

        for x1, y1, x2, y2 in text_boxes:
            # Crop and preprocess each text region
            cropped = img[y1:y2, x1:x2]
            processed = self._preprocess_image(cropped)
            
            # OCR on the cropped region
            text = pytesseract.image_to_string(
                processed, 
                config=r'--oem 3 --psm 6'
            )
            full_text.append(text.strip())

        return "\n".join(filter(None, full_text))

    def process_image(self, image_path):
        return self.extract_text(image_path)

# Example usage remains the same
if __name__ == "__main__":
    ocr = ImageToText()
    raw = ocr.process_image('APznzabTslW0F8B-lNrlhK5bsjtgWK5VyQ8weIjU5gczWaAIVD__xqJjUllf9qHoXfhbbqK_wAFFKg1VnBR-BU1UIFFN5kzAYOCKKWahY3P0-yvleEqpau1e9IU3riRQdxgFzBz0GofKZ0I7G19bvI-Jy09FTHxQoUWTX5gOIsS_xTuwerVC9ahgPu6e_page-0001.jpg')
    print("🔍 Extracted Text:\n", raw)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(raw)