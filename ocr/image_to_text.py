import cv2
import pytesseract
import requests
import os

class ImageToText:
    def __init__(self):
        pass  # No need to set tesseract_cmd path

    def extract_text(self, image_path):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found or cannot be opened.")

        # Preprocess: grayscale, threshold, blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

        # Optional: Remove small noise
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                cv2.drawContours(thresh, [cnt], -1, 0, -1)

        # OCR config and extraction
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        return text.strip()

    def correct_text(self, text):
        response = requests.post(
            "https://api.languagetool.org/v2/check",
            data={
                "text": text,
                "language": "auto"
            }
        )
        suggested_text = text
        offset_correction = 0

        for match in response.json().get("matches", []):
            if match['replacements']:
                start = match['offset'] + offset_correction
                end = start + match['length']
                replacement = match['replacements'][0]['value']
                suggested_text = suggested_text[:start] + replacement + suggested_text[end:]
                offset_correction += len(replacement) - match['length']

        return suggested_text.strip()

    def process_image(self, image_path):
        raw_text = self.extract_text(image_path)
        corrected_text = self.correct_text(raw_text)
        return raw_text, corrected_text

# Example usage:
if __name__ == "__main__":
    ocr = ImageToText()
    raw, corrected = ocr.process_image('Screenshot 2025-06-25 224812.png')
    print("🔍 Extracted Text:\n", raw)
    print("\n✅ Corrected Text:\n", corrected)