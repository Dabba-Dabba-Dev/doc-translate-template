import cv2
import pytesseract
import requests

# === Step 1: OCR with OpenCV and Tesseract ===

# Load image
img = cv2.imread('Screenshot 2025-06-25 224812.png')

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

print("🔍 Extracted Text:\n", text.strip())

# === Step 2: Send text to LanguageTool for grammar correction ===

response = requests.post(
    "https://api.languagetool.org/v2/check",
    data={
        "text": text,
        "language": "auto"  # auto-detect language
    }
)

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

print("\n✅ Corrected Text:\n", suggested_text.strip())
