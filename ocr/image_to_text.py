from PIL import Image
import pytesseract
import cv2

# Set path to tesseract if not in PATH (Windows example)
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image using OpenCV or PIL
img = cv2.imread('9fe3923b-facd-479c-a55b-b9c272e11828.jpg')

# Convert to grayscale (optional but improves accuracy)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply OCR
text = pytesseract.image_to_string(gray)

print("Extracted Text:\n", text)
