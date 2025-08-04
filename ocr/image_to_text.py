import cv2
import os
import PyPDF2
from langdetect import detect
import easyocr
import fitz  # PyMuPDF
import io

class ImageOCR:
    def __init__(self, file_path):
        self.file_path = file_path

    def detect_language(self, text):
        try:
            # Add a check for empty or whitespace-only text
            if text and not text.isspace():
                return detect(text)
            else:
                return "unknown (no text to detect)"
        except Exception as e:
            print(f"Language detection failed: {e}")
            return "unknown"

    def ocr_image(self, image_bytes):
        """Performs OCR on a single image provided as bytes."""
        # Using a broader set of languages for EasyOCR
        reader = easyocr.Reader(['ar', 'fa', 'ur', 'ug', 'en'], gpu=False)
        result = reader.readtext(image_bytes)
        return " ".join([item[1] for item in result])

    def process_file(self):
        """Processes either a PDF or an image file for OCR."""
        file_extension = os.path.splitext(self.file_path)[1].lower()
        extracted_text = ""

        if file_extension == '.pdf':
            # Extract text from the PDF
            with open(self.file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

            # Extract text from images within the PDF
            with fitz.open(self.file_path) as doc:
                for page in doc:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        extracted_text += self.ocr_image(image_bytes) + "\n"

        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # Perform OCR directly on the image
            with open(self.file_path, 'rb') as f:
                image_bytes = f.read()
            extracted_text = self.ocr_image(image_bytes)
        
        else:
            return "Unsupported file format.", "unknown"

        final_text = extracted_text.strip()
        language = self.detect_language(final_text)
        return final_text, language

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_file_path = "ocr_output1.txt"

        if os.path.exists(input_path):
            print(f"Processing {input_path}...")
            ocr_processor = ImageOCR(input_path)
            text, lang = ocr_processor.process_file()
            
            print(f"\nDetected Language: {lang}")
            
            # Save the result to a .txt file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"Successfully extracted text and saved it to {output_file_path}")
            print("\n--- Extracted Text ---")
            print(text)
        else:
            print(f"Error: File not found at {input_path}")
    else:
        print("Usage: python image_to_text.py <path_to_file>")