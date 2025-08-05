import cv2
import os
import PyPDF2
from langdetect import detect
import easyocr
import fitz  # PyMuPDF
import io
import numpy as np
import sys

class ImageOCR:
    def __init__(self, image_path):
        self.image_path = image_path

    def convert_to_jpeg(self):
        _, file_extension = os.path.splitext(self.image_path)
        if file_extension.lower() not in ['.jpg', '.jpeg']:
            img = cv2.imread(self.image_path)
            jpeg_image_path = os.path.splitext(self.image_path)[0] + '.jpeg'
            cv2.imwrite(jpeg_image_path, img)
            self.image_path = jpeg_image_path

    def process_pdf_and_summarize(self):
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf()
        print("Recognized Text from PDF:")
        print(pdf_text)

        # Extract and process images from PDF
        image_text = self.extract_and_process_images_from_pdf()
        combined_text = pdf_text + " " + image_text
        print("Combined Text from PDF and Images:")
        print(combined_text)

        original_language = self.detect_language(combined_text)
        print("Original Language:", original_language)
        return combined_text

    def extract_text_from_pdf(self):
        pdf_text = ""
        with open(self.image_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + " "
        return pdf_text.strip()

    def detect_language(self, text):
        try:
            detected_lang = detect(text)
        except:
            detected_lang = "unknown"
        return detected_lang

    def perform_easyocr(self, image_bytes):
        reader = easyocr.Reader(['en', 'pt', 'de', 'it', 'pl'], gpu=False)
        result = reader.readtext(image_bytes)
        return " ".join([item[1] for item in result])

    def extract_and_process_images_from_pdf(self):
        text_from_images = ""
        with fitz.open(self.image_path) as doc:
            for page in doc:
                image_list = page.get_images(full=True)
                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Perform OCR on the image
                    image_text = self.perform_easyocr(image_bytes)
                    text_from_images += image_text + " "
        return text_from_images.strip()

if __name__ == "__main__":
    def save_text_to_file(text, image_path):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_filename}_extracted.txt"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {output_path}")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please provide an image path as a command-line argument.")
        sys.exit(1)
        
    ocr_processor = ImageOCR(image_path)
    extracted_text = ""

    if image_path.lower().endswith('.pdf'):
        extracted_text = ocr_processor.process_pdf_and_summarize()
    elif image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        ocr_processor.convert_to_jpeg()
        extracted_text = ocr_processor.perform_easyocr(ocr_processor.image_path)
        print("Recognized Text from Image:")
        print(extracted_text)
        original_language = ocr_processor.detect_language(extracted_text)
        print("Original Language:", original_language)
    else:
        print("Invalid file format. Supported formats: PDF, JPG, JPEG, PNG.")

    if extracted_text:
        save_text_to_file(extracted_text, image_path)