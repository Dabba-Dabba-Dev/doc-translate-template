import requests
from typing import Dict, Any
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import fitz  # PyMuPDF
from PIL import Image
import io

class EnhancedOCR:
    def __init__(self):
        """
        Initializes the EnhancedOCR class, setting up the doctr OCR predictor.
        The model weights are downloaded and cached on the first run.
        """
        # Initialize doctr predictor
        self.model = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            self.model.cuda()

    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts text from an image using the doctr OCR model.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted text.
        """
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)
        
        text = result.render()
        
        return {
            "text": text.strip(),
            "stats": {} 
        }

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts text from a PDF using the doctr OCR model.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            A dictionary containing the extracted text.
        """
        doc = DocumentFile.from_pdf(pdf_path)
        result = self.model(doc)
        
        text = result.render()
        
        return {
            "text": text.strip(),
            "stats": {} 
        }

    def correct_with_languagetool(self, text: str) -> str:
        """
        Corrects the grammar of the given text using the LanguageTool API.

        Args:
            text: The text to be corrected.

        Returns:
            The corrected text.
        """
        try:
            response = requests.post(
                "https://api.languagetool.org/v2/check",
                data={
                    "text": text,
                    "language": "auto"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"LanguageTool API error: {response.status_code}")
                return text
            
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
            
            return suggested_text
            
        except Exception as e:
            print(f"LanguageTool correction failed: {e}")
            return text

# === Main execution ===
if __name__ == "__main__":
    # Set the input file path here
    input_file = "marwa_elmokni.jpg"

    try:
        # Initialize enhanced OCR
        ocr = EnhancedOCR()

        if input_file.lower().endswith('.pdf'):
            result = ocr.extract_text_from_pdf(input_file)
        else:
            result = ocr.extract_text_from_image(input_file)
        
        print("🔍 Extracted Text:")
        print("=" * 50)
        print(result["text"])
        print("=" * 50)
        
        # Apply grammar correction
        print("\n✅ Applying grammar correction...")
        corrected_text = ocr.correct_with_languagetool(result["text"])
        
        print("\n📝 Final Corrected Text:")
        print("=" * 50)
        print(corrected_text)
        print("=" * 50)
        
        # Save results
        with open("enhanced_output4.txt", "w", encoding="utf-8") as f:
            f.write("=== EXTRACTED TEXT ===\n")
            f.write(result["text"])
            f.write("\n\n=== GRAMMAR CORRECTED TEXT ===\n")
            f.write(corrected_text)
        
        print("\n💾 Results saved to 'enhanced_output.txt'")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Possible solutions:")
        print("1. Make sure the file exists and the path is correct")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Check internet connection for LanguageTool API")


