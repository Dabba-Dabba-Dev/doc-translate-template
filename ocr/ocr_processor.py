import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import pytesseract
from loguru import logger


class OCRProcessor:
    """
    Modular OCR processor supporting multiple engines with fallback mechanisms.
    
    Attributes:
        engines (Dict): Dictionary of OCR engines
        preprocessors (Dict): Dictionary of image preprocessing functions
        confidence_threshold (float): Minimum confidence for text extraction
    """
    
    def __init__(self, 
                 engines: List[str] = ["easyocr", "doctr", "tesseract"],
                 confidence_threshold: float = 0.5,
                 use_gpu: bool = False):
        """
        Initialize the OCR processor.
        
        Args:
            engines (List[str]): List of OCR engines to use
            confidence_threshold (float): Minimum confidence threshold
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.engines = {}
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        # Initialize preprocessing functions
        self.preprocessors = {
            "grayscale": self._preprocess_grayscale,
            "denoise": self._preprocess_denoise,
            "enhance": self._preprocess_enhance,
            "deskew": self._preprocess_deskew
        }
        
        # Initialize OCR engines
        self._initialize_engines(engines)
    
    def _initialize_engines(self, engine_names: List[str]):
        """Initialize specified OCR engines."""
        logger.info(f"Initializing OCR engines: {engine_names}")
        
        for engine_name in engine_names:
            try:
                if engine_name == "easyocr":
                    self.engines["easyocr"] = easyocr.Reader(
                        ['en', 'fr', 'ar'], 
                        gpu=self.use_gpu
                    )
                    logger.info("EasyOCR initialized successfully")
                    
                elif engine_name == "doctr":
                    self.engines["doctr"] = ocr_predictor(pretrained=True)
                    logger.info("docTR initialized successfully")
                    
                elif engine_name == "tesseract":
                    # Check if tesseract is installed
                    try:
                        pytesseract.get_tesseract_version()
                        self.engines["tesseract"] = True
                        logger.info("Tesseract initialized successfully")
                    except Exception as e:
                        logger.warning(f"Tesseract not available: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to initialize {engine_name}: {e}")
    
    def _preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _preprocess_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to the image."""
        return cv2.fastNlMeansDenoising(image)
    
    def _preprocess_enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def _preprocess_deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def preprocess_image(self, image: np.ndarray, 
                        steps: List[str] = ["grayscale", "denoise", "enhance"]) -> np.ndarray:
        """
        Apply preprocessing steps to the image.
        
        Args:
            image (np.ndarray): Input image
            steps (List[str]): List of preprocessing steps to apply
            
        Returns:
            np.ndarray: Preprocessed image
        """
        processed = image.copy()
        
        for step in steps:
            if step in self.preprocessors:
                processed = self.preprocessors[step](processed)
                
        return processed
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using EasyOCR.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of text regions with confidence scores
        """
        try:
            results = self.engines["easyocr"].readtext(image)
            
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence >= self.confidence_threshold:
                    extracted_texts.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "engine": "easyocr"
                    })
                    
            return extracted_texts
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_text_doctr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using docTR.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of text regions with confidence scores
        """
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Create document file
            doc = DocumentFile.from_images([pil_image])
            
            # Extract text
            result = self.engines["doctr"](doc)
            
            extracted_texts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if word.confidence >= self.confidence_threshold:
                                extracted_texts.append({
                                    "text": word.value,
                                    "confidence": word.confidence,
                                    "bbox": word.geometry,
                                    "engine": "doctr"
                                })
                                
            return extracted_texts
            
        except Exception as e:
            logger.error(f"docTR extraction failed: {e}")
            return []
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using Tesseract.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of text regions with confidence scores
        """
        try:
            # Configure tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()_+-=[]{}|;:,.<>?'
            
            # Extract text with confidence
            data = pytesseract.image_to_data(image, config=custom_config, 
                                           output_type=pytesseract.Output.DICT)
            
            extracted_texts = []
            for i, conf in enumerate(data['conf']):
                if conf > self.confidence_threshold * 100:  # Tesseract uses 0-100 scale
                    text = data['text'][i].strip()
                    if text:
                        extracted_texts.append({
                            "text": text,
                            "confidence": conf / 100.0,  # Normalize to 0-1
                            "bbox": [data['left'][i], data['top'][i], 
                                   data['left'][i] + data['width'][i], 
                                   data['top'][i] + data['height'][i]],
                            "engine": "tesseract"
                        })
                        
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def extract_text(self, image: np.ndarray, 
                    engine: str = "auto",
                    preprocess: bool = True,
                    preprocess_steps: List[str] = ["grayscale", "denoise", "enhance"]) -> Dict:
        """
        Extract text from image using specified or best available engine.
        
        Args:
            image (np.ndarray): Input image
            engine (str): OCR engine to use ("auto", "easyocr", "doctr", "tesseract")
            preprocess (bool): Whether to preprocess the image
            preprocess_steps (List[str]): Preprocessing steps to apply
            
        Returns:
            Dict: Extraction results with text, confidence, and metadata
        """
        # Preprocess image if requested
        if preprocess:
            image = self.preprocess_image(image, preprocess_steps)
        
        results = {
            "text": "",
            "confidence": 0.0,
            "engine_used": "",
            "all_results": [],
            "success": False
        }
        
        # Determine which engine to use
        if engine == "auto":
            # Try engines in order of preference
            engine_order = ["easyocr", "doctr", "tesseract"]
        else:
            engine_order = [engine]
        
        for engine_name in engine_order:
            if engine_name not in self.engines:
                continue
                
            try:
                if engine_name == "easyocr":
                    extracted = self.extract_text_easyocr(image)
                elif engine_name == "doctr":
                    extracted = self.extract_text_doctr(image)
                elif engine_name == "tesseract":
                    extracted = self.extract_text_tesseract(image)
                else:
                    continue
                
                if extracted:
                    # Combine all text
                    all_text = " ".join([item["text"] for item in extracted])
                    avg_confidence = np.mean([item["confidence"] for item in extracted])
                    
                    results.update({
                        "text": all_text,
                        "confidence": avg_confidence,
                        "engine_used": engine_name,
                        "all_results": extracted,
                        "success": True
                    })
                    
                    logger.info(f"Successfully extracted text using {engine_name}")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to extract text using {engine_name}: {e}")
                continue
        
        return results
    
    def extract_text_from_file(self, file_path: Union[str, Path],
                             engine: str = "auto",
                             preprocess: bool = True) -> Dict:
        """
        Extract text from a file (image or PDF).
        
        Args:
            file_path (Union[str, Path]): Path to the file
            engine (str): OCR engine to use
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            Dict: Extraction results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load image
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
                
            return self.extract_text(image, engine, preprocess)
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine_used": "",
                "all_results": [],
                "success": False,
                "error": str(e)
            }
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        return list(self.engines.keys())
    
    def get_engine_stats(self) -> Dict:
        """Get statistics about available engines."""
        return {
            "available_engines": self.get_available_engines(),
            "confidence_threshold": self.confidence_threshold,
            "use_gpu": self.use_gpu
        }


# Factory function for easy instantiation
def create_ocr_processor(engines: List[str] = ["easyocr", "doctr", "tesseract"],
                        confidence_threshold: float = 0.5,
                        use_gpu: bool = False) -> OCRProcessor:
    """
    Factory function to create an OCRProcessor instance.
    
    Args:
        engines (List[str]): List of OCR engines to use
        confidence_threshold (float): Minimum confidence threshold
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        OCRProcessor: Configured OCR processor instance
    """
    return OCRProcessor(engines=engines, 
                       confidence_threshold=confidence_threshold,
                       use_gpu=use_gpu)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Get file path from command-line arguments
        file_path = sys.argv[1]
        
        # Create processor and process the file
        processor = create_ocr_processor()
        results = processor.extract_text_from_file(file_path)
        
        # Print results
        print(f"Results for {file_path}:")
        print(results)
        
    else:
        # Example usage if no file is provided
        print("Usage: python ocr_processor.py <path_to_image>")
        processor = create_ocr_processor()
        print("Available engines:", processor.get_available_engines())
        print("Engine stats:", processor.get_engine_stats()) 