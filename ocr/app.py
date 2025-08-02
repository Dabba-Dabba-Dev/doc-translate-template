from flask import Flask, request, jsonify, send_file
import os
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
import pdfplumber

# Import both processors
from image_to_text import EnhancedOCRProcessor
from pdfextractor import extract_text_with_layout, has_extractable_text

app = Flask(__name__)

# Initialize OCR processor once
ocr_processor = EnhancedOCRProcessor()

def has_extractable_text(pdf_path):
    """Check if PDF has extractable text content"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:  # Check first 3 pages
                text = page.extract_text()
                if text and text.strip():
                    return True
        return False
    except:
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # Get language parameter from form data or default to None
    lang = request.form.get('language', None)
    if lang and len(lang.strip()) == 0:  # Handle empty string case
        lang = None

    temp_input_filepath = None
    output_txt_filename = None

    try:
        filename = secure_filename(file.filename)
        original_name_without_ext = os.path.splitext(filename)[0]
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_input_filepath = temp_file.name

        # Create output TXT filename in the current working directory
        output_txt_filename = "text_extracted.txt"
        output_txt_path = os.path.join(os.getcwd(), output_txt_filename)

        # Remove existing file if it exists
        if os.path.exists(output_txt_path):
            os.remove(output_txt_path)

        file_type_processed = "unknown"
        extracted_text = ""
        pages_processed = 1
        
        # Check if it's a PDF file
        if filename.lower().endswith('.pdf'):
            # First try to extract text using pdfextractor
            if has_extractable_text(temp_input_filepath):
                print("PDF has extractable text, using pdfextractor...")
                extracted_text = extract_text_with_layout(temp_input_filepath)
                file_type_processed = "pdf_text_extraction"
                
                # Write the extracted text to file
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            else:
                print("PDF appears to be image-based, falling back to OCR...")
                # Fall back to OCR for image-based PDFs
                results = ocr_processor.process_file(
                    temp_input_filepath,
                    output_txt=output_txt_path,
                    lang=lang
                )
                
                # Combine text from all pages
                all_pages_text = []
                for page in results:
                    all_pages_text.append(page['text'])
                
                extracted_text = "\n\n--- Page Break ---\n\n".join(all_pages_text)
                pages_processed = len(results)
                file_type_processed = "pdf_ocr"
                
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # Process image files using OCR
            results = ocr_processor.process_file(
                temp_input_filepath,
                output_txt=output_txt_path,
                lang=lang
            )
            
            # Combine text from all pages
            all_pages_text = []
            for page in results:
                all_pages_text.append(page['text'])
            
            extracted_text = "\n\n--- Page Break ---\n\n".join(all_pages_text)
            pages_processed = len(results)
            file_type_processed = "image_ocr"
        else:
            return jsonify({"status": "error", "message": "Unsupported file type"}), 400

        if extracted_text and extracted_text.strip():
            return jsonify({
                "status": "success",
                "message": "Extraction complete",
                "file_type": file_type_processed,
                "pages_processed": pages_processed,
                "text_file": output_txt_filename
            }), 200
        else:
            return jsonify({"status": "error", "message": "Failed to extract any text"}), 500

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500
    finally:
        # Clean up temporary input file only
        if temp_input_filepath and os.path.exists(temp_input_filepath):
            os.remove(temp_input_filepath)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='text/plain'
            )
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)