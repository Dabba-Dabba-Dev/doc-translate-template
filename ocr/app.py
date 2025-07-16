from flask import Flask, request, jsonify, send_file
import os
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime

# Import functions/classes from the attached scripts
from pdfextractor import extract_text_from_pdf
from image_to_text import EnhancedOCRWithDocx

app = Flask(__name__)

# Initialize OCR class once
ocr_processor = EnhancedOCRWithDocx()
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get language parameter from form data or default to None
    lang = request.form.get('language', None)
    if lang and len(lang.strip()) == 0:  # Handle empty string case
        lang = None

    temp_input_filepath = None
    output_docx_filename = None  # Changed from temp path to permanent path

    try:
        filename = secure_filename(file.filename)
        original_name_without_ext = os.path.splitext(filename)[0]
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_input_filepath = temp_file.name

        # Create output DOCX filename in the current working directory
        output_docx_filename = f"{original_name_without_ext}_extracted.docx"
        output_docx_path = os.path.join(os.getcwd(), output_docx_filename)

        file_type_processed = "unknown"
        
        # Determine file type based on extension and process
        if filename.lower().endswith('.pdf'):
            file_type_processed = "pdf_layout_attempt"
            extracted_text, detected_languages = extract_text_from_pdf(
                pdf_path=temp_input_filepath, 
                lang=lang, 
                output_docx_path=output_docx_path
            )
            
            if not os.path.exists(output_docx_path) or os.path.getsize(output_docx_path) == 0:
                print("PDF layout extraction failed to produce DOCX, attempting OCR via image_text_extractor...")
                file_type_processed = "pdf_ocr_fallback"
                ocr_result = ocr_processor.process_image(
                    temp_input_filepath, 
                    output_docx=output_docx_path, 
                    lang=lang
                )
                extracted_text = ocr_result.get("text", "")
                detected_languages = ocr_result.get("detected_languages", "")
            else:
                file_type_processed = "pdf_layout_success"

        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            file_type_processed = "image_ocr"
            ocr_result = ocr_processor.process_image(
                temp_input_filepath, 
                output_docx=output_docx_path, 
                lang=lang
            )
            extracted_text = ocr_result.get("text", "")
            detected_languages = ocr_result.get("detected_languages", "")
        else:
            return jsonify({"error": "Unsupported file type. Please upload a PDF or an image (png, jpg, jpeg, gif, bmp, tiff)."}), 400

        if os.path.exists(output_docx_path) and os.path.getsize(output_docx_path) > 0:
            # Return the DOCX file as a download AND keep it saved in the current directory
            return send_file(
                output_docx_path,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=os.path.basename(output_docx_path)
            )
        else:
            return jsonify({"error": "Failed to generate DOCX file. No content extracted or an internal error occurred."}), 500

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}. Please ensure all dependencies are installed and Tesseract OCR is configured correctly."}), 500
    finally:
        # Clean up temporary input file only - keep the DOCX file
        if temp_input_filepath and os.path.exists(temp_input_filepath):
            os.remove(temp_input_filepath)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
