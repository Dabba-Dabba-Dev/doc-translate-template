import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify
import traceback
from load_model import model, tokenizer

app = Flask(__name__)



def chunk_text(text, max_length=500):
    """Split text into chunks suitable for translation"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def translate_text(text, source_lang = os.getenv('OCR_LANG', 'eng_Latn'), target_lang=None, max_chunk_size=500):
    try:
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided for translation"}

        if not target_lang:
            return {"status": "error", "message": "Target language is required for translation"}

        # Use provided codes directly, no conversion
        src_lang = source_lang
        tgt_lang = target_lang

        print(f"Translating from {src_lang} to {tgt_lang}")

        chunks = chunk_text(text, max_chunk_size)
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)}")
            tokenizer.src_lang = src_lang
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True
                )

            translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)

        full_translation = " ".join(translated_chunks)

        return {
            "status": "success",
            "original_text": text,
            "translated_text": full_translation,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "chunks_processed": len(chunks)
        }

    except Exception as e:
        print(f"Translation error: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Translation failed: {str(e)}"
        }
@app.route('/translate-extracted-text', methods=['POST'])
def translate_extracted_text():
    """
    Translate the text from text_extracted.txt file
    
    Expected JSON payload:
    {
        "target_language": "fr" (REQUIRED),
        "source_language": "auto" (optional, defaults to auto)
    }
    """
    try:
        # Check if the extracted text file exists
        extracted_text_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'text_extracted.txt'))
        print(f"[DEBUG] Looking for extracted text at: {extracted_text_path}")
        print(f"[DEBUG] File exists? {os.path.exists(extracted_text_path)}")

        if not os.path.exists(extracted_text_path):
            return jsonify({
                "status": "error",
                "message": "text_extracted.txt file not found. Please upload and extract text from a file first."
            }), 404

        # Get request data
        data = request.get_json() if request.is_json else {}
        
        # Check for required target language
        target_language = data.get('target_language') or request.form.get('target_language')
        if not target_language:
            return jsonify({
                "status": "error",
                "message": "Target language is required. Please specify 'target_language' parameter.",
                "example": "POST with JSON: {'target_language': 'fr'} or form data: target_language=fr"
            }), 400

        # Get optional source language
        source_language = data.get('source_language', 'auto') or request.form.get('source_language', 'auto')

        # Read the extracted text
        try:
            with open(extracted_text_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to read text_extracted.txt: {str(e)}"
            }), 500

        if not extracted_text.strip():
            return jsonify({
                "status": "error",
                "message": "text_extracted.txt is empty"
            }), 400

        # Perform translation
        print(f"Translating extracted text to {target_language}...")
        translation_result = translate_text(
            text=extracted_text,
            source_lang=source_language,
            target_lang=target_language
        )

        if translation_result["status"] == "success":
            # Save translated text to a new file
            translated_filename = f"text_translated_{target_language}.txt"
            translated_path = os.path.join(os.getcwd(), translated_filename)
            
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translation_result["translated_text"])

            return jsonify({
                "status": "success",
                "message": "Translation completed successfully",
                "source_file": "text_extracted.txt",
                "translated_file": translated_filename,
                "translation_info": {
                    "source_language": translation_result["source_language"],
                    "target_language": translation_result["target_language"],
                    "chunks_processed": translation_result["chunks_processed"],
                    "original_text_length": len(extracted_text),
                    "translated_text_length": len(translation_result["translated_text"])
                }
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Translation failed",
                "error_details": translation_result["message"]
            }), 500

    except Exception as e:
        print(f"Error in translate_extracted_text endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Translation service is running",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)