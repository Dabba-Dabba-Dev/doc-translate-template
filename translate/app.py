import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from load_translation_model import model, tokenizer
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

def translate_line(text, src_lang="deu_XX", tgt_lang="en_XX"):  
    if not text.strip():
        return ""
    tokenizer.src_lang = src_lang  
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            no_repeat_ngram_size=3,   # prevents repeating 3-grams
            repetition_penalty=2.0,   # penalizes repeated tokens
            max_length=200,           # restrict maximum length
            early_stopping=True       # stop when end token found
        )

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    translated = translate_line(text)
    return jsonify({"translated": translated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
