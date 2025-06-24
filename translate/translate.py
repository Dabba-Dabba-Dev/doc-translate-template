from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
import torch

app = FastAPI()

# Load model at startup
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.post("/translate")
async def translate_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": output}