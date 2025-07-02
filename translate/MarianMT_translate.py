import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from transformers import MarianMTModel, MarianTokenizer
import torch
import time
import tracemalloc

# Start measuring time and memory
start_time = time.time()
tracemalloc.start()

# Load model at startup
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
text = "This is a visa document."
inputs = tokenizer(text, return_tensors="pt", padding=True)
with torch.no_grad():
    translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)
# Stop measuring memory and time
current, peak = tracemalloc.get_traced_memory()
end_time = time.time()
tracemalloc.stop()

print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
print("translation:", output)

#@app.post("/translate")
#async def translate_text(text: str):
#    inputs = tokenizer(text, return_tensors="pt", padding=True)
#    with torch.no_grad():
#        translated = model.generate(**inputs)
#    output = tokenizer.decode(translated[0], skip_special_tokens=True)
#    return {"translation": output}

