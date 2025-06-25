from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time
import tracemalloc

# Start measuring time and memory
start_time = time.time()
tracemalloc.start()

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

text = "This is a visa document."
tokenizer.src_lang = "en_XX"
inputs = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Stop measuring memory and time
current, peak = tracemalloc.get_traced_memory()
end_time = time.time()
tracemalloc.stop()

print(translated)
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")