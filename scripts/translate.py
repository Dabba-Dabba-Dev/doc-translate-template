from transformers import MarianMTModel, MarianTokenizer

# Load a pre-trained MarianMT model (English to French)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate sample text
text = "Hello, my name is Eya. I am working on a translation project."
inputs = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print("Translated text:", output)
