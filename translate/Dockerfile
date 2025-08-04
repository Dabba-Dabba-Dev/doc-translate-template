FROM python:3.9

WORKDIR /app

COPY translate/requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask

# Download the model during build (this prevents downloading at runtime)
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); \
    AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')"

COPY translate/load_model.py .
COPY ../ocr/text_extracted.txt /app/text_extracted.txt
COPY translate/app.py .

CMD ["python", "app.py"]
