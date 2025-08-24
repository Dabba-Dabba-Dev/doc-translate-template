FROM python:3.9-slim

WORKDIR /app

COPY translate/requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask

# Pre-download MBart model & tokenizer
RUN python -c "\
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast; \
MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); \
MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')"

COPY translate/load_translation_model.py .
COPY translate/app.py .
COPY ../ocr/text_extracted.txt /app/text_extracted.txt
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]