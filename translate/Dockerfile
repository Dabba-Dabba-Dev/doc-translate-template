FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY mbart_ocr.py .
COPY mbart_model.py .
COPY ocr/image_to_text.py . 

# Wrap your script in a FastAPI endpoint
CMD ["uvicorn", "mbart_ocr:app", "--host", "0.0.0.0", "--port", "8000"]