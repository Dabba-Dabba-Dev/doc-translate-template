FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY translate.py .
COPY bart.py .

# Wrap your script in a FastAPI endpoint
CMD ["uvicorn", "translate:app", "--host", "0.0.0.0", "--port", "8000"]