FROM python:3.9

WORKDIR /app

COPY translate/requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Install Flask before copying files
RUN pip install flask


COPY translate/load_model.py .
COPY translate/app.py .


CMD ["python", "app.py"]