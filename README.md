# Document Translation & OCR Pipeline

This project provides a Dockerized pipeline for extracting text from documents (PDFs, images) using various OCR engines and then translating the extracted text.

## Project Structure

```
.
├── docker-compose.yml
├── ocr/
│   ├── dockerfile
│   ├── requirements.txt
│   ├── preform_ocr.py      # Uses docTR
│   └── preform_ocr2.py     # Uses easyocr
└── translate/
    ├── Dockerfile
    ├── requirements.txt
    └── ... (translation scripts)
```

## Services

### 1. OCR Service (`ocr`)

This service is responsible for extracting text from input files. It is designed to be flexible, allowing you to switch between different OCR engines by modifying the `dockerfile` and `docker-compose.yml`.

### 2. Translation Service (`translate`)

This service takes the text output from the OCR service and translates it into the desired language. (Further documentation to be added).

---

## How to Use the OCR Service

You can choose between two OCR engines: `docTR` and `easyocr`.

### Option 1: Using `docTR` (`preform_ocr.py`)

`docTR` is a powerful OCR library, generally good for complex layouts and a variety of document types.

**1. Configure `ocr/requirements.txt`:**

Ensure the file contains the following:

```txt
python-doctr[torch]
PyMuPDF
requests
```

**2. Configure `ocr/dockerfile`:**

The Dockerfile should be set up to install `docTR`'s dependencies and run the correct script.

```dockerfile
# Base Image
FROM python:3.9-slim

# System Dependencies for docTR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run the docTR script
COPY preform_ocr.py .
CMD ["python", "preform_ocr.py"]
```

**3. Configure `docker-compose.yml`:**

Set the command to execute the `preform_ocr.py` script.

```yaml
services:
  ocr:
    build:
      context: ./ocr
      dockerfile: dockerfile
    volumes:
      - ./ocr:/app
    command: python preform_ocr.py
```

**4. Run the service:**

Modify the `input_file` variable inside `preform_ocr.py` to point to your desired test file, then run:

```bash
docker-compose build ocr
docker-compose run ocr
```

The output will be saved to `enhanced_output4.txt` in the `ocr` directory.

### Option 2: Using `easyocr` (`preform_ocr2.py`)

`easyocr` is simpler to set up and can be a good starting point for standard text recognition.

**1. Configure `ocr/requirements.txt`:**

Ensure the file contains the following:

```txt
easyocr
PyMuPDF
torch
torchvision
python-magic
```

**2. Configure `ocr/dockerfile`:**

The Dockerfile needs to be adjusted for `easyocr`'s dependencies.

```dockerfile
# Base Image
FROM python:3.9-slim

# System Dependencies for easyocr
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run the easyocr script
COPY preform_ocr2.py .
CMD ["python", "preform_ocr2.py"]
```

**3. Configure `docker-compose.yml`:**

Update the service command to run `preform_ocr2.py` and pass the input file as an argument.

```yaml
services:
  ocr:
    build:
      context: ./ocr
      dockerfile: dockerfile
    volumes:
      - ./ocr:/app
    # Pass the input file as a command-line argument
    command: python preform_ocr2.py your_document.pdf
```

**4. Run the service:**

Replace `your_document.pdf` in the `docker-compose.yml` file with the path to your test file (e.g., `test_files/contrat de travail.pdf`), then run:

```bash
docker-compose build ocr
docker-compose run ocr
```

The extracted text will be printed to the console and saved to `ocr_output.txt`.
