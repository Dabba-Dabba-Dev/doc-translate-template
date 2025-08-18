# Run the Streamlit frontend
```bash
streamlit run front.py
```
# OCR & Translation API

This project provides an Optical Character Recognition (OCR) and Translation service using **Tesseract** through RESTful APIs. It is containerized using **Docker** and can be easily launched via **Docker Compose**.

---

## 🚀 Getting Started

To run the services, make sure you have **Docker** and **Docker Compose** installed on your machine.

### Run All Services
```bash
docker-compose up --build
```

### Run Only the OCR Service
```bash
docker-compose up ocr --build
```

### Run Only the Translation Service
```bash
docker-compose up translate --build
```

# OCR and Translation API

## 📝 Usage
All OCR and translation functionalities are exposed via POST requests. To use the OCR service, you must upload an image file and specify the language code for Tesseract.

### OCR Request Example
**Endpoint:**  
`POST /ocr`

**Form Data (multipart/form-data):**
- `file`: the image file to be processed
- `lang`: the 3-letter language code (see supported languages below)

## 🌐 Supported Languages
Below is the list of supported European, Arabic, and English languages along with their corresponding mbart language codes:

| Language              | MBART Language Code |
| --------------------- | ------------------- |
| English               | `en_XX`             |
| French                | `fr_XX`             |
| German                | `de_DE`             |
| Spanish               | `es_XX`             |
| Italian               | `it_IT`             |
| Portuguese (European) | `por_XX`            |
| Dutch                 | `nl_XX`             |
| Polish                | `pl_XX`             |
| Romanian              | `ro_XX`             |
| Russian               | `ru_XX`             |
| Ukrainian             | `uk_XX`             |
| Bulgarian             | `bg_XX`             |
| Czech                 | `cs_XX`             |
| Danish                | `da_XX`             |
| Finnish               | `fi_XX`             |
| Greek                 | `el_XX`             |
| Hungarian             | `hu_XX`             |
| Latvian               | `lv_XX`             |
| Lithuanian            | `lt_XX`             |
| Norwegian             | `no_XX`             |
| Slovak                | `sk_XX`             |
| Slovenian             | `sl_XX`             |
| Swedish               | `sv_XX`             |
| Croatian              | `hr_XX`             |
| Serbian (Latin)       | `sr_XX`             |
| Arabic                | `ar_AR`             |