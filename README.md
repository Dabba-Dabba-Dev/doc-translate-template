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
| English               | `en_XX`              |
| French                | `fr_XX`              |
| German                | `de_DE`              |
| Spanish               | `es_XX`              |
| Italian               | `it_IT`              |
| Portuguese (European) | `pt_XX`              |
| Dutch                 | `nl_XX`              |
| Polish                | `pl_PL`              |
| Romanian              | `ro_RO`              |
| Russian               | `ru_RU`              |
| Ukrainian             | `uk_UA`              |
| Bulgarian             | `bg_BG`              |
| Czech                 | `cs_CZ`              |
| Danish                | `da_DK`              |
| Finnish               | `fi_FI`              |
| Greek                 | `el_GR`              |
| Hungarian             | `hu_HU`              |
| Latvian               | `lv_LV`              |
| Lithuanian            | `lt_LT`              |
| Norwegian             | `nb_NO`              |
| Slovak                | `sk_SK`              |
| Slovenian             | `sl_SI`              |
| Swedish               | `sv_SE`              |
| Croatian              | `hr_HR`              |
| Serbian (Latin)       | `sr_Latn_RS`         |
| Arabic                | `ar_AR`              |