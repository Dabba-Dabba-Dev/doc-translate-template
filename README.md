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
Below is the list of supported European, Arabic, and English languages along with their corresponding Tesseract language codes:

| Language              | Code      |
|-----------------------|-----------|
| Arabic                | ara       |
| English               | eng       |
| English (Middle)      | enm       |
| Albanian              | sqi       |
| Basque                | eus       |
| Belarusian            | bel       |
| Bosnian               | bos       |
| Bulgarian             | bul       |
| Catalan               | cat       |
| Croatian              | hrv       |
| Czech                 | ces       |
| Danish                | dan       |
| Dutch                 | nld       |
| Esperanto             | epo       |
| Estonian              | est       |
| Finnish               | fin       |
| French                | fra       |
| Frankish              | frk       |
| French (Middle)       | frm       |
| Galician              | glg       |
| German                | deu       |
| Greek                 | ell       |
| Greek (Ancient)       | grc       |
| Hungarian             | hun       |
| Icelandic             | isl       |
| Irish                 | gle       |
| Italian               | ita       |
| Italian (Old)         | ita_old   |
| Latin                 | lat       |
| Latvian               | lav       |
| Lithuanian            | lit       |
| Luxembourgish         | ltz       |
| Macedonian            | mkd       |
| Maltese               | mlt       |
| Norwegian             | nor       |
| Occitan               | oci       |
| Polish                | pol       |
| Portuguese            | por       |
| Romanian              | ron       |
| Russian               | rus       |
| Serbian               | srp       |
| Serbian (Latin)       | srp_latn  |
| Slovak                | slk       |
| Slovenian             | slv       |
| Spanish               | spa       |
| Spanish (Old)         | spa_old   |
| Swedish               | swe       |
| Ukrainian             | ukr       |
| Welsh                 | cym       |

## 📦 Tech Stack
- Python + Flask
- Tesseract OCR
- Docker / Docker Compose