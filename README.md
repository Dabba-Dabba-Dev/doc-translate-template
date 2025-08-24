
# OCR & Translation API

This project provides an Optical Character Recognition (OCR) and Translation service using **Tesseract** and **MBART** through RESTful APIs. It is containerized using **Docker** and can be easily launched via **Docker Compose**.

---

## 🚀 Getting Started

To run the services, make sure you have **Docker** and **Docker Compose** installed on your machine.

### Run All Services
```bash
docker-compose up --build
````

### Run Only the OCR Service

```bash
docker-compose up ocr --build
```

### Run Only the Translation Service

```bash
docker-compose up translate --build
```

### Run the Streamlit Frontend

```bash
streamlit run front.py
```

---

## 📝 API Usage

All functionalities are exposed via **REST endpoints**.

### 1️⃣ Process & Translate Document

**Endpoint:**
`POST http://localhost:5000/process`

**Request Body (JSON):**

```json
{
  "file": "<uploaded_file>",
  "src_lang": "en_XX",
  "tgt_lang": "fr_XX"
}
```

* `file`: Any supported file type (`.pdf`, `.png`, `.jpg`, etc.)
* `src_lang`: Source language code (see supported languages table below)
* `tgt_lang`: Target language code (see supported languages table below)

**Example with `curl`:**

```bash
curl -X POST "http://localhost:5000/process" \
  -F "file=@example.pdf" \
  -F "src_lang=en_XX" \
  -F "tgt_lang=fr_XX"
```

**Response:**
Returns a JSON containing extracted OCR text and its translation (see `front.py` for formatting details).

---

### 2️⃣ Download Final PDF

**Endpoint:**
`GET http://localhost:5000/download-final-pdf`

This endpoint returns the **translated document** as a downloadable PDF file.

Example with `curl`:

```bash
curl -O http://localhost:5000/download-final-pdf
```

---

## 🌐 Supported Languages

Below is the list of supported languages with their MBART codes:

| Language   | MBART Code |
| ---------- | ---------- |
| English    | `en_XX`    |
| French     | `fr_XX`    |
| German     | `de_DE`    |
| Spanish    | `es_XX`    |
| Italian    | `it_IT`    |
| Portuguese | `pt_XX`    |
| Dutch      | `nl_XX`    |
| Polish     | `pl_PL`    |
| Romanian   | `ro_RO`    |
| Russian    | `ru_RU`    |
| Ukrainian  | `uk_UK`    |
| Bulgarian  | `bg_BG`    |
| Czech      | `cs_CZ`    |
| Danish     | `da_DK`    |
| Finnish    | `fi_FI`    |
| Greek      | `el_EL`    |
| Hungarian  | `hu_HU`    |
| Latvian    | `lv_LV`    |
| Lithuanian | `lt_LT`    |
| Slovak     | `sk_SK`    |
| Slovenian  | `sl_SI`    |
| Swedish    | `sv_SE`    |
| Croatian   | `hr_HR`    |
| Serbian    | `sr_XX`    |
| Arabic     | `ar_AR`    |

```


