#  Doc Translate Template

This project is part of the Dabba Dabba Summer Internship (Model 2) and focuses on building a multilingual document translation engine combined with a dynamic form/template learning system.

##  Overview

The goal is to translate official documents (e.g., visa forms, certificates) between languages like English, French, Arabic, German, and Italian — while learning to recognize new templates by tagging fields through a user interface.

##  Project Structure 

doc-translate-template/

├── data/               # Raw + processed documents

├── models/             # Saved MarianMT models

├── scripts/            # Translation and training scripts

│   └── translate.py

├── templates/          # Field mapping configs

├── ui/                 # Streamlit/FastAPI (in progress)

├── requirements.txt    # All required dependencies

└── README.md

##  Key Features

- Multilingual translation using MarianMT (EN, FR, AR, etc.)
- Fine-tuning on ~50 aligned document pairs
- Field-level tagging using Label Studio
- Template learning for reusable forms (e.g., visa applications)
- Optional human validation layer for certified translations
- UI (in development) to upload, tag, and visualize translation flow

##  To run this project locally, follow these steps:

```bash
git clone https://github.com/eya2105/doc-translate-template.git
cd doc-translate-template
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

##  Docker Setup

To build and run the project using Docker:

```bash
docker-compose build --no-cache
docker compose up
```

Once running, you can access:

- Label Studio UI at: [Link Text](http://localhost:8080)

- Translator API (if exposed) at: [Link Text](http://localhost:8080)

This will launch both:

- The translator service (FastAPI backend for translation)

- The labelstudio service (for tagging fields in documents)