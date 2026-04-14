# DocDigitizer

Upload a scanned document, get the text extracted, and ask questions about it in English, Hindi, or Tamil.

---

## Why I built this

I wanted to build something that goes beyond a basic ML project. Most document tools either just show you OCR text or require expensive cloud APIs for everything. I wanted to combine local OCR with an LLM chat layer so anyone can upload a document — an invoice, a lab report, a handwritten prescription — and actually ask questions about it in their language.

PaddleOCR has reasonable support for Indic scripts which made it a natural fit for the Hindi and Tamil use case. Adding Groq as the LLM backend kept costs at zero for development since the free tier handles demo-scale traffic fine. PostgreSQL stores everything persistently so document history survives server restarts, and ChromaDB handles semantic search for longer documents where you need to find the right section before sending it to the LLM.

---

## Features

- OCR extraction from scanned images and PDFs
- Supports English, Hindi, and Tamil documents
- Handles invoices, lab reports, prescriptions, and handwritten notes
- Image preprocessing pipeline — denoising, deskewing, contrast enhancement via CLAHE — runs before OCR
- LLM-powered chat to ask questions about any uploaded document
- Multilingual chat responses using Groq LLaMA 3.3 70B via LangChain
- Document history stored in PostgreSQL with extracted field metadata
- Semantic search using ChromaDB so questions on long documents retrieve the right section
- FastAPI backend with REST endpoints and automatic Swagger docs at `/docs`
- Streamlit frontend

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| OCR | PaddleOCR + Tesseract | PaddleOCR handles Indic scripts better than Tesseract alone. Tesseract runs as a fallback when PaddleOCR returns nothing. |
| LLM | Groq LLaMA 3.3 70B via LangChain | Fast inference, free tier is enough for demo-scale usage |
| Vector DB | ChromaDB | Lightweight, runs locally, no separate server to manage |
| Database | PostgreSQL | Stores document metadata, extracted fields, and chat history persistently |
| Backend | FastAPI | Async support, automatic API docs, connection pooling via SQLAlchemy |
| Frontend | Streamlit | Fastest way to get a working UI for a document ML demo |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Small model, runs on CPU, works well enough for document chunk retrieval |

---

## Architecture

```mermaid
flowchart TD
    A[User uploads document\nPDF or Image] --> B[FastAPI\nPOST /extract]
    B --> C[OpenCV\nPreprocessing\nDenoise, deskew,\ncontrast fix]
    C --> D[PaddleOCR\nEN/HI/TA]
    D --> E{OCR returned\ntext?}
    E -->|Yes| G[extract_fields\nClean and join text]
    E -->|No| F[Tesseract\nFallback]
    F --> G
    G --> H[detect_document_type\nKeyword scoring]
    H --> I[Save to PostgreSQL\nDocument + fields]
    G --> J[ChromaDB\nEmbed text chunks]
    I --> K[Return to Frontend\nExtracted text + type]
    K --> L[Streamlit UI\nShow document + text]
    L --> M[User asks question\nin chat]
    M --> N[FastAPI\nPOST /chat]
    N --> O[ChromaDB\nFind relevant chunks]
    O --> P[LangChain\nBuild prompt with context]
    P --> Q[Groq LLaMA 3.3 70B\nGenerate answer]
    Q --> R[Save to ChatHistory\nPostgreSQL]
    R --> S[Show answer\nin chat UI]
```

---

## Project Structure

```
docdigitizer/
├── .env                        # Your local environment variables (not committed)
├── .env.example                # Copy this to .env and fill in your values
├── requirements.txt
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 097e01fa5acc_initial_tables.py
├── backend/
│   ├── main.py                 # FastAPI app, routes, middleware
│   ├── config.py               # Pydantic settings, loads from .env
│   ├── database.py             # SQLAlchemy engine, session, connection pool
│   ├── models.py               # ORM models: Document, ExtractedField, ChatHistory
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── ocr_engine.py           # PaddleOCR + Tesseract, image preprocessing
│   ├── extractor.py            # Field extraction logic per document type
│   ├── chat_engine.py          # LangChain + Groq, RAG prompt building
│   └── vector_store.py         # ChromaDB singleton, chunking, semantic search
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   ├── test_ocr.py
│   ├── test_extractor.py
│   └── test_pipeline.py
├── chroma_db/                  # Persistent ChromaDB files (gitignored)
└── uploads/                    # Temp storage for uploaded files
```

---

## Setup

### Prerequisites

Make sure these are installed before you start:

**Python 3.10+** and **PostgreSQL** (running locally).

**Groq API key** — free account at [console.groq.com](https://console.groq.com), no credit card needed.

**Tesseract OCR**
- Windows: download the installer from [UB-Mannheim GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- After installing, add `C:\Program Files\Tesseract-OCR` to your system PATH
- The backend looks for Tesseract at this exact path. If you installed it somewhere else, update the path in `backend/ocr_engine.py`

**Poppler** (required for PDF support)
- Windows: download from [oschwartz10612 GitHub releases](https://github.com/oschwartz10612/poppler-windows/releases), extract it, and add the `bin/` folder to your PATH
- Ubuntu: `sudo apt install poppler-utils`
- macOS: `brew install poppler`

---

### Step by Step

**1. Clone the repo**

```bash
git clone <repo-url>
cd docdigitizer
```

**2. Create a virtual environment** (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the spaCy model**

```bash
python -m spacy download en_core_web_sm
```

**5. Set up your `.env` file**

```bash
cp .env.example .env
```

Open `.env` and fill in your values — see the [Environment Variables](#environment-variables) section below.

**6. Create the PostgreSQL database**

Connect to PostgreSQL and run:

```sql
CREATE USER docuser WITH PASSWORD 'docpass123';
CREATE DATABASE docdigitizer OWNER docuser;
GRANT ALL PRIVILEGES ON DATABASE docdigitizer TO docuser;
```

If you use different credentials, update `DATABASE_URL` in your `.env` to match.

**7. Run database migrations**

```bash
alembic upgrade head
```

This creates the `documents`, `extracted_fields`, and `chat_history` tables.

**8. Start the backend**

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs) — handy for testing endpoints without needing Postman.

**9. Start the frontend** (open a second terminal)

```bash
streamlit run frontend/app.py
```

**10. Open the app**

Go to [http://localhost:8501](http://localhost:8501)

---

## Environment Variables

```env
# Your Groq API key — get it from console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# PostgreSQL connection string — update credentials if yours differ
DATABASE_URL=postgresql://docuser:docpass123@localhost:5432/docdigitizer

# Where ChromaDB stores its files — the default works fine, just don't delete this folder
CHROMA_PERSIST_DIR=./chroma_db

# Used internally for token signing — any random string works for local dev
APP_SECRET_KEY=your_secret_key_here

# Set to False before deploying anywhere
DEBUG=True
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check if the server is running |
| `POST` | `/extract` | Upload a document, run OCR and field extraction |
| `POST` | `/chat` | Send a message about an uploaded document |
| `GET` | `/document/{id}` | Get document details and extracted fields |
| `GET` | `/history/{id}` | Get the full chat history for a document |
| `DELETE` | `/document/{id}` | Delete a document and all associated data |

**Extract** — `multipart/form-data`:
- `file` — the document (image or PDF)
- `language` — `en`, `hi`, or `ta`

**Chat** — JSON body:
```json
{
  "doc_id": "uuid-of-the-document",
  "message": "What is the total amount on this invoice?",
  "language": "en"
}
```

---

## Supported Document Types

**Invoice / Receipt**

Extracts invoice number, GSTIN, date, vendor name, total amount, SGST, and CGST. Works well on printed invoices with clear contrast. The chat can answer questions like "what is the GSTIN?", "what is the total amount?", "when was this issued?".

**Lab Report**

Extracts lab name, patient name, report date, and test results including values, units, and normal ranges. The system also flags values as HIGH, LOW, or NORMAL based on the reference range in the report. Works best on standard printed lab formats.

**Prescription**

Extracts patient name, doctor name, date, medicines with dosages, and diagnosis if present. Printed prescriptions work reliably. Handwritten prescriptions vary — if the handwriting is clear and spaced, it usually works. Cursive is unreliable.

**Handwritten Notes**

The hardest case. The system runs OCR and returns whatever text it can extract, along with detected language (based on Unicode ranges), word count, and line count. Quality depends entirely on handwriting clarity. The LLM chat works on whatever text was extracted — if OCR missed something, the LLM won't know about it.

---

## Known Limitations

- **Handwritten accuracy depends heavily on the handwriting.** Clear, spaced printing works reasonably well. Cursive or compressed writing often returns partial or garbled text.
- **Low contrast and blurry scans extract poorly.** Preprocessing helps but it has limits — a bad scan is a bad scan.
- **Tamil and Hindi handwritten accuracy is lower than English.** PaddleOCR's Indic models are primarily trained on printed text.
- **Groq free tier has rate limits.** Fine for demos, but uploading many documents in quick succession may cause the chat endpoint to return errors.
- **If you delete the `chroma_db/` folder, old document embeddings are gone.** The text and fields in PostgreSQL remain, but semantic search won't work on those documents until they're re-indexed.
- **Tesseract path is hardcoded for Windows** (`C:\Program Files\Tesseract-OCR\tesseract.exe`). Linux and macOS users need to update this in `backend/ocr_engine.py`.

---

## Dataset Credits

These publicly available datasets were used for testing and validating the extraction pipeline. No real patient or personal data is included in the repository.

- [SROIE 2019](https://rrc.cvc.uab.es/?ch=13) — scanned receipt dataset, used for invoice extraction testing
- Medical Lab Report Dataset (Kaggle) — used for lab report field extraction testing
- Handwritten Medical Prescriptions Collection (Kaggle) — used for prescription OCR testing
- Scanned Images Dataset for OCR (Kaggle) — used for general OCR quality benchmarking

> **Note:** Do not upload real Aadhaar cards, government IDs, or sensitive personal documents. This is a demo application. Medical information extracted by OCR may contain errors — do not use it for clinical decisions.

---

## What I Learned

- PaddleOCR handles Indic scripts much better than I expected for printed text, but PP-OCRv5 changed its output format compared to v4. My parser was reading the old format and silently returning empty strings — took a while to track that down.
- ChromaDB is surprisingly straightforward. The persistent client just works: point it at a folder and it handles everything. I was expecting the setup to be more involved.
- FastAPI's automatic docs at `/docs` saved a lot of debugging time when the Streamlit frontend wasn't connecting correctly. Being able to test each endpoint directly without any extra tooling was useful.
- Tesseract needs an explicit executable path on Windows. I assumed `PATH` would be enough after installation, but `pytesseract` doesn't always pick it up automatically — that cost more time than it should have.
- PostgreSQL connection pooling matters even for small projects. Once I added async endpoints, I started hitting connection errors under light load. Setting `pool_size=5, max_overflow=10` in the SQLAlchemy engine config fixed it.

---

## License

MIT

---

## Author

Built by **Prabha**
B.Tech — Artificial Intelligence and Data Science
Sri Venkateswara College of Engineering, Graduating 2026

[LinkedIn](https://www.linkedin.com/in/your-linkedin) · [GitHub](https://github.com/your-github)
