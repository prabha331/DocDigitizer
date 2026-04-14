import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend import models
from backend.chat_engine import get_chat_engine
from backend.database import Base, engine, get_db, test_db_connection
from backend.ocr_engine import detect_document_type, extract_text, save_uploaded_file
from backend.extractor import extract_fields
from backend.schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    ExtractResponse,
)
from backend.vector_store import get_vector_store

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "pdf"}
UPLOAD_DIR = "./uploads"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("DocDigitizer starting up...")
    test_db_connection()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")
    get_vector_store()          # warm up ChromaDB + embedding model
    logger.info("DocDigitizer started.")
    yield
    # --- Shutdown ---
    logger.info("DocDigitizer shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocDigitizer API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_document_or_404(doc_id: str, db: Session) -> models.Document:
    try:
        uid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid doc_id format.")
    doc = db.query(models.Document).filter(models.Document.id == uid).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    return doc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "database": "connected", "version": "1.0.0"}


# -- POST /extract -----------------------------------------------------------

@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    language: str = Form("en"),
    db: Session = Depends(get_db),
):
    # Validate extension
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Persist upload
    file_path = await save_uploaded_file(file, UPLOAD_DIR)

    # OCR
    ocr_result = extract_text(file_path, language)
    raw_text = ocr_result["raw_text"]

    # Classify + extract structured fields
    doc_type = detect_document_type(raw_text)
    extracted_json = extract_fields(raw_text, doc_type)

    # Persist Document row
    doc_id = uuid.uuid4()
    doc = models.Document(
        id=doc_id,
        filename=file.filename,
        file_type=doc_type,
        language=language,
        raw_text=raw_text,
        extracted_json=extracted_json,
        status="completed",
    )
    db.add(doc)
    db.flush()  # get doc.id before adding children

    # Persist extracted fields as individual rows
    confidence = extracted_json.pop("confidence_score", 0.0)
    for field_name, field_value in extracted_json.items():
        if field_value is None:
            continue
        db.add(
            models.ExtractedField(
                doc_id=doc_id,
                field_name=field_name,
                field_value=(
                    json.dumps(field_value, ensure_ascii=False)
                    if not isinstance(field_value, str)
                    else field_value
                ),
                confidence_score=confidence,
            )
        )

    # Restore confidence_score so it's in extracted_json for the response
    extracted_json["confidence_score"] = confidence

    db.commit()
    db.refresh(doc)
    logger.info("Extracted and stored document doc_id=%s type=%s", doc_id, doc_type)

    # Index in vector store (non-blocking — failure doesn't break response)
    try:
        get_vector_store().add_document(
            str(doc_id),
            raw_text,
            {"doc_type": doc_type, "language": language},
        )
    except Exception as e:
        logger.error("Vector store indexing failed for doc_id=%s: %s", doc_id, e)

    return ExtractResponse(
        doc_id=doc.id,
        filename=doc.filename,
        doc_type=doc.file_type,
        language=doc.language,
        extracted_json=doc.extracted_json,
        raw_text=doc.raw_text,
        status=doc.status,
        all_lines=doc.extracted_json.get("all_lines", []),
    )


# -- POST /chat --------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    doc = _get_document_or_404(str(body.doc_id), db)

    # Build history list for LLM context
    history_rows = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.doc_id == doc.id)
        .order_by(models.ChatHistory.created_at)
        .all()
    )
    chat_history = [{"role": row.role, "message": row.message} for row in history_rows]

    # Call LLM
    reply = get_chat_engine().chat(
        doc_id=str(doc.id),
        user_message=body.message,
        doc_type=doc.file_type,
        extracted_json=doc.extracted_json or {},
        language=body.language,
        chat_history=chat_history,
    )

    # Persist both turns
    db.add(models.ChatHistory(doc_id=doc.id, role="user", message=body.message))
    db.add(models.ChatHistory(doc_id=doc.id, role="assistant", message=reply))
    db.commit()

    logger.info("Chat turn saved for doc_id=%s", doc.id)
    return ChatResponse(doc_id=doc.id, response=reply)


# -- GET /document/{doc_id} --------------------------------------------------

@app.get("/document/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = _get_document_or_404(doc_id, db)
    fields = doc.fields.all()  # lazy="dynamic" requires .all()
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        language=doc.language,
        raw_text=doc.raw_text,
        extracted_json=doc.extracted_json,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        fields=fields,
    )


# -- GET /history/{doc_id} ---------------------------------------------------

@app.get("/history/{doc_id}", response_model=list[ChatHistoryResponse])
def get_history(doc_id: str, db: Session = Depends(get_db)):
    doc = _get_document_or_404(doc_id, db)
    return (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.doc_id == doc.id)
        .order_by(models.ChatHistory.created_at)
        .all()
    )


# -- DELETE /document/{doc_id} -----------------------------------------------

@app.delete("/document/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = _get_document_or_404(doc_id, db)

    # Remove from vector store first (best-effort)
    try:
        get_vector_store().delete_document(str(doc.id))
    except Exception as e:
        logger.error("Vector store deletion failed for doc_id=%s: %s", doc.id, e)

    # Cascade deletes ExtractedField + ChatHistory via DB relationship
    db.delete(doc)
    db.commit()
    logger.info("Deleted document doc_id=%s", doc_id)
    return {"status": "deleted", "doc_id": doc_id}
