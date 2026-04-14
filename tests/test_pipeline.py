"""
Standalone pipeline test — run with:
    python tests/test_pipeline.py

Tests every layer of DocDigitizer without needing FastAPI running.
"""

import json
import sys
import uuid
from pathlib import Path

# Make sure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------

_results: dict[str, tuple[str, str]] = {}   # name -> ("PASS"|"FAIL", detail)


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    _results[name] = (status, detail)
    icon = "+" if passed else "!"
    print(f"  [{icon}] {name}: {status}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. DB Connection
# ---------------------------------------------------------------------------

def test_db_connection():
    print("\n-- 1. DB Connection ------------------------------------------")
    try:
        from backend.database import test_db_connection as _test
        _test()
        record("DB Connection", True)
    except Exception as e:
        record("DB Connection", False, str(e))


# ---------------------------------------------------------------------------
# 2. OCR Engine
# ---------------------------------------------------------------------------

def test_ocr_engine():
    print("\n-- 2. OCR Engine ---------------------------------------------")
    try:
        from backend.ocr_engine import extract_text

        # Find first usable image in test_data/
        search_dirs = [
            PROJECT_ROOT / "test_data" / "lab_reports",
            PROJECT_ROOT / "test_data" / "prescriptions",
            PROJECT_ROOT / "test_data" / "invoices",
            PROJECT_ROOT / "test_data" / "handwritten",
        ]
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".pdf"}
        image_path = None
        for d in search_dirs:
            hits = [
                p for p in d.iterdir()
                if p.suffix.lower() in IMAGE_EXTS and p.name != ".gitkeep"
            ]
            hits.sort()
            if hits:
                image_path = hits[0]
                break

        if image_path is None:
            record("OCR Engine", False, "No test images found in test_data/ — add at least one image to run this test.")
            return None

        print(f"  Using: {image_path.relative_to(PROJECT_ROOT)}")
        result = extract_text(str(image_path))
        raw_text = result.get("raw_text", "")
        print(f"  Pages   : {result.get('page_count', 1)}")
        print(f"  Blocks  : {len(result.get('blocks', []))}")
        print(f"  Raw text (first 200 chars):\n    {raw_text[:200]!r}")
        record("OCR Engine", True, f"{len(raw_text)} chars extracted")
        return raw_text

    except Exception as e:
        record("OCR Engine", False, str(e))
        return None


# ---------------------------------------------------------------------------
# 3. Document Type Detection
# ---------------------------------------------------------------------------

def test_doc_type_detection(raw_text: str | None):
    print("\n-- 3. Document Type Detection --------------------------------")
    try:
        from backend.ocr_engine import detect_document_type

        # Fall back to a synthetic lab-report text if OCR produced nothing
        text = raw_text if raw_text and raw_text.strip() else (
            "CBC Report  WBC: 7.2  Platelet: 180  Normal range  Haemoglobin 13.5 g/dL"
        )
        if not (raw_text and raw_text.strip()):
            print("  (No OCR text available — using synthetic lab text for detection test)")

        doc_type = detect_document_type(text)
        print(f"  Detected type: {doc_type}")
        record("Doc Type Detect", True, doc_type)
        return doc_type, text

    except Exception as e:
        record("Doc Type Detect", False, str(e))
        return "lab_report", raw_text or ""


# ---------------------------------------------------------------------------
# 4. Field Extractor
# ---------------------------------------------------------------------------

def test_field_extractor(raw_text: str, doc_type: str):
    print("\n-- 4. Field Extractor ----------------------------------------")
    try:
        from backend.extractor import extract_fields

        fields = extract_fields(raw_text, doc_type)
        print(f"  Extracted fields ({doc_type}):")
        print("  " + json.dumps(fields, indent=4, ensure_ascii=False).replace("\n", "\n  "))
        confidence = fields.get("confidence_score", 0.0)
        record("Field Extractor", True, f"confidence={confidence:.0%}")
        return fields

    except Exception as e:
        record("Field Extractor", False, str(e))
        return {}


# ---------------------------------------------------------------------------
# 5. ChromaDB
# ---------------------------------------------------------------------------

def test_chromadb():
    print("\n-- 5. ChromaDB -----------------------------------------------")
    dummy_id = f"pipeline-test-{uuid.uuid4()}"
    sample_text = (
        "Patient: John Doe. CBC Report. WBC: 7200. Haemoglobin: 13.5 g/dL. "
        "Platelet: 180 thou/uL. Result within normal range. "
        "Date: 01/04/2024. Lab: City Diagnostics."
    )
    try:
        from backend.vector_store import get_vector_store

        vs = get_vector_store()

        # Add
        ok = vs.add_document(dummy_id, sample_text, {"doc_type": "lab_report", "language": "en"})
        if not ok:
            raise RuntimeError("add_document returned False")
        print(f"  add_document : OK  (id={dummy_id})")

        # Search
        results = vs.search("haemoglobin normal range", dummy_id, n_results=2)
        print(f"  search       : {len(results)} chunk(s) returned")
        for i, r in enumerate(results):
            print(f"    chunk {i}: {r[:80]!r}...")

        if not results:
            raise RuntimeError("search returned 0 results — indexing may have failed")

        # Delete
        ok = vs.delete_document(dummy_id)
        if not ok:
            raise RuntimeError("delete_document returned False")
        print(f"  delete       : OK")

        # Confirm deletion
        after = vs.search("haemoglobin", dummy_id, n_results=1)
        if after:
            raise RuntimeError(f"Chunks still present after delete: {after}")
        print(f"  post-delete search: 0 results — confirmed clean")

        record("ChromaDB", True, "add / search / delete all OK")

    except Exception as e:
        record("ChromaDB", False, str(e))
        # Best-effort cleanup
        try:
            from backend.vector_store import get_vector_store
            get_vector_store().delete_document(dummy_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 6. Groq API
# ---------------------------------------------------------------------------

def test_groq_api():
    print("\n-- 6. Groq API -----------------------------------------------")
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from backend.config import get_settings

        settings = get_settings()
        if not settings.groq_api_key or settings.groq_api_key.startswith("your_"):
            record("Groq API", False, "GROQ_API_KEY not set in .env — skipped")
            return

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=settings.groq_api_key,
        )
        response = llm.invoke([HumanMessage(content="Say hello in exactly one sentence.")])
        reply = response.content.strip()
        print(f"  Model response: {reply!r}")
        record("Groq API", True, f"{len(reply)} chars received")

    except Exception as e:
        record("Groq API", False, str(e))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary():
    print("\n" + "=" * 52)
    print(f"  {'TEST':<22}  {'STATUS':<6}  DETAIL")
    print("-" * 52)
    all_pass = True
    for name, (status, detail) in _results.items():
        icon = "[+]" if status == "PASS" else "[!]"
        detail_str = (detail[:28] + "...") if len(detail) > 29 else detail
        print(f"  {icon} {name:<20}  {status:<6}  {detail_str}")
        if status == "FAIL":
            all_pass = False
    print("=" * 52)
    total = len(_results)
    passed = sum(1 for s, _ in _results.values() if s == "PASS")
    print(f"  Result: {passed}/{total} tests passed")
    print("=" * 52)
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 52)
    print("    DocDigitizer -- Pipeline Test Suite")
    print("=" * 52)

    # Suppress noisy library logs during tests
    import logging
    logging.disable(logging.WARNING)

    test_db_connection()
    raw_text        = test_ocr_engine()
    doc_type, text  = test_doc_type_detection(raw_text)
    test_field_extractor(text, doc_type)
    test_chromadb()
    test_groq_api()

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)
