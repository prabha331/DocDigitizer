import io

import requests
import streamlit as st

FASTAPI_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DocDigitizer", layout="wide", page_icon="📄")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }

    [data-testid="stFileUploader"] {
        border: 2px dashed #4f8ef7;
        border-radius: 12px;
        padding: 16px;
        background: #f0f7ff;
    }

    .badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 10px;
    }
    .badge-invoice           { background: #dbeafe; color: #1d4ed8; }
    .badge-prescription      { background: #dcfce7; color: #15803d; }
    .badge-lab_report        { background: #fef9c3; color: #a16207; }
    .badge-handwritten_notes { background: #ede9fe; color: #6d28d9; }

    /* Sidebar doc buttons — normal */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 0.83rem;
    }
    /* Sidebar doc buttons — active/selected */
    [data-testid="stSidebar"] .stButton > button[data-selected="true"],
    [data-testid="stSidebar"] .stButton > button:focus {
        border: 2px solid #4f8ef7 !important;
        background: #e8f0fe !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in [
    ("doc_data", None),
    ("messages", []),
    ("uploaded_bytes", None),
    ("uploaded_name", None),
    ("doc_history", []),          # list of full doc dicts including file_bytes
    ("selected_doc_id", None),    # currently displayed doc_id
    ("insights", {}),             # doc_id -> insight string cache
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LANG_MAP = {"English (en)": "en", "Hindi (hi)": "hi", "Tamil (ta)": "ta"}

_CHIPS = {
    "lab_report": [
        "Summarise all test results",
        "Which values are abnormal?",
        "Explain HB level",
        "What is WBC count?",
        "Explain in simple terms",
    ],
    "prescription": [
        "What medicines are prescribed?",
        "Explain dosage instructions",
        "What is 1-0-1?",
        "When to take medicines?",
        "What is diagnosis?",
    ],
    "invoice": [
        "What is total amount?",
        "Who is vendor?",
        "What taxes charged?",
        "Summarise this bill",
        "What items were purchased?",
    ],
    "handwritten_notes": [
        "Summarise these notes",
        "List key points",
        "Translate to English",
        "What is the main topic?",
        "Structure these notes",
    ],
}

_INSIGHT_PROMPTS = {
    "invoice": (
        "List the key details: vendor name, date, total amount, and items purchased "
        "in bullet points. Be concise."
    ),
    "lab_report": (
        "List all test names and their values. Flag any values that are outside the "
        "normal range with HIGH or LOW label. Be concise and use bullet points."
    ),
    "prescription": (
        "List the medicines prescribed, their dosage, and timing instructions "
        "in simple language. Use bullet points."
    ),
    "handwritten_notes": (
        "Summarize the main points from these notes in bullet points."
    ),
}

_BADGE_LABELS = {
    "invoice": "🧾 Invoice",
    "prescription": "💊 Prescription",
    "lab_report": "🔬 Lab Report",
    "handwritten_notes": "✏️ Handwritten Notes",
}

_DOC_TYPE_COLORS = {
    "invoice": "blue",
    "prescription": "green",
    "lab_report": "orange",
    "handwritten_notes": "violet",
}


def _badge_html(doc_type: str) -> str:
    label = _BADGE_LABELS.get(doc_type, doc_type.replace("_", " ").title())
    return f'<span class="badge badge-{doc_type}">{label}</span>'


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call_extract(file_bytes: bytes, filename: str, language: str) -> dict | None:
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/extract",
            files={"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            data={"language": language},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend error: {e.response.status_code} — {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Unexpected error during extraction: {e}")
    return None


def _call_chat(doc_id: str, message: str, language: str) -> str | None:
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/chat",
            json={"doc_id": doc_id, "message": message, "language": language},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Chat error: {e.response.status_code} — {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Unexpected error during chat: {e}")
    return None


def _fetch_history(doc_id: str) -> list[dict]:
    """Fetch chat history for a doc from the backend."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/history/{doc_id}", timeout=15)
        resp.raise_for_status()
        return [{"role": r["role"], "content": r["message"]} for r in resp.json()]
    except Exception:
        return []


def _send_message(prompt: str, doc_id: str, language: str):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        reply = _call_chat(doc_id, prompt, language)
    if reply:
        st.session_state["messages"].append({"role": "assistant", "content": reply})


def _get_or_generate_insight(doc_id: str, doc_type: str, language: str) -> str:
    """Return cached insight or generate it via the chat endpoint."""
    if doc_id in st.session_state["insights"]:
        return st.session_state["insights"][doc_id]
    prompt = _INSIGHT_PROMPTS.get(doc_type, _INSIGHT_PROMPTS["handwritten_notes"])
    insight = _call_chat(doc_id, prompt, language) or ""
    st.session_state["insights"][doc_id] = insight
    return insight


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("DocDigitizer")
    st.divider()

    lang_label = st.selectbox(
        "Language",
        options=list(_LANG_MAP.keys()),
        index=0,
    )
    language = _LANG_MAP[lang_label]

    st.divider()
    st.markdown("### Previous Documents")

    doc_history = st.session_state["doc_history"]
    selected_doc_id = st.session_state.get("selected_doc_id")

    if not doc_history:
        st.caption("No documents uploaded yet.")
    else:
        for entry in doc_history[-5:][::-1]:
            color = _DOC_TYPE_COLORS.get(entry.get("doc_type", ""), "gray")
            badge = _BADGE_LABELS.get(entry["doc_type"], entry["doc_type"])
            is_selected = entry["doc_id"] == selected_doc_id
            # Visually mark the active doc with a checkmark prefix
            prefix = "✓ " if is_selected else ""
            label = f"{prefix}:{color}[{badge}]  {entry['filename'][:24]}"

            if st.button(label, key=f"hist_{entry['doc_id']}"):
                if entry["doc_id"] != selected_doc_id:
                    # Load this document's data back into view
                    st.session_state["doc_data"] = {
                        "doc_id":        entry["doc_id"],
                        "filename":      entry["filename"],
                        "doc_type":      entry["doc_type"],
                        "extracted_json": entry["extracted_json"],
                        "raw_text":      entry["raw_text"],
                    }
                    st.session_state["uploaded_bytes"] = entry.get("file_bytes")
                    st.session_state["uploaded_name"]  = entry["filename"]
                    st.session_state["selected_doc_id"] = entry["doc_id"]
                    # Restore chat history from backend
                    st.session_state["messages"] = _fetch_history(entry["doc_id"])
                    st.rerun()

# ---------------------------------------------------------------------------
# SECTION 1 — Header
# ---------------------------------------------------------------------------

st.markdown("<h1 style='text-align:center'>📄 DocDigitizer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray'>Upload documents and chat with them instantly</p>",
    unsafe_allow_html=True,
)

st.warning("⚠️ Demo app — do not upload real government IDs or sensitive personal documents.")

# ---------------------------------------------------------------------------
# SECTION 2 — Upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a document (PDF, PNG, JPG)",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name != st.session_state.get("uploaded_name"):
        st.session_state["uploaded_bytes"] = file_bytes
        st.session_state["uploaded_name"] = uploaded_file.name
        st.session_state["doc_data"] = None
        st.session_state["messages"] = []

        with st.spinner("Extracting text and analysing document..."):
            result = _call_extract(file_bytes, uploaded_file.name, language)

        if result:
            st.session_state["doc_data"] = result
            st.session_state["selected_doc_id"] = result["doc_id"]
            st.success("Document processed successfully!")

            # Store full entry including file_bytes in history (avoid duplicates)
            existing_ids = [d["doc_id"] for d in st.session_state["doc_history"]]
            if result["doc_id"] not in existing_ids:
                st.session_state["doc_history"].append({
                    "doc_id":        result["doc_id"],
                    "filename":      uploaded_file.name,
                    "doc_type":      result.get("doc_type", "handwritten_notes"),
                    "extracted_json": result.get("extracted_json", {}),
                    "raw_text":      result.get("raw_text", ""),
                    "file_bytes":    file_bytes,
                })

# ---------------------------------------------------------------------------
# SECTION 3 — Results
# ---------------------------------------------------------------------------

doc_data = st.session_state.get("doc_data")

if doc_data:
    doc_id    = doc_data["doc_id"]
    doc_type  = doc_data.get("doc_type", "handwritten_notes")
    ext_json  = doc_data.get("extracted_json", {})
    full_text = ext_json.get("full_text", doc_data.get("raw_text", ""))

    st.divider()

    col_left, col_right = st.columns([0.45, 0.55])

    # --- Left column: original image ----------------------------------------
    with col_left:
        st.markdown("**Original Document**")
        img_bytes = st.session_state.get("uploaded_bytes")
        fname = st.session_state.get("uploaded_name", "")
        if img_bytes:
            if fname.lower().endswith(".pdf"):
                st.info("PDF preview not available. Upload a PNG/JPG to see the image.")
            else:
                st.image(img_bytes, use_container_width=True)
        else:
            st.caption("Image not available.")

    # --- Right column: badge + insights + extracted text --------------------
    with col_right:
        st.markdown(_badge_html(doc_type), unsafe_allow_html=True)

        # Key Insights — auto-generated on first view, cached after
        with st.spinner("Generating key insights..."):
            insight_text = _get_or_generate_insight(doc_id, doc_type, language)

        if insight_text:
            safe_insight = insight_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="background:#1a3a1a; border-left:4px solid #4CAF50;
                            padding:15px; border-radius:8px; margin-bottom:15px;">
                <b style="color:#4CAF50">&#10024; Key Insights</b><br><br>
                <span style="color:#e0e0e0; font-size:14px; line-height:1.8;">{safe_insight}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Extracted Text**")

        safe_text = (full_text or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        display_text = safe_text if safe_text.strip() else "No text could be extracted from this document."

        st.markdown(
            f"""
            <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                        font-size:13px; line-height:1.8; max-height:350px;
                        overflow-y:auto; color:#e0e0e0; border:1px solid #444;
                        font-family:monospace;">
            {display_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        word_count = len(full_text.split()) if full_text else 0
        st.caption(f"📄 {word_count} words extracted")

    # -----------------------------------------------------------------------
    # SECTION 4 — Chat
    # -----------------------------------------------------------------------

    st.divider()
    st.markdown("### 💬 Chat with your document")

    def _render_chat_panel():
        chips = _CHIPS.get(doc_type, _CHIPS["handwritten_notes"])
        chip_cols = st.columns(len(chips))
        for col, chip in zip(chip_cols, chips):
            if col.button(chip, key=f"chip_{chip}", use_container_width=True):
                _send_message(chip, doc_id, language)
                st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask anything about this document...")
        if user_input:
            _send_message(user_input, doc_id, language)
            st.rerun()

    if doc_type == "invoice":
        with st.expander("💬 Ask about this document", expanded=False):
            _render_chat_panel()
    else:
        _render_chat_panel()
