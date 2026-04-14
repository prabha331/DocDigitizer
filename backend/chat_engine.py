import json
import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.config import get_settings
from backend.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    def __init__(self):
        settings = get_settings()

        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                groq_api_key=settings.groq_api_key,
            )
            logger.info("ChatGroq initialised with model llama3-8b-8192.")
        except Exception as e:
            logger.error("Failed to initialise ChatGroq: %s", e)
            raise

        try:
            self.vector_store = get_vector_store()
            logger.info("VectorStore attached to ChatEngine.")
        except Exception as e:
            logger.error("Failed to attach VectorStore: %s", e)
            raise

    # ------------------------------------------------------------------
    # _get_system_prompt
    # ------------------------------------------------------------------

    def _get_system_prompt(
        self,
        doc_type: str,
        extracted_json: dict,
        language: str,
        raw_text: str,
    ) -> str:
        lang_name = _LANGUAGE_NAMES.get(language, "English")
        full_text = extracted_json.get("full_text") or raw_text

        return (
            f"You are a document assistant.\n"
            f"The user uploaded a {doc_type.replace('_', ' ')} document.\n"
            f"Here is the complete text extracted from the document:\n\n"
            f"{full_text}\n\n"
            f"Answer all questions based only on this document text. "
            f"Be specific and reference actual content from the document. "
            f"Respond in {lang_name}."
        )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def chat(
        self,
        doc_id: str,
        user_message: str,
        doc_type: str,
        extracted_json: dict,
        language: str,
        chat_history: list,
    ) -> str:
        """
        Run one conversational turn.

        *chat_history* is a list of dicts: [{"role": "user"|"assistant", "message": str}, ...]
        Returns the assistant reply as a plain string.
        """
        try:
            # Retrieve relevant context chunks from the vector store
            context_chunks = self.vector_store.search(user_message, doc_id, n_results=3)
            context_block = ""
            if context_chunks:
                joined = "\n---\n".join(context_chunks)
                context_block = f"\n\nRelevant context from the document:\n{joined}"

            # Build the message list
            system_prompt = self._get_system_prompt(
                doc_type, extracted_json, language, extracted_json.get("raw_text", "")
            )
            messages = [SystemMessage(content=system_prompt)]

            # Replay prior turns
            for turn in chat_history:
                role = turn.get("role", "")
                text = turn.get("message", "")
                if role == "user":
                    messages.append(HumanMessage(content=text))
                elif role == "assistant":
                    messages.append(AIMessage(content=text))

            # Current user message, augmented with retrieved context
            augmented_message = user_message + context_block
            messages.append(HumanMessage(content=augmented_message))

            response = self.llm.invoke(messages)
            reply = response.content
            logger.info(
                "chat: doc_id=%s doc_type=%s reply_len=%d", doc_id, doc_type, len(reply)
            )
            return reply

        except Exception as e:
            logger.error("chat failed for doc_id=%s: %s", doc_id, e)
            return f"Sorry, I encountered an error while processing your question: {e}"

    # ------------------------------------------------------------------
    # structure_handwritten_notes
    # ------------------------------------------------------------------

    def structure_handwritten_notes(self, raw_text: str, language: str) -> dict:
        """
        Ask the LLM to parse raw OCR text into a structured JSON dict with keys:
        title (str), key_points (list[str]), summary (str).
        """
        fallback = {"title": "Notes", "key_points": [], "summary": raw_text}
        lang_name = _LANGUAGE_NAMES.get(language, "English")

        prompt = (
            "You are a structured-notes extractor. "
            "Given the following raw text from handwritten notes, return ONLY a valid JSON object "
            "with exactly these keys:\n"
            '  "title"      : a short title or topic (string)\n'
            '  "key_points" : a list of key points as strings\n'
            '  "summary"    : a 2–3 sentence summary (string)\n\n'
            "Do not include any explanation or markdown — output raw JSON only.\n\n"
            f"Language for your response: {lang_name}\n\n"
            f"Raw text:\n{raw_text}"
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_response = response.content.strip()

            # Strip markdown code fences if the model wrapped it anyway
            if raw_response.startswith("```"):
                raw_response = raw_response.split("```")[1]
                if raw_response.startswith("json"):
                    raw_response = raw_response[4:]

            result = json.loads(raw_response)

            # Validate expected keys
            if not all(k in result for k in ("title", "key_points", "summary")):
                raise ValueError(f"Missing required keys in LLM response: {result}")

            logger.info("structure_handwritten_notes: parsed successfully.")
            return result

        except json.JSONDecodeError as e:
            logger.error("structure_handwritten_notes: JSON parse failed: %s", e)
            return fallback
        except Exception as e:
            logger.error("structure_handwritten_notes failed: %s", e)
            return fallback


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_chat_engine_instance: Optional[ChatEngine] = None


def get_chat_engine() -> ChatEngine:
    """Return the module-level singleton ChatEngine, creating it on first call."""
    global _chat_engine_instance
    if _chat_engine_instance is None:
        logger.info("Creating ChatEngine singleton.")
        _chat_engine_instance = ChatEngine()
    return _chat_engine_instance
