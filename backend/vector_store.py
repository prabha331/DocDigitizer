import logging
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping fixed-size character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self):
        settings = get_settings()
        persist_dir = settings.chroma_persist_dir

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB PersistentClient initialised at: %s", persist_dir)
        except Exception as e:
            logger.error("Failed to initialise ChromaDB client: %s", e)
            raise

        try:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer 'all-MiniLM-L6-v2' loaded.")
        except Exception as e:
            logger.error("Failed to load SentenceTransformer: %s", e)
            raise

        try:
            self._collection = self._client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection 'documents' ready (%d items).",
                self._collection.count(),
            )
        except Exception as e:
            logger.error("Failed to get/create ChromaDB collection: %s", e)
            raise

    # ------------------------------------------------------------------
    # add_document
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str, metadata: dict) -> bool:
        """
        Chunk *text*, embed each chunk, and upsert into the collection.

        metadata keys used: doc_type, language (plus doc_id and chunk_index
        are always added automatically).
        """
        try:
            chunks = _chunk_text(text)
            if not chunks:
                logger.warning("add_document: no chunks produced for doc_id=%s", doc_id)
                return False

            embeddings = self._model.encode(chunks, show_progress_bar=False).tolist()

            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "doc_id": doc_id,
                    "doc_type": metadata.get("doc_type", ""),
                    "language": metadata.get("language", "en"),
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            logger.info(
                "add_document: upserted %d chunks for doc_id=%s", len(chunks), doc_id
            )
            return True

        except Exception as e:
            logger.error("add_document failed for doc_id=%s: %s", doc_id, e)
            return False

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, doc_id: str, n_results: int = 3) -> list[str]:
        """
        Embed *query* and return the top *n_results* text chunks from the
        collection that belong to *doc_id*.
        """
        try:
            query_embedding = self._model.encode([query], show_progress_bar=False).tolist()

            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where={"doc_id": {"$eq": doc_id}},
                include=["documents"],
            )

            docs = results.get("documents", [[]])[0]
            logger.info(
                "search: returned %d chunks for doc_id=%s query=%r",
                len(docs), doc_id, query[:60],
            )
            return docs

        except Exception as e:
            logger.error("search failed for doc_id=%s: %s", doc_id, e)
            return []

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to *doc_id* from the collection."""
        try:
            # Fetch all chunk ids for this document first
            existing = self._collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])

            if not ids_to_delete:
                logger.warning(
                    "delete_document: no chunks found for doc_id=%s", doc_id
                )
                return True  # idempotent — nothing to delete is still success

            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "delete_document: deleted %d chunks for doc_id=%s",
                len(ids_to_delete), doc_id,
            )
            return True

        except Exception as e:
            logger.error("delete_document failed for doc_id=%s: %s", doc_id, e)
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the module-level singleton VectorStore, creating it on first call."""
    global _vector_store_instance
    if _vector_store_instance is None:
        logger.info("Creating VectorStore singleton.")
        _vector_store_instance = VectorStore()
    return _vector_store_instance
