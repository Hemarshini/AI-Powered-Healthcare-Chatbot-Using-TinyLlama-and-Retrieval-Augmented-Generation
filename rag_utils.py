# rag_utils.py
"""
Shared RAG utilities used by prepare_faiss.py, app.py, and eval_chatbot.py.

Centralizing this logic avoids duplicating the embedding/index-building
code in multiple files, and avoids hardcoding a machine-specific model path.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Public, portable model name (downloaded from Hugging Face Hub on first run
# and cached locally afterwards). Do NOT hardcode a local machine path here —
# it breaks the project for anyone who isn't you.
EMBEDDING_MODEL = os.environ.get(
    "HEALTHBOT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

DATA_FILE = os.environ.get("HEALTHBOT_DATA_FILE", "healthcare_faqs.txt")
INDEX_DIR = os.environ.get("HEALTHBOT_INDEX_DIR", "faiss_index")


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model (portable, no local path)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index(embeddings: HuggingFaceEmbeddings, data_file: str = DATA_FILE) -> FAISS:
    """Build a FAISS index from the healthcare FAQ knowledge base.

    Each Q&A pair becomes its own chunk (split on blank lines, which is how
    healthcare_faqs.txt separates entries) rather than using fixed-length
    character chunking. Fixed-length chunking can jam 2-3 unrelated Q&A
    pairs into a single chunk, which confuses smaller models at generation
    time — keeping each chunk to exactly one self-contained Q&A pair gives
    much cleaner, more precise retrieval.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Knowledge base file '{data_file}' not found. "
            "Make sure it's in the project root before running this script."
        )

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    docs = [Document(page_content=block) for block in blocks]

    if not docs:
        raise ValueError(f"No content could be loaded from '{data_file}'.")

    return FAISS.from_documents(docs, embeddings)


def load_or_build_index(embeddings: HuggingFaceEmbeddings, index_dir: str = INDEX_DIR) -> FAISS:
    """Load a saved FAISS index if it exists, otherwise build and save a new one."""
    index_file = os.path.join(index_dir, "index.faiss")

    if os.path.exists(index_dir) and os.path.exists(index_file):
        return FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

    db = build_index(embeddings)
    db.save_local(index_dir)
    return db
