"""
Smoke tests for the Healthcare Chatbot (RAG).

Scoped deliberately to the RETRIEVAL layer only — knowledge base integrity,
embedding generation, and FAISS index correctness. We intentionally do NOT
test the LLM generation step here, since TinyLlama (~2.2GB) would need to be
downloaded fresh on every CI run, making the pipeline slow and unreliable.
Retrieval correctness is the actual core of RAG's grounding guarantee, so
validating it automatically is still a meaningful, honest quality gate.

NOTE: adjust import names below (rag_utils, prepare_faiss) if your actual
function/module names differ slightly from what's assumed here.
"""
import os
import pytest

KB_PATH = "healthcare_faqs.txt"
EXPECTED_MIN_ENTRIES = 129  # per project README


def test_knowledge_base_file_exists():
    assert os.path.exists(KB_PATH), f"Missing knowledge base file at {KB_PATH}"


def test_knowledge_base_has_expected_entry_count():
    """
    Confirms the curated Q&A knowledge base hasn't been accidentally truncated
    or corrupted — entries are separated by blank lines per the README's
    chunking strategy.
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    entries = [e for e in content.split("\n\n") if e.strip()]
    assert len(entries) >= EXPECTED_MIN_ENTRIES, (
        f"Expected at least {EXPECTED_MIN_ENTRIES} Q&A entries, found {len(entries)}. "
        "This likely means the knowledge base file was truncated or the chunking "
        "delimiter assumption in this test doesn't match the actual file format."
    )


def test_rag_utils_importable():
    """Confirms the embedding + FAISS index logic module imports cleanly."""
    import rag_utils  # noqa: F401


def test_prepare_faiss_importable():
    """Confirms the FAISS index-building script imports cleanly."""
    import prepare_faiss  # noqa: F401


def test_llm_utils_importable():
    """Confirms the LLM backend selector module imports cleanly (doesn't
    require actually loading a model, just that the module itself is valid)."""
    import llm_utils  # noqa: F401


def test_build_index_raises_clear_error_on_missing_file():
    """
    Fast, no-download test: confirms build_index() fails with a clear,
    actionable error rather than a cryptic one if the knowledge base file
    is ever accidentally missing or misnamed.
    """
    import rag_utils

    with pytest.raises(FileNotFoundError):
        rag_utils.build_index(embeddings=None, data_file="nonexistent_file.txt")


@pytest.mark.slow
def test_faiss_index_builds_and_retrieves_relevant_result():
    """
    End-to-end check of the retrieval layer only: builds real sentence-transformer
    embeddings, indexes the knowledge base with FAISS, and confirms a clearly
    on-topic query actually retrieves a relevant chunk. This catches silent
    retrieval failures (e.g., an embedding model mismatch or broken chunking)
    without needing the LLM generation step at all.
    """
    import rag_utils

    embeddings = rag_utils.get_embeddings()
    db = rag_utils.build_index(embeddings)

    query = "What are the symptoms of a heart attack?"
    results = db.similarity_search(query, k=3)

    assert len(results) > 0, "Retrieval returned no results for a clearly in-domain query"
    combined_text = " ".join(r.page_content for r in results).lower()
    assert "heart" in combined_text or "cardiac" in combined_text or "chest" in combined_text, (
        "Top retrieved results don't appear relevant to the query — possible "
        "embedding model or chunking issue"
    )
