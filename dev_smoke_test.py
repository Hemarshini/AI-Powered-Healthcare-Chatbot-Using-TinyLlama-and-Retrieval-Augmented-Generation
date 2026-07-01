# dev_smoke_test.py
"""
Quick sanity check that the embedding model and FAISS setup work end-to-end,
independent of the real healthcare knowledge base.

This is a DEV-ONLY script (not part of the app). It replaces the old
create_faiss.py, which built a throwaway index from unrelated placeholder
text (e.g. "The capital of France is Paris") and could be confused for
part of the real pipeline. If you don't need a standalone smoke test,
it's safe to delete this file from the repo entirely.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag_utils import EMBEDDING_MODEL


def run_smoke_test():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    sample_docs = [
        "Regular exercise helps maintain a healthy heart.",
        "FAISS is a library for fast vector similarity search.",
        "Always consult a doctor before starting a new treatment.",
    ]

    db = FAISS.from_texts(sample_docs, embeddings)
    results = db.similarity_search("What helps keep your heart healthy?", k=1)

    assert results, "Smoke test failed: no results returned from FAISS search."
    print("✅ Smoke test passed. Embedding + FAISS pipeline is working.")
    print(f"Top match: {results[0].page_content}")


if __name__ == "__main__":
    run_smoke_test()
