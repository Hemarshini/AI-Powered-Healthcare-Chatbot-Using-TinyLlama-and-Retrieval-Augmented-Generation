# prepare_faiss.py
"""
Builds and saves the FAISS vector store from healthcare_faqs.txt.
Run this once before starting the app, or whenever the knowledge base changes.
"""

from rag_utils import get_embeddings, build_index, INDEX_DIR


def prepare_faiss():
    embeddings = get_embeddings()
    faiss_db = build_index(embeddings)
    faiss_db.save_local(INDEX_DIR)
    print(f"✅ FAISS vector store created successfully at '{INDEX_DIR}/'")


if __name__ == "__main__":
    prepare_faiss()
