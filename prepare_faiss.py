# prepare_faiss.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def prepare_faiss():
    # Load the data file
    loader = TextLoader("healthcare_faqs.txt")
    documents = loader.load()

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Use free HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    faiss_db = FAISS.from_documents(docs, embeddings)

    # Save FAISS database
    faiss_db.save_local("faiss_index")
    print("✅ FAISS vector store created successfully!")

if __name__ == "__main__":
    prepare_faiss()
