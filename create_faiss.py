from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_faiss_index():
    # Load local embeddings model (adjust path as needed)
    embeddings = HuggingFaceEmbeddings(model_name=r"C:\Users\divve\OneDrive\Desktop\all-MiniLM-L6-v2")

    # Your documents — put your healthcare-related texts here
    docs = [
        "The capital of France is Paris.",
        "Mount Everest is the tallest mountain in the world.",
        "Python is a popular programming language.",
        "FAISS is a library for fast vector similarity search.",
        "Healthcare advice: Always consult a doctor before taking any medicine."
    ]

    # Create the FAISS index from texts
    db = FAISS.from_texts(docs, embeddings)

    # Save the index locally in folder 'faiss_index'
    db.save_local("faiss_index")

    print("FAISS index created and saved successfully!")

if __name__ == "__main__":
    create_faiss_index()
