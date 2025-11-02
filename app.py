import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 AI Healthcare Dashboard")
    st.markdown("Navigate through tools and insights for healthcare support.")
    st.markdown("---")
    st.markdown("**Pages**")
    st.markdown("- Chatbot")
    st.markdown("- Appointments (coming soon)")
    st.markdown("- Health Records (coming soon)")
    st.markdown("---")
    st.markdown("© 2025 HealthBot AI")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .chat-container {
            background-color: #f5f5f5;
            padding: 30px;
            border-radius: 12px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .user-msg {
            background-color: #d1f5d3;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 10px;
            font-weight: 500;
        }
        .bot-msg {
            background-color: #f0f0f0;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Embedding and Index Setup ----------
INDEX_DIR = "faiss_index"

@st.cache_resource
def get_embeddings():
    model_path = r"C:\Users\divve\OneDrive\Desktop\all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_path)

def create_and_save_index(embeddings):
    loader = TextLoader("healthcare_faqs.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_DIR)
    return db

@st.cache_resource
def load_faiss():
    embeddings = get_embeddings()
    if not os.path.exists(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        st.warning("Setting up knowledge base...")
        db = create_and_save_index(embeddings)
    else:
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return db

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Main App ----------
def main():
    st.markdown("## 💬 Healthcare Chatbot")
    st.markdown("Ask health-related questions powered by **TinyLLaMA** and get instant responses.")

    db = load_faiss()
    retriever = db.as_retriever()
    llm = Ollama(model="tinyllama")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    user_input = st.text_input("Type your health question here:", key="input")

    if user_input:
        answer = qa.run(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, msg in st.session_state.history:
        if speaker == "You":
            st.markdown(f'<div class="user-msg">🧑‍⚕️ {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

