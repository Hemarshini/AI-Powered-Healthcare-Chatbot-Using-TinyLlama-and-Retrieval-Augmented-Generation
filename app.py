import os
import streamlit as st

from rag_utils import get_embeddings, load_or_build_index
from llm_utils import (
    get_llm,
    generate_answer_ollama,
    generate_answer_hf,
    SIMILARITY_THRESHOLD,
)

st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: #0a0f1e;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1528;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.sidebar-logo .logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
}
.sidebar-logo h2 {
    color: #4ecca3 !important;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}
.sidebar-logo p {
    color: #7a8aaa !important;
    font-size: 0.78rem;
    margin: 0.25rem 0 0;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 1rem 0;
}

.how-it-works {
    background: #111d35;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.how-it-works h4 {
    color: #4ecca3 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.75rem;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
}
.step-num {
    background: #4ecca3;
    color: #0a0f1e;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    font-size: 0.7rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    color: #a0aec0 !important;
    font-size: 0.8rem;
    line-height: 1.45;
}

.tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.75rem;
}
.tech-badge {
    background: #1a2a45;
    color: #4ecca3 !important;
    border: 1px solid #2a4060;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}

.disclaimer {
    background: #111d35;
    border-left: 3px solid #e67e22;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.8rem;
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: #a0aec0 !important;
    line-height: 1.5;
}

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 50%, #081220 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(78,204,163,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.header-tag {
    display: inline-block;
    background: rgba(78,204,163,0.15);
    color: #4ecca3;
    border: 1px solid rgba(78,204,163,0.3);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}
.main-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.main-header h1 span { color: #4ecca3; }
.main-header p {
    color: #8899b5;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
    max-width: 600px;
}
.header-stats {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.25rem;
}
.stat-item { text-align: left; }
.stat-num {
    color: #4ecca3;
    font-size: 1.3rem;
    font-weight: 700;
    display: block;
}
.stat-label {
    color: #6a7a95;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Input area ── */
.input-wrapper {
    background: #0d1528;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}
.input-label {
    color: #4ecca3;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    display: block;
}
.stTextInput > div > div > input {
    background: #111d35 !important;
    border: 1px solid #2a4060 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4ecca3 !important;
    box-shadow: 0 0 0 2px rgba(78,204,163,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4a5a75 !important;
}

/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.message-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.message-row.user { flex-direction: row-reverse; }
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.user-av { background: #1a3a5c; }
.avatar.bot-av { background: #0d3325; }
.bubble {
    max-width: 75%;
    padding: 0.85rem 1.1rem;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.user-bubble {
    background: #1a3a5c;
    color: #d0e8ff;
    border-bottom-right-radius: 4px;
}
.bubble.bot-bubble {
    background: #111d35;
    color: #c8d8e8;
    border: 1px solid #1e3a5f;
    border-bottom-left-radius: 4px;
}
.bubble.ood-bubble {
    background: #1f1a0d;
    color: #d4b896;
    border: 1px solid #3d2e0a;
    border-bottom-left-radius: 4px;
}
.bubble-time {
    font-size: 0.68rem;
    color: #4a5a75;
    margin-top: 0.3rem;
    text-align: right;
}

/* ── Clear button ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #2a4060 !important;
    color: #6a7a95 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #4ecca3 !important;
    color: #4ecca3 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4ecca3 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🩺</span>
        <h2>HealthBot AI</h2>
        <p>RAG-powered medical assistant</p>
    </div>
    <hr class="sidebar-divider">
    <div class="how-it-works">
        <h4>How it works</h4>
        <div class="step-item">
            <div class="step-num">1</div>
            <div class="step-text">Your question is embedded into a semantic vector</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div class="step-text">FAISS retrieves the most relevant medical Q&A chunks</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div class="step-text">Llama 3.2 generates a grounded response using only that context</div>
        </div>
    </div>
    <div class="tech-badges">
        <span class="tech-badge">Llama 3.2</span>
        <span class="tech-badge">RAG</span>
        <span class="tech-badge">FAISS</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">Streamlit</span>
    </div>
    <hr class="sidebar-divider">
    <div class="disclaimer">
        ⚠️ General health information only. Always consult a qualified medical professional for personal advice.
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)
    st.caption("Built by **Divvela Hemarshini** · MIT License")

# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings_cached():
    return get_embeddings()

@st.cache_resource
def load_vector_store_cached():
    embeddings = load_embeddings_cached()
    with st.spinner("Setting up knowledge base..."):
        return load_or_build_index(embeddings)

@st.cache_resource
def load_llm_cached():
    with st.spinner("Loading Llama 3.2..."):
        return get_llm()

# ── Domain check + generation ────────────────────────────────────────────────
def is_medical_query(question, db, threshold=SIMILARITY_THRESHOLD):
    results = db.similarity_search_with_relevance_scores(question, k=3)
    if not results:
        return False, 0.0, []
    top_score = results[0][1]
    docs = [r[0] for r in results]
    return top_score >= threshold, top_score, docs

def generate_answer(question, db, llm_tuple):
    backend_type, llm = llm_tuple
    is_relevant, score, docs = is_medical_query(question, db)
    if not is_relevant:
        return ("ood", "I'm a healthcare assistant and can only help with health-related questions. "
                "Your question doesn't appear to be related to medical or wellness topics. "
                "Please ask about symptoms, conditions, nutrition, mental health, "
                "preventive care, or other health topics.")
    context = "\n\n".join(doc.page_content for doc in docs)
    try:
        if backend_type == "ollama":
            return ("bot", generate_answer_ollama(llm, context, question))
        else:
            return ("bot", generate_answer_hf(llm, context, question))
    except Exception as e:
        if backend_type == "ollama":
            return ("bot", f"Could not reach Ollama. Make sure it's running and Llama 3.2 is pulled.\n\nError: {e}")
        return ("bot", f"Language model error: {e}")

# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-tag">🤖 AI-Powered · RAG Architecture</div>
        <h1>AI <span>Healthcare</span> Chatbot</h1>
        <p>
            Ask health-related questions and get accurate, grounded responses.
            Every answer is retrieved from a curated medical knowledge base —
            not generated from unchecked model memory.
        </p>
        <div class="header-stats">
            <div class="stat-item">
                <span class="stat-num">129+</span>
                <span class="stat-label">Medical Q&A pairs</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">RAG</span>
                <span class="stat-label">Architecture</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">Llama 3.2</span>
                <span class="stat-label">Language model</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = load_vector_store_cached()
    llm_tuple = load_llm_cached()

    # Input area
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">Ask a health question</span>', unsafe_allow_html=True)
    user_input = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. What are the symptoms of diabetes? How can I lower my blood pressure?",
        key="input",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if user_input:
        with st.spinner("Retrieving context and generating response..."):
            msg_type, answer = generate_answer(user_input, db, llm_tuple)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append((msg_type, answer))

    # Chat display
    if st.session_state.history:
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"""
                <div class="message-row user">
                    <div class="avatar user-av">👤</div>
                    <div>
                        <div class="bubble user-bubble">{msg}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            elif role == "ood":
                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar bot-av">🩺</div>
                    <div>
                        <div class="bubble ood-bubble">{msg}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar bot-av">🩺</div>
                    <div>
                        <div class="bubble bot-bubble">{msg}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()