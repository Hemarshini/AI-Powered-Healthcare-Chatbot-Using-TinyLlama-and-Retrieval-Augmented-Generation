# 🤖 AI-Powered Healthcare Chatbot Using TinyLlama and Retrieval-Augmented Generation (RAG)

This project is an **AI-powered healthcare chatbot** built with **Streamlit**, **LangChain**, and **TinyLlama**. It uses **Retrieval-Augmented Generation (RAG)** to provide accurate and context-aware healthcare information from a curated knowledge base.

---

## 🩺 Features
- 💬 Conversational chatbot for healthcare-related questions.
- 🧠 Uses TinyLlama for text generation.
- 📚 Employs RAG with FAISS for relevant information retrieval.
- 🌐 Streamlit web app interface.
- ⚡ Real-time Q&A on health and lifestyle topics.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/AI-Powered-Healthcare-Chatbot.git
cd AI-Powered-Healthcare-Chatbot
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare FAISS vector store
```bash
python prepare_faiss.py
```
This will create a folder named `faiss_index/` used by the chatbot.

### 4️⃣ Run the Chatbot App
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧩 Project Structure

| File | Description |
|------|--------------|
| `app.py` | Streamlit interface for the chatbot |
| `prepare_faiss.py` | Creates FAISS index from healthcare_faqs.txt |
| `create_faiss.py` | Sample FAISS index creation test script |
| `healthcare_faqs.txt` | Knowledge base for healthcare responses |
| `requirements.txt` | Required Python libraries |
| `screenshots/` | Contains UI demo screenshots |
| `LICENSE` | Project license (MIT) |

---

## 📸 Demo Screenshots
| Chat Interface | Bot Response Example |
|----------------|----------------------|
| ![Chat UI](screenshots/chat_ui.png) | ![Response](screenshots/demo_response.png) |

---

## 🧠 Technologies Used
- Python 3.10+
- Streamlit
- LangChain
- FAISS
- TinyLlama (via Ollama)
- HuggingFace Transformers
- Sentence Transformers

---

## 🚀 Future Enhancements
- Add medical appointment scheduling
- Integrate multilingual support
- Deploy chatbot on cloud (Streamlit Cloud / HuggingFace Spaces)

---

## 👩‍💻 Author
**Divvela Hemarshini**  
B.Tech CSE | AI & ML Enthusiast  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/your-link)

---

## 🪪 License
This project is licensed under the MIT License. See `LICENSE` for details.
