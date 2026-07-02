---
title: Healthcare Chatbot RAG
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# AI-Powered Healthcare Chatbot using TinyLlama and RAG

A domain-specific healthcare chatbot that answers medical and lifestyle questions using Retrieval-Augmented Generation (RAG), grounding every response in a **hand-curated 129-entry medical knowledge base** rather than relying on the raw, unverified knowledge of the underlying language model.

**[Live Demo](https://huggingface.co/spaces/Hemarshini/healthcare-chatbot-rag)** &nbsp;·&nbsp; **[Source Code](https://github.com/Hemarshini/AI-Powered-Healthcare-Chatbot)** &nbsp;·&nbsp; Built by [Divvela Hemarshini](https://www.linkedin.com/in/divvelahemarshini)

---

## Overview

General-purpose LLMs are prone to hallucinating confident-sounding but incorrect medical information, which is a serious problem in a healthcare context. This project builds an end-to-end RAG pipeline that retrieves relevant, verified content from a curated knowledge base before generating a response, so answers stay grounded in real source material instead of the model's unchecked internal knowledge.

The system was built independently from knowledge base curation through retrieval pipeline design, LLM integration, and evaluation, with a dual-backend LLM design that supports both fast local development and free cloud deployment (see [Deployment](#deployment) below).

## Key Results

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |

**Test set:** 41 labeled queries — 31 in-domain questions (including paraphrased variants to test semantic generalization beyond exact keyword matching) and 10 out-of-domain questions (to verify the system correctly rejects unrelated queries rather than hallucinating a medical answer). Zero misses. Computed via `eval_chatbot.py`, fully reproducible.

> **Note:** These metrics measure retrieval quality — whether the correct knowledge base chunks are fetched for each query — not generation quality. A 1.00 retrieval F1 means the FAISS + sentence-transformer pipeline correctly identifies relevant context for every in-domain query and correctly rejects every out-of-domain query. Retrieval is evaluated separately because it is the component that can be assessed objectively and reproducibly without human annotation of generated text.

## How It Works

The pipeline takes a user question through four stages:

1. **Knowledge base indexing** — the curated healthcare Q&A corpus (`healthcare_faqs.txt`) is chunked with a recursive text splitter and embedded using a sentence-transformer model
2. **Vector storage** — embeddings are stored in a FAISS index for fast similarity search
3. **Retrieval** — on each query, the top-k most relevant chunks are retrieved from the FAISS index based on semantic similarity, not just keyword overlap
4. **Generation** — the retrieved context is passed to TinyLlama through a LangChain `RetrievalQA` chain, which generates a response grounded in that retrieved content

### Why RAG instead of a plain LLM chatbot

A standalone LLM answering health questions from parametric memory alone has no way to cite where an answer came from, and no mechanism to admit "this isn't in my knowledge base." The retrieval step constrains the model's context to verified source material, and the architecture is set up so the retrieved chunks are inspectable at every step — useful both for debugging and for the evaluation harness, which checks retrieval quality directly against the underlying documents rather than only the final generated text.

### Dual LLM backend

The app supports two interchangeable LLM backends via `llm_utils.py`:

| Backend | How it runs | Use case |
|---|---|---|
| `hf_pipeline` (default) | Loads TinyLlama directly via `transformers`, in-process | Deployable anywhere, including free CPU hosting — no external server needed |
| `ollama` | Calls a locally-running Ollama server | Faster iteration during local development |

Switch backends with an environment variable:
```bash
HEALTHBOT_LLM_BACKEND=ollama streamlit run app.py
```

## Tech Stack

`Python` `LangChain` `TinyLlama` `Transformers` `FAISS` `Sentence Transformers` `Streamlit` `scikit-learn`

## Dataset

The knowledge base (`healthcare_faqs.txt`) contains **129 hand-curated Q&A pairs** covering cardiovascular health, respiratory conditions, mental health, nutrition, musculoskeletal issues, women's and pediatric health, infectious disease, dental care, first aid, and preventive care. Each entry is a plain-text Q&A pair, chunked and embedded at index-build time.

## Project Structure

```
AI-Powered-Healthcare-Chatbot/
├── app.py                 # Streamlit chatbot interface
├── rag_utils.py            # Shared embedding + FAISS index logic
├── llm_utils.py            # LLM backend selector (transformers / Ollama)
├── prepare_faiss.py       # Builds the FAISS index from healthcare_faqs.txt
├── eval_chatbot.py        # Retrieval evaluation — precision, recall, F1-score
├── dev_smoke_test.py      # Quick sanity check for the embedding/FAISS pipeline
├── healthcare_faqs.txt    # Curated healthcare knowledge base (129 Q&A entries)
├── requirements.txt       # Python dependencies
├── screenshots/           # UI demo screenshots
├── LICENSE
└── README.md
```

## Getting Started (local development)

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Hemarshini/AI-Powered-Healthcare-Chatbot.git
cd AI-Powered-Healthcare-Chatbot
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Build the FAISS index

```bash
python prepare_faiss.py
```

### Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. By default this uses the `hf_pipeline` backend (downloads TinyLlama on first run, ~2.2GB). For faster local iteration, install [Ollama](https://ollama.com), run `ollama pull tinyllama`, then launch with `HEALTHBOT_LLM_BACKEND=ollama streamlit run app.py`.

### Run the evaluation suite

```bash
python eval_chatbot.py
```

## Deployment

This app is deployed on **Hugging Face Spaces** (free CPU tier) using the `hf_pipeline` backend, since Spaces can't run an external Ollama server process. To deploy your own copy:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), and choose SDK: **Streamlit**.
2. Push these files to the Space repo: `app.py`, `rag_utils.py`, `llm_utils.py`, `prepare_faiss.py`, `healthcare_faqs.txt`, `requirements.txt`, and this `README.md` (the YAML frontmatter at the top of this file configures the Space).
3. The Space will auto-build and launch `streamlit run app.py`.
4. First load downloads the TinyLlama model (~2.2GB) — this can take a couple of minutes, and the app shows a loading spinner during this step.
5. Responses run on CPU, so generation is slower than a GPU-backed API (typically 10–30 seconds per response) — acceptable for a portfolio demo, not optimized for production traffic.

## Future Enhancements

- Expand the knowledge base toward 300+ curated records
- Add GPU-backed hosted inference for faster response times at scale
- Add medical appointment scheduling
- Add multilingual support

## Disclaimer

This chatbot provides general health information for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with questions about a medical condition.

## License

This project is released under the [MIT License](LICENSE).

## Author

**Divvela Hemarshini**
AI & Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/divvelahemarshini) &nbsp;·&nbsp; [GitHub](https://github.com/Hemarshini)
