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

# AI-Powered Healthcare Chatbot using RAG

A domain-specific healthcare chatbot that answers medical and lifestyle questions using Retrieval-Augmented Generation (RAG), grounding every response in a **hand-curated 129-entry medical knowledge base** rather than relying on the raw, unverified knowledge of the underlying language model. Off-topic questions and health-related-but-uncovered questions are detected via a two-tier similarity check and given distinct, honest responses rather than an ungrounded guess.

[![CI](https://github.com/Hemarshini/REPO-NAME/actions/workflows/ci.yml/badge.svg)](https://github.com/Hemarshini/REPO-NAME/actions/workflows/ci.yml)


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

**Test set:** 41 labeled queries — 31 in-domain questions (including paraphrased variants, to test generalization beyond exact keyword matching) and 10 out-of-domain questions (to verify the system doesn't falsely surface content for unrelated queries). Computed via `eval_chatbot.py`, fully reproducible — retrieval scores this cleanly in large part due to the per-Q&A-pair chunking strategy (see [How It Works](#how-it-works)), which keeps each retrieved chunk self-contained rather than mixing unrelated topics.

## How It Works

The pipeline takes a user question through five stages:

1. **Knowledge base indexing** — the curated healthcare Q&A corpus (`healthcare_faqs.txt`) is split into one chunk per Q&A pair (using blank lines as natural boundaries) and embedded using a sentence-transformer model. Keeping each chunk to a single self-contained Q&A pair — rather than fixed-length character chunking — avoids jamming unrelated topics together in one retrieved chunk, which meaningfully improves answer accuracy for smaller models.
2. **Vector storage** — embeddings are stored in a FAISS index for fast similarity search
3. **Two-tier domain check** — the top retrieval similarity score is compared against two tuned thresholds:
   - **≥ 0.25** → confident match, answered normally from retrieved context
   - **-0.065 to 0.25** → plausibly health-related but not covered by the knowledge base (e.g. "Is it bad to skip breakfast?") — the app says so honestly rather than falsely claiming the question is off-topic
   - **< -0.065** → genuinely unrelated to health (e.g. "What's the capital of France?") — declined with a different, accurate message

   These thresholds were tuned empirically by logging real similarity scores across a mix of on-topic, adjacent, and off-topic test questions, not guessed.
4. **Retrieval** — for confident matches, the top-k most relevant chunks are retrieved from the FAISS index based on semantic similarity, not just keyword overlap
5. **Generation** — the retrieved context is passed to the active language model via a constrained prompt (with stop-sequence and output-cleanup logic to prevent the model from echoing extra Q&A pairs from the context), which generates a response grounded in that retrieved content

### Why RAG instead of a plain LLM chatbot

A standalone LLM answering health questions from parametric memory alone has no way to cite where an answer came from, and no mechanism to admit "this isn't in my knowledge base." The retrieval step constrains the model's context to verified source material, and the two-tier domain check adds a second layer of protection: instead of the model quietly guessing at an off-topic question — or falsely telling a user their legitimate health question "doesn't appear to be health-related" just because the knowledge base doesn't cover it — the app gives an honest, differentiated response in each case.

### Dual LLM backend

The app supports two interchangeable LLM backends via `llm_utils.py`:

| Backend | Model | How it runs | Use case |
|---|---|---|---|
| `ollama` (local default) | Llama 3.2 | Calls a locally-running Ollama server | Best answer quality for local development |
| `hf_pipeline` (deployment default) | TinyLlama | Loads directly via `transformers`, in-process | Deployable anywhere, including free CPU hosting — no external server needed |

Switch backends with an environment variable:
```bash
HEALTHBOT_LLM_BACKEND=hf_pipeline streamlit run app.py
```

The UI always displays whichever model is actually active (via `get_model_display_name()`), so it never claims to be running a model that isn't loaded.

## Tech Stack

`Python` `LangChain` `Llama 3.2` `TinyLlama` `Ollama` `Transformers` `FAISS` `Sentence Transformers` `Streamlit` `scikit-learn`

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

Opens at `http://localhost:8501`. By default this uses the `ollama` backend with **Llama 3.2** for best answer quality — install [Ollama](https://ollama.com) and run `ollama pull llama3.2` first. To run without Ollama (uses TinyLlama via `transformers` instead — this is also what the deployed version uses):
```bash
HEALTHBOT_LLM_BACKEND=hf_pipeline streamlit run app.py
```

### Run the evaluation suite

```bash
python eval_chatbot.py
```
## Docker & CI/CD

The app is containerized with Docker, defaulting to the lightweight `hf_pipeline` backend so it runs without requiring a GPU or an external Ollama server. A GitHub Actions CI pipeline validates the retrieval layer — knowledge base integrity, embedding generation, and FAISS index relevance — on every push, deliberately scoped to avoid downloading the full LLM in CI to keep the pipeline fast and reliable.

### Run with Docker

```bash
docker build -t healthcare-chatbot-rag .
docker run -p 8501:8501 healthcare-chatbot-rag
```

Then open `http://localhost:8501` in your browser.

### CI Pipeline

Every push to `main` triggers:
1. **Fast smoke tests** — confirms the knowledge base, embedding module, and FAISS logic all import and load correctly
2. **Retrieval quality gate** — builds real embeddings and a FAISS index, then confirms a clearly on-topic query retrieves a relevant result, catching silent embedding or chunking regressions
3. **Docker build** — gated behind both test stages passing

See the workflow file at `.github/workflows/ci.yml`.

## Deployment

This app is deployed on **Hugging Face Spaces** (free CPU tier) using the `hf_pipeline` backend, since Spaces can't run an external Ollama server process. To deploy your own copy:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), and choose SDK: **Streamlit**.
2. Push these files to the Space repo: `app.py`, `rag_utils.py`, `llm_utils.py`, `prepare_faiss.py`, `healthcare_faqs.txt`, `requirements.txt`, and this `README.md` (the YAML frontmatter at the top of this file configures the Space).
3. **Important:** in the Space's **Settings → Variables and secrets**, add these variables:
   - `HEALTHBOT_LLM_BACKEND` = `hf_pipeline` — without this, the app defaults to the `ollama` backend, which will fail on Spaces since there's no Ollama server available there.
   - `HEALTHBOT_SIMILARITY_LOW_THRESHOLD` = `-0.065` — the tuned low threshold for the two-tier domain check (see [How It Works](#how-it-works)). Without this it falls back to a less-accurate default.
4. The Space will auto-build and launch `streamlit run app.py`.
5. First load downloads the TinyLlama model (~2.2GB) — this can take a couple of minutes, and the app shows a loading spinner during this step.
6. Responses run on CPU, so generation is slower than a GPU-backed API (typically 10–30 seconds per response) — acceptable for a portfolio demo, not optimized for production traffic.

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
