# --- Healthcare Chatbot (RAG): Production Container ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force the lightweight, no-GPU-required backend — the same one used for the
# live Hugging Face Spaces deployment. Without this, the app would default to
# the 'ollama' backend and fail immediately, since no Ollama server exists
# inside this container.
ENV HEALTHBOT_LLM_BACKEND=hf_pipeline
ENV HEALTHBOT_SIMILARITY_LOW_THRESHOLD=-0.065

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit's default port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
