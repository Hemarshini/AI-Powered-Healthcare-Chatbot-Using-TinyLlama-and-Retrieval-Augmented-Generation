# llm_utils.py
"""
LLM backend selector.

Supports two backends:
- "ollama" (default for local): calls a locally-running Ollama server.
  Use Llama 3.2 (3B) for best results — far better instruction following
  than TinyLlama. Run: ollama pull llama3.2
- "hf_pipeline": loads a model directly via transformers. Slower on CPU
  but works without Ollama installed.

Switch backends:
    HEALTHBOT_LLM_BACKEND=ollama streamlit run app.py     (default)
    HEALTHBOT_LLM_BACKEND=hf_pipeline streamlit run app.py

Model used: Llama 3.2 (3B) via Ollama — strong instruction following,
runs locally, free, no API key needed.
"""

import os

# Similarity threshold below which a query is rejected as out-of-domain
SIMILARITY_THRESHOLD = float(os.environ.get("HEALTHBOT_SIMILARITY_THRESHOLD", "0.25"))

# Default models
DEFAULT_OLLAMA_MODEL = os.environ.get("HEALTHBOT_OLLAMA_MODEL", "llama3.2")
DEFAULT_HF_MODEL = os.environ.get("HEALTHBOT_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def build_prompt(context: str, question: str) -> str:
    """
    Build a clean, instruction-following prompt.
    Llama 3.2 uses a simple prompt format — no special chat tags needed.
    """
    return f"""You are a helpful and accurate healthcare assistant.

Your job is to answer the user's health question using ONLY the information provided in the Context section below.

Rules:
- Answer directly and concisely in 2-4 sentences
- Use only information from the Context — do not add anything from outside it
- If the Context does not contain enough information to answer, say: "I don't have enough information on that topic. Please consult a qualified medical professional."
- Never make up medical facts
- Stop writing immediately after your answer. Do NOT generate any further "Question:" or "Answer:" pairs.
- Do NOT include any preamble like "Sure!" or "Here's the answer". Do NOT restate the question. Do NOT use labels like "Q:" or "A:". Write only the plain answer sentences, nothing else.

Context:
{context}

Question: {question}

Answer:"""


import re

# Markers that signal the model has started a genuinely NEW Q&A pair from
# the context (not the leaked "Q:" label on the current answer, which is
# handled separately by _clean_answer). Deliberately excludes a bare "\nQ:"
# since that pattern also matches the leaked label on the real answer itself.
_STOP_MARKERS = ["\nQuestion:", "\n\nQuestion", "\nQuestion "]

# Preamble/label patterns models sometimes leak despite prompt instructions.
_PREAMBLE_PATTERNS = [
    r"^sure!?\s*here'?s the answer[^:]*:?\s*",
    r"^here'?s the answer[^:]*:?\s*",
    r"^sure!?\s*",
    r"^q:\s*.*?a:\s*",  # leaked "Q: ... A:" label pair, same line or across lines
]


def _clean_answer(text: str) -> str:
    """Strip common preamble/label artifacts a small model may leak despite
    prompt instructions, e.g. 'Sure! Here's the answer: Q: ... A: ...'."""
    cleaned = text.strip()
    for pattern in _PREAMBLE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _truncate_at_stop(text: str) -> str:
    """Cut the generated text off at the first sign it's started echoing
    additional Q&A pairs from the context, rather than stopping cleanly."""
    earliest = len(text)
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            earliest = min(earliest, idx)
    truncated = text[:earliest].strip()
    return _clean_answer(truncated)


def get_llm():
    backend = os.environ.get("HEALTHBOT_LLM_BACKEND", "ollama")

    if backend == "ollama":
        from langchain_community.llms import Ollama
        model = DEFAULT_OLLAMA_MODEL
        # stop= tells Ollama to halt generation the moment it emits one of
        # these sequences, instead of relying only on post-hoc truncation.
        return ("ollama", Ollama(model=model, temperature=0.1, stop=_STOP_MARKERS))

    # HF pipeline fallback
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False,
        return_full_text=False,
    )
    return ("hf_pipeline", pipe)


def generate_answer_ollama(llm, context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return _truncate_at_stop(response)


def generate_answer_hf(pipe, context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    outputs = pipe(prompt)
    return _truncate_at_stop(outputs[0]["generated_text"])


def get_model_display_name(backend_type: str) -> str:
    """Human-readable label for whichever model is actually running —
    used in the UI so it never claims a model that isn't active."""
    if backend_type == "ollama":
        return f"{DEFAULT_OLLAMA_MODEL} (Ollama)"
    short_name = DEFAULT_HF_MODEL.split("/")[-1]
    return f"{short_name} (Transformers)"
