# eval_chatbot.py
"""
Evaluates retrieval quality of the RAG pipeline using precision, recall, and F1-score.

Methodology:
- POSITIVE cases: health-related questions that SHOULD match content in the
  knowledge base. A retrieval is scored "relevant" (1) if at least one of the
  top-k retrieved chunks contains one of the question's expected keywords.
- NEGATIVE cases: out-of-domain questions (not health-related) that the system
  should NOT confidently match to any knowledge-base content. A retrieval is
  scored "relevant" (1) if the top match's similarity score is still high
  despite being irrelevant (a false positive / hallucination risk case).

This gives an honest, reproducible metric instead of an unverified claim.
Run this after `python prepare_faiss.py` has built the index.
"""

from rag_utils import get_embeddings, load_or_build_index
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

TOP_K = 3
# Below this similarity distance threshold, a negative-case match is considered
# a false positive (system wrongly treated an unrelated question as answerable).
NEGATIVE_MATCH_DISTANCE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Positive test cases: (question, expected_keywords, ground_truth_label=1)
# Includes both direct and paraphrased versions of KB questions, to test
# generalization rather than exact-string lookup.
# ---------------------------------------------------------------------------
POSITIVE_CASES = [
    ("What are the symptoms of diabetes?", ["thirst", "urination", "fatigue", "blurred vision"]),
    ("How do I know if I might have diabetes?", ["thirst", "urination", "fatigue", "hunger"]),
    ("How can I avoid getting a cold?", ["hands", "sleep", "diet", "sick individuals"]),
    ("What is high blood pressure?", ["hypertension", "artery", "blood pressure"]),
    ("What are heart attack warning signs?", ["chest pain", "shortness of breath", "sweating"]),
    ("How do I lower my blood pressure naturally?", ["salt", "weight", "exercise", "smoking"]),
    ("What foods help reduce cholesterol?", ["heart-healthy", "physical activity", "saturated fats"]),
    ("Is exercise good for my heart?", ["heart muscle", "circulation", "cholesterol"]),
    ("What causes chest pain?", ["heart", "lung", "digestive", "muscle strain", "anxiety"]),
    ("How do I manage asthma?", ["triggers", "inhaler", "action plan"]),
    ("How can pneumonia be prevented?", ["vaccinated", "hygiene", "smoking", "immune system"]),
    ("Can allergies cause breathing issues?", ["wheezing", "coughing", "shortness of breath"]),
    ("What does smoking do to your lungs?", ["lung tissue", "lung capacity", "mucus", "cancer"]),
    ("What are early signs of lung disease?", ["cough", "shortness of breath", "wheezing", "fatigue"]),
    ("How do I prevent osteoporosis?", ["calcium", "vitamin d", "weight-bearing", "smoking"]),
    ("What foods are high in calcium?", ["dairy", "leafy greens", "fortified cereals", "fatty fish"]),
    ("How do I treat a pulled muscle at home?", ["rest", "ice", "compression", "elevate"]),
    ("Does walking help with joint pain?", ["circulation", "muscles", "stiffness"]),
    ("How do I prevent back pain from sitting?", ["breaks", "posture", "ergonomic", "stretch"]),
    ("What are natural ways to reduce stress?", ["breathing", "exercise", "meditate", "sleep"]),
    ("What are the signs of depression?", ["sadness", "loss of interest", "fatigue", "appetite"]),
    ("Why is sleep important for mental health?", ["mood", "focus", "memory", "resilience"]),
    ("Can anxiety cause physical symptoms?", ["heartbeat", "sweating", "trembling", "dizziness"]),
    ("What are healthy ways to cope with stress?", ["exercise", "friends", "hobbies", "mindfulness"]),
    ("What vaccines do adults need?", ["influenza", "tetanus", "shingles", "pneumococcal"]),
    ("Why should I get an annual check-up?", ["early", "monitors", "preventive"]),
    ("How often should diabetics check blood sugar?", ["doctor", "multiple times", "recommended"]),
    ("Is drinking water important?", ["hydrated", "digestion", "temperature", "joint"]),
    ("Does handwashing prevent illness?", ["germs", "bacteria", "viruses", "infections"]),
    ("What are dehydration symptoms?", ["thirst", "dry mouth", "dizziness", "dark-colored urine"]),
    ("How can I strengthen my immune system?", ["nutrient", "exercise", "sleep", "stress"]),
]

# ---------------------------------------------------------------------------
# Negative test cases: out-of-domain questions with no expected relevant
# content in the healthcare knowledge base.
# ---------------------------------------------------------------------------
NEGATIVE_CASES = [
    "What is the capital of France?",
    "Who won the last World Cup?",
    "How do I fix a JavaScript null pointer error?",
    "What's the best recipe for chocolate cake?",
    "How do I change a car tire?",
    "What year did World War II end?",
    "How do I invest in the stock market?",
    "What's the tallest mountain in the world?",
    "How do I train a neural network in PyTorch?",
    "What's the plot of the movie Inception?",
]


def evaluate():
    embeddings = get_embeddings()
    db = load_or_build_index(embeddings)

    y_true = []
    y_pred = []
    details = []

    # Positive cases
    for question, keywords in POSITIVE_CASES:
        results = db.similarity_search(question, k=TOP_K)
        retrieved_text = " ".join(r.page_content.lower() for r in results)
        hit = any(kw.lower() in retrieved_text for kw in keywords)

        y_true.append(1)
        y_pred.append(1 if hit else 0)
        details.append((question, "positive", hit))

    # Negative cases
    for question in NEGATIVE_CASES:
        results = db.similarity_search_with_score(question, k=1)
        # Lower distance score = more similar. A low distance on an unrelated
        # question means the system would have falsely treated it as relevant.
        false_positive = bool(results) and results[0][1] < NEGATIVE_MATCH_DISTANCE_THRESHOLD

        y_true.append(0)
        y_pred.append(1 if false_positive else 0)
        details.append((question, "negative", false_positive))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)
    print(f"Test set size : {len(y_true)}  ({len(POSITIVE_CASES)} positive, {len(NEGATIVE_CASES)} negative)")
    print(f"Accuracy      : {acc:.2f}")
    print(f"Precision     : {precision:.2f}")
    print(f"Recall        : {recall:.2f}")
    print(f"F1-score      : {f1:.2f}")
    print("=" * 60)

    print("\nMisses (for debugging):")
    for question, case_type, result in details:
        expected = True if case_type == "positive" else False
        if result != expected:
            print(f"  [{case_type}] FAILED: {question}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    evaluate()
