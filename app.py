import streamlit as st
import docx2txt
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# LOAD LLM
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# LOAD DOCUMENT
# -------------------------------
text = docx2txt.process("sample.docx")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

text = clean_text(text)

# -------------------------------
# SENTENCE SPLITTING
# -------------------------------
def get_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

# -------------------------------
# REMOVE DUPLICATES (IMPORTANT FIX)
# -------------------------------
def remove_duplicate_sentences(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    seen = set()
    result = []

    for s in sentences:
        if s.lower() not in seen:
            seen.add(s.lower())
            result.append(s)

    return ". ".join(result)

# -------------------------------
# LIMIT TO 2 SENTENCES ONLY
# -------------------------------
def limit_to_two_sentences(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:2])

# -------------------------------
# RETRIEVAL (IMPROVED)
# -------------------------------
def search_answer(query, text):
    sentences = get_sentences(text)
    query_words = set(query.lower().split())

    best_score = 0
    best_index = 0

    for i, sentence in enumerate(sentences):
        sentence_words = set(sentence.lower().split())
        score = len(query_words.intersection(sentence_words))

        if score > best_score:
            best_score = score
            best_index = i

    # expand context (better understanding)
    start = max(0, best_index - 2)
    end = min(len(sentences), best_index + 3)

    context_window = sentences[start:end]
    context = " ".join(context_window)

    # 🔥 FIX: remove duplicates
    context = remove_duplicate_sentences(context)

    return context

# -------------------------------
# LLM GENERATION
# -------------------------------
def generate_answer(context, question):
    context = limit_to_two_sentences(context)  # 🔥 FORCE 2 SENTENCES

    prompt = f"""
You are an expert assistant.

Answer the question clearly in 1-2 sentences only.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=120,
        min_length=30,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄RAG Chatbot")

query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:
    query_lower = query.lower()

    if any(x in query_lower for x in ["llm", "model", "what are you", "which model"]):
        result = "I am a RAG-based chatbot using FLAN-T5 Small"

    else:
        context = search_answer(query, text)

        if context:
            result = generate_answer(context, query)
        else:
            result = "I couldn't find relevant information in the document."

    st.subheader("Answer:")
    st.write(result)