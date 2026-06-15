import streamlit as st
import docx2txt
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# LOAD MODEL
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
# BETTER SENTENCE SPLITTING
# -------------------------------
def get_sentences(text):
    # FIX: better splitting (prevents broken context)
    return re.split(r'(?<=[.!?])\s+', text)

# -------------------------------
# IMPROVED RETRIEVAL (IMPORTANT FIX)
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

    # 🔥 FIX: expand context window more (VERY IMPORTANT)
    start = max(0, best_index - 2)
    end = min(len(sentences), best_index + 3)

    context_window = sentences[start:end]

    return " ".join(context_window)

# -------------------------------
# LLM GENERATION (FOR BETTER ANSWERS)
# -------------------------------
def generate_answer(context, question):
    context = context[:800]  # more context = better answers

    prompt = f"""
You are an expert assistant.

Use the context below to answer in a FULL, DETAILED, EXPLANATORY way.

Do NOT give short answers.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=256,
        min_length=80,      # 🔥 forces longer answers
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