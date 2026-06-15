import streamlit as st
import docx2txt
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# LOAD LLM (FLAN-T5 SMALL)
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
# SPLIT INTO SENTENCES
# -------------------------------
def get_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# -------------------------------
# IMPROVED RETRIEVAL (FIXED)
# -------------------------------
def search_answer(query, text):
    sentences = get_sentences(text)
    query_words = set(query.lower().split())

    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())

        # overlap scoring
        score = len(query_words.intersection(sentence_words))

        # boost meaningful sentences
        if len(sentence.split()) > 8:
            score += 1

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence

# -------------------------------
# LLM GENERATION (IMPROVED)
# -------------------------------
def generate_answer(context, question):
    context = context[:600]  # prevent overload

    prompt = f"""
    Use the context below to give a detailed and complete answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 Improved RAG Chatbot (FLAN-T5 Small)")


query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:
    query_lower = query.lower()

    # system identity fix (optional but useful)
    if any(x in query_lower for x in ["llm", "model", "what are you", "which model"]):
        result = "I am a RAG-based chatbot using FLAN-T5 Small."

    else:
        context = search_answer(query, text)

        if context:
            result = generate_answer(context, query)
        else:
            result = "I couldn't find relevant information in the document."

    st.subheader("Answer:")
    st.write(result)