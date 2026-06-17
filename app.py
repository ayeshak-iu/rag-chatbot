import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from google import genai

# ===============================
# GEMINI SETUP (NEW SDK FIXED)
# ===============================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ===============================
# LOAD FLAN-T5
# ===============================
@st.cache_resource
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, t5_model = load_llm()

# ===============================
# LOAD EMBEDDER
# ===============================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ===============================
# LOAD DOCUMENT
# ===============================
@st.cache_data
def load_document(path):
    if not os.path.exists(path):
        return ""
    return docx2txt.process(path)

text = load_document("sample.docx")

# ===============================
# CHUNKING
# ===============================
def split_into_chunks(text, chunk_size=80, overlap=15):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

chunks = split_into_chunks(text)

if len(chunks) == 0:
    chunks = ["No document loaded."]

# ===============================
# FAISS INDEX
# ===============================
chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# ===============================
# RETRIEVAL
# ===============================
def get_context(query):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.array(q).astype("float32")

    scores, indices = index.search(q, k=1)
    return chunks[indices[0][0]], float(scores[0][0])

# ===============================
# FLAN-T5 RAG ANSWER
# ===============================
def doc_answer(context, question):
    prompt = f"""
Answer briefly using ONLY the context.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = t5_model.generate(
        **inputs,
        max_length=120
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# GEMINI ANSWER (FIXED)
# ===============================
def gemini_answer(question):
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=question
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ===============================
# MODEL QUESTION DETECTOR
# ===============================
def is_model_question(query):
    q = query.lower()
    keywords = [
        "which model",
        "what model",
        "which llm",
        "what llm",
        "are you gemini",
        "are you flan",
        "who are you",
        "your model",
        "what ai are you"
    ]
    return any(k in q for k in keywords)

# ===============================
# STREAMLIT UI
# ===============================
st.title("RAG Chatbot ")

query = st.text_input("Ask a question:")

# ===============================
# MAIN LOGIC
# ===============================
if query:

    # ===============================
    # MODEL INFO HANDLING
    # ===============================
    if is_model_question(query):
        answer = (
            "I am a hybrid AI system.\n\n"
            "🔹 FLAN-T5 → Used for document-based answers (RAG)\n"
            "🔹 Gemini 1.5 Flash → Used for general questions\n\n"
            "The system automatically chooses the best model based on similarity score."
        )
        model_used = "SYSTEM INFO RESPONSE"

    else:
        context, score = get_context(query)

        if score > 0.60 and len(text.strip()) > 0:
            answer = doc_answer(context, query)
            model_used = "📄 FLAN-T5 (RAG - Document)"
        else:
            answer = gemini_answer(query)
            model_used = "🌐 Gemini 1.5 Flash"

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("Answer:")
    st.write(answer)

    st.caption(f"Model Used: {model_used}")