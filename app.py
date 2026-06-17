import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import google.generativeai as genai

# ===============================
# GEMINI SETUP (FIXED + SAFE)
# ===============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Use stable model (avoids 404 issues)
gemini_model = genai.GenerativeModel("gemini-pro")

# ===============================
# FLAN-T5 MODEL
# ===============================
@st.cache_resource
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, t5_model = load_llm()

# ===============================
# EMBEDDER
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
# CHUNKING (IMPROVED)
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

# ===============================
# FAISS (COSINE SIMILARITY)
# ===============================
chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# ===============================
# RETRIEVAL FUNCTION
# ===============================
def get_context(query):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.array(q).astype("float32")

    scores, indices = index.search(q, k=1)

    best_chunk = chunks[indices[0][0]]
    score = scores[0][0]

    return best_chunk, score

# ===============================
# DOCUMENT ANSWER (FLAN-T5)
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

    outputs = t5_model.generate(**inputs, max_length=120)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# GEMINI FALLBACK
# ===============================
def gemini_answer(question):
    try:
        response = gemini_model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ===============================
# STREAMLIT UI (CLEAN)
# ===============================
st.title("RAG Chatbot")

query = st.text_input("Ask a question:")

# ===============================
# MAIN LOGIC
# ===============================
if query:

    context, score = get_context(query)

    #  DECISION RULE
    if score > 0.60 and len(text.strip()) > 0:
        answer = doc_answer(context, query)
    else:
        answer = gemini_answer(query)

    # ANSWER 
    st.subheader("Answer:")
    st.write(answer)