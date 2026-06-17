import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import google.generativeai as genai

# -------------------------------
# GEMINI SETUP
# -------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# LOAD FLAN-T5 MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, t5_model = load_llm()

# -------------------------------
# LOAD EMBEDDER (CACHED)
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# -------------------------------
# LOAD DOCUMENT
# -------------------------------
@st.cache_data
def load_document(path):
    if not os.path.exists(path):
        return ""
    return docx2txt.process(path)

text = load_document("sample.docx")

# -------------------------------
# BETTER CHUNKING (WORD-BASED)
# -------------------------------
def split_into_chunks(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

chunks = split_into_chunks(text)

# -------------------------------
# EMBEDDINGS + FAISS
# -------------------------------
chunk_embeddings = embedder.encode(chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(chunk_embeddings).astype("float32"))

# -------------------------------
# SEARCH FUNCTION (RAG)
# -------------------------------
def search_answer(query):
    query_embedding = embedder.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k=3)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(retrieved_chunks)

    return context, distances[0][0]

# -------------------------------
# FLAN-T5 ANSWER (DOCUMENT)
# -------------------------------
def generate_doc_answer(context, question):
    prompt = f"""
Answer in 1-2 sentences based on the context.

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

# -------------------------------
# GEMINI ANSWER (GENERAL)
# -------------------------------
def generate_gemini_answer(question):
    try:
        response = gemini_model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"
# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 RAG Chatbot")

query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:

    context, score = search_answer(query)

    # safer similarity scaling
    similarity = 1 / (1 + score)

    if similarity > 0.75 and len(text) > 0:
        result = generate_doc_answer(context, query)
        source = "📄 Document (RAG)"
    else:
        result = generate_gemini_answer(query)
        source = "🌐 Gemini"

    st.subheader("Answer:")
    st.write(result)

    st.caption(f"Source: {source}")