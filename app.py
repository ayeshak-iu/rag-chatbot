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

# safer model fallback (avoids 404 issues)
try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except:
    gemini_model = genai.GenerativeModel("gemini-pro")

# -------------------------------
# LOAD FLAN-T5
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, t5_model = load_llm()

# -------------------------------
# LOAD EMBEDDER
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
# BETTER CHUNKING (LESS NOISE)
# -------------------------------
def split_into_chunks(text, chunk_size=80, overlap=15):
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
# EMBEDDINGS + FAISS (COSINE)
# -------------------------------
chunk_embeddings = embedder.encode(
    chunks,
    normalize_embeddings=True
)

chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def search_answer(query):
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, k=1)

    best_chunk = chunks[indices[0][0]]
    score = scores[0][0]

    return best_chunk, score

# -------------------------------
# FLAN-T5 ANSWER (DOCUMENT ONLY)
# -------------------------------
def generate_doc_answer(context, question):
    prompt = f"""
Answer in 1-2 sentences based ONLY on the context.

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
# GEMINI FALLBACK
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
st.title("RAG Chatbot ")

query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:

    context, score = search_answer(query)

    st.write("Similarity Score:", score)
    st.write("Retrieved Context:", context)

    # STRICT threshold (important fix)
    if score > 0.60 and len(text.strip()) > 0:
        result = generate_doc_answer(context, query)
        source = "📄 Document (RAG)"
    else:
        result = generate_gemini_answer(query)
        source = "🌐 Gemini"

    st.subheader("Answer:")
    st.write(result)

    st.caption(f"Source: {source}")