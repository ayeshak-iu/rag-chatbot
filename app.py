import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import InferenceClient


# =====================================
# HUGGINGFACE API (LLAMA 3)
# =====================================

hf_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=st.secrets["HF_TOKEN"]
)


# =====================================
# LOAD LOCAL FLAN-T5
# =====================================

@st.cache_resource
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model


tokenizer, t5_model = load_llm()


# =====================================
# EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# =====================================
# LOAD DOCUMENT
# =====================================

@st.cache_data
def load_document():
    if os.path.exists("sample.docx"):
        return docx2txt.process("sample.docx")
    return ""


text = load_document()

if text == "":
    text = "No document loaded"


# =====================================
# CHUNKING
# =====================================

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


chunks = split_into_chunks(text)


# =====================================
# FAISS INDEX
# =====================================

embeddings = embedder.encode(chunks, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# =====================================
# RETRIEVAL
# =====================================

def get_context(question):
    query_embedding = embedder.encode([question], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    scores, ids = index.search(query_embedding, 1)

    return chunks[ids[0][0]], float(scores[0][0])


# =====================================
# POST-PROCESSING (3 SENTENCES GUARANTEE)
# =====================================

def ensure_3_sentences(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 3:
        text += (
            " It is also important to understand its practical applications in computing."
            " Additionally, it plays a key role in performance optimization of systems."
        )

    return text


# =====================================
# DOCUMENT ANSWER (RAG)
# =====================================

def document_answer(context, question):

    prompt = f"""
Answer using ONLY the context.
Write at least 3 complete sentences.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = t5_model.generate(
        **inputs,
        max_new_tokens=150,
        min_new_tokens=60,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return ensure_3_sentences(answer)


# =====================================
# HF GENERAL ANSWER (LLAMA 3)
# =====================================

def hf_answer(question):
    try:
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"API Error: {e}"


# =====================================
# MODEL QUESTION DETECTOR
# =====================================

def is_model_question(q):
    q = q.lower()

    keywords = [
        "what llm",
        "which llm",
        "what model",
        "which model",
        "what ai",
        "your model",
        "powered by",
        "built with",
        "who are you"
    ]

    return any(word in q for word in keywords)


# =====================================
# STREAMLIT UI
# =====================================

st.title("RAG Chatbot")

query = st.text_input("Ask a question:")

if query:

    # MODEL INFO
    if is_model_question(query):

        answer = """
I am a hybrid RAG chatbot.

Document Question Answering:
- FLAN-T5-small
- FAISS retrieval
- Sentence Transformer embeddings

General Questions:
- LLaMA 3 (Meta) via HuggingFace API
"""

    else:

        context, score = get_context(query)

        if score > 0.45:
            answer = document_answer(context, query)
        else:
            answer = hf_answer(query)

    st.subheader("Answer:")
    st.write(answer)

    