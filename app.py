import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import InferenceClient


# =====================================
# HUGGINGFACE API
# =====================================

hf_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=st.secrets["HF_TOKEN"]
)



# =====================================
# LOAD LOCAL FLAN-T5
# =====================================

@st.cache_resource
def load_llm():

    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small"
    )

    return tokenizer, model


tokenizer, t5_model = load_llm()



# =====================================
# EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_embedder():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


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

def split_into_chunks(
        text,
        chunk_size=500,
        overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        chunks.append(
            text[start:start+chunk_size]
        )

        start += chunk_size - overlap


    return chunks



chunks = split_into_chunks(text)



# =====================================
# FAISS
# =====================================

embeddings = embedder.encode(
    chunks,
    normalize_embeddings=True
)


embeddings = np.array(
    embeddings
).astype("float32")


index = faiss.IndexFlatIP(
    embeddings.shape[1]
)


index.add(embeddings)



# =====================================
# RETRIEVAL
# =====================================

def get_context(question):

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    scores, ids = index.search(
        query_embedding,
        1
    )


    return (
        chunks[ids[0][0]],
        float(scores[0][0])
    )



# =====================================
# DOCUMENT ANSWER
# =====================================

def document_answer(context, question):

    prompt = f"""

Answer using ONLY the context.

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
        max_length=150
    )


    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )



# =====================================
# HUGGINGFACE GENERAL ANSWER
# =====================================

def hf_answer(question):

    try:

        response = hf_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": question
                }
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


    return any(
        word in q
        for word in keywords
    )



# =====================================
# STREAMLIT UI
# =====================================

st.title(
    "RAG Chatbot"
)


query = st.text_input(
    "Ask a question:"
)



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
- Mistral-7B-Instruct through HuggingFace API

"""

        used = "SYSTEM INFO"



    else:


        context, score = get_context(query)


        


        # document retrieval

        if score > 0.45:


            answer = document_answer(
                context,
                query
            )


            used = "FLAN-T5 (RAG Document)"



        else:


            answer = hf_answer(
                query
            )


            used = "HuggingFace Mistral API"



    st.subheader(
        "Answer:"
    )

    st.write(answer)


    