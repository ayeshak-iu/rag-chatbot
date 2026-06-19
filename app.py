import streamlit as st
import docx2txt
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import InferenceClient


# ===============================
# HUGGINGFACE API SETUP
# ===============================

hf_client = InferenceClient(
    model="google/flan-t5-base",
    token=st.secrets["HF_TOKEN"]
)


# ===============================
# LOAD LOCAL FLAN-T5
# ===============================

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



# ===============================
# EMBEDDINGS
# ===============================

@st.cache_resource
def load_embedder():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


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

def split_into_chunks(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[start:start+chunk_size]

        chunks.append(chunk)

        start += chunk_size - overlap


    return chunks



chunks = split_into_chunks(text)


if len(chunks) == 0:

    chunks = ["No document loaded"]



# ===============================
# FAISS VECTOR STORE
# ===============================

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



# ===============================
# RETRIEVAL
# ===============================

def get_context(query):

    q = embedder.encode(
        [query],
        normalize_embeddings=True
    )


    q = np.array(q).astype("float32")


    scores, ids = index.search(
        q,
        k=1
    )


    return (
        chunks[ids[0][0]],
        float(scores[0][0])
    )



# ===============================
# DOCUMENT ANSWER
# ===============================

def document_answer(context, question):

    prompt = f"""

Answer only using the context.

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


    output = t5_model.generate(
        **inputs,
        max_length=120
    )


    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )



# ===============================
# HUGGINGFACE FALLBACK LLM
# ===============================

def hf_answer(question):

    try:

        response = hf_client.text_generation(
            question,
            max_new_tokens=120
        )

        return response


    except Exception as e:

        return f"HuggingFace Error: {e}"




# ===============================
# MODEL QUESTION HANDLER
# ===============================

def is_model_question(query):

    q = query.lower()


    keywords = [

        "which model",
        "what model",
        "which llm",
        "what llm",
        "what ai",
        "your model",
        "powered by",
        "built with",
        "who are you",
        "are you flan"

    ]


    return any(
        word in q
        for word in keywords
    )




# ===============================
# UI
# ===============================

st.title("RAG Chatbot")

query = st.text_input(
    "Ask a question:"
)



if query:


    if is_model_question(query):

        answer = """

I am a hybrid RAG chatbot.

Document questions:
→ google/flan-t5-small + FAISS retrieval


General questions:
→ google/flan-t5-base through HuggingFace API


Embeddings:
→ all-MiniLM-L6-v2

"""

        model_used = "SYSTEM INFO"



    else:


        context, score = get_context(query)



        # threshold matches report

        if score > 0.75:

            answer = document_answer(
                context,
                query
            )

            model_used = "FLAN-T5 (RAG Document)"



        else:

            answer = hf_answer(
                query
            )

            model_used = "FLAN-T5 HuggingFace API"



    st.subheader("Answer:")

    st.write(answer)


    st.caption(
        f"Model Used: {model_used}"
    )