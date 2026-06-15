import streamlit as st
import docx2txt
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# LOAD LLM (LIGHTWEIGHT)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"   # ✅ small model
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
# SIMPLE RETRIEVER
# -------------------------------
def search_answer(query, text):
    sentences = get_sentences(text)
    query_lower = query.lower()

    best_score = 0
    best_sentence = ""

    for sentence in sentences:
        score = 0
        for word in query_lower.split():
            if word in sentence.lower():
                score += 2

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence

# -------------------------------
# LLM GENERATION (OPTIMIZED)
# -------------------------------
def generate_answer(context, question):
    # limit context for speed
    context = context[:500]

    prompt = f"""
    Answer the question based on the context below.

    Context: {context}

    Question: {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=100,     # ✅ shorter output
        do_sample=False     # ✅ stable answers
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 RAG Chatbot ")

st.write("This chatbot retrieves information from a document and uses a lightweight language model to generate answers.")

query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:
    query_lower = query.lower()

    
    if any(x in query_lower for x in ["llm", "model", "which model", "what are you"]):
        result = "I am a RAG-based chatbot using FLAN-T5 Small"

    else:
        context = search_answer(query, text)

        if context:
            result = generate_answer(context, query)
        else:
            result = "I couldn't find relevant information in the document."

    st.subheader("Answer:")
    st.write(result)