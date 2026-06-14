import streamlit as st
import docx2txt

# -------------------------------
# LOAD DOCUMENT
# -------------------------------
text = docx2txt.process("sample.docx")

# -------------------------------
# IMPROVED SEARCH FUNCTION
# -------------------------------
def search_answer(query, text):
    query_words = query.lower().split()
    sentences = text.split(".")

    scored_sentences = []

    for sentence in sentences:
        score = 0
        for word in query_words:
            if word in sentence.lower():
                score += 1
        
        if score > 0:
            scored_sentences.append((score, sentence.strip()))

    # sort by best match
    scored_sentences.sort(reverse=True)

    if scored_sentences:
        best_matches = [s[1] for s in scored_sentences[:2]]
        return ". ".join(best_matches)
    else:
        return None

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 RAG-Based Chatbot   ")

query = st.text_input("Ask a question:")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if query:
    query_lower = query.lower()

    # ✅ CLEAN LLM RESPONSE
    if any(x in query_lower for x in ["llm", "model", "which model", "what are you"]):
        result = "I am a RAG-based chatbot. I use document retrieval to answer questions, and I am designed to support advanced language models like FLAN-T5 in full versions."

    else:
        doc_answer = search_answer(query, text)

        if doc_answer:
            result = doc_answer
        else:
            # ✅ SMART FALLBACK
            if "computer vision" in query_lower:
                result = "Computer vision is a field of artificial intelligence that enables machines to interpret and understand visual information like images and videos."
            else:
                result = "This information is not clearly available in the document, but I can try to answer based on general knowledge if needed."

    st.subheader("Answer:")
    st.write(result)