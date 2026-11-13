import os
import re
import math
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

# Page Config
st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="centered")
st.title("📄 AI Document Q&A Chatbot (RAG)")
st.caption("Upload any PDF and ask questions grounded strictly in your document.")

# Gemini Client Setup
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1beta'}
)
CHAT_MODEL = "gemini-3.5-flash"

# Helper Functions: Chunking & TF-IDF Vectorization
def create_chunks(text, chunk_size=500, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def tokenize(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

@st.cache_resource(show_spinner="PDF process aur index ho rahi hai...")
def process_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = create_chunks(full_text)
    
    tokenized_chunks = [tokenize(c) for c in chunks]
    vocab = sorted(list(set(w for doc in tokenized_chunks for w in doc)))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    N = len(chunks)
    idf = np.array([
        math.log((N + 1) / (sum(1 for doc in tokenized_chunks if w in doc) + 1)) + 1
        for w in vocab
    ])
    
    def get_vec(text):
        vec = np.zeros(len(vocab))
        for w in tokenize(text):
            if w in word_to_idx:
                vec[word_to_idx[w]] += 1
        return vec * idf

    chunk_vectors = [get_vec(c) for c in chunks]
    return chunks, vocab, word_to_idx, idf, chunk_vectors

def retrieve_top_chunks(query, chunks, vocab, word_to_idx, idf, chunk_vectors, top_k=3):
    q_vec = np.zeros(len(vocab))
    for w in tokenize(query):
        if w in word_to_idx:
            q_vec[word_to_idx[w]] += 1
    q_vec = q_vec * idf
    
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return chunks[:top_k]
        
    scores = []
    for idx, d_vec in enumerate(chunk_vectors):
        d_norm = np.linalg.norm(d_vec)
        sim = (np.dot(q_vec, d_vec) / (q_norm * d_norm)) if d_norm > 0 else 0
        scores.append((sim, chunks[idx]))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:top_k]]

# Sidebar: PDF Upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Apni PDF file yahan upload karo", type=["pdf"])
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

# Main Chat Interface
if uploaded_file is not None:
    chunks, vocab, word_to_idx, idf, chunk_vectors = process_pdf(uploaded_file)
    st.info(f"✅ Document indexed into **{len(chunks)} chunks**. Neeche apna sawal poocho!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("PDF se koi bhi sawal poocho..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Document mein jawab dhoondh raha hoon..."):
                relevant_chunks = retrieve_top_chunks(
                    question, chunks, vocab, word_to_idx, idf, chunk_vectors
                )
                context = "\n---\n".join(relevant_chunks)
                prompt = f"""You are a helpful AI Document Assistant. Answer the question strictly using ONLY the provided context below. If the answer is not present in the context, say "I don't know based on the uploaded document."

Context:
{context}

Question: {question}
"""
                response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
                answer = response.text
                st.markdown(answer)
                
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for i, c in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {c}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("👈 Shuru karne ke liye left sidebar se koi PDF file (jaise `sample.pdf`) upload karo!")