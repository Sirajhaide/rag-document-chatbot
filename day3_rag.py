import os
import re
import math
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

# 1. Gemini Chat Client Setup
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1beta'}
)
CHAT_MODEL = "gemini-3.5-flash"

# 2. PDF Load & Chunking
reader = PdfReader("sample.pdf")
full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = create_chunks(full_text)
print(f"PDF ke {len(chunks)} chunks ban gaye. Ab TF-IDF vectors bana rahe hain...")

# 3. Pure-Python TF-IDF Vectorizer (Zero external dependencies!)
def tokenize(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

tokenized_chunks = [tokenize(c) for c in chunks]
vocab = sorted(list(set(word for doc in tokenized_chunks for word in doc)))
word_to_idx = {word: i for i, word in enumerate(vocab)}

# IDF (Inverse Document Frequency) calculate karna
N = len(chunks)
idf = np.zeros(len(vocab))
for i, word in enumerate(vocab):
    doc_count = sum(1 for doc in tokenized_chunks if word in doc)
    idf[i] = math.log((N + 1) / (doc_count + 1)) + 1

def get_vector(text):
    tokens = tokenize(text)
    vec = np.zeros(len(vocab))
    for word in tokens:
        if word in word_to_idx:
            vec[word_to_idx[word]] += 1
    return vec * idf

chunk_vectors = [get_vector(c) for c in chunks]
print("Vectors ready! Ab sawal poochte hain...\n")

# 4. Cosine Similarity se Top 3 Relevant Chunks dhoondhna
def retrieve_top_chunks(query, top_k=3):
    query_vec = get_vector(query)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return chunks[:top_k]
    
    scores = []
    for idx, doc_vec in enumerate(chunk_vectors):
        d_norm = np.linalg.norm(doc_vec)
        similarity = (np.dot(query_vec, doc_vec) / (q_norm * d_norm)) if d_norm > 0 else 0
        scores.append((similarity, chunks[idx]))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

# 5. Sawal Poochna aur Grounded Answer Generate Karna
question = "Where did the candidate do their summer internship and what was the project?"
relevant_chunks = retrieve_top_chunks(question)
context = "\n---\n".join(relevant_chunks)

prompt = f"""You are a helpful document assistant. Answer the question strictly using ONLY the provided context below. If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
"""

response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
print("Sawal:", question)
print("\nRAG Bot ka Jawab:")
print(response.text)