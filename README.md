# 📄 AI Document Q&A Chatbot (RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** web application built with **Python**, **Google Gemini API**, and **Streamlit** that allows users to upload any PDF document and ask questions strictly grounded in its content.

## 🚀 Key Features
- **Custom Overlapping Chunking:** Splits extracted PDF text into 500-character sliding windows with a 100-character overlap to preserve semantic context across boundaries.
- **Pure-Python Vector Retrieval:** Implements a custom **TF-IDF Vectorizer** and **Cosine Similarity** ranking engine using `NumPy` to retrieve the top-3 most relevant passages.
- **Zero-Hallucination Grounding:** Uses strict prompt constraints with **Google Gemini Flash** (`google-genai` SDK) to ensure answers come solely from the uploaded document.
- **Interactive Web UI:** Built with **Streamlit**, featuring dynamic PDF uploads, chat history, and an expandable source-chunk viewer.

## 🛠️ Tech Stack
- **Language:** Python 3
- **LLM & SDK:** Google Gemini Flash (`google-genai`)
- **Retrieval & Math:** NumPy, TF-IDF, Cosine Similarity
- **PDF Parsing & UI:** `pypdf`, `streamlit`, `python-dotenv`

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/Sirajhaide/rag-document-chatbot.git](https://github.com/Sirajhaide/rag-document-chatbot.git)
   cd rag-document-chatbot
