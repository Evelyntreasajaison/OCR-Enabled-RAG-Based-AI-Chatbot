# OCR-Enabled-RAG-Based-AI-Chatbot
This project is a Retrieval-Augmented Generation (RAG) based AI chatbot that answers questions from custom PDF documents, including scanned image PDFs, using OCR and semantic search. The chatbot runs fully offline using HuggingFace models and provides a real-time chat interface with Streamlit.
# Features:
- Supports text-based PDFs
- Supports scanned and image PDFs using OCR
- Semantic search with FAISS
- Local embeddings using HuggingFace
- Interactive Streamlit chat UI
- No paid APIs required
# Architecture:
  PDFs -> Text Extraction + OCR -> Chunking -> Embeddings -> FAISS -> Retriever -> LLM ->Answer
# Tech Stack:
  Python, LangChain, FAISS, HuggingFace Transformers, Sentence-Transformers,Tesseract OCR, pdf2image, Streamlit
# Project Structure:
  data/ : PDF documents
  vectorstore/ : FAISS index
  ingest.py : Document ingestion + OCR
  ui.py : Streamlit chatbot UI
# Setup:
  1. Create virtual environment
  2. Install dependencies
  3. Install Tesseract OCR and Poppler
  4. Add PDFs to data/
  5. Run ingest.py
  6. Run Streamlit UI
# Usage:
  - Ask questions related to uploaded PDFs
  - Supports re-ingestion when documents change
