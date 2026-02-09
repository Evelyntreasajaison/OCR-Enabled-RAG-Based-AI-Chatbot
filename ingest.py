import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract

load_dotenv()

DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/"

def ocr_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

def ingest_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)

            
            loader = PyPDFLoader(path)
            pages = loader.load()

            extracted_text = "".join([p.page_content for p in pages])

            if extracted_text.strip():
                documents.extend(pages)
            else:
                print(f"🔍 OCR applied on: {file}")
                text = ocr_pdf(path)
                documents.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("✅ Documents ingested (Text + OCR PDFs supported)")

if __name__ == "__main__":
    ingest_documents()
