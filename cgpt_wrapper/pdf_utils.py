from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re
import os

# ---- Extract raw text from PDF ----
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text.strip()

# ---- Extract numbered questions like 1. ..., 2) ..., etc. ----
def extract_numbered_questions(text):
    pattern = r"(?:(?:^|\n)(\d+[\.\)]\s+.*?))(?:\n|$)"
    matches = re.findall(pattern, text)
    return matches

# ---- Extract unnumbered questions by looking for '?' ----
def extract_unnumbered_questions(text):
    # Remove numbered ones to avoid duplication
    numbered = extract_numbered_questions(text)
    cleaned_text = text
    for q in numbered:
        cleaned_text = cleaned_text.replace(q, "")
    questions = re.findall(r"([^.?!]*\?)", cleaned_text)
    return [q.strip() for q in questions if len(q.strip()) > 5]

# ---- Create vector store from PDF ----
def create_vector_store_from_pdf(pdf_file):
    full_text = extract_text_from_pdf(pdf_file)
    documents = [Document(page_content=full_text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

# ---- Combined question extraction (optional use) ----
def extract_all_questions(text):
    numbered = extract_numbered_questions(text)
    unnumbered = extract_unnumbered_questions(text)
    return {
        "numbered": numbered,
        "unnumbered": unnumbered
    }
