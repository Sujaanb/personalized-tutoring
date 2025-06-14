# pdf_utils.py

import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.vectorstores import FAISS

def extract_numbered_questions(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)

    pattern = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, full_text, re.DOTALL)
    return {int(num): text.strip().replace("\n", " ") for num, text in matches}

def create_vector_store_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = MistralAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)
def extract_unnumbered_questions(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)

    # First, remove known numbered questions
    numbered_pattern = r"\d+\.\s*.*?(?=\n\d+\.|\Z)"
    clean_text = re.sub(numbered_pattern, "", full_text, flags=re.DOTALL)

    # Then extract unnumbered questions based on '?'
    possible_questions = re.findall(r"(.*?\?)", clean_text, re.DOTALL)
    return [q.strip().replace("\n", " ") for q in possible_questions if len(q.strip()) > 10]
