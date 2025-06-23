import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load Mistral model
def get_llm():
    return ChatMistralAI(model="mistral-small", temperature=0.4)

# Create vector store from PDF
def create_vector_store_from_pdf(uploaded_pdf):
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_file.name

    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()
    
    # (Continue with your existing chunking and embedding logic...)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    return vectordb

# Answer with PDF context
def answer_with_pdf(vectordb, query):
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(), retriever=retriever, return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result["result"]

# Answer without PDF
def answer_without_pdf(query):
    llm = get_llm()
    messages = [
        SystemMessage(content="You are a helpful and patient AI tutor. Explain concepts clearly."),
        HumanMessage(content=query)
    ]
    result = llm.invoke(messages)
    return result.content

# Main App UI
st.title("📘 AI Tutor - Ask Your Questions")

option = st.radio("Choose mode:", ("With PDF", "Without PDF"))

# Option to clear chat
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

if option == "With PDF":
    pdf_file = st.file_uploader("Upload your PDF with questions", type="pdf")
    if pdf_file is not None:
        vectordb = create_vector_store_from_pdf(pdf_file)
        query = st.text_input("Ask a question based on the PDF:")
        if st.button("Get Answer") and query:
            answer = answer_with_pdf(vectordb, query)
            st.session_state.chat_history.append((query, answer))
            st.success(answer)

elif option == "Without PDF":
    query = st.text_input("Ask any academic question:")
    if st.button("Get Answer") and query:
        answer = answer_without_pdf(query)
        st.session_state.chat_history.append((query, answer))
        st.success(answer)

# View Chat History
if st.checkbox("📜 View Chat History"):
    if st.session_state.chat_history:
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Tutor:** {a}")
    else:
        st.info("No chat history yet.")
