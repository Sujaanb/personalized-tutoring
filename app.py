import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage
import os
import tempfile
from dotenv import load_dotenv
import csv
import datetime

# Load .env for API keys
load_dotenv()

# Session state init
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat history saving functions
def save_to_markdown(history, filename="chat_history.md"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, (q, a) in enumerate(history, 1):
            f.write(f"### Q{i}: {q}\n")
            f.write(f"{a}\n\n")

def save_to_csv(history, filename="chat_history.csv"):
    with open(filename, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])
        for q, a in history:
            writer.writerow([q, a])

# PDF → Vector store
def create_vector_store_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

# Get Mistral LLM
def get_llm():
    return ChatMistralAI(model="mistral-small", temperature=0.4)

# Direct LLM QA
def ask_direct_llm(llm, query):
    messages = [
        SystemMessage(content="You are a helpful and patient AI tutor."),
        HumanMessage(content=query)
    ]
    return llm.invoke(messages).content

# PDF-based QA
def ask_from_pdf(vectordb, llm, query):
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Main UI
st.set_page_config(page_title="AI Tutor", page_icon="🎓")
st.title("🎓 AI Tutor (Mistral-based)")

mode = st.radio("Choose Mode:", ["📄 With PDF", "💬 Without PDF"])

llm = get_llm()

# With PDF mode
if mode == "📄 With PDF":
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file:
        vectordb = create_vector_store_from_pdf(pdf_file)
        question = st.text_input("Ask a question from the PDF")
        if question:
            answer = ask_from_pdf(vectordb, llm, question)
            st.write("✅ Answer:", answer)
            st.session_state.chat_history.append((question, answer))

# Without PDF mode
elif mode == "💬 Without PDF":
    question = st.text_input("Ask your question")
    if question:
        answer = ask_direct_llm(llm, question)
        st.write("✅ Answer:", answer)
        st.session_state.chat_history.append((question, answer))

# Chat history display
if st.sidebar.checkbox("📜 Show Chat History"):
    st.markdown("### 💬 Chat History")
    if not st.session_state.chat_history:
        st.info("No chat history yet.")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Q{i}: {q}"):
            st.markdown(f"**Answer:** {a}")

# Save history
if st.session_state.chat_history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("💾 **Save Chat History**")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if st.sidebar.button("⬇️ Download as Markdown"):
        filename = f"chat_{timestamp}.md"
        save_to_markdown(st.session_state.chat_history, filename)
        with open(filename, "rb") as f:
            st.sidebar.download_button("📥 Markdown", f, file_name=filename)

    if st.sidebar.button("⬇️ Download as CSV"):
        filename = f"chat_{timestamp}.csv"
        save_to_csv(st.session_state.chat_history, filename)
        with open(filename, "rb") as f:
            st.sidebar.download_button("📥 CSV", f, file_name=filename)

# Clear history
if st.sidebar.button("🧹 Clear Chat History"):
    if st.sidebar.checkbox("⚠️ Confirm Clear"):
        st.session_state.chat_history.clear()
        st.success("Chat history cleared.")

