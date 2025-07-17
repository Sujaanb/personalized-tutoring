import streamlit as st
import os
import glob
from dotenv import load_dotenv

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, END

# === Environment & API Key ===
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
os.environ["MISTRAL_API_KEY"] = mistral_api_key

# === Embedding & LLM ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(temperature=0.2)

# === Vectorstores ===
doc_vectorstore = None
retriever = None
memory_vectorstore = Chroma(collection_name="memory", embedding_function=embedding_model, persist_directory="./memory_store")

# === File Upload ===
st.set_page_config(page_title="RAG Memory Agent", layout="wide")
st.title("📚 RAG with Long-Term Memory")

uploaded_files = st.file_uploader("Upload one or more `.txt` files", type="txt", accept_multiple_files=True)

# === Process Uploaded Files ===
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    # Split & Index
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    doc_vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_store", collection_name="docs")
    retriever = doc_vectorstore.as_retriever()
    st.success("Documents processed and indexed!")

# === LangGraph State ===
class RAGState(dict):
    input: str
    docs: list
    memory: list
    draft: str
    answer: str

# === Nodes ===
def retrieve(state):
    docs = retriever.invoke(state["input"]) if retriever else []
    mem_docs = memory_vectorstore.similarity_search(state["input"], k=2)
    return {"input": state["input"], "docs": docs, "memory": mem_docs}

def generate(state):
    context = "\n\n".join(f"- {doc.page_content}" for doc in state["docs"])
    memory_context = "\n\n".join(f"- {doc.page_content}" for doc in state["memory"])

    combined_context = f"""[Retrieved Document Context]\n{context}\n\n[Relevant Memory]\n{memory_context}"""

    # Stage 1
    draft_prompt = f"""The following context and memories have been retrieved:

{combined_context}

Now, answer the following question clearly and completely:

Question: {state['input']}"""
    draft_response = llm.invoke(draft_prompt).content

    # Stage 2
    refine_prompt = f"""Refine the following draft answer using only the given context and memory.
Ensure it is well-written, clear, and grounded in source content.

Context:
{combined_context}

Draft Answer:
{draft_response}

Refined Answer:"""
    final_response = llm.invoke(refine_prompt).content

    # Save memory
    memory_vectorstore.add_documents([
        Document(page_content=f"Q: {state['input']}\nA: {final_response}", metadata={"source": "memory"})
    ])

    return {
        "input": state["input"],
        "docs": state["docs"],
        "memory": state["memory"],
        "draft": draft_response,
        "answer": final_response
    }

# === LangGraph Setup ===
graph = StateGraph(RAGState)
graph.add_node("retriever", retrieve)
graph.add_node("generator", generate)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)
app = graph.compile()

# === User Interface ===
query = st.text_input("🔍 Ask a question:")
if st.button("Submit") and query and retriever:
    with st.spinner("Processing..."):
        state = app.invoke({"input": query})

        st.subheader("📝 Draft Answer")
        st.write(state["draft"])

        st.subheader("🧠 Retrieved Memory")
        for doc in state["memory"]:
            st.markdown(f"> {doc.page_content[:300]}...")

        st.subheader("✅ Refined Answer")
        st.success(state["answer"])

        st.caption("Memory saved to ./memory_store/")
