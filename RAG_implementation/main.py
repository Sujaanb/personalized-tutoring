# === 1. Environment Setup ===
import os
import glob
from dotenv import load_dotenv

# Load API key
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("Missing MISTRAL_API_KEY in .env")
os.environ["MISTRAL_API_KEY"] = mistral_api_key

# === 2. LangChain & LangGraph Imports ===
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# === 3. Load .txt Documents ===
folder_path = "texts/"
file_paths = glob.glob(f"{folder_path}/*.txt")

documents = []
for file_path in file_paths:
    loader = TextLoader(file_path, encoding="utf-8")
    documents.extend(loader.load())

# === 4. Split Documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# === 5. Embedding and Vectorstores ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Main knowledge base
doc_vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_store")
retriever = doc_vectorstore.as_retriever()

# Memory store (for Q&A)
memory_vectorstore = Chroma(collection_name="memory", embedding_function=embedding_model, persist_directory="./memory_store")

# === 6. Mistral LLM ===
llm = ChatMistralAI(temperature=0.2)

# === 7. LangGraph State ===
class RAGState(dict):
    input: str
    docs: list
    memory: list
    draft: str
    answer: str

# === 8. Nodes ===
def retrieve(state):
    docs = retriever.invoke(state["input"])
    mem_docs = memory_vectorstore.similarity_search(state["input"], k=2)
    return {"input": state["input"], "docs": docs, "memory": mem_docs}

def generate(state):
    context = "\n\n".join(f"- {doc.page_content}" for doc in state["docs"])
    memory_context = "\n\n".join(f"- {doc.page_content}" for doc in state["memory"])

    combined_context = f"""[Retrieved Document Context]\n{context}\n\n[Relevant Memory]\n{memory_context}"""

    # Stage 1: Draft
    draft_prompt = f"""The following context and memories have been retrieved:

{combined_context}

Now, answer the following question clearly and completely:

Question: {state['input']}
"""
    draft_response = llm.invoke(draft_prompt).content

    # Stage 2: Refine
    refine_prompt = f"""Refine the following draft answer using only the given context and memory.
Ensure it is well-written, clear, and grounded in source content.

Context:
{combined_context}

Draft Answer:
{draft_response}

Refined Answer:"""

    final_response = llm.invoke(refine_prompt).content

    # Save to memory
    memory_doc = Document(
        page_content=f"Q: {state['input']}\nA: {final_response}",
        metadata={"source": "memory"}
    )
    memory_vectorstore.add_documents([memory_doc])

    return {
        "input": state["input"],
        "docs": state["docs"],
        "memory": state["memory"],
        "draft": draft_response,
        "answer": final_response
    }

# === 9. Graph Setup ===
graph = StateGraph(RAGState)
graph.add_node("retriever", retrieve)
graph.add_node("generator", generate)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)

app = graph.compile()

# === 10. Run the Agent ===
def chat():
    print("Mistral Memory Agent (type 'exit' to quit)\n")
    while True:
        user_query = input("Ask a question: ")
        if user_query.strip().lower() in ("exit", "quit"):
            break
        final_state = app.invoke({"input": user_query})
        print("\n📝 Draft Answer:\n", final_state["draft"])
        print("\n🧠 Retrieved Memory:\n", "\n".join(doc.page_content for doc in final_state["memory"]))
        print("\n✅ Refined Answer:\n", final_state["answer"])
    print("\n✅ Memory stored in ./memory_store/")

if __name__ == "__main__":
    chat()