import streamlit as st
import os
import pyppeteer
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langchain.docstore.document import Document
from pathlib import Path
import tempfile
import traceback

# Streamlit page configuration
st.set_page_config(
    page_title="🤖 PERSONALISED AI TUTOR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .user-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-left: 5px solid #0077be;
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #ff8a80;
        color: #2c3e50;
        margin-right: 2rem;
    }
    
    .user-message strong, .assistant-message strong {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .profile-setup-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .status-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: #2c3e50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stAlert > div {
        padding: 0.75rem;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    
    .debug-info {
        background: linear-gradient(135deg, #f6f9fc 0%, #f1f4f8 100%);
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 1rem;
        border-left: 4px solid #6c5ce7;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 1rem;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #2c3e50;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components with comprehensive error handling"""
    try:
        # Load environment
        load_dotenv()
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            st.error("❌ Missing MISTRAL_API_KEY in .env file")
            st.stop()

        # Import required modules
        from langchain_mistralai.chat_models import ChatMistralAI
        from langchain_mistralai.embeddings import MistralAIEmbeddings
        from langchain_chroma import Chroma
        from langgraph.graph import StateGraph, END
        from langchain.memory.vectorstore import VectorStoreRetrieverMemory
        from langchain.schema import HumanMessage
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # State Schema
        class AgentState(TypedDict):
            input: str
            memory: Optional[str]
            knowledge: Optional[str]
            response: Optional[str]

        embedding = MistralAIEmbeddings(api_key=mistral_api_key)

        # Vector Stores
        memory_vectorstore = Chroma(
            collection_name="long_term_memory",
            embedding_function=embedding,
            persist_directory="./chroma_storage"
        )

        knowledge_vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=embedding,
            persist_directory="./kb_storage"
        )

        # Retrievers
        memory = VectorStoreRetrieverMemory(
            retriever=memory_vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        knowledge_retriever = knowledge_vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM with error handling
        llm = ChatMistralAI(
            model="mistral-small",
            api_key=mistral_api_key,
            temperature=0.7,
            max_retries=3,
            timeout=30
        )

        # Test LLM connection (silently)
        try:
            test_response = llm.invoke([HumanMessage(content="Hello, test message")])
            # Removed success message
        except Exception as e:
            st.error(f"❌ LLM connection failed: {str(e)}")
            raise e

        # Node functions with enhanced error handling
        def retrieve_memory_node(state: AgentState) -> AgentState:
            try:
                query = state["input"]
                memory_result = memory.load_memory_variables({"input": query})
                return {**state, "memory": memory_result.get("history", "")}
            except Exception as e:
                st.warning(f"⚠️ Memory retrieval issue: {str(e)}")
                return {**state, "memory": ""}

        def retrieve_knowledge_node(state: AgentState) -> AgentState:
            try:
                query = state["input"]
                docs = knowledge_retriever.invoke(query)
                knowledge = "\n".join(doc.page_content for doc in docs)
                return {**state, "knowledge": knowledge}
            except Exception as e:
                st.warning(f"⚠️ Knowledge retrieval issue: {str(e)}")
                return {**state, "knowledge": ""}

        def generate_response_node(state: AgentState) -> AgentState:
            try:
                memory = state.get("memory", "")
                knowledge = state.get("knowledge", "")
                query = state["input"]

                # Build prompt with user profile context
                prompt_parts = []
                
                # Add user profile context
                user_context = f"User Profile: {st.session_state.user_position} working in {st.session_state.user_occupation}"
                prompt_parts.append(user_context)
                prompt_parts.append("Please tailor your response to be relevant for this user's professional background and experience level.")
                
                if knowledge.strip():
                    prompt_parts.append(f"Relevant Knowledge:\n{knowledge}")
                
                if memory.strip():
                    prompt_parts.append(f"Previous Conversation:\n{memory}")
                
                prompt_parts.append(f"Current User Question: {query}")
                prompt_parts.append("Provide a helpful, accurate, and professionally relevant response based on the user's background and available information.")
                
                prompt = "\n\n".join(prompt_parts)
                
                # Debug info for troubleshooting
                if st.session_state.get("show_debug", False):
                    st.write("🔍 **Debug - Prompt sent to LLM:**")
                    st.code(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                
                # Generate response with timeout handling
                response = llm.invoke([HumanMessage(content=prompt)])
                
                if not response or not response.content:
                    return {**state, "response": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."}
                
                return {**state, "response": response.content}
                
            except Exception as e:
                error_msg = f"I encountered an error while generating a response: {str(e)}"
                st.error(f"❌ Response generation error: {str(e)}")
                return {**state, "response": error_msg}

        def update_memory_node(state: AgentState) -> AgentState:
            try:
                if state.get("response"):
                    memory.save_context(
                        {"input": state["input"]},
                        {"output": state["response"]}
                    )
                return {}
            except Exception as e:
                st.warning(f"⚠️ Memory update issue: {str(e)}")
                return {}

        # LangGraph Setup
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_memory", retrieve_memory_node)
        graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
        graph.add_node("generate_response", generate_response_node)
        graph.add_node("update_memory", update_memory_node)

        graph.set_entry_point("retrieve_memory")
        graph.add_edge("retrieve_memory", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)

        memory_agent = graph.compile()

        # Removed success message

        return {
            'agent': memory_agent,
            'knowledge_vectorstore': knowledge_vectorstore,
            'memory_vectorstore': memory_vectorstore,
            'AgentState': AgentState,
            'llm': llm
        }
    
    except Exception as e:
        st.error(f"❌ Critical error initializing RAG system: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

def upload_text_files_to_kb(uploaded_files, knowledge_vectorstore):
    """Process uploaded files and add them to knowledge base with better error handling"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        documents = []
        processed_files = 0
        
        for uploaded_file in uploaded_files:
            try:
                # Read file content with encoding handling
                content = str(uploaded_file.read(), "utf-8").strip()
                if not content:
                    st.warning(f"⚠️ File {uploaded_file.name} is empty, skipping...")
                    continue

                # Split into chunks
                split_docs = splitter.create_documents(
                    [content], 
                    metadatas=[{"source": uploaded_file.name}]
                )
                documents.extend(split_docs)
                processed_files += 1
                
            except UnicodeDecodeError:
                st.error(f"❌ Could not decode file {uploaded_file.name}. Please ensure it's a valid UTF-8 text file.")
                continue
            except Exception as e:
                st.error(f"❌ Error processing file {uploaded_file.name}: {str(e)}")
                continue

        if documents:
            knowledge_vectorstore.add_documents(documents)
            return len(documents), processed_files
        else:
            return 0, 0
    
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        st.code(traceback.format_exc())
        return 0, 0

def main():
    # Main header with enhanced design
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PERSONALISED AI TUTOR</h1>
        <p>Your Intelligent Knowledge Companion</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "user_profile_set" not in st.session_state:
        st.session_state.user_profile_set = False
    if "user_occupation" not in st.session_state:
        st.session_state.user_occupation = ""
    if "user_position" not in st.session_state:
        st.session_state.user_position = ""

    # User Profile Setup (shown only once at the beginning)
    if not st.session_state.user_profile_set:
        st.markdown("""
        <div class="profile-setup-card">
            <h2>👤 Welcome! Let's Set Up Your Profile</h2>
            <p>Provide your professional details to get personalized, relevant responses tailored to your expertise level.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            occupation = st.text_input(
                "🏢 Your Occupation/Field",
                placeholder="e.g., Software Engineer, Doctor, Teacher, Marketing, Finance, etc.",
                help="Enter your professional field or occupation"
            )
        
        with col2:
            position = st.text_input(
                "📈 Your Position/Level",
                placeholder="e.g., Senior, Junior, Manager, Director, Student, Intern, etc.",
                help="Enter your current position or experience level"
            )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("✅ Start Chat Session", type="primary", use_container_width=True):
                if occupation.strip() and position.strip():
                    st.session_state.user_occupation = occupation.strip()
                    st.session_state.user_position = position.strip()
                    st.session_state.user_profile_set = True
                    st.rerun()
                else:
                    st.error("Please fill in both occupation and position fields")
        
        # Skip option
        st.markdown("---")
        if st.button("⏭️ Skip Profile Setup (Continue without personalization)", type="secondary"):
            st.session_state.user_occupation = "General User"
            st.session_state.user_position = "Not Specified"
            st.session_state.user_profile_set = True
            st.rerun()
        
        st.stop()  # Stop here until profile is set

    # Initialize the RAG system (only after profile is set)
    if st.session_state.user_profile_set:
        # Display current user profile with enhanced design
        st.markdown(f"""
        <div class="profile-card">
            <h3>👤 Active User Profile</h3>
            <p><strong>{st.session_state.user_position}</strong> in <strong>{st.session_state.user_occupation}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Initializing RAG system..."):
            rag_components = initialize_rag_system()
    
    agent = rag_components['agent']
    knowledge_vectorstore = rag_components['knowledge_vectorstore']
    memory_vectorstore = rag_components['memory_vectorstore']
    llm = rag_components['llm']

    # Sidebar for file upload and controls
    with st.sidebar:
        st.markdown("### 📂 Knowledge Base Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload text files to knowledge base",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload .txt files to add to the knowledge base"
        )
        
        if uploaded_files:
            if st.button("📤 Process Uploaded Files", type="primary"):
                with st.spinner("Processing files..."):
                    chunks_added, files_processed = upload_text_files_to_kb(
                        uploaded_files, knowledge_vectorstore
                    )
                    if chunks_added > 0:
                        st.markdown(f'<div class="success-badge">✅ Added {chunks_added} chunks from {files_processed} file(s)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-badge">⚠️ No valid content found in uploaded files</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Debug options
        st.markdown("### 🔧 Debug Options")
        st.session_state.show_debug = st.checkbox("Show debug information", value=st.session_state.show_debug)
        
        # User profile management
        st.markdown("### 👤 Profile Management")
        st.write(f"**Current Profile:**")
        st.write(f"• Occupation: {st.session_state.user_occupation}")
        st.write(f"• Position: {st.session_state.user_position}")
        
        if st.button("✏️ Edit Profile"):
            st.session_state.user_profile_set = False
            st.rerun()
        
        if st.button("🧪 Test LLM Connection"):
            try:
                with st.spinner("Testing LLM..."):
                    test_response = llm.invoke([HumanMessage(content="Say 'Connection successful!'")])
                    st.markdown(f'<div class="success-badge">✅ LLM Test: {test_response.content}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ LLM Test Failed: {str(e)}")

        st.markdown("---")
        
        # Reset options
        st.markdown("### 🧹 Reset Options")
        if st.button("🗑️ Clear Memory", help="Clear conversation memory"):
            try:
                memory_vectorstore.reset_collection()
                st.markdown('<div class="success-badge">✅ Memory cleared!</div>', unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing memory: {str(e)}")
            
        if st.button("🗑️ Clear Knowledge Base", help="Clear knowledge base"):
            try:
                knowledge_vectorstore.reset_collection()
                st.markdown('<div class="success-badge">✅ Knowledge base cleared!</div>', unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing knowledge base: {str(e)}")

        if st.button("🔄 Clear Chat History", help="Clear chat display"):
            st.session_state.messages = []
            st.markdown('<div class="success-badge">✅ Chat history cleared!</div>', unsafe_allow_html=True)
            st.rerun()

    # Display chat history with enhanced design
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">')
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You</strong>
                    <div style="margin-top: 0.5rem;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant</strong>
                    <div style="margin-top: 0.5rem;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>')

    # Chat input
    user_input = st.chat_input("💬 Ask me anything...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with chat_container:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong>
                <div style="margin-top: 0.5rem;">{user_input}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate response with comprehensive error handling
        with st.spinner("🤔 Thinking..."):
            try:
                # Validate input
                if not user_input.strip():
                    response = "Please provide a valid question or input."
                else:
                    # Invoke agent with timeout
                    result = agent.invoke({"input": user_input})
                    response = result.get("response", "I apologize, but I couldn't generate a response. Please try again.")
                
                # Validate response
                if not response or response.strip() == "":
                    response = "I'm having trouble generating a response right now. Could you please rephrase your question?"
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with chat_container:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant</strong>
                        <div style="margin-top: 0.5rem;">{response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                error_msg = f"❌ I encountered an error while processing your request: {str(e)}"
                st.error(error_msg)
                
                # Add error to chat history for context
                st.session_state.messages.append({"role": "assistant", "content": "I apologize, but I encountered an error processing your request. Please try again or rephrase your question."})
                
                # Show detailed error in debug mode
                if st.session_state.show_debug:
                    st.code(traceback.format_exc())

    # Footer with enhanced system status
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="status-card">💾 Memory<br><small>Persistent Storage</small></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="status-card">📚 Knowledge<br><small>Document Based</small></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="status-card">🧠 AI Engine<br><small>Mistral Small</small></div>', unsafe_allow_html=True)
    with col4:
        if st.session_state.show_debug:
            st.markdown('<div class="status-card">🔧 Debug<br><small>Enabled</small></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card">🔧 Debug<br><small>Disabled</small></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()