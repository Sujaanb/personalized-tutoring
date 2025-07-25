# AI Tutor - RAG Assistant with Streamlit Interface

This is an AI-powered tutoring system with a Retrieval-Augmented Generation (RAG) architecture that now includes a user-friendly Streamlit web interface.

## Features

### 🤖 Chat Interface
- Interactive chat with the AI tutor
- Context-aware responses based on your uploaded documents
- Conversation history within the session
- Real-time processing indicator

### 📚 Document Management
- Upload PDF and TXT files directly through the web interface
- Bulk processing of files from the uploads directory
- Real-time status updates of the knowledge base
- File listing and management

### 🎯 Interactive Quiz Mode
- Generate quiz questions based on your uploaded content
- Multiple-choice questions with instant feedback
- Score tracking and percentage calculation
- Random question generation from knowledge base

### 📊 Knowledge Base Monitoring
- Real-time status of document chunks
- File type breakdown (PDF/TXT)
- Easy refresh of status information

## How to Use

### 1. Starting the Application

#### Option A: Using the Batch File (Recommended)
Double-click `run_streamlit.bat` to start the application automatically.

#### Option B: Manual Start
Open command prompt in the project directory and run:
```bash
C:/Python313/python.exe -m streamlit run streamlit_app.py
```

### 2. Accessing the Interface
Once started, open your web browser and go to: `http://localhost:8501`

### 3. Setting Up Your Knowledge Base

1. **Upload Documents**: 
   - Use the file uploader in the sidebar to select PDF or TXT files
   - Click "Process Uploaded Files" to add them to the knowledge base
   
2. **Quick Actions**:
   - Use "Process PDFs" to process only PDF files from the uploads directory
   - Use "Process TXTs" to process only text files from the uploads directory

3. **Monitor Status**:
   - Click "Refresh Status" to see current knowledge base statistics
   - Click "List Files" to see available files in the uploads directory

### 4. Chatting with the AI

1. Type your questions in the chat input at the bottom
2. The AI will provide responses based on your uploaded documents
3. Conversation history is maintained during your session
4. Use "Clear Chat" to reset the conversation history

### 5. Taking Quizzes

1. Click "Start Quiz" in the quiz panel
2. Answer multiple-choice questions generated from your documents
3. Receive instant feedback on your answers
4. Track your score and percentage
5. Click "End Quiz" when finished or continue for more questions

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Original RAG system with LangGraph workflow
- **Vector Store**: ChromaDB for document embeddings and memory
- **LLM**: Mistral AI for response generation and quiz creation
- **Document Processing**: LangChain for PDF and text processing

### File Structure
```
├── streamlit_app.py          # Main Streamlit interface
├── main.py                   # Original RAG assistant logic
├── vector_store_manager.py   # Vector store management
├── document_processor.py     # Document processing utilities
├── upload_service.py         # File upload and processing
├── graph_nodes.py           # LangGraph workflow nodes
├── config.py                # Configuration settings
├── run_streamlit.bat        # Windows batch file to start app
├── .env                     # Environment variables (create this)
├── uploads/                 # Directory for uploaded files
├── chroma_storage/          # ChromaDB memory storage
└── kb_storage/              # ChromaDB knowledge base storage
```

### Environment Setup
Create a `.env` file in the project directory with:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

## Original Functionality Preserved
All original functionality from the command-line version is preserved:
- Document processing and chunking
- Vector store management
- Memory persistence
- Quiz generation
- Knowledge retrieval
- Context-aware responses

The Streamlit interface is a wrapper that provides a user-friendly way to access all these features without changing the underlying logic.

## Troubleshooting

1. **Port already in use**: If port 8501 is busy, Streamlit will automatically try the next available port
2. **Missing API key**: Ensure your `.env` file contains a valid `MISTRAL_API_KEY`
3. **Upload errors**: Check that the uploads directory exists and has proper permissions
4. **Memory issues**: Large documents may take time to process; wait for the spinner to complete

## Credits
- **Created by**: smgrizzlybear
- **Date**: 2025-07-12
- **Updated**: 2025-07-24 (Streamlit interface added)
