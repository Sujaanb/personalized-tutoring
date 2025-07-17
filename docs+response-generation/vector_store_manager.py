from typing import List, Dict, Any
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.docstore.document import Document

class VectorStoreManager:
    """Manages vector stores for memory and knowledge base."""
    
    def __init__(self, api_key: str, memory_dir: str, kb_dir: str):
        self.embedding = MistralAIEmbeddings(api_key=api_key)
        
        # Initialize vector stores
        self.memory_vectorstore = Chroma(
            collection_name="long_term_memory",
            embedding_function=self.embedding,
            persist_directory=memory_dir
        )
        
        self.knowledge_vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embedding,
            persist_directory=kb_dir
        )
        
        # Initialize retrievers
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.memory_vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        self.knowledge_retriever = self.knowledge_vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def reset_collections(self):
        """Reset both memory and knowledge collections."""
        self.memory_vectorstore.reset_collection()
        self.knowledge_vectorstore.reset_collection()
        print("🧹 Chroma memory and knowledge base have been reset.")
    
    def add_documents_to_knowledge_base(self, documents: List[Document]) -> bool:
        """Add documents to the knowledge base."""
        try:
            self.knowledge_vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"❌ Error adding documents to knowledge base: {str(e)}")
            return False
    
    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Get status information about the knowledge base."""
        try:
            collection = self.knowledge_vectorstore.get()
            doc_count = len(collection['documents']) if collection['documents'] else 0
            
            # Count file types
            file_types = {}
            if doc_count > 0:
                metadatas = collection.get('metadatas', [])
                for metadata in metadatas:
                    if metadata and 'file_type' in metadata:
                        file_type = metadata['file_type']
                        file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                'total_chunks': doc_count,
                'file_types': file_types,
                'success': True
            }
        except Exception as e:
            return {
                'total_chunks': 0,
                'file_types': {},
                'success': False,
                'error': str(e)
            }
    
    def save_memory_context(self, input_text: str, output_text: str):
        """Save conversation context to memory."""
        self.memory.save_context(
            {"input": input_text},
            {"output": output_text}
        )
    
    def load_memory_variables(self, input_text: str) -> str:
        """Load relevant memory for given input."""
        memory_result = self.memory.load_memory_variables({"input": input_text})
        return memory_result.get("history", "")
    
    def retrieve_knowledge(self, query: str) -> str:
        """Retrieve relevant knowledge for given query."""
        docs = self.knowledge_retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)