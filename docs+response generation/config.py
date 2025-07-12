import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the RAG system."""
    
    def __init__(self):
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("Missing MISTRAL_API_KEY in .env")
        
        # Directory settings
        self.uploads_directory = "./uploads"
        self.chroma_memory_dir = "./chroma_storage"
        self.chroma_kb_dir = "./kb_storage"
        
        # Model settings
        self.model_name = "mistral-small"
        self.temperature = 0.7
        
        # Chunking settings
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Retrieval settings
        self.memory_k = 5
        self.knowledge_k = 3

# Global config instance
config = Config()