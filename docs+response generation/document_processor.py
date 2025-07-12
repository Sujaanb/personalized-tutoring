from pathlib import Path
from typing import List, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DocumentProcessor:
    """Handles text extraction from PDF and TXT files."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF {pdf_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text content from a TXT file."""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return content
        except Exception as e:
            print(f"❌ Error reading TXT {txt_path}: {str(e)}")
            return ""
    
    def process_file(self, file_path: Path) -> Tuple[List[Document], bool]:
        """Process a single file and return documents and success status."""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            content = self.extract_text_from_pdf(str(file_path))
            file_type = "pdf"
        elif file_extension == '.txt':
            content = self.extract_text_from_txt(str(file_path))
            file_type = "txt"
        else:
            return [], False
        
        if not content:
            return [], False
        
        # Split into chunks
        split_docs = self.splitter.create_documents(
            [content], 
            metadatas=[{
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_type
            }]
        )
        
        return split_docs, True
    
    def get_files_by_type(self, directory: str, file_types: List[str] = None) -> List[Path]:
        """Get files of specified types from directory."""
        if file_types is None:
            file_types = ['.pdf', '.txt']
        
        directory_path = Path(directory)
        if not directory_path.exists():
            return []
        
        files = []
        for file_type in file_types:
            files.extend(directory_path.glob(f"*{file_type}"))
        
        return files