from typing import List
from pathlib import Path
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager

class UploadService:
    """Service for handling document uploads to the knowledge base."""
    
    def __init__(self, processor: DocumentProcessor, vector_manager: VectorStoreManager):
        self.processor = processor
        self.vector_manager = vector_manager
    
    def upload_documents(self, directory: str = "./uploads", file_types: List[str] = None) -> dict:
        """Process and upload documents of specified types."""
        if file_types is None:
            file_types = ['.pdf', '.txt']
        
        files = self.processor.get_files_by_type(directory, file_types)
        
        if not files:
            return {
                'success': False,
                'message': f"No files with extensions {file_types} found in {directory}",
                'processed_files': 0,
                'total_chunks': 0
            }
        
        all_documents = []
        processed_files = 0
        file_type_counts = {'.pdf': 0, '.txt': 0}
        
        for file_path in files:
            file_ext = file_path.suffix.lower()
            print(f"{'📄' if file_ext == '.pdf' else '📝'} Processing {file_ext.upper()}: {file_path.name}")
            
            documents, success = self.processor.process_file(file_path)
            
            if success:
                all_documents.extend(documents)
                processed_files += 1
                file_type_counts[file_ext] += 1
            else:
                print(f"⚠️ No text extracted from {file_path.name}")
        
        if all_documents:
            success = self.vector_manager.add_documents_to_knowledge_base(all_documents)
            if success:
                message = f"✅ Added {len(all_documents)} chunks from {processed_files} file(s) to the knowledge base."
                for ext, count in file_type_counts.items():
                    if count > 0:
                        message += f"\n   {'📄' if ext == '.pdf' else '📝'} {ext.upper()} files processed: {count}"
                
                return {
                    'success': True,
                    'message': message,
                    'processed_files': processed_files,
                    'total_chunks': len(all_documents),
                    'file_type_counts': file_type_counts
                }
        
        return {
            'success': False,
            'message': "⚠️ No valid files found or all were empty.",
            'processed_files': 0,
            'total_chunks': 0
        }
    
    def upload_pdf_files(self, directory: str = "./uploads") -> dict:
        """Upload only PDF files."""
        return self.upload_documents(directory, ['.pdf'])
    
    def upload_txt_files(self, directory: str = "./uploads") -> dict:
        """Upload only TXT files."""
        return self.upload_documents(directory, ['.txt'])
    
    def list_available_files(self, directory: str = "./uploads") -> dict:
        """List available files in the uploads directory."""
        uploads_dir = Path(directory)
        
        if not uploads_dir.exists():
            return {
                'exists': False,
                'message': f"{directory} directory does not exist",
                'pdf_files': [],
                'txt_files': []
            }
        
        pdf_files = list(uploads_dir.glob("*.pdf"))
        txt_files = list(uploads_dir.glob("*.txt"))
        
        return {
            'exists': True,
            'pdf_files': [f.name for f in pdf_files],
            'txt_files': [f.name for f in txt_files],
            'total_files': len(pdf_files) + len(txt_files)
        }