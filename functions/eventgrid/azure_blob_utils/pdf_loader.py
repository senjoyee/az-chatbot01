import tempfile
import fitz
import logging
from typing import List
from langchain_core.documents import Document
from .base_loader import BaseDocumentLoader, DocumentLoaderFactory

class PDFLoader(BaseDocumentLoader):
    def load(self, blob_name: str) -> List[Document]:
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(temp_file)
            temp_path = temp_file.name
            
        try:
            doc = fitz.open(temp_path)
            full_text = []
            
            # Combine all pages into one text string
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    full_text.append(text)
            
            # Create single document with combined text
            if full_text:
                metadata = {
                    'source': f"{self.container_name}/{blob_name}",
                    'total_pages': len(doc),
                    'file_type': 'pdf'
                }
                return [Document(page_content='\n'.join(full_text), metadata=metadata)]
            return []
            
        except Exception as e:
            logging.error(f"Error processing PDF {blob_name}: {str(e)}")
            raise
        finally:
            import os
            os.unlink(temp_path)

# Register the loader
DocumentLoaderFactory.register_loader('.pdf', PDFLoader)