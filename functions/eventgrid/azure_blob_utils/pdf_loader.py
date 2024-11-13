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
            
            for page in doc:
                # Get text blocks with layout analysis
                blocks = page.get_text("blocks")
                # Sort blocks top to bottom, left to right
                blocks.sort(key=lambda b: (-b[1], b[0]))
                
                # Process each text block
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        full_text.append(text)
            
            if full_text:
                # Join blocks with double newlines to preserve paragraph structure
                content = '\n\n'.join(full_text)
                metadata = {
                    'source': f"{self.container_name}/{blob_name}",
                    'total_pages': len(doc),
                    'file_type': 'pdf'
                }
                return [Document(page_content=content, metadata=metadata)]
            return []
            
        except Exception as e:
            logging.error(f"Error processing PDF {blob_name}: {str(e)}")
            raise
        finally:
            import os
            os.unlink(temp_path)

# Register the loader
DocumentLoaderFactory.register_loader('.pdf', PDFLoader)