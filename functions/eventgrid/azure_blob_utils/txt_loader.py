
import logging
from typing import List
from langchain_core.documents import Document
from .base_loader import BaseDocumentLoader, DocumentLoaderFactory

class TxtLoader(BaseDocumentLoader):
    def load(self, blob_name: str) -> List[Document]:
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        try:
            # Download and decode text directly without tempfile
            blob_data = blob_client.download_blob()
            text = blob_data.content_as_text()
            
            if text.strip():
                metadata = {
                    'source': f"{self.container_name}/{blob_name}",
                    'file_type': 'txt'
                }
                return [Document(page_content=text, metadata=metadata)]
            return []
            
        except Exception as e:
            logging.error(f"Error processing TXT {blob_name}: {str(e)}")
            raise

# Register the loader
DocumentLoaderFactory.register_loader('.txt', TxtLoader)
DocumentLoaderFactory.register_loader('.md', TxtLoader)  # Also handle markdown files
DocumentLoaderFactory.register_loader('.log', TxtLoader)  # Also handle log files