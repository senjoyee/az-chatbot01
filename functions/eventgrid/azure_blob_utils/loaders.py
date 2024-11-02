import logging
import os
from pathlib import Path
from typing import List
import tempfile
from azure.storage.blob import BlobClient, ContainerClient
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

class EnhancedAzureBlobStorageLoader(BaseLoader):
    def __init__(self, conn_str: str, container: str, blob_name: str):
        self.conn_str = conn_str
        self.container = container
        self.blob = blob_name.strip()  # Ensure blob name is not empty

    def _get_file_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower()

    def _process_pdf(self, file_path: str) -> List[Document]:
        try:
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "Could not import pymupdf4llm package. "
                "Please install it with `pip install pymupdf4llm`."
            )
        
        # Extract full document text without page chunks
        md_content = pymupdf4llm.to_markdown(
            doc=file_path,
            page_chunks=False  # Changed to False to get full document
        )

        # Handle the single document case
        content = md_content[0] if isinstance(md_content, list) else md_content
        
        if isinstance(content, dict):
            text = content.get('text', '')
        else:
            text = content

        logging.info(f"Extracted {len(text)} characters from PDF: {self.blob}")
        
        return [
            Document(
                page_content=text,
                metadata={
                    'source': self.blob,
                    'file_path': file_path,
                }
            )
        ]

    def _process_docx(self, file_path: str) -> List[Document]:
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "Could not import docx2txt package. "
                "Please install it with `pip install docx2txt`."
            )
        
        text = docx2txt.process(file_path)
        logging.info(f"Extracted {len(text)} characters from DOCX: {self.blob}")
        
        return [
            Document(
                page_content=text,
                metadata={
                    'source': self.blob,
                    'file_path': file_path,
                    'format': 'docx'
                }
            )
        ]

    def _process_text(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        logging.info(f"Extracted {len(text)} characters from text file: {self.blob}")
        
        return [
            Document(
                page_content=text,
                metadata={
                    'source': self.blob,
                    'file_path': file_path,
                    'format': 'text'
                }
            )
        ]

    def load(self) -> List[Document]:
        client = BlobClient.from_connection_string(
            conn_str=self.conn_str,
            container_name=self.container,
            blob_name=self.blob
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.container}/{self.blob}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as file:
                blob_data = client.download_blob()
                blob_data.readinto(file)

            file_extension = self._get_file_extension(self.blob)
            
            if file_extension == '.pdf':
                documents = self._process_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                documents = self._process_docx(file_path)
            elif file_extension in ['.txt', '.text']:
                documents = self._process_text(file_path)
            else:
                documents = self._process_pdf(file_path)

            # Verify that each document has 'source' in metadata
            for idx, doc in enumerate(documents):
                if not doc.metadata.get('source'):
                    logging.error(f"Loaded document at index {idx} is missing 'source' in metadata")
                    doc.metadata['source'] = self.blob or "unknown"
                else:
                    logging.info(f"Loaded document {idx} has source: {doc.metadata['source']}")

            return documents

class EnhancedAzureBlobStorageContainerLoader(BaseLoader):
    def __init__(self, conn_str: str, container: str, prefix: str = ""):
        self.conn_str = conn_str
        self.container = container
        self.prefix = prefix

    def load(self) -> List[Document]:
        container = ContainerClient.from_connection_string(
            conn_str=self.conn_str, 
            container_name=self.container
        )
        
        docs = []
        blob_list = container.list_blobs(name_starts_with=self.prefix)
        
        for blob in blob_list:
            loader = EnhancedAzureBlobStorageLoader(
                self.conn_str,
                self.container,
                blob.name,
            )
            docs.extend(loader.load())
            
        return docs