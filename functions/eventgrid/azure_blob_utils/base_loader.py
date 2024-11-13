from abc import ABC, abstractmethod
from typing import List
import logging
import os
from langchain_core.documents import Document
from azure.storage.blob import BlobServiceClient

class BaseDocumentLoader(ABC):
    def __init__(self, conn_string: str, container_name: str):
        self.conn_string = conn_string  # Store the connection string
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        self.container_name = container_name

    @abstractmethod
    def load(self, blob_name: str) -> List[Document]:
        """Load single document and return list of Document objects"""
        pass

    def load_from_container(self, prefix: str = "") -> List[Document]:
        """Load all supported documents from container with optional prefix"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            documents = []
            
            # List all blobs with given prefix
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            
            for blob in blob_list:
                try:
                    # Get file extension
                    extension = DocumentLoaderFactory.get_file_extension(blob.name)
                    
                    # Skip if extension not supported
                    if extension not in DocumentLoaderFactory._loaders:
                        logging.warning(f"Skipping unsupported file type: {blob.name}")
                        continue
                    
                    # Get appropriate loader for this file type using stored connection string
                    loader = DocumentLoaderFactory.get_loader(
                        file_extension=extension,
                        conn_string=self.conn_string,  # Use stored connection string
                        container_name=self.container_name
                    )
                    
                    # Load and append documents
                    docs = loader.load(blob.name)
                    documents.extend(docs)
                    logging.info(f"Successfully loaded {blob.name}")
                    
                except Exception as e:
                    logging.error(f"Error loading {blob.name}: {str(e)}")
                    continue
                    
            return documents
            
        except Exception as e:
            logging.error(f"Error loading from container: {str(e)}")
            raise

class DocumentLoaderFactory:
    _loaders = {}

    @classmethod
    def register_loader(cls, extension: str, loader_class):
        """Register a loader class for a file extension"""
        cls._loaders[extension] = loader_class

    @classmethod
    def get_loader(cls, file_extension: str, conn_string: str, container_name: str) -> BaseDocumentLoader:
        """Get appropriate loader instance for file extension"""
        loader_class = cls._loaders.get(file_extension.lower())
        if not loader_class:
            raise ValueError(f"No loader registered for extension: {file_extension}")
        return loader_class(conn_string, container_name)

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Extract file extension from filename"""
        return os.path.splitext(filename.lower())[1]