
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from azure.storage.blob import BlobServiceClient

class BaseDocumentLoader(ABC):
    def __init__(self, conn_string: str, container_name: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        self.container_name = container_name

    @abstractmethod
    def load(self, blob_name: str) -> List[Document]:
        """Load document and return list of Document objects"""
        pass

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