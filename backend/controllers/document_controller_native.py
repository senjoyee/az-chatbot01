"""
Document controller using native Azure AI Search SDK.
This replaces the langchain-based controller with direct SDK calls.
"""

from fastapi import HTTPException
from datetime import datetime
import asyncio
import logging
from typing import List

from models.schemas import BlobEvent, DocumentIn, FileProcessingStatus
from models.enums import ProcessingStatus
from services.document_service_native import DocumentServiceNative
from services.azure_storage import AzureStorageManager
from services.file_processor import FileProcessor

logger = logging.getLogger(__name__)

class DocumentControllerNative:
    """
    Document controller using native Azure AI Search SDK.
    Handles document operations using the native document service.
    """
    
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.file_processor = FileProcessor()
        # Use the native document service
        self.document_service = DocumentServiceNative()
    
    async def process_uploaded_files(self, event: BlobEvent):
        """
        Process a file that was previously uploaded.
        
        Args:
            event: The blob event containing the file name
            
        Returns:
            FileProcessingStatus: The status of the file processing
        """
        file_name = event.file_name
        can_process, error_message = await self.storage_manager.check_and_acquire_processing_lock(
            self.storage_manager.container_name, file_name)
        if not can_process:
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                error_message=error_message
            )
        try:
            # Start the processing task
            asyncio.create_task(self.file_processor.process_file_async(event))
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                start_time=datetime.utcnow()
            )
        except Exception as e:
            await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def index_documents(self, documents_in: List[DocumentIn]):
        """
        Index documents in the vector store.
        
        Args:
            documents_in: The documents to index
            
        Returns:
            dict: The result of the indexing operation
        """
        return await self.document_service.index_documents(documents_in)
        
    async def delete_document(self, filename: str):
        """
        Delete a document from the vector store.
        
        Args:
            filename: The name of the file to delete
            
        Returns:
            dict: The result of the deletion operation
        """
        return await self.document_service.delete_from_vector_store(filename)
