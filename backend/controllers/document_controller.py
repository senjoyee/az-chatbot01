from fastapi import HTTPException
from datetime import datetime
import asyncio
import logging

from models.schemas import BlobEvent, DocumentIn, FileProcessingStatus
from models.enums import ProcessingStatus
from services.document_service import DocumentService
from services.azure_storage import AzureStorageManager
from services.file_processor import FileProcessor

logger = logging.getLogger(__name__)

class DocumentController:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.file_processor = FileProcessor()
        self.document_service = DocumentService()
    
    async def process_uploaded_files(self, event: BlobEvent):
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
    
    async def index_documents(self, documents_in):
        return await self.document_service.index_documents(documents_in)