from typing import Optional, List
from fastapi import HTTPException

from models.enums import ProcessingStatus
from models.schemas import FileProcessingStatus, BlobEvent
from services.azure_storage import AzureStorageManager
from services.file_processor import FileProcessor
from services.document_service import DocumentService

class FileController:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.file_processor = FileProcessor()
        self.document_service = DocumentService()

    async def get_file_status(self, file_name: str) -> FileProcessingStatus:
        status, error_message = await self.storage_manager.get_status(file_name)
        return FileProcessingStatus(
            status=status,
            file_name=file_name,
            error_message=error_message
        )

    async def get_file_statuses(self, page: int, page_size: int, status_filter: Optional[ProcessingStatus] = None):
        try:
            all_statuses = await self.storage_manager.get_all_statuses(status_filter)
            all_statuses.sort(key=lambda x: x["last_updated"], reverse=True)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_statuses = all_statuses[start_idx:end_idx]
            return {
                "total": len(all_statuses),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(all_statuses) + page_size - 1) // page_size,
                "statuses": paginated_statuses
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def reset_file_status(self, filename: str):
        try:
            await self.storage_manager.reset_status(filename)
            return {"message": f"Successfully reset status for {filename}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
            import asyncio
            asyncio.create_task(self.file_processor.process_file_async(event))
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                start_time=datetime.utcnow()
            )
        except Exception as e:
            await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
            raise HTTPException(status_code=500, detail=str(e))