from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from models.schemas import FileProcessingStatus
from models.enums import ProcessingStatus
from azure_storage import AzureStorageManager

logger = logging.getLogger(__name__)
router = APIRouter()
storage_manager = AzureStorageManager()

@router.get("/file_status/{file_name}")
async def get_file_status(file_name: str) -> FileProcessingStatus:
    status, error_message = await storage_manager.get_status(file_name)
    return FileProcessingStatus(
        status=status,
        file_name=file_name,
        error_message=error_message
    )

@router.get("/file_statuses")
async def get_file_statuses(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[ProcessingStatus] = None
):
    try:
        all_statuses = await storage_manager.get_all_statuses(status_filter)
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
        logger.error(f"Error getting file statuses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset_file_status/{filename}")
async def reset_file_status(filename: str):
    try:
        await storage_manager.reset_status(filename)
        return {"message": f"Successfully reset status for {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))