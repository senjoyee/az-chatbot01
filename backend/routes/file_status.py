from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional

from models.enums import ProcessingStatus
from models.schemas import FileProcessingStatus
from controllers.file_controller import FileController

router = APIRouter(tags=["file_status"])

@router.get("/file_status/{file_name}")
async def get_file_status(file_name: str, controller: FileController = Depends()) -> FileProcessingStatus:
    return await controller.get_file_status(file_name)

@router.get("/file_statuses")
async def get_file_statuses(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[ProcessingStatus] = None,
    controller: FileController = Depends()
):
    return await controller.get_file_statuses(page, page_size, status_filter)

@router.post("/reset_file_status/{filename}")
async def reset_file_status(filename: str, controller: FileController = Depends()):
    return await controller.reset_file_status(filename)